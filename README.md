# Dynamic Inter-Layer Bridges for RoBERTa-Large (RTE)

This repo implements and evaluates **cross-layer communication mechanisms** on top of `roberta-large` for the GLUE RTE task. 
We test two families of *bridges* — **QKV** (classical cross-attention across layers) and **HDIM** (higher-dimensional token–pair messages) — along with **hybrids** that combine both. 

All runs are **deterministic**, with seeds fixed and PyTorch set to reproducible mode. Results are reported on RTE dev with no cherry-picking: every ablation, successful or collapsed, is logged.

> **Baseline:** plain `roberta-large` on RTE dev ≈ **86.6%**.  
> **This repo:** HDIM-only H9 reaches **87.36%** (↑), hybrids A8 (**86.28%**) and A3 (**85.92%**) are competitive.

## Contents

- [Model Overview](#model-overview)
- [Mathematical Formulation](#mathematical-formulation)
  - [1. Routing (select source layers)](#1-routing-select-source-layers)
  - [2. QKV cross-layer message](#2-qkv-cross-layer-message)
  - [3. HDIM (higher-dimensional) message](#3-hdim-higher-dimensional-message)
  - [4. Blending, normalization, and injection (pre-ffn)](#4-blending-normalization-and-injection-pre-ffn)
  - [5. Expressivity note](#5-expressivity-note)
- [Training Setup (defaults used in sweeps)](#training-setup-defaults-used-in-sweeps)
- [Reading Usage](#reading-usage)
- [Knobs To Tune](#knobs-to-tune)
- [Results — All 50 Ablations (RTE, RoBERTa-large with Cross-Layer Bridges)](#results--all-50-ablations-rte-roberta-large-with-cross-layer-bridges)
  - [Headline: Top 10 Runs](#headline-top-10-runs)
  - [1.1–1.10 (HDIM-only)](#11–110-hdim-only)
  - [1.11–1.20 (HDIM-only advanced)](#111–120-hdim-only-advanced)
  - [1.21–1.30 (HDIM-only: masks/router/gates)](#121–130-hdim-only-masksroutergates)
  - [2.1–2.10 (Hybrid QKV + HDIM, early-stop rules)](#21–210-hybrid-qkv--hdim-early-stop-rules)
  - [2.11–2.20 (Hybrid, with live usage metrics)](#211–220-hybrid-with-live-usage-metrics)
- [Reproducibility](#reproducibility)


## Model Overview
On late target layers, a learned **router** selects earlier source layers. Each routed layer receives two possible messages:

- **QKV bridge:** target queries attend to source keys/values (like normal attention, but *across layers*).
- **HDIM bridge:** a small MLP kernel computes nonlinear token–pair messages from source to target.

Per-layer gates blend them:
\[
\boxed{\text{add\_ctx}=\alpha^{(j)}_{\text{attn}}\cdot \text{ctx}_{\text{qkv}}+\alpha^{(j)}_{\text{hdim}}\cdot \text{msg}_{\text{hdim}}}
\]

This `add_ctx` is injected into the residual stream of the target layer, pre-FFN.


| Approach | What it does | Strengths | Caveats |
|---|---|---|---|
| **QKV bridge** | Target Q attends to source K/V across layers | Simple, strong baseline; reuses attention geometry | Fixed **bilinear** similarity |
| **HDIM bridge** | Nonlinear token-pair kernel via tiny MLPs | **More expressive** similarity/value fusion | Slightly higher compute; tune dims & gates |
| **Hybrid (QKV+HDIM)** | Compute both; blend with learned per-layer gates | Lets model **pick per layer**; robust | If one path is useless, gates suppress it (watch usage) |

---
# Mathematical Formulation

We augment a depth-$L$ transformer (RoBERTa-large) with **cross-layer bridges**. Let:
- $B$ = batch size, $S$ = sequence length, $H$ = hidden size,
- $h$ = number of attention heads, $d_h = H/h$ (assumed integer),
- $d_r$ = router projection size, $D$ = HDIM projection size.

For each layer $j\in\{0,\dots,L-1\}$, denote its token states by
\[
H_j \in \mathbb{R}^{B\times S\times H}, \qquad H_j[b,s,:] = \text{hidden of token } s.
\]
Let $M\in\{0,1\}^{B\times S}$ be the (1=keep) attention mask; we use the standard **extended mask**
\[
\mathrm{ExtMask}(M) \in \mathbb{R}^{B\times 1\times 1\times S}, \quad
\mathrm{ExtMask}(M)[b,1,1,t] =
\begin{cases}
0, & M[b,t]=1\\
-\infty, & M[b,t]=0.
\end{cases}
\]

We only **route** on the last $N$ layers (e.g., $N=4$): $j \in \{L-N,\dots,L-1\}$, and allow sources $i<j$.

---

## 1. Routing (select source layers)

Define a length-aware **pool** over tokens:
\[
\mathrm{pool}(H_j, M) =
\begin{cases}
\text{CLS}(H_j) = H_j[:,0,:] \in \mathbb{R}^{B\times H}, & \text{if CLS}\\[3pt]
\text{MaskedMean}(H_j,M) = \frac{\sum_{t} M[:,t]\, H_j[:,t,:]}{\sum_t M[:,t]} \in \mathbb{R}^{B\times H}, & \text{if mean.}
\end{cases}
\]
Project pooled summaries into a router space:
\[
q_j = W_q\, \mathrm{pool}(H_j,M) \in \mathbb{R}^{B\times d_r}, \qquad
k_i = W_k\, \mathrm{pool}(H_i,M) \in \mathbb{R}^{B\times d_r}.
\]
Compute **routing logits** (temperature $\tau>0$):
\[
\ell_{ij} = \frac{\langle q_j, k_i\rangle}{\tau} \in \mathbb{R}^{B}.
\]
Let $\mathrm{TopK}_j \subset \{0,\dots,j-1\}$ be the $k$ source indices with largest $\ell_{ij}$ (per-batch). Define **mixing weights** by a masked softmax **restricted to** the selected sources:
\[
w_{ij} =
\frac{\exp(\ell_{ij})\,\mathbf{1}[i\in \mathrm{TopK}_j]}{\sum_{u\in \mathrm{TopK}_j}\exp(\ell_{uj})}
\quad\text{for } i<j.
\]
(For $k=1$, $w_{ij}$ is 1 at the top source and 0 elsewhere.)

---

## 2. QKV cross-layer message

Let $\text{Attn}(Q,K,V,M)$ be standard scaled dot-product attention with mask $M$:
\[
\text{Attn}(Q,K,V,M) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_h}} + \mathrm{ExtMask}(M)\Big)\,V.
\]
Use **target queries** and **source keys/values**:
\[
Q_j = \mathrm{reshape\_heads}(H_j W_Q^{(j)}) \in \mathbb{R}^{B\times h\times S\times d_h},\quad
K_i = \mathrm{reshape\_heads}(H_i W_K^{(i)}),\quad
V_i = \mathrm{reshape\_heads}(H_i W_V^{(i)}).
\]
Per source $i\!\to\! j$:
\[
\mathrm{ctx}_{\text{qkv}}^{(i\to j)} = \mathrm{merge\_heads}\Big(\text{Attn}(Q_j,K_i,V_i,M)\Big) \in \mathbb{R}^{B\times S\times H}.
\]
Mix across the routed sources:
\[
\boxed{\;\mathrm{ctx}_{\text{qkv}} = \sum_{i<j} w_{ij}\, \mathrm{ctx}_{\text{qkv}}^{(i\to j)}\; } \qquad \in \mathbb{R}^{B\times S\times H}.
\]

---

## 3. HDIM (higher-dimensional) message

Per token, project to a compact space:
\[
Z_j = H_j P_j \in \mathbb{R}^{B\times S\times D},\qquad Z_i = H_i P_i \in \mathbb{R}^{B\times S\times D}.
\]
For each target token $s$ and source token $t$, build **pairwise features**
\[
\phi_{st} = \big[\, Z_j[s],\; Z_i[t],\; Z_j[s]\odot Z_i[t],\; |Z_j[s]-Z_i[t]| \,\big] \in \mathbb{R}^{4D}.
\]
A small MLP scorer $f_\theta:\mathbb{R}^{4D}\!\to\!\mathbb{R}$ produces **alignment logits**
\[
a_{st}^{(i\to j)} = f_\theta(\phi_{st}) \in \mathbb{R}.
\]
Apply a masked softmax over **source tokens** $t$:
\[
\alpha_{st}^{(i\to j)} = \frac{\exp\!\big(a_{st}^{(i\to j)} + \mathrm{MaskTok}(M,t)\big)}{\sum_{u}\exp\!\big(a_{su}^{(i\to j)} + \mathrm{MaskTok}(M,u)\big)},
\]
where $\mathrm{MaskTok}(M,t)=0$ if $M[:,t]=1$ else $-\infty$ (broadcast over $B$).

Pool the source hidden states (in the **original** $H$ space):
\[
\mathrm{ctx}_i[s] = \sum_{t=1}^{S} \alpha_{st}^{(i\to j)}\, H_i[t] \in \mathbb{R}^{H}.
\]
Fuse with the target via a value MLP $g_\phi:\mathbb{R}^{3H}\!\to\!\mathbb{R}^{H}$:
\[
\mathrm{msg}_{\text{hdim}}^{(i\to j)}[s] = g_\phi\!\big(\,[\;\mathrm{ctx}_i[s],\; H_j[s],\; \mathrm{ctx}_i[s]\odot H_j[s]\;]\,\big).
\]
Mix across routed sources:
\[
\boxed{\;\mathrm{msg}_{\text{hdim}} = \sum_{i<j} w_{ij}\, \mathrm{msg}_{\text{hdim}}^{(i\to j)}\; } \qquad \in \mathbb{R}^{B\times S\times H}.
\]

---

## 4. Blending, normalization, and injection (pre-FFN)

Let \(A_j=\mathrm{SA}^{(j)}(H_j)\) denote the **post-attention output** at layer \(j\).
Each routed layer \(j\) learns scalar gates \(\alpha^{(j)}_{\text{attn}},\alpha^{(j)}_{\text{hdim}}\in\mathbb{R}\) (initialized small, e.g., 0.15 and 0.05). Form:

\[
\mathrm{add\_ctx}_j \;=\; \alpha^{(j)}_{\text{attn}}\,\mathrm{ctx}_{\text{qkv}} \;+\; \alpha^{(j)}_{\text{hdim}}\,\mathrm{msg}_{\text{hdim}} \;\in \mathbb{R}^{B\times S\times H}.
\]

Apply per-layer LayerNorm and a near-zero linear projection (keeps init close to base model), then dropout:

\[
\widetilde{\mathrm{add\_ctx}}_j \;=\; \mathrm{Dropout}\!\big( W_{\text{proj}}^{(j)}(\mathrm{LN}^{(j)}(\mathrm{add\_ctx}_j)) \big).
\]

Inject **before** the FFN of layer \(j\):

\[
H_j^{\text{preFFN}} = A_j + \widetilde{\mathrm{add\_ctx}}_j,\qquad
H_{j+1}=\mathrm{FFN}^{(j)}(H_j^{\text{preFFN}}).
\]


---

## 5. Expressivity note

The QKV scorer is **bilinear** in $(q,k)$: $s_{ij} = q^\top k$.  
HDIM exposes elementwise products and absolute differences and passes them through an MLP $(f_\theta, g_\phi)$, i.e., a **learned nonlinear kernel** over $(Z_j,Z_i,H_j,H_i)$.  
Thus HDIM can subsume dot-product behaviors (by collapsing to a linear map) and represent nonlinear similarity rules that dot product cannot.

---

**Defaults used in our sweeps:** $k{=}1$, $N{=}4$, $d_r{=}128$, $D{=}24$; routing pool = mean (hybrids sometimes prefer CLS); $\tau{=}0.7$; gates initialized to small positive values.

---

## Training Setup (defaults used in sweeps)
- Backbone: `roberta-large`, MAX_LEN=128
- Optim: AdamW (encoder **1.5e-5**, head **3e-4**), weight decay 0.01
- Scheduler: linear with **10% warmup**
- Label smoothing: **0.05**
- Batch size: **8**
- Epochs: HDIM-only **3**; Hybrid **5** (early-stop if epoch-1 < 0.70)
- AMP + grad clip (1.0); reproducible seeds

---

## Reading Usage
Instrumented scripts log per-layer usage:
- **Gates**: `alpha_attn`, `alpha_hdim`  
- **Contribution norms**: `qkv_norm_mean`, `hdim_norm_mean`  
- **Routing histograms**: which sources were picked  
Interpretation: if norms/gates skew, that path dominates; if both non-trivial, hybridization is active.


---

## Knobs To Tune
- `topk` (1–2 was best here)  
- `route_last_n` (e.g. 4)  
- `route_dim` (128 baseline, 256 tested)  
- `route_pool` = `mean` or `cls`  
- `route_temp` (too sharp collapsed on RTE)  
- **HDIM fusion**: `concat_only` consistently strongest  
- **HDIM scorer**: 4-term MLP > bilinear  
- Gate inits (small but >0 so paths can grow)

---


# Results — All 50 Ablations (RTE, RoBERTa-large with Cross-Layer Bridges)

---

## Headline: Top 10 Runs

| Rank | ID | Brief | Best Acc |
| --- | --- | --- | ---: |
| 1 | **H9** | HDIM-only, 4-term, **val=concat_only**, mean routing | **0.8736** |
| 2 | **A8** | Hybrid (QKV+HDIM), **router CLS** | **0.8628** |
| 3 | **A3** | Hybrid, **QKV-only** | 0.8592 |
| 4 | H3 | HDIM-only, stronger start (gate=0.10) | 0.8484 |
| 5 | H27 | HDIM-only, **alpha softplus** | 0.8412 |
| 6 | H13 | HDIM-only, cosine + concat_only | 0.8412 |
| 7 | H8 | HDIM-only, pair=bilinear | 0.8375 |
| 8 | H10 | HDIM-only, share Zj/Zi projections | 0.8303 |
| 9 | H23 | HDIM-only, cross-only + concat_diff_hadam | 0.8303 |
|10 | H18 | HDIM-only, value residual | 0.8267 |

**Takeaway:** Router design and value fusion dominate. QKV-only is strong; HDIM-only wins when fusion is lean (`concat_only`) and gates are sane.

---

## 1.1–1.10 (HDIM-only)

| ID | Config highlight | Best Acc |
| --- | --- | ---: |
| H9_val_concat_only | **val=concat_only** | **0.8736** |
| H3_gate0p10 | init_gate_hdim=0.10 | 0.8484 |
| H8_pair_bilinear | pair=bilinear | 0.8375 |
| H10_share_proj | share projections p_i=p_j | 0.8303 |
| H1_proj32_mlp128 | proj=32, mlp=128 | 0.8159 |
| H6_last6 | route_last_n=6 | 0.7834 |
| H7_router_temp0p5 | route_temp=0.5 | 0.7040 |
| H2/H4/H5 | (small/proj/drop0/CLS pool) | 0.5271 |

---

## 1.11–1.20 (HDIM-only advanced)

| ID | Config highlight | Best Acc |
| --- | --- | ---: |
| H13_cosine_concat_only | cosine + **concat_only** | **0.8412** |
| H18_val_residual | residual on value path | 0.8267 |
| H14_gate0p10_drop0p05 | stronger gate + 0.05 drop | 0.8267 |
| H16_entreg_1e3 | ent-regularized alignments | 0.7545 |
| H15_last3 | route_last_n=3 | 0.7112 |
| H11/H12/H17/H19/H20 | others | 0.5271–0.8051 |

---

## 1.21–1.30 (HDIM-only: masks/router/gates)

| ID | Config highlight | Best Acc |
| --- | --- | ---: |
| H27_alpha_softplus | **positive (softplus) gate** | **0.8412** |
| H23_cross_only_concat_diff_hadam | cross-only + richer fusion | 0.8303 |
| H21_cross_only | cross sentence pairs only | 0.7690 |
| H28_val_gelu | GELU in value MLP | 0.7437 |
| H22/H24/H25/H26/H29/H30 | others | 0.5271–0.5704 |

---

## 2.1–2.10 (Hybrid QKV + HDIM, early-stop rules)

| ID | Config highlight | Best Acc | Notes |
| --- | --- | ---: | --- |
| **A8_Router_CLS** | **router CLS** | **0.8628** | best hybrid |
| **A3_QKV_only** | **QKV-only** | **0.8592** | strong baseline |
| A1/A2/A4/A5/A6/A7/A9/A10 | various | early-stop / <0.70 ep1 | see note |

---

## 2.11–2.20 (Hybrid, with live usage metrics)

| ID | Config highlight | Best Acc |
| --- | --- | ---: |
| Original | baseline hybrid (topk=1, lastN=4, mean) | **0.8592** |
| topk=2 | two sources | 0.8412 |
| topk=3 | three sources | 0.8375 |
| route_dim=256 | larger router key/query | 0.8303 |
| route_last_n=2 | fewer targets | – |
| route_last_n=6 | more targets | – |
| route_temp=0.5 | sharper router | – |
| route_pool=cls | CLS pooling | – |
| ablate_attn_bridge | disable QKV bridge (α_attn=0) | – |
| ablate_hdim_bridge | disable HDIM bridge (α_hdim=0) | – |

**Note:** A dash (–) = early-stop triggered after epoch 1 (<0.70), no meaningful best.

---

- **Worked:** lean HDIM fusion (`concat_only`), QKV-only, CLS router.  
- **Collapsed:** overly sharp routing (temp=0.5), over-constrained masks, HDIM sharing. 

---

## Reproducibility

Each script prints:
- per-epoch train loss/EMA/acc  
- validation accuracy/F1  
- (hybrid metrics script) **live bridge usage**: QKV/HDIM norms, gate values, source histograms.
