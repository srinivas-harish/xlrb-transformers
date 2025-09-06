#!/usr/bin/env python3
# RTE — HDIM-only ablations H21–H30 (3 epochs each), save stats only.
# mean-pool routing, lastN=4, topk=1, pre-FFN injection, dropout=0.1, init_gate_hdim=0.05.
# pip install -q transformers==4.44.2 datasets==2.21.0 scikit-learn==1.5.2 tqdm==4.66.4

import os, json, csv, math, time, random, numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
    get_scheduler,
    RobertaModel, RobertaConfig
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput

# -----------------------
# Reproducibility
# -----------------------
GLOBAL_SEED = 42
def set_all_seeds(seed: int):
    set_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_all_seeds(GLOBAL_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------
# Data
# -----------------------
raw = load_dataset("glue", "rte")
checkpoint = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
MAX_LEN = 128

def preprocess(batch):
    enc = tokenizer(batch["sentence1"], batch["sentence2"],
                    truncation=True, max_length=MAX_LEN)
    enc["labels"] = batch["label"]
    return enc

cols_to_remove = [c for c in raw["train"].column_names if c not in ("idx",)]
tokenized = raw.map(preprocess, batched=True, remove_columns=cols_to_remove)
keep_cols = ["input_ids", "attention_mask", "labels"]
tokenized = tokenized.with_format("torch", columns=keep_cols)

collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
BATCH_SIZE = 8
train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collator, num_workers=2, pin_memory=True)
val_loader   = DataLoader(tokenized["validation"], batch_size=32, shuffle=False,
                          collate_fn=collator, num_workers=2, pin_memory=True)

SEP_ID = tokenizer.sep_token_id
SPECIAL_IDS = set(tokenizer.all_special_ids)

# -----------------------
# Configs
# -----------------------
@dataclass
class BridgeCfg:
    # Router
    topk: int = 1
    route_dim: int = 128
    route_temp: float = 0.7        # smaller => sharper (since we divide by temp)
    route_pool: str = "mean"       # "mean" or "cls"
    route_last_n: int = 4
    router_augmented: bool = False # NEW: segment-aware router query

    # Injection & dropout
    dropout: float = 0.1
    inject_site: str = "pre_ffn"   # fixed to pre-FFN for HDIM-only
    tgt_last_m: int = 0            # NEW: if >0, only inject into last-m targets (sources still last_n)

    # HDIM gate
    init_gate_attn: float = 0.15   # irrelevant (QKV off)
    init_gate_hdim: float = 0.05
    alpha_positive: bool = False   # NEW: softplus gate if True

    # HDIM architecture
    hdim_proj_dim: int = 24
    hdim_mlp_hidden: int = 96
    hdim_pair_mode: str = "4term"        # "4term" | "bilinear" | "cosine"
    hdim_val_mode: str = "concat_only"   # BASELINE = concat_only (H9)
    hdim_share_projections: bool = False
    hdim_use_proj_ln: bool = False       # (kept False; you found it bad)
    hdim_val_residual: bool = False      # (off by default)
    hdim_val_residual_scale: float = 0.10
    val_activation: str = "relu"         # NEW: "relu" | "gelu"

    # Token alignment softmax
    hdim_token_temp: float = 1.0
    # Entropy regularization (kept off; you found it harmful)
    ent_reg_weight: float = 0.0

    # Pair masking
    hdim_cross_only: bool = False        # NEW: allow only cross-sentence pairs
    hdim_mask_specials: bool = False     # NEW: exclude special tokens from pairs
    sep_id: int = SEP_ID
    special_ids: Tuple[int, ...] = tuple(SPECIAL_IDS)

    # Sharing across target layers
    hdim_share_across_layers: bool = False

    # Norm match for injection
    norm_match: bool = False
    norm_match_scale: float = 1.0

    # Toggles
    use_qkv: bool = False
    use_hdim: bool = True
    per_head_gates: bool = False
    adapter_hidden: int = 256            # unused here

class GeneralizedCrossMessage(nn.Module):
    """
    Flexible HDIM message with:
      - Pair scoring: 4term-MLP | bilinear | cosine
      - Token softmax temperature (hdim_token_temp)
      - Pair masking (cross-only, no specials)
      - Value fusion: concat_only | concat_hadam | concat_diff_hadam
      - Optional residual fusion: + beta*(ctx - H)
      - Value MLP activation: ReLU or GELU
    """
    def __init__(self, hidden_size:int, cfg: BridgeCfg):
        super().__init__()
        self.H = hidden_size
        self.D = cfg.hdim_proj_dim
        self.cfg = cfg

        # projections
        self.p_j = nn.Linear(self.H, self.D, bias=False)
        if cfg.hdim_share_projections:
            self.p_i = self.p_j
        else:
            self.p_i = nn.Linear(self.H, self.D, bias=False)

        # scorer
        if cfg.hdim_pair_mode == "4term":
            in_feats = 4 * self.D
            self.score_mlp = nn.Sequential(
                nn.Linear(in_feats, cfg.hdim_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.hdim_mlp_hidden, 1)
            )
            self.W_bilinear = None
        elif cfg.hdim_pair_mode == "bilinear":
            self.W_bilinear = nn.Parameter(torch.empty(self.D, self.D))
            nn.init.xavier_uniform_(self.W_bilinear)
            self.score_mlp = None
        elif cfg.hdim_pair_mode == "cosine":
            self.W_bilinear = None
            self.score_mlp = None
        else:
            raise ValueError(f"Unknown hdim_pair_mode: {cfg.hdim_pair_mode}")

        # value fusion MLP
        if cfg.hdim_val_mode == "concat_only":
            val_in = 2*self.H  # [ctx, H]
        elif cfg.hdim_val_mode == "concat_hadam":
            val_in = 3*self.H  # [ctx, H, ctx*H]
        elif cfg.hdim_val_mode == "concat_diff_hadam":
            val_in = 4*self.H  # [ctx, H, |diff|, ctx*H]
        else:
            raise ValueError(f"Unknown hdim_val_mode: {cfg.hdim_val_mode}")

        out_hid = 2 * max(128, self.H // 2)
        act = nn.GELU() if cfg.val_activation.lower() == "gelu" else nn.ReLU(inplace=True)
        self.val_mlp = nn.Sequential(
            nn.Linear(val_in, out_hid),
            act,
            nn.Linear(out_hid, self.H)
        )

        self.drop = nn.Dropout(cfg.dropout)
        self.alpha = nn.Parameter(torch.tensor(float(cfg.init_gate_hdim)))
        self._ent_reg = None  # (unused; kept for completeness)

    def _expand_mask_last(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (1.0 - mask[:, None, :]) * torch.finfo(dtype).min

    def _pair_logits(self, Zj: torch.Tensor, Zi: torch.Tensor,
                     attn_mask: Optional[torch.Tensor],
                     pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, S, D = Zj.shape
        mode = self.cfg.hdim_pair_mode
        if mode == "4term":
            Zj_e = Zj.unsqueeze(2)               # [B,S,1,D]
            Zi_e = Zi.unsqueeze(1)               # [B,1,S,D]
            Zj_pair = Zj_e.expand(-1, S, S, -1)  # [B,S,S,D]
            Zi_pair = Zi_e.expand(-1, S, S, -1)
            hadam = Zj_pair * Zi_pair
            diff  = torch.abs(Zj_pair - Zi_pair)
            pair  = torch.cat([Zj_pair, Zi_pair, hadam, diff], dim=-1)   # [B,S,S,4D]
            logits = self.score_mlp(pair).squeeze(-1)                    # [B,S,S]
        elif mode == "bilinear":
            ZjW = torch.matmul(Zj, self.W_bilinear)      # [B,S,D]
            logits = torch.matmul(ZjW, Zi.transpose(1, 2))
        elif mode == "cosine":
            Zj_n = F.normalize(Zj, p=2, dim=-1)
            Zi_n = F.normalize(Zi, p=2, dim=-1)
            logits = torch.matmul(Zj_n, Zi_n.transpose(1, 2)) * 10.0
        else:
            raise RuntimeError

        if attn_mask is not None:
            logits = logits + self._expand_mask_last(attn_mask, logits.dtype)
        if pair_mask is not None:
            logits = logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)
        if self.cfg.hdim_token_temp != 1.0:
            logits = logits / max(1e-6, self.cfg.hdim_token_temp)
        return logits

    def forward(self, H_j: torch.Tensor, H_i: torch.Tensor,
                attn_mask: Optional[torch.Tensor],
                pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, S, H = H_j.shape
        Zj = self.p_j(H_j)
        Zi = self.p_i(H_i)

        logits = self._pair_logits(Zj, Zi, attn_mask, pair_mask)
        probs  = F.softmax(logits, dim=-1)              # [B,S,S]
        ctx_i  = torch.matmul(probs, H_i)               # [B,S,H]

        # value fusion
        if self.cfg.hdim_val_mode == "concat_only":
            msg_in = torch.cat([ctx_i, H_j], dim=-1)
        elif self.cfg.hdim_val_mode == "concat_hadam":
            msg_in = torch.cat([ctx_i, H_j, ctx_i * H_j], dim=-1)
        elif self.cfg.hdim_val_mode == "concat_diff_hadam":
            msg_in = torch.cat([ctx_i, H_j, torch.abs(ctx_i - H_j), ctx_i * H_j], dim=-1)
        else:
            raise RuntimeError

        msg = self.val_mlp(msg_in)                      # [B,S,H]

        # optional residual fusion
        if self.cfg.hdim_val_residual:
            msg = msg + self.cfg.hdim_val_residual_scale * (ctx_i - H_j)

        return self.alpha * self.drop(msg)

# -----------------------
# Utilities for pair masks (segments & specials)
# -----------------------
def build_pair_mask(input_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    sep_id: int,
                    special_ids: set,
                    cross_only: bool,
                    mask_specials: bool) -> Optional[torch.Tensor]:
    """
    Returns [B,S,S] boolean mask of allowed pairs (True=allow), or None if no constraints.
    Always respects attention_mask (no pad pairs).
    """
    if not cross_only and not mask_specials:
        # Only padding will be suppressed later via attention mask; return None and let attn_mask handle it
        B = input_ids.size(0)
        return None

    B, S = input_ids.shape
    aid = input_ids.cpu().numpy()
    amask = attention_mask.bool().cpu().numpy()

    pm = np.ones((B, S, S), dtype=bool)
    # base: disallow padding pairs
    for b in range(B):
        valid = amask[b]  # [S]
        pm[b] = np.logical_and.outer(valid, valid)

    # segment masks
    if cross_only or mask_specials:
        A = np.zeros((B, S), dtype=bool)
        Bm = np.zeros((B, S), dtype=bool)
        specials = np.zeros((B, S), dtype=bool)
        for b in range(B):
            ids = aid[b]
            seps = np.where(ids == sep_id)[0]
            # Robust fallback if unexpected formatting
            if len(seps) >= 2:
                sep0, sep1 = int(seps[0]), int(seps[1])
                last_sep = int(seps[-1])
                a_start, a_end = 1, max(1, sep0) - 1
                b_start, b_end = min(S-1, sep1 + 1), max(sep1 + 1, last_sep) - 1
                if a_end >= a_start:
                    A[b, a_start:a_end+1] = True
                if b_end >= b_start:
                    Bm[b, b_start:b_end+1] = True
            # mark specials
            specials[b] = np.isin(ids, list(special_ids))
        if cross_only:
            cross = np.logical_or(np.logical_and.outer(A, Bm),
                                  np.logical_and.outer(Bm, A))
            pm = np.logical_and(pm, cross)
        if mask_specials:
            keep = np.logical_not(specials)
            nospec = np.logical_and.outer(keep, keep)
            pm = np.logical_and(pm, nospec)

    pm_t = torch.from_numpy(pm)
    return pm_t

# -----------------------
# Encoder (HDIM-only)
# -----------------------
class RobertaHybridEncoder(nn.Module):
    def __init__(self, model_name: str, cfg: BridgeCfg):
        super().__init__()
        self.cfg = cfg
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.config: RobertaConfig = self.roberta.config
        self.embeddings = self.roberta.embeddings
        self.layers = nn.ModuleList([self.roberta.encoder.layer[i] for i in range(self.config.num_hidden_layers)])
        self.L = self.config.num_hidden_layers
        H = self.config.hidden_size

        # Router
        self.route_q = nn.Linear(H, cfg.route_dim)
        self.route_k = nn.Linear(H, cfg.route_dim)
        if cfg.router_augmented:
            self.route_q_mlp = nn.Sequential(
                nn.Linear(4*H, 2*H),
                nn.ReLU(inplace=True),
                nn.Linear(2*H, cfg.route_dim)
            )
        else:
            self.route_q_mlp = None

        # Near-zero projection for injected context
        self.proj_by_tgt = nn.ModuleList([nn.Linear(H, H) for _ in range(self.L)])
        for lin in self.proj_by_tgt:
            nn.init.normal_(lin.weight, mean=0.0, std=1e-3); nn.init.zeros_(lin.bias)

        self.pre_ln_by_tgt = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.L)])

        # HDIM modules per target
        if cfg.hdim_share_across_layers:
            shared = GeneralizedCrossMessage(H, cfg)
            self.hdim_by_tgt = nn.ModuleList([shared for _ in range(self.L)])
        else:
            self.hdim_by_tgt = nn.ModuleList([GeneralizedCrossMessage(H, cfg) for _ in range(self.L)])

        # Gates (QKV disabled)
        self.alpha_attn = nn.Parameter(torch.zeros(self.L), requires_grad=False)
        self.alpha_hdim = nn.Parameter(torch.full((self.L,), float(cfg.init_gate_hdim)))
        self.drop = nn.Dropout(cfg.dropout)

    def _extend_mask(self, attention_mask: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
        if attention_mask is None: return None
        ext = attention_mask[:, None, None, :]
        return (1.0 - ext) * torch.finfo(dtype).min

    def _layer_summary(self, H_l: torch.Tensor, pool: str, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pool == "cls":
            return H_l[:, 0]
        if attn_mask is None:
            return H_l.mean(dim=1)
        m = attn_mask.float(); denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (H_l * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, input_ids, attention_mask=None, **_):
        emb = self.embeddings(input_ids=input_ids,
                              token_type_ids=torch.zeros_like(input_ids),
                              position_ids=None)
        dtype = emb.dtype
        device = emb.device
        ext_mask = self._extend_mask(attention_mask, dtype) if attention_mask is not None else None

        # Build pair mask once (B,S,S)
        pair_mask = build_pair_mask(
            input_ids, attention_mask,
            sep_id=self.cfg.sep_id,
            special_ids=set(self.cfg.special_ids),
            cross_only=self.cfg.hdim_cross_only,
            mask_specials=self.cfg.hdim_mask_specials
        )
        if pair_mask is not None:
            pair_mask = pair_mask.to(device)

        # segment means (for router_augmented)
        def segment_means(hidden_states, attn_mask, inp_ids):
            with torch.no_grad():
                B, S, H = hidden_states.shape
                aid = inp_ids.cpu().numpy()
                am = attn_mask.bool().cpu().numpy()
                mA = torch.zeros(B, H, device=hidden_states.device)
                mB = torch.zeros(B, H, device=hidden_states.device)
                for b in range(B):
                    ids = aid[b]; seps = np.where(ids == self.cfg.sep_id)[0]
                    # defaults = full mean split in half if missing seps
                    if len(seps) >= 2:
                        sep0, sep1 = int(seps[0]), int(seps[1])
                        last_sep = int(seps[-1])
                        a_start, a_end = 1, max(1, sep0) - 1
                        b_start, b_end = min(S-1, sep1 + 1), max(sep1 + 1, last_sep) - 1
                        if a_end >= a_start:
                            A = hidden_states[b, a_start:a_end+1, :].mean(dim=0)
                        else:
                            A = hidden_states[b].mean(dim=0)
                        if b_end >= b_start:
                            Bv = hidden_states[b, b_start:b_end+1, :].mean(dim=0)
                        else:
                            Bv = hidden_states[b].mean(dim=0)
                    else:
                        A = hidden_states[b].mean(dim=0)
                        Bv = hidden_states[b].mean(dim=0)
                    mA[b] = A; mB[b] = Bv
                return mA, mB

        hidden_states = emb
        layer_out_cache: Dict[int, torch.Tensor] = {}
        layer_sum_cache: Dict[int, torch.Tensor] = {}

        for li in range(self.L):
            layer = self.layers[li]
            sa_outs = layer.attention(hidden_states, attention_mask=ext_mask, output_attentions=False)
            attn_out = sa_outs[0]

            # routing decision only on last-N targets
            apply_routing = li > 0 and (li >= self.L - self.cfg.route_last_n)
            add_ctx = None
            if apply_routing:
                H_j_in = hidden_states            # [B,S,H]
                B, S, H = H_j_in.shape
                # router key/query
                if self.cfg.router_augmented:
                    mA, mB = segment_means(H_j_in, attention_mask, input_ids)
                    feats = torch.cat([mA, mB, torch.abs(mA-mB), mA*mB], dim=-1)  # [B,4H]
                    q_r = self.route_q_mlp(feats)                                  # [B,Dr]
                else:
                    q_sum = self._layer_summary(H_j_in, self.cfg.route_pool, attention_mask)
                    q_r = self.route_q(q_sum)

                src_indices = list(range(li))
                if len(src_indices) > 0:
                    k_list = []
                    for i in src_indices:
                        if i not in layer_sum_cache:
                            H_i_out = layer_out_cache[i]
                            layer_sum_cache[i] = self._layer_summary(H_i_out, self.cfg.route_pool, attention_mask)
                        k_list.append(self.route_k(layer_sum_cache[i]))
                    K_route = torch.stack(k_list, dim=1)                                 # [B,num_src,Dr]
                    logits_r = torch.einsum("bd,bnd->bn", q_r, K_route) / max(1e-6, self.cfg.route_temp)
                    k = min(self.cfg.topk, logits_r.size(1))
                    top_vals, top_idx = torch.topk(logits_r, k=k, dim=1)                 # [B,k]
                    weights = torch.softmax(top_vals, dim=1)

                    # Gather H_i for selected sources
                    H_stack = torch.stack([layer_out_cache[i] for i in src_indices], dim=1)  # [B,num_src,S,H]
                    idx_H = top_idx.view(B, k, 1, 1).expand(B, k, S, H)
                    H_sel = torch.take_along_dim(H_stack, idx_H, dim=1)                       # [B,k,S,H]
                    msg_hdim = torch.zeros_like(H_j_in)
                    for sel in range(k):
                        msg = self.hdim_by_tgt[li](H_j_in, H_sel[:, sel, :, :], attention_mask, pair_mask)
                        msg_hdim = msg_hdim + weights[:, sel].view(B,1,1) * msg

                    # layer gate (optionally constrained positive)
                    if self.cfg.alpha_positive:
                        alpha_layer = F.softplus(self.alpha_hdim[li])
                    else:
                        alpha_layer = self.alpha_hdim[li]

                    add_ctx = alpha_layer * msg_hdim
                    add_ctx = self.pre_ln_by_tgt[li](add_ctx)
                    add_ctx = self.proj_by_tgt[li](add_ctx)
                    add_ctx = self.drop(add_ctx)

            # optional target-injection mask (inject only into last-m)
            inject_here = True
            if self.cfg.tgt_last_m > 0:
                inject_here = (li >= self.L - self.cfg.tgt_last_m)

            # Norm match if enabled
            if add_ctx is not None and inject_here and self.cfg.norm_match:
                # RMS scale match to attn_out
                rms_out = torch.sqrt(torch.mean(attn_out.pow(2)) + 1e-12)
                rms_add = torch.sqrt(torch.mean(add_ctx.pow(2)) + 1e-12)
                scale = (rms_out / rms_add) * self.cfg.norm_match_scale
                add_ctx = add_ctx * scale

            if add_ctx is not None and inject_here and self.cfg.inject_site == "pre_ffn":
                attn_out = attn_out + add_ctx

            inter = layer.intermediate(attn_out)
            layer_output = layer.output(inter, attn_out)
            hidden_states = layer_output
            layer_out_cache[li] = hidden_states

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states[:,0],
        )

# -----------------------
# Classifier wrapper
# -----------------------
class CLSHead(nn.Module):
    def __init__(self, in_dim:int, n_classes:int=2, p:float=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, max(128, in_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(max(128, in_dim // 2), n_classes)
        )
    def forward(self, x): return self.net(x)

class RobertaHybridForSequenceClassification(nn.Module):
    def __init__(self, model_name:str, cfg:BridgeCfg, num_labels:int=2, dropout:float=0.1):
        super().__init__()
        self.encoder = RobertaHybridEncoder(model_name, cfg)
        H = self.encoder.config.hidden_size
        self.head = CLSHead(H, n_classes=num_labels, p=dropout)
    def gradient_checkpointing_enable(self):
        self.encoder.roberta.gradient_checkpointing_enable()
    def forward(self, input_ids, attention_mask=None, labels=None):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = enc_out.last_hidden_state[:, 0]
        logits = self.head(cls)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(logits=logits, loss=loss)

# -----------------------
# Training config
# -----------------------
LR_ENCODER = 1.5e-5
LR_HEAD    = 3e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_FRAC = 0.10
LABEL_SMOOTH = 0.05
MAX_GRAD_NORM = 1.0

scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
ce_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

def split_params(mod: nn.Module):
    decay, no_decay = [], []
    for n, p in mod.named_parameters():
        if not p.requires_grad: continue
        if any(nd in n for nd in ["bias", "LayerNorm.weight"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay

@torch.no_grad()
def evaluate(model: nn.Module, loader) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
    y_pred = np.concatenate(all_preds); y_true = np.concatenate(all_labels)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}

def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_names = ["roberta.embeddings", "roberta.encoder.layer"]
    base = 0
    for n,p in model.named_parameters():
        if any(n.startswith(bn) for bn in base_names):
            base += p.numel()
    extra = total - base
    return {"total": total, "trainable": trainable, "approx_base": base, "approx_extra": extra}

@torch.no_grad()
def gate_summaries(enc: RobertaHybridEncoder) -> Dict[str, Dict]:
    hdim = enc.alpha_hdim.detach().float().cpu().numpy()
    return {
        "alpha_hdim": {
            "shape": list(hdim.shape),
            "mean": float(hdim.mean()), "std": float(hdim.std()),
            "min": float(hdim.min()), "max": float(hdim.max())
        }
    }

def build_model(cfg: BridgeCfg) -> RobertaHybridForSequenceClassification:
    model = RobertaHybridForSequenceClassification(checkpoint, cfg, num_labels=2, dropout=0.1)
    model.gradient_checkpointing_enable()
    model.to(device)
    return model

def build_optim_sched(model: nn.Module, num_training_steps: int):
    enc_decay, enc_no_decay = split_params(model.encoder)
    head_decay, head_no_decay = split_params(model.head)
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_decay, "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY},
            {"params": enc_no_decay, "lr": LR_ENCODER, "weight_decay": 0.0},
            {"params": head_decay, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
            {"params": head_no_decay, "lr": LR_HEAD, "weight_decay": 0.0},
        ]
    )
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(WARMUP_FRAC * num_training_steps),
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler

# -----------------------
# Base HDIM-only cfg = H9
# -----------------------
def base_cfg() -> BridgeCfg:
    return BridgeCfg(
        topk=1,
        route_dim=128,
        route_temp=0.7,
        route_pool="mean",
        route_last_n=4,
        router_augmented=False,
        dropout=0.1,
        inject_site="pre_ffn",
        tgt_last_m=0,
        init_gate_attn=0.15,
        init_gate_hdim=0.05,
        alpha_positive=False,
        hdim_proj_dim=24,
        hdim_mlp_hidden=96,
        hdim_pair_mode="4term",
        hdim_val_mode="concat_only",        # <-- H9 best
        hdim_share_projections=False,
        hdim_use_proj_ln=False,
        hdim_val_residual=False,
        hdim_val_residual_scale=0.10,
        val_activation="relu",
        hdim_token_temp=1.0,
        ent_reg_weight=0.0,
        hdim_cross_only=False,
        hdim_mask_specials=False,
        sep_id=SEP_ID,
        special_ids=tuple(SPECIAL_IDS),
        hdim_share_across_layers=False,
        norm_match=False,
        norm_match_scale=1.0,
        use_qkv=False,
        use_hdim=True,
        per_head_gates=False,
        adapter_hidden=256
    )

# -----------------------
# H21–H30 ablations (each flips one idea)
# -----------------------
ABLATIONS = [
    # 21) Cross-sentence pairs only
    {"name":"H21_cross_only","desc":"pair mask: cross-sentence only",
     "apply": lambda c: setattr(c,"hdim_cross_only",True)},
    # 22) Cross-only + exclude specials
    {"name":"H22_cross_only_no_specials","desc":"cross-only & exclude special tokens",
     "apply": lambda c: (setattr(c,"hdim_cross_only",True), setattr(c,"hdim_mask_specials",True))},
    # 23) Cross-only + richer fusion
    {"name":"H23_cross_only_concat_diff_hadam","desc":"cross-only & val=concat_diff_hadam",
     "apply": lambda c: (setattr(c,"hdim_cross_only",True), setattr(c,"hdim_val_mode","concat_diff_hadam"))},
    # 24) Segment-aware router query
    {"name":"H24_router_augmented","desc":"router_augmented=True",
     "apply": lambda c: setattr(c,"router_augmented",True)},
    # 25) Inject only into last-2 targets (sources remain last-4)
    {"name":"H25_tgt_last2","desc":"tgt_last_m=2",
     "apply": lambda c: setattr(c,"tgt_last_m",2)},
    # 26) Norm-match the injected message (0.5 scale)
    {"name":"H26_norm_match0p5","desc":"norm_match=True, scale=0.5",
     "apply": lambda c: (setattr(c,"norm_match",True), setattr(c,"norm_match_scale",0.5))},
    # 27) Positive layer gate (softplus)
    {"name":"H27_alpha_softplus","desc":"alpha_positive=True",
     "apply": lambda c: setattr(c,"alpha_positive",True)},
    # 28) GELU in value MLP
    {"name":"H28_val_gelu","desc":"val_activation=gelu",
     "apply": lambda c: setattr(c,"val_activation","gelu")},
    # 29) Slightly softer token alignments
    {"name":"H29_toktemp1p05","desc":"hdim_token_temp=1.05",
     "apply": lambda c: setattr(c,"hdim_token_temp",1.05)},
    # 30) Share HDIM weights across targets
    {"name":"H30_share_across_layers","desc":"hdim_share_across_layers=True",
     "apply": lambda c: setattr(c,"hdim_share_across_layers",True)},
]

# -----------------------
# Save-only stats paths
# -----------------------
os.makedirs("stats_hdim2130", exist_ok=True)
json_path = os.path.join("stats_hdim2130", "hdim2130_stats.json")
csv_path  = os.path.join("stats_hdim2130", "hdim2130_stats.csv")

def run_experiment(exp_name: str, exp_desc: str, cfg: BridgeCfg) -> Dict:
    set_all_seeds(GLOBAL_SEED)
    model = build_model(cfg)
    num_update_steps_per_epoch = math.ceil(len(train_loader))
    num_training_steps = NUM_EPOCHS * num_update_steps_per_epoch
    optimizer, scheduler = build_optim_sched(model, num_training_steps)

    pcounts = count_params(model)
    static = {
        "exp_name": exp_name,
        "exp_desc": exp_desc,
        "seed": GLOBAL_SEED,
        "cfg": asdict(cfg),
        "param_counts": pcounts,
        "tokenizer": checkpoint,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "train_steps_per_epoch": num_update_steps_per_epoch,
        "total_training_steps": num_training_steps,
        "hparams": {
            "lr_encoder": LR_ENCODER, "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY, "epochs": NUM_EPOCHS,
            "warmup_frac": WARMUP_FRAC, "label_smoothing": LABEL_SMOOTH,
            "max_grad_norm": MAX_GRAD_NORM
        }
    }

    epoch_logs = []
    best = {"val_acc": -1.0, "epoch": -1}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        start = time.time()
        ema_loss = None
        correct = 0
        seen = 0

        # capture LR
        lr_enc = None
        for g in optimizer.param_groups:
            if "lr" in g:
                lr_enc = g["lr"]; break

        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"[{exp_name}] Epoch {epoch}/{NUM_EPOCHS}", dynamic_ncols=True, leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = out.logits
                loss = ce_loss(logits, batch["labels"])

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            seen += batch["labels"].size(0)
            acc_running = correct / max(1, seen)

            loss_val = float(loss.item())
            ema_loss = loss_val if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_val

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "ema": f"{ema_loss:.4f}", "acc": f"{acc_running:.4f}"})

        train_acc_epoch = correct / max(1, seen)
        val_metrics = evaluate(model, val_loader)
        elapsed = time.time() - start

        gates = gate_summaries(model.encoder)

        epoch_logs.append({
            "epoch": epoch,
            "time_sec": round(elapsed, 2),
            "train_acc": float(train_acc_epoch),
            "train_loss_ema": float(ema_loss) if ema_loss is not None else None,
            "val_acc": float(val_metrics["accuracy"]),
            "val_f1_macro": float(val_metrics["f1_macro"]),
            "lr": float(lr_enc) if lr_enc is not None else None,
            "gates": gates
        })

        if val_metrics["accuracy"] > best["val_acc"]:
            best["val_acc"] = float(val_metrics["accuracy"])
            best["epoch"] = epoch

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    final = {**static, "epochs": epoch_logs, "best": best}
    return final

# -----------------------
# Run H21–H30
# -----------------------
all_results: List[Dict] = []
for ab in ABLATIONS:
    cfg = base_cfg()
    ab["apply"](cfg)
    # enforce HDIM-only invariants
    cfg.use_qkv = False; cfg.use_hdim = True; cfg.inject_site = "pre_ffn"
    print(f"Running {ab['name']}: {ab['desc']}")
    result = run_experiment(ab["name"], ab["desc"], cfg)
    all_results.append(result)
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# Save JSON
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved detailed JSON stats to {json_path}")

# Save CSV
csv_fields = [
    "exp_name", "exp_desc", "epoch", "time_sec",
    "train_acc", "train_loss_ema", "val_acc", "val_f1_macro",
    "lr",
    "alpha_hdim_shape", "alpha_hdim_mean", "alpha_hdim_std", "alpha_hdim_min", "alpha_hdim_max",
    "param_total", "param_trainable", "param_base_approx", "param_extra_approx",
    "topk", "route_temp", "route_pool", "route_last_n", "router_augmented",
    "tgt_last_m", "route_dim",
    "hdim_proj_dim", "hdim_mlp_hidden", "hdim_pair_mode", "hdim_val_mode",
    "val_activation",
    "hdim_share_projections", "hdim_use_proj_ln", "hdim_val_residual", "hdim_val_residual_scale",
    "hdim_token_temp", "hdim_cross_only", "hdim_mask_specials",
    "alpha_positive", "hdim_share_across_layers", "norm_match", "norm_match_scale",
]
os.makedirs("stats_hdim2130", exist_ok=True)
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for res in all_results:
        base = {
            "exp_name": res["exp_name"], "exp_desc": res["exp_desc"],
            "param_total": res["param_counts"]["total"],
            "param_trainable": res["param_counts"]["trainable"],
            "param_base_approx": res["param_counts"]["approx_base"],
            "param_extra_approx": res["param_counts"]["approx_extra"],
            "topk": res["cfg"]["topk"], "route_temp": res["cfg"]["route_temp"],
            "route_pool": res["cfg"]["route_pool"], "route_last_n": res["cfg"]["route_last_n"],
            "router_augmented": res["cfg"]["router_augmented"],
            "tgt_last_m": res["cfg"]["tgt_last_m"],
            "route_dim": res["cfg"]["route_dim"],
            "hdim_proj_dim": res["cfg"]["hdim_proj_dim"], "hdim_mlp_hidden": res["cfg"]["hdim_mlp_hidden"],
            "hdim_pair_mode": res["cfg"]["hdim_pair_mode"], "hdim_val_mode": res["cfg"]["hdim_val_mode"],
            "val_activation": res["cfg"]["val_activation"],
            "hdim_share_projections": res["cfg"]["hdim_share_projections"],
            "hdim_use_proj_ln": res["cfg"]["hdim_use_proj_ln"],
            "hdim_val_residual": res["cfg"]["hdim_val_residual"],
            "hdim_val_residual_scale": res["cfg"]["hdim_val_residual_scale"],
            "hdim_token_temp": res["cfg"]["hdim_token_temp"],
            "hdim_cross_only": res["cfg"]["hdim_cross_only"],
            "hdim_mask_specials": res["cfg"]["hdim_mask_specials"],
            "alpha_positive": res["cfg"]["alpha_positive"],
            "hdim_share_across_layers": res["cfg"]["hdim_share_across_layers"],
            "norm_match": res["cfg"]["norm_match"],
            "norm_match_scale": res["cfg"]["norm_match_scale"],
        }
        for ep in res["epochs"]:
            row = dict(base)
            row.update({
                "epoch": ep["epoch"], "time_sec": ep["time_sec"],
                "train_acc": ep["train_acc"], "train_loss_ema": ep["train_loss_ema"],
                "val_acc": ep["val_acc"], "val_f1_macro": ep["val_f1_macro"],
                "lr": ep["lr"],
                "alpha_hdim_shape": str(ep["gates"]["alpha_hdim"]["shape"]),
                "alpha_hdim_mean": ep["gates"]["alpha_hdim"]["mean"],
                "alpha_hdim_std": ep["gates"]["alpha_hdim"]["std"],
                "alpha_hdim_min": ep["gates"]["alpha_hdim"]["min"],
                "alpha_hdim_max": ep["gates"]["alpha_hdim"]["max"],
            })
            writer.writerow(row)

print(f"Saved per-epoch CSV table to {csv_path}")

# Summary
for res in all_results:
    print(f"{res['exp_name']} | best_acc={res['best']['val_acc']:.4f} at epoch {res['best']['epoch']} "
          f"| val={res['cfg']['hdim_val_mode']} | pair={res['cfg']['hdim_pair_mode']} "
          f"| crossOnly={res['cfg']['hdim_cross_only']}| noSpec={res['cfg']['hdim_mask_specials']} "
          f"| routerAug={res['cfg']['router_augmented']} | tgtLastM={res['cfg']['tgt_last_m']} "
          f"| ttemp={res['cfg']['hdim_token_temp']} | act={res['cfg']['val_activation']} "
          f"| shareAcross={res['cfg']['hdim_share_across_layers']} | normMatch={res['cfg']['norm_match']}")
