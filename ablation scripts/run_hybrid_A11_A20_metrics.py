 
# RoBERTa-large on GLUE RTE — Hybrid bridges (QKV + lightweight HDIM) WITH LIVE + EPOCH-END USAGE METRICS
# - Trainer-free loop (fp16, clip, scheduler), NO warm starts
# - TQDM shows live per-step usage; after each epoch we print aggregated distribution
# - Memory-safe defaults: last-4 routing, topk=1, MAX_LEN=128

# %%capture
#!pip install -q transformers==4.44.2 datasets==2.21.0 scikit-learn==1.5.2 tqdm==4.66.4

# %%
import os, math, random, numpy as np, torch, time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
    get_scheduler,
    RobertaModel, RobertaConfig
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput
)

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------
# Data
# -----------------------
raw = load_dataset("glue", "rte")

checkpoint = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

MAX_LEN = 128  # keep S^2 in check for HDIM
def preprocess(batch):
    enc = tokenizer(
        batch["sentence1"], batch["sentence2"],
        truncation=True, max_length=MAX_LEN,
    )
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

# -----------------------
# Lightweight HDIM message
# -----------------------
class GeneralizedCrossMessage(nn.Module):
    def __init__(self, hidden_size:int, proj_dim:int=24, mlp_hidden:int=96, dropout:float=0.1, gate_init:float=0.05):
        super().__init__()
        self.H = hidden_size; self.D = proj_dim
        self.p_j = nn.Linear(self.H, self.D, bias=False)
        self.p_i = nn.Linear(self.H, self.D, bias=False)
        self.score_mlp = nn.Sequential(
            nn.Linear(4*self.D, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, 1)
        )
        out_hid = 2 * max(128, self.H // 2)
        self.val_mlp = nn.Sequential(
            nn.Linear(3*self.H, out_hid), nn.ReLU(inplace=True), nn.Linear(out_hid, self.H)
        )
        self.drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(float(gate_init)))  # small >0

    def _expand_mask_last(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (1.0 - mask[:, None, :]) * torch.finfo(dtype).min

    def forward(self, H_j: torch.Tensor, H_i: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, S, H = H_j.shape
        Zj = self.p_j(H_j); Zi = self.p_i(H_i)       # [B,S,D]
        Zj_e = Zj.unsqueeze(2); Zi_e = Zi.unsqueeze(1)
        Zj_pair = Zj_e.expand(-1, S, S, -1); Zi_pair = Zi_e.expand(-1, S, S, -1)
        hadam = Zj_pair * Zi_pair
        diff  = torch.abs(Zj_pair - Zi_pair)
        pair  = torch.cat([Zj_pair, Zi_pair, hadam, diff], dim=-1)   # [B,S,S,4D]
        logits = self.score_mlp(pair).squeeze(-1)                     # [B,S,S]
        if attn_mask is not None:
            logits = logits + self._expand_mask_last(attn_mask, logits.dtype)
        probs = F.softmax(logits, dim=-1)                             # [B,S,S]
        ctx_i = torch.matmul(probs, H_i)                               # [B,S,H]
        msg_in = torch.cat([ctx_i, H_j, ctx_i * H_j], dim=-1)
        msg = self.val_mlp(msg_in)                                     # [B,S,H]
        return self.alpha * self.drop(msg)

# -----------------------
# Hybrid encoder: QKV + HDIM (both gated, LN, near-zero proj) + DEBUG
# -----------------------
@dataclass
class BridgeCfg:
    topk: int = 1
    route_dim: int = 128
    route_temp: float = 0.7
    route_pool: str = "mean"
    route_last_n: int = 4
    dropout: float = 0.1
    init_gate_attn: float = 0.15
    init_gate_hdim: float = 0.05
    hdim_proj_dim: int = 24
    hdim_mlp_hidden: int = 96

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
        self.nh = self.config.num_attention_heads
        assert H % self.nh == 0
        self.hd = H // self.nh
        self.scale = 1.0 / math.sqrt(self.hd)

        self.route_q = nn.Linear(H, cfg.route_dim)
        self.route_k = nn.Linear(H, cfg.route_dim)

        self.proj_by_tgt = nn.ModuleList([nn.Linear(H, H) for _ in range(self.L)])
        for lin in self.proj_by_tgt:
            nn.init.normal_(lin.weight, mean=0.0, std=1e-3); nn.init.zeros_(lin.bias)

        self.pre_ln_by_tgt = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.L)])

        self.hdim_by_tgt = nn.ModuleList([
            GeneralizedCrossMessage(H, proj_dim=cfg.hdim_proj_dim, mlp_hidden=cfg.hdim_mlp_hidden,
                                    dropout=cfg.dropout, gate_init=cfg.init_gate_hdim)
            for _ in range(self.L)
        ])

        self.alpha_attn = nn.Parameter(torch.full((self.L,), float(cfg.init_gate_attn)))
        self.alpha_hdim = nn.Parameter(torch.full((self.L,), float(cfg.init_gate_hdim)))
        self.drop = nn.Dropout(cfg.dropout)

        # ---- DEBUG HOOKS ----
        self.debug = False
        self.last_debug = None

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        return x.view(B, S, self.nh, self.hd).permute(0, 2, 1, 3).contiguous()

    def _extend_mask(self, attention_mask: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
        if attention_mask is None: return None
        ext = attention_mask[:, None, None, :]  # [B,1,1,S]
        return (1.0 - ext) * torch.finfo(dtype).min

    def _layer_summary(self, H_l: torch.Tensor, pool: str, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pool == "cls":
            return H_l[:, 0]
        else:
            if attn_mask is None: return H_l.mean(dim=1)
            m = attn_mask.float(); denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (H_l * m.unsqueeze(-1)).sum(dim=1) / denom

    def _attend(self, Q, K, V, ext_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale   # [B,nh,S,S]
        if ext_mask is not None: scores = scores + ext_mask
        P = torch.softmax(scores, dim=-1)
        P = self.drop(P)
        ctx = torch.matmul(P, V)                                     # [B,nh,S,hd]
        return ctx.permute(0, 2, 1, 3).contiguous().view(Q.size(0), Q.size(2), -1)  # [B,S,H]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        emb = self.embeddings(
            input_ids=input_ids,
            token_type_ids=torch.zeros_like(input_ids) if token_type_ids is None else token_type_ids,
            position_ids=position_ids,
        )
        dtype = emb.dtype
        device = emb.device
        ext_mask = self._extend_mask(attention_mask, dtype).to(device) if attention_mask is not None else None

        hidden_states = emb
        all_hidden: List[torch.Tensor] = [] if output_hidden_states else None
        all_attn: List[torch.Tensor] = [] if output_attentions else None

        layer_out_cache: Dict[int, torch.Tensor] = {}
        layer_sum_cache: Dict[int, torch.Tensor] = {}

        dbg = [] if self.debug else None

        for li in range(self.L):
            layer = self.layers[li]

            sa_outs = layer.attention(
                hidden_states,
                attention_mask=ext_mask,
                head_mask=head_mask[li] if head_mask is not None else None,
                output_attentions=output_attentions,
            )
            attn_out = sa_outs[0]
            if output_attentions:
                all_attn.append(sa_outs[1])

            apply_routing = li > 0 and (li >= self.L - self.cfg.route_last_n)
            if apply_routing:
                H_j_in = hidden_states
                B, S, H = H_j_in.shape

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
                    K_route = torch.stack(k_list, dim=1)
                    logits = torch.einsum("bd,bnd->bn", q_r, K_route) / max(1e-6, self.cfg.route_temp)
                    k = min(self.cfg.topk, logits.size(1))
                    top_vals, top_idx = torch.topk(logits, k=k, dim=1)
                    weights = torch.softmax(top_vals, dim=1)

                    # QKV message
                    Qj = self._shape_heads(layer.attention.self.query(H_j_in))
                    K_stack, V_stack = [], []
                    for i in src_indices:
                        H_i_out = layer_out_cache[i]
                        lsrc = self.layers[i].attention.self
                        K_stack.append(self._shape_heads(lsrc.key(H_i_out)))
                        V_stack.append(self._shape_heads(lsrc.value(H_i_out)))
                    K_stack = torch.stack(K_stack, dim=1)
                    V_stack = torch.stack(V_stack, dim=1)

                    nh, hd = Qj.size(1), Qj.size(-1)
                    idx = top_idx.view(B, k, 1, 1, 1).expand(B, k, nh, S, hd)
                    K_sel = torch.take_along_dim(K_stack, idx, dim=1)
                    V_sel = torch.take_along_dim(V_stack, idx, dim=1)
                    Q_rep = Qj.unsqueeze(1).expand(B, k, nh, S, hd)

                    Bk = B * k
                    Q_bk = Q_rep.reshape(Bk, nh, S, hd)
                    K_bk = K_sel.reshape(Bk, nh, S, hd)
                    V_bk = V_sel.reshape(Bk, nh, S, hd)
                    ext_mask_bk = ext_mask.repeat_interleave(k, dim=0) if ext_mask is not None else None
                    C_bk = self._attend(Q_bk, K_bk, V_bk, ext_mask_bk)
                    C_bk = C_bk.view(B, k, S, H)
                    ctx_qkv = (weights.view(B, k, 1, 1) * C_bk).sum(dim=1)

                    # HDIM message
                    H_stack = torch.stack([layer_out_cache[i] for i in src_indices], dim=1)
                    idx_H = top_idx.view(B, k, 1, 1).expand(B, k, S, H)
                    H_sel = torch.take_along_dim(H_stack, idx_H, dim=1)
                    msg_hdim = torch.zeros_like(ctx_qkv)
                    for sel in range(k):
                        msg_hdim = msg_hdim + weights[:, sel].view(B,1,1) * self.hdim_by_tgt[li](H_j_in, H_sel[:, sel, :, :], attention_mask)

                    add_ctx = self.alpha_attn[li] * ctx_qkv + self.alpha_hdim[li] * msg_hdim
                    add_ctx = self.pre_ln_by_tgt[li](add_ctx)
                    add_ctx = self.drop(self.proj_by_tgt[li](add_ctx))
                    attn_out = attn_out + add_ctx

                    if dbg is not None:
                        qkv_norm = ctx_qkv.norm(dim=-1).mean(dim=-1)   # [B]
                        hdim_norm = msg_hdim.norm(dim=-1).mean(dim=-1) # [B]
                        if self.cfg.topk == 1:
                            src_choice = top_idx.squeeze(1)             # [B]
                            vals, counts = torch.unique(src_choice, return_counts=True)
                            src_major = int(vals[counts.argmax()].item())
                        else:
                            src_major = -1
                        dbg.append({
                            "layer": li,
                            "alpha_attn": float(self.alpha_attn[li].item()),
                            "alpha_hdim": float(self.alpha_hdim[li].item()),
                            "qkv_norm_mean": float(qkv_norm.mean().item()),
                            "hdim_norm_mean": float(hdim_norm.mean().item()),
                            "src_major": src_major,
                        })

            inter = layer.intermediate(attn_out)
            layer_output = layer.output(inter, attn_out)
            hidden_states = layer_output

            layer_out_cache[li] = hidden_states
            if output_hidden_states:
                all_hidden.append(hidden_states)

        if self.debug:
            self.last_debug = dbg

        pooled = hidden_states[:, 0]

        if not return_dict:
            out = (hidden_states, pooled)
            if output_hidden_states: out = out + (tuple(all_hidden),)
            if output_attentions:   out = out + (tuple(all_attn),)
            return out

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=tuple(all_hidden) if output_hidden_states else None,
            attentions=tuple(all_attn) if output_attentions else None,
            cross_attentions=None,
        )

# -----------------------
# Classifier head + wrapper
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
        return SequenceClassifierOutput(logits=logits, loss=loss,
                                        hidden_states=enc_out.hidden_states, attentions=enc_out.attentions)

# -----------------------
# Usage aggregator (epoch-end distribution)
# -----------------------
class UsageMeter:
    def __init__(self, L:int, last_n:int):
        start = max(0, L - last_n)
        self.layers = list(range(start, L))
        self.reset()
    def reset(self):
        self.sums = {j: {"q":0.0,"h":0.0,"aq":0.0,"ah":0.0,"n":0} for j in self.layers}
        self.src_hist = {j: {} for j in self.layers}
    def update(self, dbg_list: Optional[List[Dict]]):
        if not dbg_list: return
        for r in dbg_list:
            j = r["layer"]
            if j not in self.sums: continue
            s = self.sums[j]
            s["q"]  += r["qkv_norm_mean"]
            s["h"]  += r["hdim_norm_mean"]
            s["aq"] += r["alpha_attn"]
            s["ah"] += r["alpha_hdim"]
            s["n"]  += 1
            src = int(r.get("src_major", -1))
            self.src_hist[j][src] = self.src_hist[j].get(src, 0) + 1
    def summary(self) -> List[Tuple[int,float,float,float,float,Dict[int,int]]]:
        rows = []
        for j in self.layers:
            s = self.sums[j]; n = max(1, s["n"])
            rows.append((j, s["q"]/n, s["h"]/n, s["aq"]/n, s["ah"]/n, self.src_hist[j]))
        return rows
    def print(self, tag:str):
        print(f"\n=== {tag}: Bridge usage (avg over steps) ===")
        print("layer  qkv_norm  hdim_norm  gate_qkv  gate_hdim   top_src_hist")
        for j,q,h,aq,ah,hist in self.summary():
            top = sorted(hist.items(), key=lambda x: -x[1])[:3]
            print(f"{j:>5}  {q:8.3f}  {h:9.3f}   {aq:7.3f}   {ah:8.3f}   {top}")

# -----------------------
# Hyperparams
# -----------------------
LR_ENCODER = 1.5e-5
LR_HEAD    = 3e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
WARMUP_FRAC = 0.10
LABEL_SMOOTH = 0.05
max_grad_norm = 1.0

# Define 10 different ablations as BridgeCfg variations
ablations = [
    {"name": "Original", "cfg": BridgeCfg(topk=1, route_dim=128, route_temp=0.7, route_pool="mean", route_last_n=4, dropout=0.1, init_gate_attn=0.15, init_gate_hdim=0.05, hdim_proj_dim=24, hdim_mlp_hidden=96)},
    {"name": "topk=2", "cfg": BridgeCfg(topk=2, route_dim=128, route_temp=0.7, route_pool="mean", route_last_n=4, dropout=0.1, init_gate_attn=0.15, init_gate_hdim=0.05, hdim_proj_dim=24, hdim_mlp_hidden=96)},
    {"name": "topk=3", "cfg": BridgeCfg(topk=3, route_dim=128, route_temp=0.7, route_pool="mean", route_last_n=4, dropout=0.1, init_gate_attn=0.15, init_gate_hdim=0.05, hdim_proj_dim=24, hdim_mlp_hidden=96)},
    {"name": "route_last_n=2", "cfg": BridgeCfg(topk=1, route_dim=128, route_temp=0.7, route_pool="mean", route_last_n=2, dropout=0.1, init_gate_attn=0.15, init_gate_hdim=0.05, hdim_proj_dim=24, hdim_mlp_hidden=96)},
    {"name": "route_last_n=6", "cfg": BridgeCfg(topk=1, route_dim=128, route_temp=0.7, route_pool="mean", route_last_n=6, dropout=0.1, init_gate_attn=0.15, init_gate_hdim=0.05, hdim_proj_dim=24, hdim_mlp_hidden=96)},
    {"name": "route_dim=256", "cfg": BridgeCfg(topk=1, route_dim=256, route_temp=0.7, route_pool="mean", route_last_n=4, dropout=0.1, init_gate_attn=0.15, init_gate_hdim=0.05, hdim_proj_dim=24, hdim_mlp_hidden=96)},
    {"name": "route_temp=0.5", "cfg": BridgeCfg(topk=1, route_dim=128, route_temp=0.5, route_pool="mean", route_last_n=4, dropout=0.1, init_gate_attn=0.15, init_gate_hdim=0.05, hdim_proj_dim=24, hdim_mlp_hidden=96)},
    {"name": "route_pool=cls", "cfg": BridgeCfg(topk=1, route_dim=128, route_temp=0.7, route_pool="cls", route_last_n=4, dropout=0.1, init_gate_attn=0.15, init_gate_hdim=0.05, hdim_proj_dim=24, hdim_mlp_hidden=96)},
    {"name": "ablate_attn_bridge", "cfg": BridgeCfg(topk=1, route_dim=128, route_temp=0.7, route_pool="mean", route_last_n=4, dropout=0.1, init_gate_attn=0.0, init_gate_hdim=0.05, hdim_proj_dim=24, hdim_mlp_hidden=96)},
    {"name": "ablate_hdim_bridge", "cfg": BridgeCfg(topk=1, route_dim=128, route_temp=0.7, route_pool="mean", route_last_n=4, dropout=0.1, init_gate_attn=0.15, init_gate_hdim=0.0, hdim_proj_dim=24, hdim_mlp_hidden=96)},
]

# -----------------------
# Eval
# -----------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, collect_usage: bool=False, tag: str="VAL") -> Dict[str, float]:
    model.eval()
    # optionally aggregate usage over eval
    meter = UsageMeter(model.encoder.encoder.L if hasattr(model.encoder, "encoder") else model.encoder.L,
                       model.encoder.cfg.route_last_n)
    if collect_usage:
        model.encoder.debug = True
    all_preds, all_labels = [], []
    for batch in tqdm(loader, total=len(loader), desc=tag, dynamic_ncols=True, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
        if collect_usage:
            meter.update(model.encoder.last_debug)
    if collect_usage:
        model.encoder.debug = False
        meter.print(f"{tag} usage")
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}

# -----------------------
# Helper: format live usage into a compact tqdm string
# -----------------------
def format_usage(enc: RobertaHybridEncoder) -> str:
    if enc.last_debug is None or len(enc.last_debug) == 0:
        return ""
    recs = sorted(enc.last_debug, key=lambda r: r["layer"])
    Ls  = ",".join(str(r["layer"]) for r in recs)
    aqS = "/".join(f"{r['alpha_attn']:.2f}" for r in recs)
    ahS = "/".join(f"{r['alpha_hdim']:.2f}" for r in recs)
    qS  = "/".join(f"{r['qkv_norm_mean']:.3f}" for r in recs)
    hS  = "/".join(f"{r['hdim_norm_mean']:.3f}" for r in recs)
    sS  = ",".join(str(r['src_major']) for r in recs)
    return f"L[{Ls}] aq={aqS} ah={ahS} | q={qS} h={hS} | src=[{sS}]"

# -----------------------
# Train loop for each ablation
# -----------------------
results = {}
for ab_idx, ablation in enumerate(ablations):
    print(f"\n=== Starting Ablation {ab_idx+1}/10: {ablation['name']} ===")
    bridge_cfg = ablation["cfg"]
    
    # Model + Optim + Sched
    model = RobertaHybridForSequenceClassification(checkpoint, bridge_cfg, num_labels=2, dropout=0.1)
    model.gradient_checkpointing_enable()
    model.to(device)

    def split_params(mod: nn.Module):
        decay, no_decay = [], []
        for n, p in mod.named_parameters():
            if not p.requires_grad: continue
            if any(nd in n for nd in ["bias", "LayerNorm.weight"]):
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay

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

    num_update_steps_per_epoch = math.ceil(len(train_loader))
    num_training_steps = NUM_EPOCHS * num_update_steps_per_epoch

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(WARMUP_FRAC * num_training_steps),
        num_training_steps=num_training_steps,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    ce_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_acc = -1.0
    save_dir = f"roberta_hybrid_rte_best_{ablation['name'].replace('=', '_')}"
    os.makedirs(save_dir, exist_ok=True)

    no_improve = 0
    EARLY_STOP_PATIENCE = 2

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        start = time.time()
        ema_loss = None
        correct = 0
        seen = 0

        # per-epoch usage meter (train)
        train_usage = UsageMeter(model.encoder.L, model.encoder.cfg.route_last_n)

        # turn on debug collection during training
        model.encoder.debug = True

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{NUM_EPOCHS}", dynamic_ncols=True)
        for step_idx, batch in enumerate(pbar):
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

            loss_val = loss.item()
            ema_loss = loss_val if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_val

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # update per-epoch usage aggregates
            train_usage.update(model.encoder.last_debug)

            # print usage every few steps
            usage = ""
            if (step_idx % 5) == 0:
                usage = format_usage(model.encoder)
            pbar.set_postfix_str(f"loss={loss_val:.4f} ema={ema_loss:.4f} acc={acc_running:.3f}  {usage}")

        # end of epoch: disable debug and print train usage distribution
        model.encoder.debug = False
        train_usage.print(f"TRAIN epoch {epoch}")

        # Evaluate + (optional) collect usage on dev too
        val_metrics = evaluate(model, val_loader, collect_usage=True, tag=f"VAL epoch {epoch}")
        elapsed = time.time() - start
        print(f"\nEpoch {epoch} | time {elapsed:.1f}s | val_acc {val_metrics['accuracy']:.4f} | val_f1 {val_metrics['f1_macro']:.4f}")

        # Early stop if epoch 1 < 70%
        if epoch == 1 and val_metrics["accuracy"] < 0.70:
            print(f"Early stopping ablation {ablation['name']} after epoch 1 (val_acc {val_metrics['accuracy']:.4f} < 0.70)")
            break

        # Save best
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            print(f"New best acc {best_acc:.4f} -> saving to {save_dir}")
            torch.save({
                "encoder_state": model.encoder.state_dict(),
                "head_state": model.head.state_dict(),
                "bridge_cfg": bridge_cfg.__dict__,
            }, os.path.join(save_dir, "model.pt"))
            tokenizer.save_pretrained(save_dir)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best acc {best_acc:.4f})")
                break

    print(f"\nAblation {ablation['name']} complete. Best validation accuracy: {best_acc:.4f}")
    results[ablation['name']] = best_acc

# -----------------------
# Summary of all ablations
# -----------------------
print("\n=== Ablation Results ===")
for name, acc in results.items():
    print(f"{name}: Best Val Acc = {acc:.4f}")

 