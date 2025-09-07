# app.py
# One-file API + Celery + Trainer for RoBERTa-large RTE with Hybrid QKV/HDIM bridges.
# Supports presets (A1–A10, H1–H20) and arbitrary overrides for "anything under the sun".
#
# Requirements (examples):
#   pip install fastapi uvicorn celery[redis] redis pydantic datasets scikit-learn tqdm transformers==4.44.2
#   pip install "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" "torchaudio==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
#
# Run:
#   1) Redis up (eg: docker run -p 6379:6379 redis)
#   2) Celery worker: celery -A app.celery_app worker --loglevel=INFO
#   3) API server:    uvicorn app:app --host 0.0.0.0 --port 8000
#
# API:
#   POST /run
#     {
#       "ablation": "A4_HDIM_only",         # optional preset name
#       "overrides": {"hdim_proj_dim": 32}, # any BridgeCfg or train knob
#       "epochs": 3,
#       "batch_size": 8,
#       "max_len": 128,
#       "save_dir": "runs/A4",
#       "save_artifacts": true,
#       "early_stop": {"epoch1_val_below": 0.7}  # optional
#     }
#   GET  /status/{task_id}
#   GET  /result/{task_id}

# main.py
import os, math, time, random, numpy as np, json
from dataclasses import dataclass, asdict, replace
from typing import Dict, Optional, List, Any

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
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput
)

# ---------------- Repro + device helpers ----------------
GLOBAL_SEED = 42

def set_all_seeds(seed: int = GLOBAL_SEED):
    set_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pick_device(force: Optional[str] = None) -> torch.device:
    if force == "cpu":  return torch.device("cpu")
    if force == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def effective_num_workers(requested: int | None) -> int:
    # If running inside Celery or not explicitly set, be safe
    if os.environ.get("IN_CELERY", "0") == "1":
        return 0
    return 0 if requested is None else max(0, int(requested))

# ---------------- NVTX helper ----------------
class nvtx_range:
    def __init__(self, name: str, enabled: bool):
        self.name = name
        self.enabled = enabled and torch.cuda.is_available()
    def __enter__(self):
        if self.enabled:
            try:
                torch.cuda.nvtx.range_push(self.name)
            except Exception:
                pass
    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
# ---------------- Config (superset of your knobs) ----------------
@dataclass
class BridgeCfg:
    topk: int = 1
    route_dim: int = 128
    route_temp: float = 0.7
    route_pool: str = "mean"     # "mean" | "cls"
    route_last_n: int = 4

    dropout: float = 0.1
    inject_site: str = "pre_ffn"

    init_gate_attn: float = 0.15
    init_gate_hdim: float = 0.05

    hdim_proj_dim: int = 24
    hdim_mlp_hidden: int = 96
    hdim_pair_mode: str = "4term"        # "4term" | "bilinear" | "cosine"
    hdim_val_mode: str = "concat_hadam"  # "concat_hadam" | "concat_only" | "concat_diff_hadam"
    hdim_share_projections: bool = False
    hdim_use_proj_ln: bool = False
    hdim_val_residual: bool = False
    hdim_val_residual_scale: float = 0.10
    hdim_token_temp: float = 1.0

    ent_reg_weight: float = 0.0

    enable_qkv: bool = True
    enable_hdim: bool = True
    qkv_target_weights: bool = False
    use_bridge_WV: bool = False
    inject_last_m: Optional[int] = None

    per_head_gates: bool = False
    adapter_hidden: int = 256

# ---------------- HDIM module ----------------
class GeneralizedCrossMessage(nn.Module):
    def __init__(self, hidden_size: int, cfg: BridgeCfg):
        super().__init__()
        self.H = hidden_size; self.D = cfg.hdim_proj_dim; self.cfg = cfg
        self.p_j = nn.Linear(self.H, self.D, bias=False)
        self.p_i = self.p_j if cfg.hdim_share_projections else nn.Linear(self.H, self.D, bias=False)
        self.proj_ln = nn.LayerNorm(self.D) if cfg.hdim_use_proj_ln else None

        if cfg.hdim_pair_mode == "4term":
            self.score_mlp = nn.Sequential(
                nn.Linear(4*self.D, cfg.hdim_mlp_hidden), nn.ReLU(inplace=True), nn.Linear(cfg.hdim_mlp_hidden, 1)
            ); self.W_bilinear = None
        elif cfg.hdim_pair_mode == "bilinear":
            self.W_bilinear = nn.Parameter(torch.empty(self.D, self.D)); nn.init.xavier_uniform_(self.W_bilinear)
            self.score_mlp = None
        elif cfg.hdim_pair_mode == "cosine":
            self.W_bilinear = None; self.score_mlp = None
        else: raise ValueError(f"Unknown hdim_pair_mode {cfg.hdim_pair_mode}")

        if cfg.hdim_val_mode == "concat_hadam":   val_in = 3*self.H
        elif cfg.hdim_val_mode == "concat_only":  val_in = 2*self.H
        elif cfg.hdim_val_mode == "concat_diff_hadam": val_in = 4*self.H
        else: raise ValueError(f"Unknown hdim_val_mode {cfg.hdim_val_mode}")

        out_hid = 2 * max(128, self.H // 2)
        self.val_mlp = nn.Sequential(nn.Linear(val_in, out_hid), nn.ReLU(inplace=True), nn.Linear(out_hid, self.H))
        self.val_residual_scale = nn.Parameter(torch.tensor(float(cfg.hdim_val_residual_scale)))
        self.drop = nn.Dropout(cfg.dropout)
        self.alpha = nn.Parameter(torch.tensor(float(cfg.init_gate_hdim)))
        self._ent_reg = None

    def _expand_mask_last(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (1.0 - mask[:, None, :]) * torch.finfo(dtype).min

    def _pair_logits(self, Zj, Zi, attn_mask):
        if self.proj_ln is not None: Zj = self.proj_ln(Zj); Zi = self.proj_ln(Zi)
        mode = self.cfg.hdim_pair_mode; B, S, D = Zj.shape
        if mode == "4term":
            Zj_e = Zj.unsqueeze(2); Zi_e = Zi.unsqueeze(1)
            Zj_pair = Zj_e.expand(-1, S, S, -1); Zi_pair = Zi_e.expand(-1, S, S, -1)
            hadam = Zj_pair * Zi_pair; diff = torch.abs(Zj_pair - Zi_pair)
            pair = torch.cat([Zj_pair, Zi_pair, hadam, diff], dim=-1)
            logits = self.score_mlp(pair).squeeze(-1)
        elif mode == "bilinear":
            logits = torch.matmul(torch.matmul(Zj, self.W_bilinear), Zi.transpose(1, 2))
        else:  # cosine
            Zj_n = F.normalize(Zj, p=2, dim=-1); Zi_n = F.normalize(Zi, p=2, dim=-1)
            logits = torch.matmul(Zj_n, Zi_n.transpose(1, 2)) * 10.0
        if attn_mask is not None: logits = logits + self._expand_mask_last(attn_mask, logits.dtype)
        if self.cfg.hdim_token_temp != 1.0: logits = logits / max(1e-6, self.cfg.hdim_token_temp)
        return logits

    def forward(self, H_j, H_i, attn_mask):
        Zj = self.p_j(H_j); Zi = self.p_i(H_i)
        logits = self._pair_logits(Zj, Zi, attn_mask)
        probs = F.softmax(logits, dim=-1)
        ctx_i = torch.matmul(probs, H_i)

        if self.training and self.cfg.ent_reg_weight > 0.0:
            self._ent_reg = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
        else:
            self._ent_reg = None

        if self.cfg.hdim_val_mode == "concat_hadam":
            msg_in = torch.cat([ctx_i, H_j, ctx_i * H_j], dim=-1)
        elif self.cfg.hdim_val_mode == "concat_only":
            msg_in = torch.cat([ctx_i, H_j], dim=-1)
        else:
            msg_in = torch.cat([ctx_i, H_j, torch.abs(ctx_i - H_j), ctx_i * H_j], dim=-1)

        msg = self.val_mlp(msg_in)
        if self.cfg.hdim_val_residual:
            msg = msg + self.val_residual_scale * (ctx_i - H_j)
        return self.alpha * self.drop(msg)

# ---------------- Hybrid encoder (QKV + HDIM) ----------------
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

        self.route_q = nn.Linear(H, self.cfg.route_dim)
        self.route_k = nn.Linear(H, self.cfg.route_dim)

        self.proj_by_tgt = nn.ModuleList([nn.Linear(H, H) for _ in range(self.L)])
        for lin in self.proj_by_tgt:
            nn.init.normal_(lin.weight, mean=0.0, std=1e-3); nn.init.zeros_(lin.bias)
        self.pre_ln_by_tgt = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.L)])
        self.hdim_by_tgt = nn.ModuleList([GeneralizedCrossMessage(H, self.cfg) for _ in range(self.L)])

        self.alpha_attn = nn.Parameter(torch.full((self.L,), float(self.cfg.init_gate_attn)))
        self.alpha_hdim = nn.Parameter(torch.full((self.L,), float(self.cfg.init_gate_hdim)))
        self.drop = nn.Dropout(self.cfg.dropout)

        self.bridge_WV_by_tgt = nn.ModuleList([nn.Linear(self.hd, self.hd, bias=False) for _ in range(self.L)]) \
            if self.cfg.use_bridge_WV else None

    def _shape_heads(self, x):  # [B,S,H] -> [B,nh,S,hd]
        B, S, H = x.shape
        return x.view(B, S, self.nh, self.hd).permute(0, 2, 1, 3).contiguous()

    def _extend_mask(self, attention_mask, dtype):
        if attention_mask is None: return None
        ext = attention_mask[:, None, None, :]
        return (1.0 - ext) * torch.finfo(dtype).min

    def _layer_summary(self, H_l, pool, attn_mask):
        if pool == "cls": return H_l[:, 0]
        if attn_mask is None: return H_l.mean(dim=1)
        m = attn_mask.float(); denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (H_l * m.unsqueeze(-1)).sum(dim=1) / denom

    def _attend(self, Q, K, V, ext_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) * (1.0 / math.sqrt(Q.size(-1)))
        if ext_mask is not None: scores = scores + ext_mask
        P = torch.softmax(scores, dim=-1); P = self.drop(P)
        ctx = torch.matmul(P, V)
        return ctx.permute(0, 2, 1, 3).contiguous().view(Q.size(0), Q.size(2), -1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        emb = self.embeddings(
            input_ids=input_ids,
            token_type_ids=torch.zeros_like(input_ids) if token_type_ids is None else token_type_ids,
            position_ids=position_ids,
        )
        dtype = emb.dtype; device = emb.device
        ext_mask = self._extend_mask(attention_mask, dtype).to(device) if attention_mask is not None else None

        hidden_states = emb
        all_hidden = [] if output_hidden_states else None
        all_attn = [] if output_attentions else None
        layer_out_cache: Dict[int, torch.Tensor] = {}
        layer_sum_cache: Dict[int, torch.Tensor] = {}

        for li in range(self.L):
            layer = self.layers[li]
            sa_outs = layer.attention(hidden_states, attention_mask=ext_mask,
                                      head_mask=head_mask[li] if head_mask is not None else None,
                                      output_attentions=output_attentions)
            attn_out = sa_outs[0]
            if output_attentions: all_attn.append(sa_outs[1])

            apply_routing = li > 0 and (li >= self.L - self.cfg.route_last_n)
            inject_here = True if self.cfg.inject_last_m is None else (li >= self.L - self.cfg.inject_last_m)

            if apply_routing and inject_here:
                H_j_in = hidden_states
                B, S, H = H_j_in.shape
                q_sum = self._layer_summary(H_j_in, self.cfg.route_pool, attention_mask)
                q_r = self.route_q(q_sum)
                src_indices = list(range(li))
                if src_indices:
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

                    ctx_qkv = None
                    if self.cfg.enable_qkv:
                        Qj = self._shape_heads(layer.attention.self.query(H_j_in))
                        K_stack, V_stack = [], []
                        if self.cfg.qkv_target_weights:
                            tgt_self = layer.attention.self
                            for i in src_indices:
                                H_i_out = layer_out_cache[i]
                                K_stack.append(self._shape_heads(tgt_self.key(H_i_out)))
                                V_stack.append(self._shape_heads(tgt_self.value(H_i_out)))
                        else:
                            for i in src_indices:
                                H_i_out = layer_out_cache[i]
                                lsrc = self.layers[i].attention.self
                                K_stack.append(self._shape_heads(lsrc.key(H_i_out)))
                                V_stack.append(self._shape_heads(lsrc.value(H_i_out)))
                        K_stack = torch.stack(K_stack, dim=1); V_stack = torch.stack(V_stack, dim=1)
                        nh, hd = Qj.size(1), Qj.size(-1)
                        idx = top_idx.view(B, k, 1, 1, 1).expand(B, k, nh, S, hd)
                        K_sel = torch.take_along_dim(K_stack, idx, dim=1)
                        V_sel = torch.take_along_dim(V_stack, idx, dim=1)
                        if self.bridge_WV_by_tgt is not None:
                            WV = self.bridge_WV_by_tgt[li]
                            V_flat = V_sel.reshape(-1, hd); V_flat = WV(V_flat); V_sel = V_flat.view(B, k, nh, S, hd)
                        Q_rep = Qj.unsqueeze(1).expand(B, k, nh, S, hd)
                        Bk = B * k
                        Q_bk = Q_rep.reshape(Bk, nh, S, hd)
                        K_bk = K_sel.reshape(Bk, nh, S, hd)
                        V_bk = V_sel.reshape(Bk, nh, S, hd)
                        ext_mask_bk = ext_mask.repeat_interleave(k, dim=0) if ext_mask is not None else None
                        C_bk = self._attend(Q_bk, K_bk, V_bk, ext_mask_bk)
                        C_bk = C_bk.view(B, k, S, H)
                        ctx_qkv = (weights.view(B, k, 1, 1) * C_bk).sum(dim=1)

                    msg_hdim = None
                    if self.cfg.enable_hdim:
                        H_stack = torch.stack([layer_out_cache[i] for i in src_indices], dim=1)
                        idx_H = top_idx.view(B, k, 1, 1).expand(B, k, S, H)
                        H_sel = torch.take_along_dim(H_stack, idx_H, dim=1)
                        msg_hdim = torch.zeros_like(H_j_in)
                        for sel in range(k):
                            m = self.hdim_by_tgt[li](H_j_in, H_sel[:, sel, :, :], attention_mask)
                            msg_hdim = msg_hdim + weights[:, sel].view(B, 1, 1) * m

                    add_ctx = torch.zeros_like(H_j_in)
                    if ctx_qkv is not None: add_ctx = add_ctx + self.alpha_attn[li] * ctx_qkv
                    if msg_hdim is not None: add_ctx = add_ctx + self.alpha_hdim[li] * msg_hdim
                    add_ctx = self.pre_ln_by_tgt[li](add_ctx)
                    add_ctx = self.drop(self.proj_by_tgt[li](add_ctx))
                    attn_out = attn_out + add_ctx

            inter = layer.intermediate(attn_out)
            hidden_states = layer.output(inter, attn_out)
            layer_out_cache[li] = hidden_states
            if output_hidden_states: all_hidden.append(hidden_states)

        pooled = hidden_states[:, 0]
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states, pooler_output=pooled,
            hidden_states=tuple(all_hidden) if output_hidden_states else None,
            attentions=tuple(all_attn) if output_attentions else None,
        )

# ---------------- Classifier wrapper ----------------
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
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = enc.last_hidden_state[:, 0]
        logits = self.head(cls)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(logits=logits, loss=loss)

# ---------------- Data & utils ----------------
def build_loaders(batch_size:int=8, max_len:int=128, dataloader_workers:int|None=None, device=None):
    raw = load_dataset("glue", "rte")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
    def preprocess(batch):
        enc = tokenizer(batch["sentence1"], batch["sentence2"], truncation=True, max_length=max_len)
        enc["labels"] = batch["label"]; return enc
    cols_to_remove = [c for c in raw["train"].column_names if c not in ("idx",)]
    tokenized = raw.map(preprocess, batched=True, remove_columns=cols_to_remove)
    tokenized = tokenized.with_format("torch", columns=["input_ids","attention_mask","labels"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    nw = effective_num_workers(dataloader_workers)
    pin = (device is not None and device.type == "cuda")

    train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(tokenized["validation"], batch_size=32, shuffle=False,
                              collate_fn=collator, num_workers=nw, pin_memory=pin)
    return tokenizer, train_loader, val_loader


def split_params(mod: nn.Module):
    decay, no_decay = [], []
    for n, p in mod.named_parameters():
        if not p.requires_grad: continue
        (no_decay if any(nd in n for nd in ["bias","LayerNorm.weight"]) else decay).append(p)
    return decay, no_decay

@torch.no_grad()
def evaluate(device: torch.device, model: nn.Module, loader) -> Dict[str, float]:
    model.eval(); preds, labels = [], []
    for batch in loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
            logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        preds.append(logits.argmax(-1).cpu()); labels.append(batch["labels"].cpu())
    y_pred = torch.cat(preds).numpy(); y_true = torch.cat(labels).numpy()
    return {"accuracy": float(accuracy_score(y_true,y_pred)),
            "f1_macro": float(f1_score(y_true,y_pred, average="macro"))}

def count_params(model: nn.Module) -> Dict[str,int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_names = ["roberta.embeddings","roberta.encoder.layer"]; base = 0
    for n,p in model.named_parameters():
        if any(n.startswith(bn) for bn in base_names): base += p.numel()
    return {"total": total, "trainable": trainable, "approx_base": base, "approx_extra": total - base}

@torch.no_grad()
def gate_summaries(enc: RobertaHybridEncoder) -> Dict[str, Dict]:
    def stats(x):
        return {"shape": list(x.shape), "mean": float(x.mean()), "std": float(x.std()),
                "min": float(x.min()), "max": float(x.max())}
    return {"alpha_attn": stats(enc.alpha_attn.detach().float().cpu().numpy()),
            "alpha_hdim": stats(enc.alpha_hdim.detach().float().cpu().numpy())}

# ---------------- Presets (A1–A10, H1–H20) ----------------
def base_cfg_hybrid() -> BridgeCfg:
    return BridgeCfg(
        topk=1, route_dim=128, route_temp=0.7, route_pool="mean", route_last_n=4,
        dropout=0.1, inject_site="pre_ffn",
        init_gate_attn=0.15, init_gate_hdim=0.05,
        hdim_proj_dim=24, hdim_mlp_hidden=96, hdim_pair_mode="4term", hdim_val_mode="concat_hadam",
        hdim_share_projections=False, hdim_use_proj_ln=False, hdim_val_residual=False, hdim_val_residual_scale=0.10,
        hdim_token_temp=1.0, ent_reg_weight=0.0,
        enable_qkv=True, enable_hdim=True, qkv_target_weights=False, use_bridge_WV=False,
        inject_last_m=None, per_head_gates=False, adapter_hidden=256
    )

ABLATIONS: Dict[str, Any] = {
    "A1_QKV_WVbridge":      lambda c: replace(c, use_bridge_WV=True),
    "A2_QKV_TargetW":       lambda c: replace(c, qkv_target_weights=True),
    "A3_QKV_only":          lambda c: replace(c, enable_qkv=True, enable_hdim=False),
    "A4_HDIM_only":         lambda c: replace(c, enable_qkv=False, enable_hdim=True),
    "A5_HDIM_bilinear":     lambda c: replace(c, hdim_pair_mode="bilinear"),
    "A6_HDIM_proj32_mlp128":lambda c: replace(c, hdim_proj_dim=32, hdim_mlp_hidden=128),
    "A7_HDIM_GELU":         lambda c: c,  # placeholder (you can extend to add hdim_act if desired)
    "A8_Router_CLS":        lambda c: replace(c, route_pool="cls"),
    "A9_Router_Temp0p5":    lambda c: replace(c, route_temp=0.5),
    "A10_Inject_Last2":     lambda c: replace(c, inject_last_m=2),

    "H1_proj32_mlp128":     lambda c: replace(c, hdim_proj_dim=32, hdim_mlp_hidden=128, enable_qkv=False, enable_hdim=True),
    "H2_proj16_mlp64":      lambda c: replace(c, hdim_proj_dim=16, hdim_mlp_hidden=64,  enable_qkv=False, enable_hdim=True),
    "H3_gate0p10":          lambda c: replace(c, init_gate_hdim=0.10,                  enable_qkv=False, enable_hdim=True),
    "H4_dropout0":          lambda c: replace(c, dropout=0.0,                           enable_qkv=False, enable_hdim=True),
    "H5_routepool_cls":     lambda c: replace(c, route_pool="cls",                      enable_qkv=False, enable_hdim=True),
    "H6_last6":             lambda c: replace(c, route_last_n=6,                        enable_qkv=False, enable_hdim=True),
    "H7_router_temp0p5":    lambda c: replace(c, route_temp=0.5,                        enable_qkv=False, enable_hdim=True),
    "H8_pair_bilinear":     lambda c: replace(c, hdim_pair_mode="bilinear",             enable_qkv=False, enable_hdim=True),
    "H9_val_concat_only":   lambda c: replace(c, hdim_val_mode="concat_only",           enable_qkv=False, enable_hdim=True),
    "H10_share_proj":       lambda c: replace(c, hdim_share_projections=True,           enable_qkv=False, enable_hdim=True),

    "H11_token_temp0p8":        lambda c: replace(c, hdim_token_temp=0.8,               enable_qkv=False, enable_hdim=True),
    "H12_pair_cosine":          lambda c: replace(c, hdim_pair_mode="cosine",           enable_qkv=False, enable_hdim=True),
    "H13_cosine_concat_only":   lambda c: replace(c, hdim_pair_mode="cosine", hdim_val_mode="concat_only", enable_qkv=False, enable_hdim=True),
    "H14_gate0p10_drop0p05":    lambda c: replace(c, init_gate_hdim=0.10, dropout=0.05, enable_qkv=False, enable_hdim=True),
    "H15_last3":                lambda c: replace(c, route_last_n=3,                    enable_qkv=False, enable_hdim=True),
    "H16_entreg_1e3":           lambda c: replace(c, ent_reg_weight=1e-3,               enable_qkv=False, enable_hdim=True),
    "H17_proj_LN":              lambda c: replace(c, hdim_use_proj_ln=True,             enable_qkv=False, enable_hdim=True),
    "H18_val_residual":         lambda c: replace(c, hdim_val_residual=True,            enable_qkv=False, enable_hdim=True),
    "H19_route_dim64":          lambda c: replace(c, route_dim=64,                      enable_qkv=False, enable_hdim=True),
    "H20_proj32_concat_only":   lambda c: replace(c, hdim_proj_dim=32, hdim_val_mode="concat_only", enable_qkv=False, enable_hdim=True),
}

# ---------------- Training core ----------------
def train_and_eval(
    task_ctx: Optional[Any],
    device: torch.device,
    cfg: BridgeCfg,
    epochs: int = 3,
    batch_size: int = 8,
    max_len: int = 128,
    lr_encoder: float = 1.5e-5,
    lr_head: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.10,
    label_smoothing: float = 0.05,
    max_grad_norm: float = 1.0,
    save_dir: Optional[str] = None,
    save_artifacts: bool = False,
    early_stop: Optional[Dict[str, float]] = None,
    gradient_checkpointing: bool = True,
    max_train_batches: Optional[int] = None,
    progress_path: Optional[str] = None,
) -> Dict[str, Any]:

    set_all_seeds(GLOBAL_SEED)
    tokenizer, train_loader, val_loader = build_loaders(batch_size=batch_size, max_len=max_len)

    model = RobertaHybridForSequenceClassification("roberta-large", cfg, num_labels=2, dropout=0.1)
    if gradient_checkpointing: model.gradient_checkpointing_enable()
    model.to(device)

    enc_decay, enc_no_decay = split_params(model.encoder)
    head_decay, head_no_decay = split_params(model.head)
    optimizer = torch.optim.AdamW(
        [{"params": enc_decay, "lr": lr_encoder, "weight_decay": weight_decay},
         {"params": enc_no_decay, "lr": lr_encoder, "weight_decay": 0.0},
         {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay},
         {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0}]
    )
    num_update_steps_per_epoch = math.ceil(len(train_loader))
    num_training_steps = epochs * num_update_steps_per_epoch
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=int(warmup_frac * num_training_steps), num_training_steps=num_training_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if save_dir: os.makedirs(save_dir, exist_ok=True)
    logs = []; best = {"val_acc": -1.0, "epoch": -1}
    early_1_thresh = float(early_stop["epoch1_val_below"]) if (early_stop and "epoch1_val_below" in early_stop) else None

    for epoch in range(1, epochs + 1):
        model.train(); start = time.time(); ema_loss = None; correct = 0; seen = 0
        with nvtx_range(f"epoch_{epoch}", enabled=(device.type == "cuda")):
            for ib, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", dynamic_ncols=True, leave=False)):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with nvtx_range("forward", enabled=(device.type == "cuda")):
                    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                        loss = ce_loss(out.logits, batch["labels"])
                        if cfg.ent_reg_weight > 0.0:
                            regs = [m._ent_reg for m in model.encoder.hdim_by_tgt if getattr(m, "_ent_reg", None) is not None]
                            if regs: loss = loss + cfg.ent_reg_weight * torch.stack(regs).mean()
                preds = out.logits.argmax(-1); correct += (preds == batch["labels"]).sum().item(); seen += len(preds)
                loss_val = float(loss.item()); ema_loss = loss_val if ema_loss is None else 0.9*ema_loss + 0.1*loss_val
                with nvtx_range("backward+step", enabled=(device.type == "cuda")):
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer); scaler.update(); scheduler.step()
                if max_train_batches is not None and (ib + 1) >= max_train_batches:
                    break

        train_acc_epoch = correct / max(1, seen)
        val_metrics = evaluate(device, model, val_loader)
        gates = gate_summaries(model.encoder)
        ep_row = {"epoch": epoch, "time_sec": round(time.time()-start,2),
                  "train_acc": float(train_acc_epoch), "train_loss_ema": float(ema_loss) if ema_loss is not None else None,
                  "val_acc": float(val_metrics["accuracy"]), "val_f1_macro": float(val_metrics["f1_macro"]),
                  "lr": float(optimizer.param_groups[0]["lr"]), "gates": gates}
        logs.append(ep_row)

        # Optional: write incremental progress for external tailing
        if progress_path:
            try:
                with open(progress_path, "a") as pf:
                    pf.write(json.dumps(ep_row) + "\n")
            except Exception:
                pass

        if task_ctx is not None:
            task_ctx.update_state(state="STARTED", meta={"progress_epoch": epoch, "epochs_total": epochs, "last_val": val_metrics})

        if val_metrics["accuracy"] > best["val_acc"]:
            best = {"val_acc": float(val_metrics["accuracy"]), "epoch": epoch}
            if save_dir and save_artifacts:
                torch.save({"encoder_state": model.encoder.state_dict(),
                            "head_state": model.head.state_dict(),
                            "bridge_cfg": asdict(cfg)}, os.path.join(save_dir, "model.pt"))

        if epoch == 1 and (early_1_thresh is not None) and (val_metrics["accuracy"] < early_1_thresh):
            break

    return {"seed": GLOBAL_SEED, "device": str(device), "cfg": asdict(cfg),
            "param_counts": count_params(model), "best": best, "epochs": logs,
            "save_dir": save_dir if save_dir else None}

__all__ = [
    "BridgeCfg", "base_cfg_hybrid", "ABLATIONS",
    "train_and_eval", "pick_device", "set_all_seeds"
]
