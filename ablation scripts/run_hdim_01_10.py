#!/usr/bin/env python3
# RTE — 10 HDIM-only ablations (3 epochs each), save stats only.
# Requirements:
# pip install -q transformers==4.44.2 datasets==2.21.0 scikit-learn==1.5.2 tqdm==4.66.4

import os, json, csv, math, time, random, numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List

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
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
# HDIM module (configurable)
# -----------------------
@dataclass
class BridgeCfg:
    # Router
    topk: int = 1
    route_dim: int = 128
    route_temp: float = 0.7
    route_pool: str = "mean"      # "mean" or "cls"
    route_last_n: int = 4

    # Training / injection
    dropout: float = 0.1
    inject_site: str = "pre_ffn"  # keep fixed to pre_ffn for this HDIM-only study

    # Gates (QKV disabled anyway)
    init_gate_attn: float = 0.15  # unused in HDIM-only, kept for completeness
    init_gate_hdim: float = 0.05

    # HDIM architecture
    hdim_proj_dim: int = 24
    hdim_mlp_hidden: int = 96
    hdim_pair_mode: str = "4term"       # "4term" (concat + hadam + diff into MLP), "bilinear", "cosine"
    hdim_val_mode: str = "concat_hadam" # "concat_hadam", "concat_only", "concat_diff_hadam"
    hdim_share_projections: bool = False
    hdim_token_temp: float = 1.0        # temperature for token-token softmax inside HDIM

    # Toggles
    use_qkv: bool = False     # HDIM-only
    use_hdim: bool = True
    per_head_gates: bool = False
    adapter_hidden: int = 256 # unused here; injection is pre_ffn

class GeneralizedCrossMessage(nn.Module):
    """
    Flexible HDIM message:
      - Pair scoring: 4term-MLP (default), bilinear, or cosine
      - Value fusion: concat_hadam (default), concat_only, concat_diff_hadam
      - Optional shared projections p_j == p_i
      - Token softmax temperature hdim_token_temp
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
            in_feats = 4 * self.D  # [Zj, Zi, hadam, diff]
            self.score_mlp = nn.Sequential(
                nn.Linear(in_feats, cfg.hdim_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.hdim_mlp_hidden, 1)
            )
            self.W_bilinear = None
        elif cfg.hdim_pair_mode == "bilinear":
            # score = Zj^T W Zi (scalar)
            self.W_bilinear = nn.Parameter(torch.empty(self.D, self.D))
            nn.init.xavier_uniform_(self.W_bilinear)
            self.score_mlp = None
        elif cfg.hdim_pair_mode == "cosine":
            self.score_mlp = None
            self.W_bilinear = None
        else:
            raise ValueError(f"Unknown hdim_pair_mode: {cfg.hdim_pair_mode}")

        # value fusion MLP
        if cfg.hdim_val_mode == "concat_hadam":
            val_in = 3*self.H  # [ctx_i, H_j, ctx_i*H_j]
        elif cfg.hdim_val_mode == "concat_only":
            val_in = 2*self.H  # [ctx_i, H_j]
        elif cfg.hdim_val_mode == "concat_diff_hadam":
            val_in = 4*self.H  # [ctx_i, H_j, |diff|, hadam]
        else:
            raise ValueError(f"Unknown hdim_val_mode: {cfg.hdim_val_mode}")

        out_hid = 2 * max(128, self.H // 2)
        self.val_mlp = nn.Sequential(
            nn.Linear(val_in, out_hid),
            nn.ReLU(inplace=True),
            nn.Linear(out_hid, self.H)
        )

        self.drop = nn.Dropout(cfg.dropout)
        self.alpha = nn.Parameter(torch.tensor(float(cfg.init_gate_hdim)))

    def _expand_mask_last(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (1.0 - mask[:, None, :]) * torch.finfo(dtype).min

    def _pair_logits(self, Zj: torch.Tensor, Zi: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Zj, Zi: [B,S,D] each
        returns logits [B,S,S]
        """
        B, S, D = Zj.shape
        mode = self.cfg.hdim_pair_mode

        if mode == "4term":
            Zj_e = Zj.unsqueeze(2)               # [B,S,1,D]
            Zi_e = Zi.unsqueeze(1)               # [B,1,S,D]
            Zj_pair = Zj_e.expand(-1, S, S, -1)  # [B,S,S,D]
            Zi_pair = Zi_e.expand(-1, S, S, -1)  # [B,S,S,D]
            hadam = Zj_pair * Zi_pair
            diff  = torch.abs(Zj_pair - Zi_pair)
            pair  = torch.cat([Zj_pair, Zi_pair, hadam, diff], dim=-1)   # [B,S,S,4D]
            logits = self.score_mlp(pair).squeeze(-1)                    # [B,S,S]
        elif mode == "bilinear":
            # logits[b,s,t] = Zj[b,s,:]^T W Zi[b,t,:]
            W = self.W_bilinear
            ZjW = torch.matmul(Zj, W)            # [B,S,D]
            logits = torch.matmul(ZjW, Zi.transpose(1, 2))  # [B,S,S]
        elif mode == "cosine":
            # cosine similarity across token pairs
            Zj_n = F.normalize(Zj, p=2, dim=-1)  # [B,S,D]
            Zi_n = F.normalize(Zi, p=2, dim=-1)
            logits = torch.matmul(Zj_n, Zi_n.transpose(1, 2))  # [B,S,S]
            # scale up a bit so softmax isn't too flat
            logits = logits * 10.0
        else:
            raise RuntimeError

        if attn_mask is not None:
            logits = logits + self._expand_mask_last(attn_mask, logits.dtype)
        # apply token temperature
        if self.cfg.hdim_token_temp != 1.0:
            logits = logits / max(1e-6, self.cfg.hdim_token_temp)
        return logits

    def forward(self, H_j: torch.Tensor, H_i: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, S, H = H_j.shape
        Zj = self.p_j(H_j)   # [B,S,D]
        Zi = self.p_i(H_i)   # [B,S,D]

        logits = self._pair_logits(Zj, Zi, attn_mask)   # [B,S,S]
        probs  = F.softmax(logits, dim=-1)              # [B,S,S]
        ctx_i  = torch.matmul(probs, H_i)               # [B,S,H]

        if self.cfg.hdim_val_mode == "concat_hadam":
            msg_in = torch.cat([ctx_i, H_j, ctx_i * H_j], dim=-1)
        elif self.cfg.hdim_val_mode == "concat_only":
            msg_in = torch.cat([ctx_i, H_j], dim=-1)
        elif self.cfg.hdim_val_mode == "concat_diff_hadam":
            msg_in = torch.cat([ctx_i, H_j, torch.abs(ctx_i - H_j), ctx_i * H_j], dim=-1)
        else:
            raise RuntimeError

        msg = self.val_mlp(msg_in)                      # [B,S,H]
        return self.alpha * self.drop(msg)

# -----------------------
# Encoder (HDIM-only routing, QKV disabled)
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
        self.nh = self.config.num_attention_heads
        assert H % self.nh == 0
        self.hd = H // self.nh

        # Router
        self.route_q = nn.Linear(H, cfg.route_dim)
        self.route_k = nn.Linear(H, cfg.route_dim)

        # Near-zero projection for injected context
        self.proj_by_tgt = nn.ModuleList([nn.Linear(H, H) for _ in range(self.L)])
        for lin in self.proj_by_tgt:
            nn.init.normal_(lin.weight, mean=0.0, std=1e-3); nn.init.zeros_(lin.bias)

        self.pre_ln_by_tgt = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.L)])

        # HDIM modules per target
        self.hdim_by_tgt = nn.ModuleList([
            GeneralizedCrossMessage(H, cfg) for _ in range(self.L)
        ])

        # Gates (QKV path disabled)
        self.alpha_attn = nn.Parameter(torch.zeros(self.L), requires_grad=False)
        self.alpha_hdim = nn.Parameter(torch.full((self.L,), float(cfg.init_gate_hdim)))

        self.drop = nn.Dropout(cfg.dropout)

        # Disable non-used pieces
        for p in self.alpha_attn.parameters() if hasattr(self.alpha_attn, "parameters") else []:
            p.requires_grad = False

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

        for li in range(self.L):
            layer = self.layers[li]

            # Self-attention
            sa_outs = layer.attention(
                hidden_states,
                attention_mask=ext_mask,
                head_mask=head_mask[li] if head_mask is not None else None,
                output_attentions=output_attentions,
            )
            attn_out = sa_outs[0]
            if output_attentions:
                all_attn.append(sa_outs[1])

            # Routing only last N targets
            apply_routing = li > 0 and (li >= self.L - self.cfg.route_last_n)
            add_ctx = None
            if apply_routing:
                H_j_in = hidden_states         # [B,S,H]
                B, S, H = H_j_in.shape
                q_sum = self._layer_summary(H_j_in, self.cfg.route_pool, attention_mask)  # [B,H]
                q_r = self.route_q(q_sum)                                                # [B,Dr]
                src_indices = list(range(li))
                if len(src_indices) > 0:
                    k_list = []
                    for i in src_indices:
                        if i not in layer_sum_cache:
                            H_i_out = layer_out_cache[i]
                            layer_sum_cache[i] = self._layer_summary(H_i_out, self.cfg.route_pool, attention_mask)
                        k_list.append(self.route_k(layer_sum_cache[i]))                  # [B,Dr]
                    K_route = torch.stack(k_list, dim=1)                                 # [B,num_src,Dr]
                    logits_r = torch.einsum("bd,bnd->bn", q_r, K_route) / max(1e-6, self.cfg.route_temp)
                    k = min(self.cfg.topk, logits_r.size(1))
                    top_vals, top_idx = torch.topk(logits_r, k=k, dim=1)                 # [B,k]
                    weights = torch.softmax(top_vals, dim=1)                              # [B,k]

                    # Gather H_i for selected sources
                    H_stack = torch.stack([layer_out_cache[i] for i in src_indices], dim=1)  # [B,num_src,S,H]
                    idx_H = top_idx.view(B, k, 1, 1).expand(B, k, S, H)
                    H_sel = torch.take_along_dim(H_stack, idx_H, dim=1)                       # [B,k,S,H]
                    msg_hdim = torch.zeros_like(H_j_in)
                    for sel in range(k):
                        msg = self.hdim_by_tgt[li](H_j_in, H_sel[:, sel, :, :], attention_mask)  # [B,S,H]
                        msg_hdim = msg_hdim + weights[:, sel].view(B,1,1) * msg

                    add_ctx = self.alpha_hdim[li] * msg_hdim
                    add_ctx = self.pre_ln_by_tgt[li](add_ctx)
                    add_ctx = self.proj_by_tgt[li](add_ctx)
                    add_ctx = self.drop(add_ctx)

            # Injection pre-FFN (fixed)
            if add_ctx is not None and self.cfg.inject_site == "pre_ffn":
                attn_out = attn_out + add_ctx

            inter = layer.intermediate(attn_out)
            layer_output = layer.output(inter, attn_out)
            hidden_states = layer_output

            layer_out_cache[li] = hidden_states
            if output_hidden_states:
                all_hidden.append(hidden_states)

        pooled = hidden_states[:, 0]  # CLS

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
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }

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
# Base HDIM-only cfg (A2-like)
# -----------------------
def base_cfg() -> BridgeCfg:
    return BridgeCfg(
        topk=1,
        route_dim=128,
        route_temp=0.7,
        route_pool="mean",
        route_last_n=4,
        dropout=0.1,
        inject_site="pre_ffn",
        init_gate_attn=0.15,  # irrelevant
        init_gate_hdim=0.05,
        hdim_proj_dim=24,
        hdim_mlp_hidden=96,
        hdim_pair_mode="4term",
        hdim_val_mode="concat_hadam",
        hdim_share_projections=False,
        hdim_token_temp=1.0,
        use_qkv=False,
        use_hdim=True,
        per_head_gates=False,
        adapter_hidden=256
    )

# -----------------------
# 10 HDIM ablations
# -----------------------
ABLATIONS = [
    # Capacity up
    {"name": "H1_proj32_mlp128", "desc":"Proj=32, MLP=128",
     "apply": lambda c: (setattr(c,"hdim_proj_dim",32), setattr(c,"hdim_mlp_hidden",128))},
    # Capacity down (regularization)
    {"name": "H2_proj16_mlp64", "desc":"Proj=16, MLP=64",
     "apply": lambda c: (setattr(c,"hdim_proj_dim",16), setattr(c,"hdim_mlp_hidden",64))},
    # Stronger gate start
    {"name": "H3_gate0p10", "desc":"init_gate_hdim=0.10",
     "apply": lambda c: setattr(c,"init_gate_hdim",0.10)},
    # No dropout inside HDIM
    {"name": "H4_dropout0", "desc":"dropout=0.0",
     "apply": lambda c: setattr(c,"dropout",0.0)},
    # CLS routing instead of mean
    {"name": "H5_routepool_cls", "desc":"route_pool=cls",
     "apply": lambda c: setattr(c,"route_pool","cls")},
    # Consider more sources (last 6)
    {"name": "H6_last6", "desc":"route_last_n=6",
     "apply": lambda c: setattr(c,"route_last_n",6)},
    # Sharper routing (higher temp -> less division effect in logits here we increase temp? For router we *divide* by temp; larger temp softens; smaller temp sharpens. Keep sharper by 0.5)
    {"name": "H7_router_temp0p5", "desc":"route_temp=0.5 (sharper)",
     "apply": lambda c: setattr(c,"route_temp",0.5)},
    # Change pair scoring to bilinear
    {"name": "H8_pair_bilinear", "desc":"pair=bilinear",
     "apply": lambda c: setattr(c,"hdim_pair_mode","bilinear")},
    # Value fusion: concat only (no hadam)
    {"name": "H9_val_concat_only", "desc":"val=concat_only",
     "apply": lambda c: setattr(c,"hdim_val_mode","concat_only")},
    # Share Zj/Zi projections
    {"name": "H10_share_proj", "desc":"hdim_share_projections=True",
     "apply": lambda c: setattr(c,"hdim_share_projections",True)},
]

# -----------------------
# Save-only stats paths
# -----------------------
os.makedirs("stats_hdim10", exist_ok=True)
json_path = os.path.join("stats_hdim10", "hdim10_stats.json")
csv_path  = os.path.join("stats_hdim10", "hdim10_stats.csv")

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

        # capture LR at epoch start
        lr_enc = None
        for g in optimizer.param_groups:
            if "lr" in g:
                lr_enc = g["lr"]; break

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[{exp_name}] Epoch {epoch}/{NUM_EPOCHS}", dynamic_ncols=True, leave=False)
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
# Run all 10 ablations
# -----------------------
all_results: List[Dict] = []
for ab in ABLATIONS:
    cfg = base_cfg()
    ab["apply"](cfg)
    # enforce HDIM-only
    cfg.use_qkv = False
    cfg.use_hdim = True
    cfg.inject_site = "pre_ffn"
    print(f"Running {ab['name']}: {ab['desc']}")
    result = run_experiment(ab["name"], ab["desc"], cfg)
    all_results.append(result)
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# Save JSON
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved detailed JSON stats to {json_path}")

# Save CSV (flat per-epoch)
csv_fields = [
    "exp_name", "exp_desc", "epoch", "time_sec",
    "train_acc", "train_loss_ema", "val_acc", "val_f1_macro",
    "lr",
    "alpha_hdim_shape", "alpha_hdim_mean", "alpha_hdim_std", "alpha_hdim_min", "alpha_hdim_max",
    "param_total", "param_trainable", "param_base_approx", "param_extra_approx",
    "topk", "route_temp", "route_pool", "route_last_n",
    "hdim_proj_dim", "hdim_mlp_hidden", "hdim_pair_mode", "hdim_val_mode",
    "hdim_share_projections", "hdim_token_temp",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for res in all_results:
        base = {
            "exp_name": res["exp_name"],
            "exp_desc": res["exp_desc"],
            "param_total": res["param_counts"]["total"],
            "param_trainable": res["param_counts"]["trainable"],
            "param_base_approx": res["param_counts"]["approx_base"],
            "param_extra_approx": res["param_counts"]["approx_extra"],
            "topk": res["cfg"]["topk"],
            "route_temp": res["cfg"]["route_temp"],
            "route_pool": res["cfg"]["route_pool"],
            "route_last_n": res["cfg"]["route_last_n"],
            "hdim_proj_dim": res["cfg"]["hdim_proj_dim"],
            "hdim_mlp_hidden": res["cfg"]["hdim_mlp_hidden"],
            "hdim_pair_mode": res["cfg"]["hdim_pair_mode"],
            "hdim_val_mode": res["cfg"]["hdim_val_mode"],
            "hdim_share_projections": res["cfg"]["hdim_share_projections"],
            "hdim_token_temp": res["cfg"]["hdim_token_temp"],
        }
        for ep in res["epochs"]:
            row = dict(base)
            row.update({
                "epoch": ep["epoch"],
                "time_sec": ep["time_sec"],
                "train_acc": ep["train_acc"],
                "train_loss_ema": ep["train_loss_ema"],
                "val_acc": ep["val_acc"],
                "val_f1_macro": ep["val_f1_macro"],
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
          f"| proj={res['cfg']['hdim_proj_dim']} | mlp={res['cfg']['hdim_mlp_hidden']} "
          f"| pair={res['cfg']['hdim_pair_mode']} | val={res['cfg']['hdim_val_mode']} "
          f"| pool={res['cfg']['route_pool']} | lastN={res['cfg']['route_last_n']} | temp={res['cfg']['route_temp']} "
          f"| shareP={res['cfg']['hdim_share_projections']}")
