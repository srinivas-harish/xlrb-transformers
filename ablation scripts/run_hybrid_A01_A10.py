# RoBERTa-large on GLUE RTE — Hybrid bridges (QKV cross-layer + lightweight HDIM), 10 ablations
# Deterministic runner. Early-stop ablation if epoch 1 < 70% acc.
 
# %%capture
#!pip install -q transformers==4.44.2 datasets==2.21.0 scikit-learn==1.5.2 tqdm==4.66.4

# %%
import os, math, random, numpy as np, torch, time
from dataclasses import dataclass, replace
from typing import Dict, Optional, List

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
os.environ["PYTHONHASHSEED"] = str(SEED)
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# cuDNN determinism knobs (safe for this model)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

# Deterministic DataLoader shuffling
g = torch.Generator()
g.manual_seed(SEED)
def seed_worker(worker_id):
    wseed = SEED + worker_id
    random.seed(wseed)
    np.random.seed(wseed)
    torch.manual_seed(wseed)

BATCH_SIZE = 8
train_loader = DataLoader(
    tokenized["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
    num_workers=2,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g
)
val_loader   = DataLoader(
    tokenized["validation"],
    batch_size=32,
    shuffle=False,
    collate_fn=collator,
    num_workers=2,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g
)

# -----------------------
# Lightweight HDIM message  
# -----------------------
class GeneralizedCrossMessage(nn.Module):
    def __init__(
        self,
        hidden_size:int,
        proj_dim:int=24,
        mlp_hidden:int=96,
        dropout:float=0.1,
        gate_init:float=0.05,
        pair_mode:str="4term",           # "4term" | "bilinear"
        act:str="relu"                   # "relu" | "gelu"
    ):
        super().__init__()
        self.H = hidden_size; self.D = proj_dim
        self.p_j = nn.Linear(self.H, self.D, bias=False)
        self.p_i = nn.Linear(self.H, self.D, bias=False)

        # scorer
        self.pair_mode = pair_mode
        if pair_mode == "4term":
            act_layer = nn.GELU() if act.lower() == "gelu" else nn.ReLU(inplace=True)
            self.score_mlp = nn.Sequential(
                nn.Linear(4*self.D, mlp_hidden), act_layer, nn.Linear(mlp_hidden, 1)
            )
            self.W_bilinear = None
        elif pair_mode == "bilinear":
            self.score_mlp = None
            self.W_bilinear = nn.Parameter(torch.empty(self.D, self.D))
            nn.init.xavier_uniform_(self.W_bilinear)
        else:
            raise ValueError("pair_mode must be '4term' or 'bilinear'")

        # value fusion MLP (concat_hadam fixed like your script)
        out_hid = 2 * max(128, self.H // 2)
        act_layer_v = nn.GELU() if act.lower() == "gelu" else nn.ReLU(inplace=True)
        self.val_mlp = nn.Sequential(
            nn.Linear(3*self.H, out_hid), act_layer_v, nn.Linear(out_hid, self.H)
        )

        self.drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(float(gate_init)))  # small >0

    def _expand_mask_last(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (1.0 - mask[:, None, :]) * torch.finfo(dtype).min

    def forward(self, H_j: torch.Tensor, H_i: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, S, H = H_j.shape
        Zj = self.p_j(H_j); Zi = self.p_i(H_i)       # [B,S,D]

        if self.pair_mode == "4term":
            Zj_e = Zj.unsqueeze(2); Zi_e = Zi.unsqueeze(1)
            Zj_pair = Zj_e.expand(-1, S, S, -1); Zi_pair = Zi_e.expand(-1, S, S, -1)
            hadam = Zj_pair * Zi_pair
            diff  = torch.abs(Zj_pair - Zi_pair)
            pair  = torch.cat([Zj_pair, Zi_pair, hadam, diff], dim=-1)   # [B,S,S,4D]
            logits = self.score_mlp(pair).squeeze(-1)                    # [B,S,S]
        else:  # bilinear
            ZjW = torch.matmul(Zj, self.W_bilinear)                      # [B,S,D]
            logits = torch.matmul(ZjW, Zi.transpose(1, 2))               # [B,S,S]

        if attn_mask is not None:
            logits = logits + self._expand_mask_last(attn_mask, logits.dtype)
        probs = F.softmax(logits, dim=-1)                                 # [B,S,S]
        ctx_i = torch.matmul(probs, H_i)                                   # [B,S,H]
        msg_in = torch.cat([ctx_i, H_j, ctx_i * H_j], dim=-1)
        msg = self.val_mlp(msg_in)                                         # [B,S,H]
        return self.alpha * self.drop(msg)

# -----------------------
# Hybrid encoder: QKV cross-layer + lightweight HDIM (both gated, LN, near-zero proj) 
# -----------------------
@dataclass
class BridgeCfg:
    topk: int = 1
    route_dim: int = 128
    route_temp: float = 0.7
    route_pool: str = "mean"     # "mean" | "cls"
    route_last_n: int = 4        # only last 4 targets
    dropout: float = 0.1
    # gates (per-target learned scalars start slightly on)
    init_gate_attn: float = 0.15
    init_gate_hdim: float = 0.05
    # hdim dims
    hdim_proj_dim: int = 24
    hdim_mlp_hidden: int = 96
    hdim_pair_mode: str = "4term"    # "4term" | "bilinear"
    hdim_act: str = "relu"           # "relu" | "gelu"
    # ablation toggles
    enable_qkv: bool = True
    enable_hdim: bool = True
    qkv_target_weights: bool = False     # use target layer's key/value for all sources
    use_bridge_WV: bool = False          # dedicated learned W_V on the bridge path
    inject_last_m: Optional[int] = None  # if set, inject only into last m targets

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

        # Router
        self.route_q = nn.Linear(H, cfg.route_dim)
        self.route_k = nn.Linear(H, cfg.route_dim)

        # Per-target projection for injected ctx (near-zero init)
        self.proj_by_tgt = nn.ModuleList([nn.Linear(H, H) for _ in range(self.L)])
        for lin in self.proj_by_tgt:
            nn.init.normal_(lin.weight, mean=0.0, std=1e-3); nn.init.zeros_(lin.bias)

        # Pre-LN for add_ctx
        self.pre_ln_by_tgt = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.L)])

        # HDIM modules per target
        self.hdim_by_tgt = nn.ModuleList([
            GeneralizedCrossMessage(
                H,
                proj_dim=cfg.hdim_proj_dim,
                mlp_hidden=cfg.hdim_mlp_hidden,
                dropout=cfg.dropout,
                gate_init=cfg.init_gate_hdim,
                pair_mode=cfg.hdim_pair_mode,
                act=cfg.hdim_act
            )
            for _ in range(self.L)
        ])

        # Learned per-target gates for QKV and HDIM components
        self.alpha_attn = nn.Parameter(torch.full((self.L,), float(cfg.init_gate_attn)))
        self.alpha_hdim = nn.Parameter(torch.full((self.L,), float(cfg.init_gate_hdim)))
        self.drop = nn.Dropout(cfg.dropout)

        # Optional dedicated bridge W_V (per target layer, per head-dim)
        if cfg.use_bridge_WV:
            self.bridge_WV_by_tgt = nn.ModuleList([nn.Linear(self.hd, self.hd, bias=False) for _ in range(self.L)])
        else:
            self.bridge_WV_by_tgt = None

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

            # Routing (only last N targets)
            apply_routing = li > 0 and (li >= self.L - self.cfg.route_last_n)
            inject_here = True if self.cfg.inject_last_m is None else (li >= self.L - self.cfg.inject_last_m)

            if apply_routing and inject_here:
                H_j_in = hidden_states                                   # [B,S,H]
                B, S, H = H_j_in.shape

                # Router summaries
                q_sum = self._layer_summary(H_j_in, self.cfg.route_pool, attention_mask)  # [B,H]
                q_r = self.route_q(q_sum)                                                # [B,Dr]
                src_indices = list(range(li))
                if len(src_indices) > 0:
                    # keys for routing
                    k_list = []
                    for i in src_indices:
                        if i not in layer_sum_cache:
                            H_i_out = layer_out_cache[i]
                            layer_sum_cache[i] = self._layer_summary(H_i_out, self.cfg.route_pool, attention_mask)
                        k_list.append(self.route_k(layer_sum_cache[i]))                  # [B,Dr]
                    K_route = torch.stack(k_list, dim=1)                                 # [B,num_src,Dr]
                    logits = torch.einsum("bd,bnd->bn", q_r, K_route) / max(1e-6, self.cfg.route_temp)
                    k = min(self.cfg.topk, logits.size(1))
                    top_vals, top_idx = torch.topk(logits, k=k, dim=1)                   # [B,k]
                    weights = torch.softmax(top_vals, dim=1)                              # [B,k]

                    # -------- QKV message (vectorized over k), optional toggles --------
                    ctx_qkv = None
                    if self.cfg.enable_qkv:
                        Qj = self._shape_heads(layer.attention.self.query(H_j_in))       # [B,nh,S,hd]

                        # Source K/V: either per-source weights or target layer's weights
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

                        K_stack = torch.stack(K_stack, dim=1)                             # [B,num_src,nh,S,hd]
                        V_stack = torch.stack(V_stack, dim=1)

                        nh, hd = Qj.size(1), Qj.size(-1)
                        idx = top_idx.view(B, k, 1, 1, 1).expand(B, k, nh, S, hd)
                        K_sel = torch.take_along_dim(K_stack, idx, dim=1)                 # [B,k,nh,S,hd]
                        V_sel = torch.take_along_dim(V_stack, idx, dim=1)                 # [B,k,nh,S,hd]

                        # Optional dedicated bridge W_V on the selected V
                        if self.bridge_WV_by_tgt is not None:
                            WV = self.bridge_WV_by_tgt[li]
                            V_flat = V_sel.reshape(-1, hd)                                 # [(B*k*nh*S), hd]
                            V_flat = WV(V_flat)
                            V_sel = V_flat.view(B, k, nh, S, hd)

                        Q_rep = Qj.unsqueeze(1).expand(B, k, nh, S, hd)
                        Bk = B * k
                        Q_bk = Q_rep.reshape(Bk, nh, S, hd)
                        K_bk = K_sel.reshape(Bk, nh, S, hd)
                        V_bk = V_sel.reshape(Bk, nh, S, hd)
                        ext_mask_bk = ext_mask.repeat_interleave(k, dim=0) if ext_mask is not None else None
                        C_bk = self._attend(Q_bk, K_bk, V_bk, ext_mask_bk)                # [Bk,S,H]
                        C_bk = C_bk.view(B, k, S, H)
                        ctx_qkv = (weights.view(B, k, 1, 1) * C_bk).sum(dim=1)            # [B,S,H]

                    # -------- HDIM message (weighted over k), optional --------
                    msg_hdim = None
                    if self.cfg.enable_hdim:
                        H_stack = torch.stack([layer_out_cache[i] for i in src_indices], dim=1)  # [B,num_src,S,H]
                        idx_H = top_idx.view(B, k, 1, 1).expand(B, k, S, H)
                        H_sel = torch.take_along_dim(H_stack, idx_H, dim=1)                       # [B,k,S,H]
                        msg_hdim = torch.zeros_like(H_j_in)
                        for sel in range(k):
                            msg_hdim = msg_hdim + weights[:, sel].view(B,1,1) * self.hdim_by_tgt[li](H_j_in, H_sel[:, sel, :, :], attention_mask)

                    # -------- Combine (according to which are enabled), LN, proj, inject --------
                    add_ctx = torch.zeros_like(H_j_in)
                    if ctx_qkv is not None:
                        add_ctx = add_ctx + self.alpha_attn[li] * ctx_qkv
                    if msg_hdim is not None:
                        add_ctx = add_ctx + self.alpha_hdim[li] * msg_hdim

                    add_ctx = self.pre_ln_by_tgt[li](add_ctx)
                    add_ctx = self.drop(self.proj_by_tgt[li](add_ctx))
                    attn_out = attn_out + add_ctx

            # FFN
            inter = layer.intermediate(attn_out)
            layer_output = layer.output(inter, attn_out)
            hidden_states = layer_output

            # cache post-layer states
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
        return SequenceClassifierOutput(logits=logits, loss=loss,
                                        hidden_states=enc_out.hidden_states, attentions=enc_out.attentions)

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

base_cfg = BridgeCfg(
    topk=1,
    route_dim=128,
    route_temp=0.7,
    route_pool="mean",
    route_last_n=4,
    dropout=0.1,
    init_gate_attn=0.15,
    init_gate_hdim=0.05,
    hdim_proj_dim=24,
    hdim_mlp_hidden=96,
    hdim_pair_mode="4term",
    hdim_act="relu",
    enable_qkv=True,
    enable_hdim=True,
    qkv_target_weights=False,
    use_bridge_WV=False,
    inject_last_m=None,
)

# -----------------------
# Utils
# -----------------------
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
    for batch in tqdm(loader, total=len(loader), desc="Valid", dynamic_ncols=True, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}

def train_one(cfg: BridgeCfg, save_dir: str, early_stop_if_epoch1_under: float = 0.70) -> Dict[str, float]:
    # reset seeds per run
    set_seed(SEED); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    model = RobertaHybridForSequenceClassification(checkpoint, cfg, num_labels=2, dropout=0.1)
    model.gradient_checkpointing_enable()
    model.to(device)

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

    os.makedirs(save_dir, exist_ok=True)

    best_acc = -1.0
    no_improve = 0
    EARLY_STOP_PATIENCE = 2

    epoch1_val = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        start = time.time()
        ema_loss = None
        correct = 0
        seen = 0

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{NUM_EPOCHS}", dynamic_ncols=True, leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = out.logits
                loss = ce_loss(logits, batch["labels"])

            # running stats
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            seen += batch["labels"].size(0)
            acc_running = correct / max(1, seen)

            # ema loss
            loss_val = loss.item()
            ema_loss = loss_val if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_val

            # step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "ema": f"{ema_loss:.4f}", "acc": f"{acc_running:.4f}"})

        # Evaluate
        val_metrics = evaluate(model, val_loader)
        elapsed = time.time() - start
        print(f"\nEpoch {epoch} | time {elapsed:.1f}s | val_acc {val_metrics['accuracy']:.4f} | val_f1 {val_metrics['f1_macro']:.4f}")

        #  if epoch 1 < threshold
        if epoch == 1:
            epoch1_val = val_metrics['accuracy']
            if epoch1_val < early_stop_if_epoch1_under:
                print(f"Early-stop: epoch-1 acc {epoch1_val:.4f} < {early_stop_if_epoch1_under:.2f}.")
                break

        # Save best
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            print(f"New best acc {best_acc:.4f} -> saving to {save_dir}")
            torch.save({
                "encoder_state": model.encoder.state_dict(),
                "head_state": model.head.state_dict(),
                "bridge_cfg": cfg.__dict__,
            }, os.path.join(save_dir, "model.pt"))
            tokenizer.save_pretrained(save_dir)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best acc {best_acc:.4f})")
                break

    return {"best_acc": float(best_acc), "epoch1_acc": float(epoch1_val if epoch1_val is not None else -1.0)}

# -----------------------
# 10 Ablations (names + toggles)
# -----------------------
ABLATIONS = [
    # 1) Dedicated W_V on bridge path
    ("A1_QKV_WVbridge", lambda c: replace(c, use_bridge_WV=True)),
    # 2) Use target layer's K/V for all sources
    ("A2_QKV_TargetW", lambda c: replace(c, qkv_target_weights=True)),
    # 3) QKV only
    ("A3_QKV_only", lambda c: replace(c, enable_qkv=True, enable_hdim=False)),
    # 4) HDIM only
    ("A4_HDIM_only", lambda c: replace(c, enable_qkv=False, enable_hdim=True)),
    # 5) HDIM bilinear scorer
    ("A5_HDIM_bilinear", lambda c: replace(c, hdim_pair_mode="bilinear")),
    # 6) HDIM higher capacity
    ("A6_HDIM_proj32_mlp128", lambda c: replace(c, hdim_proj_dim=32, hdim_mlp_hidden=128)),
    # 7) HDIM GELU nonlinearity
    ("A7_HDIM_GELU", lambda c: replace(c, hdim_act="gelu")),
    # 8) Router CLS pooling
    ("A8_Router_CLS", lambda c: replace(c, route_pool="cls")),
    # 9) Router temp 0.5 (sharper)
    ("A9_Router_Temp0p5", lambda c: replace(c, route_temp=0.5)),
    # 10) Inject only into last 2 target layers
    ("A10_Inject_Last2", lambda c: replace(c, inject_last_m=2)),
]

 
results = []
for name, apply in ABLATIONS:
    print("\n" + "="*80)
    print(f"Running ablation: {name}")
    print("="*80)
    cfg = apply(base_cfg)
    save_dir = f"rte_hybrid_best_{name}"
    out = train_one(cfg, save_dir=save_dir, early_stop_if_epoch1_under=0.70)
    results.append((name, out["best_acc"], out["epoch1_acc"]))

print("\n" + "#"*80)
print("Ablation summary (best_acc | epoch1_acc)")
for name, best_acc, ep1 in results:
    print(f"{name:>20s}  | best={best_acc:.4f} | ep1={ep1:.4f}")
print("#"*80)

 
if len(ABLATIONS) > 0:
    last_name = ABLATIONS[-1][0]
    ckpt_path = os.path.join(f"rte_hybrid_best_{last_name}", "model.pt")
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_state_dict=False, map_location="cpu")
        reload_model = RobertaHybridForSequenceClassification(
            checkpoint,
            BridgeCfg(**sd["bridge_cfg"]),
            num_labels=2,
            dropout=0.1
        )
        reload_model.encoder.load_state_dict(sd["encoder_state"])
        reload_model.head.load_state_dict(sd["head_state"])
        reload_model.to(device).eval()
        print(f"Reloaded best model from {ckpt_path}")
