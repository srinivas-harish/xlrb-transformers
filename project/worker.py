# worker.py
import os
from typing import Any, Dict
from celery import Celery

from main import (
    base_cfg_hybrid, ABLATIONS, train_and_eval,
    pick_device, set_all_seeds
)
 
os.environ["IN_CELERY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "ablation_service",
    broker=os.getenv("CELERY_BROKER_URL", REDIS_URL),
    backend=os.getenv("CELERY_RESULT_BACKEND", REDIS_URL),
)

@celery_app.task(bind=True)
def run_ablation_task(self, payload: Dict[str, Any]):
    ablation = payload.get("ablation")
    overrides = payload.get("overrides", {}) or {}
    epochs = int(payload.get("epochs", 3))
    batch_size = int(payload.get("batch_size", 8))
    max_len = int(payload.get("max_len", 128))
    save_dir = payload.get("save_dir")
    save_artifacts = bool(payload.get("save_artifacts", False))
    early_stop = payload.get("early_stop")
    gradient_checkpointing = bool(payload.get("gradient_checkpointing", True))
    force_device = payload.get("device")  # "cpu" | "cuda" | None

    # training hyperparams optionally in overrides:
    lr_encoder = float(overrides.pop("lr_encoder", 1.5e-5))
    lr_head = float(overrides.pop("lr_head", 3e-4))
    weight_decay = float(overrides.pop("weight_decay", 0.01))
    warmup_frac = float(overrides.pop("warmup_frac", 0.10))
    label_smoothing = float(overrides.pop("label_smoothing", 0.05))
    max_grad_norm = float(overrides.pop("max_grad_norm", 1.0))

    device = pick_device(force_device)
    set_all_seeds()

    cfg = base_cfg_hybrid()
    if ablation:
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation '{ablation}'.")
        cfg = ABLATIONS[ablation](cfg)

    # arbitrary BridgeCfg overrides:
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    result = train_and_eval(
        task_ctx=self, device=device, cfg=cfg,
        epochs=epochs, batch_size=batch_size, max_len=max_len,
        lr_encoder=lr_encoder, lr_head=lr_head, weight_decay=weight_decay,
        warmup_frac=warmup_frac, label_smoothing=label_smoothing, max_grad_norm=max_grad_norm,
        save_dir=save_dir, save_artifacts=save_artifacts,
        early_stop=early_stop, gradient_checkpointing=gradient_checkpointing,
    )
    return result
