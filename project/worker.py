# worker.py
import os, json
from celery import Celery

from main import (
    train_and_eval, base_cfg_hybrid, ABLATIONS, BridgeCfg, pick_device
)
from db import (
    get_session, update_status, append_epoch, complete_run, add_artifact, init_db
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("ablation_service", broker=REDIS_URL, backend=REDIS_URL)

# mark we're inside a Celery worker so DataLoader uses workers=0 (your main.py honors IN_CELERY)
os.environ["IN_CELERY"] = "1"

# ensure DB exists in worker process too
init_db()

@celery_app.task(name="worker.run_ablation_task", bind=True)
def run_ablation_task(self, run_id: str, req: dict):
    """
    req is the exact POST body (RunRequest). We:
      1) mark RUNNING
      2) run training
      3) persist per-epoch rows + artifacts
      4) mark COMPLETE (or FAILED)
    """
    try:
        with get_session() as s:
            update_status(s, run_id, "RUNNING")

        device = pick_device(req.get("device"))
        cfg = base_cfg_hybrid()

        # Apply preset
        ab = req.get("ablation")
        if ab and ab in ABLATIONS:
            cfg = ABLATIONS[ab](cfg)

        # Apply overrides to BridgeCfg if present
        for k, v in (req.get("overrides") or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        # Trainer call
        result = train_and_eval(
            task_ctx=self,
            device=device,
            cfg=cfg,
            epochs=req.get("epochs", 3),
            batch_size=req.get("batch_size", 8),
            max_len=req.get("max_len", 128),
            lr_encoder=(req.get("overrides") or {}).get("lr_encoder", 1.5e-5),
            lr_head=(req.get("overrides") or {}).get("lr_head", 3e-4),
            weight_decay=(req.get("overrides") or {}).get("weight_decay", 0.01),
            warmup_frac=(req.get("overrides") or {}).get("warmup_frac", 0.10),
            label_smoothing=(req.get("overrides") or {}).get("label_smoothing", 0.05),
            max_grad_norm=(req.get("overrides") or {}).get("max_grad_norm", 1.0),
            save_dir=req.get("save_dir"),
            save_artifacts=req.get("save_artifacts", False),
            early_stop=req.get("early_stop"),
            gradient_checkpointing=req.get("gradient_checkpointing", True),
        )

        # Optional: write a report.json next to artifacts
        save_dir = result.get("save_dir")
        report_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            report_path = os.path.join(save_dir, "report.json")
            with open(report_path, "w") as f:
                json.dump(result, f, indent=2)

        # Persist to DB
        with get_session() as s:
            for ep in result.get("epochs", []):
                append_epoch(s, run_id, ep)
            if save_dir and report_path and os.path.exists(report_path):
                add_artifact(s, run_id, "report", report_path, bytes=os.path.getsize(report_path))
            model_path = os.path.join(save_dir, "model.pt") if save_dir else None
            if model_path and os.path.exists(model_path):
                add_artifact(s, run_id, "checkpoint", model_path, bytes=os.path.getsize(model_path))
            complete_run(s, run_id, result_json=result, best_val_acc=(result.get("best") or {}).get("val_acc"))

        return result

    except Exception as e:
        with get_session() as s:
            update_status(s, run_id, "FAILED", error=str(e))
        raise
