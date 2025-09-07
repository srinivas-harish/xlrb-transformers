import argparse
import json
import os
from typing import Any, Dict, Optional

from main import train_and_eval, base_cfg_hybrid, ABLATIONS, pick_device


def run_from_req(req: Dict[str, Any], *, run_id: Optional[str] = None, save_dir: Optional[str] = None, progress_file: Optional[str] = None) -> Dict[str, Any]:
    device = pick_device(req.get("device"))
    cfg = base_cfg_hybrid()

    ab = req.get("ablation")
    if ab and ab in ABLATIONS:
        cfg = ABLATIONS[ab](cfg)

    # Apply overrides onto cfg and training knobs
    overrides = req.get("overrides") or {}
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    epochs = int(req.get("epochs", 3))
    batch_size = int(req.get("batch_size", 8))
    max_len = int(req.get("max_len", 128))
    lr_encoder = float(overrides.get("lr_encoder", 1.5e-5))
    lr_head = float(overrides.get("lr_head", 3e-4))
    weight_decay = float(overrides.get("weight_decay", 0.01))
    warmup_frac = float(overrides.get("warmup_frac", 0.10))
    label_smoothing = float(overrides.get("label_smoothing", 0.05))
    max_grad_norm = float(overrides.get("max_grad_norm", 1.0))
    gradient_checkpointing = bool(req.get("gradient_checkpointing", True))
    early_stop = req.get("early_stop")

    # Optional runner-specific overrides
    max_train_batches = req.get("max_train_batches")
    if max_train_batches is not None:
        try:
            max_train_batches = int(max_train_batches)
        except Exception:
            max_train_batches = None

    # Decide save_dir
    eff_save_dir = save_dir or req.get("save_dir")
    if eff_save_dir:
        os.makedirs(eff_save_dir, exist_ok=True)

    result = train_and_eval(
        task_ctx=None,
        device=device,
        cfg=cfg,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay,
        warmup_frac=warmup_frac,
        label_smoothing=label_smoothing,
        max_grad_norm=max_grad_norm,
        save_dir=eff_save_dir,
        save_artifacts=bool(req.get("save_artifacts", False)),
        early_stop=early_stop,
        gradient_checkpointing=gradient_checkpointing,
        max_train_batches=max_train_batches,
        progress_path=progress_file,
    )

    # Always write a report.json if save_dir present
    if eff_save_dir:
        out_path = os.path.join(eff_save_dir, "report.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    p = argparse.ArgumentParser(description="Runner for training (with optional external profilers)")
    p.add_argument("--infile", required=True, help="Path to JSON request payload (same as API body)")
    p.add_argument("--run-id", required=False, help="Run ID (for directory naming only)")
    p.add_argument("--save-dir", required=False, help="Override save_dir in payload")
    p.add_argument("--progress-file", required=False, help="If set, write per-epoch JSON lines for live progress")
    args = p.parse_args()

    with open(args.infile) as f:
        req = json.load(f)

    run_from_req(req, run_id=args.run_id, save_dir=args.save_dir, progress_file=args.progress_file)


if __name__ == "__main__":
    main()
