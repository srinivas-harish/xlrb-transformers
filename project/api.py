# api.py
import os
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from db import (
    init_db, get_session, create_run, set_task_id,
    list_runs as db_list_runs, get_run_row, serialize_run,
)
from worker import run_ablation_task

app = FastAPI(title="Ablation API", version="0.2.0")

# ensure tables exist
init_db()

class EarlyStop(BaseModel):
    epoch1_val_below: Optional[float] = None

class RunRequest(BaseModel):
    ablation: Optional[str] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)
    epochs: int = 3
    batch_size: int = 8
    max_len: int = 128
    save_dir: Optional[str] = None
    save_artifacts: bool = False
    early_stop: Optional[EarlyStop] = None
    gradient_checkpointing: bool = True
    device: Optional[str] = None

@app.get("/")
def root():
    return {"ok": True, "msg": "Ablation service ready", "endpoints": ["/runs (POST)", "/runs (GET)", "/runs/{run_id} (GET)"]}

@app.post("/runs")
def submit_run(req: RunRequest):
    with get_session() as s:
        run = create_run(
            s,
            ablation=req.ablation,
            overrides=req.overrides,
            epochs=req.epochs,
            batch_size=req.batch_size,
            max_len=req.max_len,
            device=req.device or "auto",
            save_dir=req.save_dir,
            save_artifacts=req.save_artifacts,
            early_stop=req.early_stop.model_dump() if req.early_stop else None,
        )
        run_id = run.id

    # enqueue Celery task carrying the run_id and full request payload
    async_result = run_ablation_task.delay(run_id, req.model_dump())

    with get_session() as s:
        set_task_id(s, run_id, async_result.id)

    return {"run_id": run_id, "task_id": async_result.id, "status": "QUEUED"}

@app.get("/runs")
def list_runs(limit: int = Query(50, ge=1, le=500), status: Optional[str] = None):
    with get_session() as s:
        rows = db_list_runs(s, limit=limit, status=status)
        return [serialize_run(r, with_children=False) for r in rows]

@app.get("/runs/{run_id}")
def get_run(run_id: str):
    with get_session() as s:
        row = get_run_row(s, run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        return serialize_run(row, with_children=True)
