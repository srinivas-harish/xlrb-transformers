# FastAPI app + routes, imports celery_app + task from worker.py

# api.py
from typing import Any, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from celery.result import AsyncResult

from worker import celery_app, run_ablation_task
 

app = FastAPI(title="Ablation Service")

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "ablation",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


# ---- Pydantic models (API-facing) ----
class EarlyStopSpec(BaseModel):
    epoch1_val_below: float = Field(..., description="If val acc after epoch 1 is below this, stop early")

class RunRequest(BaseModel):
    ablation: Optional[str] = Field(None, description="Preset like A4_HDIM_only or H1_proj32_mlp128")
    overrides: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary knobs (BridgeCfg or train knobs)")
    epochs: int = 3
    batch_size: int = 8
    max_len: int = 128
    save_dir: Optional[str] = None
    save_artifacts: bool = False
    early_stop: Optional[EarlyStopSpec] = None
    gradient_checkpointing: bool = True
    device: Optional[str] = Field(None, description='"cpu" or "cuda" (auto if omitted)')
    dataloader_workers: int | None = None  # <— NEW

class RunResponse(BaseModel):
    task_id: str

class StatusResponse(BaseModel):
    status: str
    meta: Optional[Dict[str, Any]] = None

# ---- Routes ----
@app.post("/run", response_model=RunResponse)
def run_job(req: RunRequest):
    payload = req.dict()
    task = run_ablation_task.delay(payload)
    return RunResponse(task_id=task.id)

@app.get("/status/{task_id}", response_model=StatusResponse)
def get_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if res.state == "PENDING":
        return StatusResponse(status="PENDING")
    if res.state == "STARTED":
        return StatusResponse(status="STARTED", meta=res.info if isinstance(res.info, dict) else {"info": str(res.info)})
    if res.state == "FAILURE":
        return StatusResponse(status="FAILURE", meta={"error": str(res.result)})
    if res.state == "SUCCESS":
        return StatusResponse(status="SUCCESS", meta={"ready": True})
    return StatusResponse(status=res.state)

@app.get("/result/{task_id}")
def get_result(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if not res.ready():
        return {"status": res.state}
    if res.failed():
        return {"status": "FAILURE", "error": str(res.result)}
    return {"status": "SUCCESS", "result": res.result}
