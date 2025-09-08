# api.py
import os
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from db import (
    init_db, get_session, create_run, set_task_id,
    list_runs as db_list_runs, get_run_row, serialize_run,
)
from worker import run_ablation_task
from profiling import (
    parse_nsys_sqlite, export_nsys_stats, parse_nsys_csvs, parse_ncu_raw,
    have_nsys, have_ncu, run_cuda_kernel_test
)
import glob, json, os

app = FastAPI(title="Ablation API", version="0.2.0")

# CORS for local frontend dev (Vite on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
        "http://localhost:5175", "http://127.0.0.1:5175",
        "http://0.0.0.0:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/runs/{run_id}/refresh_profiling")
def refresh_profiling(run_id: str):
    """
    Best-effort backfill of profiling summary/details for an existing run by
    parsing the saved Nsight artifacts (sqlite or rep).
    """
    with get_session() as s:
        row = get_run_row(s, run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        result = (row.result_json or {})
        profiling = result.setdefault("profiling", {})

        # Try to find a sqlite timeline in artifacts or save_dir
        sqlite_paths: list[str] = []
        for a in (row.artifacts or []):
            p = a.path or ""
            if p.endswith(".sqlite") and os.path.exists(p):
                sqlite_paths.append(p)
        # fallback: search in save_dir/nsys
        if not sqlite_paths and row.save_dir:
            sqlite_paths.extend(glob.glob(os.path.join(row.save_dir, "nsys", "*.sqlite")))

        summary = None
        details = None
        diag: dict = {
            "detected_tools": {"nsys": have_nsys(), "ncu": have_ncu()},
            "artifacts": {"nsys_rep": False, "nsys_sqlite": False, "ncu_rep": False},
            "csv_reports": [],
            "kernels_source": None,
            "counts": {},
            "notes": [],
        }
        try:
            import platform, os as _os
            rel = platform.release().lower()
            if "microsoft" in rel or _os.environ.get("WSL_DISTRO_NAME"):
                diag.setdefault("env", {})["is_wsl"] = True
        except Exception:
            pass
        # Existing artifacts snapshot
        for a in (row.artifacts or []):
            p = (a.path or "").lower()
            if p.endswith(".qdrep") or p.endswith(".nsys-rep"):
                diag["artifacts"]["nsys_rep"] = True
            if p.endswith(".sqlite"):
                diag["artifacts"]["nsys_sqlite"] = True
            if p.endswith(".ncu-rep"):
                diag["artifacts"]["ncu_rep"] = True
        # First try sqlite parser
        for sp in sqlite_paths:
            try:
                parsed = parse_nsys_sqlite(sp)
                if parsed:
                    summary_keys = {
                        "gpu_busy_pct", "total_gpu_time_ms", "memcpy_time_ms", "memset_time_ms",
                        "transfer_overhead_pct", "num_unique_kernels"
                    }
                    summary = {k: v for k, v in parsed.items() if k in summary_keys}
                    details = {k: v for k, v in parsed.items() if k not in summary_keys}
                    try:
                        diag["counts"]["nsys_sqlite_kernels"] = len((details or {}).get("kernels") or [])
                        diag["kernels_source"] = diag.get("kernels_source") or ("nsys_sqlite" if (details or {}).get("kernels") else None)
                    except Exception:
                        pass
                    break
            except Exception:
                continue
        # If still nothing, attempt CSV export if a rep exists
        if not summary:
            rep_paths: list[str] = []
            for a in (row.artifacts or []):
                p = a.path or ""
                if (p.endswith(".qdrep") or p.endswith(".nsys-rep")) and os.path.exists(p):
                    rep_paths.append(p)
            for rp in rep_paths:
                try:
                    outb = os.path.splitext(rp)[0] + ".stats.refresh"
                    stats = export_nsys_stats(rp, outb)
                    if stats.get("csv"):
                        parsed = parse_nsys_csvs(stats["csv"]) or {}
                        try:
                            diag["csv_reports"] = sorted(list((stats.get("csv") or {}).keys()))
                        except Exception:
                            pass
                        if parsed:
                            summary_keys = {
                                "gpu_busy_pct", "total_gpu_time_ms", "memcpy_time_ms", "memset_time_ms",
                                "transfer_overhead_pct", "num_unique_kernels"
                            }
                            summary = {k: v for k, v in parsed.items() if k in summary_keys}
                            details = {k: v for k, v in parsed.items() if k not in summary_keys}
                            try:
                                diag["counts"]["nsys_csv_kernels"] = len((details or {}).get("kernels") or [])
                                if (details or {}).get("kernels") and not diag.get("kernels_source"):
                                    diag["kernels_source"] = "nsys_csv"
                            except Exception:
                                pass
                            break
                except Exception:
                    continue

        # If we still don't have kernel details, try NCU raw CSV import from any .ncu-rep
        if not details:
            ncu_paths: list[str] = []
            for a in (row.artifacts or []):
                p = a.path or ""
                if p.endswith(".ncu-rep") and os.path.exists(p):
                    ncu_paths.append(p)
            # fallback: search in save_dir/ncu
            if not ncu_paths and row.save_dir:
                ncu_paths.extend(glob.glob(os.path.join(row.save_dir, "ncu", "*.ncu-rep")))
            for np in ncu_paths:
                try:
                    parsed_ncu = parse_ncu_raw(np)
                    if parsed_ncu and parsed_ncu.get("kernels"):
                        details = (details or {})
                        details["kernels"] = parsed_ncu["kernels"]
                        # Make sure summary exists and has num_unique_kernels
                        summary = summary or {}
                        if "num_unique_kernels" not in summary or summary["num_unique_kernels"] in (None, 0):
                            summary["num_unique_kernels"] = int(parsed_ncu.get("num_unique_kernels") or len(parsed_ncu.get("kernels") or []))
                        try:
                            diag["counts"]["ncu_kernels"] = len(parsed_ncu.get("kernels") or [])
                            if not diag.get("kernels_source"):
                                diag["kernels_source"] = "ncu_raw"
                        except Exception:
                            pass
                        break
                except Exception:
                    continue

        if not (summary or details):
            raise HTTPException(status_code=422, detail="No Nsight artifacts could be parsed")

        if summary is not None:
            profiling["nsys_summary"] = summary
        if details is not None:
            profiling["nsys_details"] = details
        profiling["debug"] = diag

        # Persist back into DB
        row.result_json = result

        return {"ok": True, "profiling": profiling}


@app.post("/diag/kernel_test")
def diag_kernel_test():
    """
    Run a dummy CUDA workload under Nsight Compute to verify kernel capture.
    Returns detailed diagnostics including kernel list and an optional CSV preview.
    """
    try:
        res = run_cuda_kernel_test()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
