# worker.py
import os, json, shutil
from typing import Optional
from celery import Celery

from main import (
    base_cfg_hybrid, ABLATIONS, BridgeCfg, pick_device
)
from db import (
    get_session, update_status, append_epoch, complete_run, add_artifact, init_db, get_run_row
)
from profiling import (
    have_nsys, have_ncu, run_with_nsys, export_nsys_stats, parse_nsys_csvs, run_with_ncu, parse_nsys_sqlite, parse_ncu_raw
)
import subprocess
import threading
import time

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

        # Decide save_dir (default to project/runs/<run_id> if not provided)
        save_dir = req.get("save_dir") or os.path.join(os.path.dirname(__file__), "runs", run_id)
        os.makedirs(save_dir, exist_ok=True)

        # Persist the exact request used
        req_path = os.path.join(save_dir, "run_req.json")
        with open(req_path, "w") as f:
            json.dump(req, f, indent=2)

        # Build base runner command
        pyexe = shutil.which("python") or "python"
        runner_py = os.path.join(os.path.dirname(__file__), "runner.py")
        # progress file path for live epoch updates
        progress_path = os.path.join(save_dir, "progress.jsonl")
        base_cmd = [pyexe, runner_py, "--infile", req_path, "--run-id", run_id, "--save-dir", save_dir, "--progress-file", progress_path]

        # 1) Run full training under Nsight Systems if available
        nsys_dir = os.path.join(save_dir, "nsys")
        ncu_dir = os.path.join(save_dir, "ncu")
        os.makedirs(nsys_dir, exist_ok=True)
        os.makedirs(ncu_dir, exist_ok=True)

        report_path = os.path.join(save_dir, "report.json")
        profiling_summary = None
        ncu_kernels: Optional[dict] = None
        # Diagnostics container for frontend visibility when kernels=0
        diag: dict = {
            "detected_tools": {"nsys": have_nsys(), "ncu": have_ncu()},
            "artifacts": {"nsys_rep": False, "nsys_sqlite": False, "ncu_rep": False},
            "csv_reports": [],
            "kernels_source": None,
            "counts": {},
            "notes": [],
        }
        try:
            import platform
            rel = platform.release().lower()
            if "microsoft" in rel or os.environ.get("WSL_DISTRO_NAME"):
                diag.setdefault("env", {})["is_wsl"] = True
        except Exception:
            pass

        # Tail progress file in a background thread to push epochs to DB live
        stop_evt = threading.Event()

        def tail_progress(path: str, rid: str, stop_event: threading.Event):
            last_size = 0
            seen_epochs = set()
            while not stop_event.is_set():
                try:
                    if os.path.exists(path):
                        size = os.path.getsize(path)
                        if size < last_size:
                            # file truncated/rotated
                            last_size = 0
                        if size > last_size:
                            with open(path, "r") as f:
                                f.seek(last_size)
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        obj = json.loads(line)
                                        ep = int(obj.get("epoch"))
                                        if ep not in seen_epochs:
                                            with get_session() as s:
                                                append_epoch(s, rid, obj)
                                            seen_epochs.add(ep)
                                    except Exception:
                                        pass
                                last_size = f.tell()
                except Exception:
                    pass
                time.sleep(0.5)

        t = threading.Thread(target=tail_progress, args=(progress_path, run_id, stop_evt), daemon=True)
        t.start()
        if have_nsys():
            out_base = os.path.join(nsys_dir, run_id)
            nsys_res = run_with_nsys(base_cmd, out_base)
            # Export CSV stats and parse
            rep_path = nsys_res.get("qdrep") or nsys_res.get("nsys_rep")
            if rep_path:
                stats_out_base = os.path.join(nsys_dir, f"{run_id}.stats")
                stats_res = export_nsys_stats(rep_path, stats_out_base)
                diag["artifacts"]["nsys_rep"] = True
                if stats_res.get("csv"):
                    try:
                        diag["csv_reports"] = sorted(list((stats_res.get("csv") or {}).keys()))
                    except Exception:
                        pass
                    profiling_summary = parse_nsys_csvs(stats_res["csv"]) or {}
                    try:
                        diag["counts"]["nsys_csv_kernels"] = len((profiling_summary or {}).get("kernels") or [])
                    except Exception:
                        pass
                    if (profiling_summary or {}).get("kernels") and not diag.get("kernels_source"):
                        diag["kernels_source"] = "nsys_csv"
                    # If CSV parsing produced weak/empty summary fields, enrich from SQLite if present
                    sqlite_path = nsys_res.get("sqlite")
                    if sqlite_path and os.path.exists(sqlite_path):
                        try:
                            parsed_sql = parse_nsys_sqlite(sqlite_path) or {}
                            diag["artifacts"]["nsys_sqlite"] = True
                            if parsed_sql:
                                for k, v in parsed_sql.items():
                                    if (k not in profiling_summary) or (profiling_summary.get(k) in (None, [], {}, 0) and v not in (None, [], {})):
                                        profiling_summary[k] = v
                                try:
                                    if parsed_sql.get("kernels") and not diag.get("kernels_source"):
                                        diag["kernels_source"] = "nsys_sqlite"
                                    diag["counts"]["nsys_sqlite_kernels"] = len(parsed_sql.get("kernels") or [])
                                except Exception:
                                    pass
                        except Exception:
                            pass
                # Fallback: try parsing the .sqlite directly if CSV export failed entirely
                if not profiling_summary:
                    sqlite_path = nsys_res.get("sqlite")
                    if sqlite_path and os.path.exists(sqlite_path):
                        try:
                            profiling_summary = parse_nsys_sqlite(sqlite_path) or {}
                            diag["artifacts"]["nsys_sqlite"] = True
                            try:
                                if (profiling_summary or {}).get("kernels") and not diag.get("kernels_source"):
                                    diag["kernels_source"] = "nsys_sqlite"
                                diag["counts"]["nsys_sqlite_kernels"] = len((profiling_summary or {}).get("kernels") or [])
                            except Exception:
                                pass
                        except Exception:
                            profiling_summary = {}
                # Register artifacts
                if rep_path and os.path.exists(rep_path):
                    with get_session() as s:
                        add_artifact(s, run_id, "nsys", rep_path, bytes=os.path.getsize(rep_path))
                # Also register the SQLite timeline if present
                sqlite_path = nsys_res.get("sqlite")
                if sqlite_path and os.path.exists(sqlite_path):
                    with get_session() as s:
                        add_artifact(s, run_id, "nsys-sqlite", sqlite_path, bytes=os.path.getsize(sqlite_path))
                for name, path in (stats_res.get("csv") or {}).items():
                    if os.path.exists(path):
                        with get_session() as s:
                            add_artifact(s, run_id, "csv", path, bytes=os.path.getsize(path))
                # If CSV export failed, persist stderr for debugging and still surface a hint
                if not profiling_summary:
                    try:
                        if stats_res.get("stderr"):
                            err_path = os.path.join(nsys_dir, f"{run_id}.stats.stderr.txt")
                            with open(err_path, "w") as ef:
                                ef.write(stats_res.get("stderr") or "")
                            with get_session() as s:
                                add_artifact(s, run_id, "txt", err_path, bytes=os.path.getsize(err_path))
                    except Exception:
                        pass
            else:
                # Fallback: run without nsys if something failed
                subprocess.run(base_cmd, check=True)
        else:
            # No Nsight Systems; run directly
            subprocess.run(base_cmd, check=True)

        # 2) Optionally run Nsight Compute on a short window (epochs=1, limited batches)
        if have_ncu():
            # Write a short-run req override file
            short_req = dict(req)
            short_req["epochs"] = 1
            short_req["max_train_batches"] = int(min(30, max(5, int(req.get("batch_size", 8)))))
            short_req_path = os.path.join(save_dir, "run_req_ncu.json")
            with open(short_req_path, "w") as f:
                json.dump(short_req, f, indent=2)
            short_cmd = [pyexe, runner_py, "--infile", short_req_path, "--run-id", f"{run_id}-ncu", "--save-dir", save_dir]
            ncu_out_base = os.path.join(ncu_dir, run_id)
            ncu_res = run_with_ncu(short_cmd, ncu_out_base)
            if ncu_res.get("ncu_rep") and os.path.exists(ncu_res["ncu_rep"]):
                with get_session() as s:
                    add_artifact(s, run_id, "ncu", ncu_res["ncu_rep"], bytes=os.path.getsize(ncu_res["ncu_rep"]))
                # Parse kernels from the NCU report (best-effort)
                try:
                    parsed_ncu = parse_ncu_raw(ncu_res["ncu_rep"]) or None
                except Exception:
                    parsed_ncu = None
                diag["artifacts"]["ncu_rep"] = True
                if parsed_ncu and parsed_ncu.get("kernels"):
                    ncu_kernels = parsed_ncu
                    # record counts and source
                    try:
                        diag["counts"]["ncu_kernels"] = len(parsed_ncu.get("kernels") or [])
                    except Exception:
                        pass
                    if not diag.get("kernels_source"):
                        diag["kernels_source"] = "ncu_raw"
            else:
                # Persist stderr to help diagnose missing NCU report (e.g., restricted counters on Windows/WSL)
                try:
                    err_txt = (ncu_res or {}).get("stderr")
                    if err_txt:
                        err_path = os.path.join(ncu_dir, f"{run_id}.ncu.stderr.txt")
                        with open(err_path, "w") as ef:
                            ef.write(err_txt)
                        with get_session() as s:
                            add_artifact(s, run_id, "txt", err_path, bytes=os.path.getsize(err_path))
                    # Add a brief hint to diagnostics
                    tried_sets = (ncu_res or {}).get("tried_sets") or []
                    if tried_sets:
                        diag["notes"].append(f"Nsight Compute did not produce a report; tried sets: {' '.join(sum(tried_sets, []))}.")
                    else:
                        diag["notes"].append("Nsight Compute did not produce a report. If on Windows/WSL, enable GPU performance counters in NVIDIA Control Panel or try running again.")
                except Exception:
                    pass

        # Load report
        if not os.path.exists(report_path):
            raise RuntimeError(f"Training report not found at {report_path}")
        with open(report_path) as f:
            result = json.load(f)
        result.setdefault("profiling", {})
        profiling_out = result["profiling"]

        if profiling_summary:
            # Split concise and detailed views for the UI
            # Summary keys are a subset; details include lists
            summary_keys = {
                "gpu_busy_pct", "total_gpu_time_ms", "memcpy_time_ms", "memset_time_ms",
                "transfer_overhead_pct", "num_unique_kernels"
            }
            nsys_sum = {k: v for k, v in profiling_summary.items() if k in summary_keys}
            nsys_details = {k: v for k, v in profiling_summary.items() if k not in summary_keys}
            profiling_out["nsys_summary"] = nsys_sum
            if nsys_details:
                profiling_out["nsys_details"] = nsys_details
        else:
            # If we have an Nsight artifact but no summary, surface a marker so UI can show guidance
            profiling_out.setdefault("nsys_summary", None)

        # If we extracted kernels from NCU, merge them into details so UI shows them
        if ncu_kernels and ncu_kernels.get("kernels"):
            # Ensure details object
            details = profiling_out.setdefault("nsys_details", {})
            if not details.get("kernels"):
                details["kernels"] = ncu_kernels["kernels"]
            # Ensure summary exists (object, not None) so the UI doesn't show the empty notice
            if profiling_out.get("nsys_summary") is None:
                profiling_out["nsys_summary"] = {}
            # Fill num_unique_kernels if missing
            try:
                if "num_unique_kernels" not in profiling_out["nsys_summary"] or profiling_out["nsys_summary"]["num_unique_kernels"] in (None, 0):
                    profiling_out["nsys_summary"]["num_unique_kernels"] = int(ncu_kernels.get("num_unique_kernels") or len(ncu_kernels.get("kernels") or []))
            except Exception:
                pass

        # Attach diagnostics; include a hint if kernels still look zero
        try:
            details = profiling_out.get("nsys_details") or {}
            kernels_list = (details.get("kernels") or [])
            num_from_sum = None
            try:
                num_from_sum = profiling_out.get("nsys_summary", {}).get("num_unique_kernels")
            except Exception:
                num_from_sum = None
            if (not kernels_list) and (not num_from_sum or int(num_from_sum) == 0):
                if diag.get("env", {}).get("is_wsl"):
                    diag["notes"].append("WSL detected: Nsight Systems may not capture CUPTI kernel tables; relying on Nsight Compute artifacts.")
                if diag["artifacts"].get("ncu_rep") and diag["counts"].get("ncu_kernels", 0) == 0:
                    diag["notes"].append("Nsight Compute report found but no kernels parsed. Ensure GPU performance counters are enabled in Windows NVIDIA Control Panel, and that the NCU run captured kernels.")
                    diag["notes"].append("Hint: older runs may have used a restrictive kernel_regex; current config profiles all kernels (regex ). Re-run to populate kernels.")    
        except Exception:
            pass
        profiling_out["debug"] = diag

        # Optional: over-write report.json to include profiling summary if we added it
        if result.get("save_dir") and (profiling_summary is not None or ncu_kernels is not None):
            try:
                with open(report_path, "w") as f:
                    json.dump(result, f, indent=2)
            except Exception:
                pass

        # Stop progress tailer and persist to DB
        stop_evt.set()
        t.join(timeout=2.0)

        # Persist to DB
        with get_session() as s:
            existing_epochs = set()
            try:
                row = get_run_row(s, run_id)
                if row and row.epochs_rel:
                    existing_epochs = {int(e.epoch) for e in row.epochs_rel}
            except Exception:
                existing_epochs = set()
            for ep in result.get("epochs", []):
                try:
                    if int(ep.get("epoch")) in existing_epochs:
                        continue
                except Exception:
                    pass
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
