import os
import json
import shutil
import subprocess
from typing import Dict, Any, Optional, List, Tuple
import re
import io
import glob
import sys
import uuid
import tempfile


def _which(exe: str) -> Optional[str]:
    """
    More robust which() that also searches common CUDA install locations so the
    Celery/uvicorn environment can find Nsight tools even if PATH is minimal.
    """
    p = shutil.which(exe)
    if p:
        return p
    # Try CUDA_HOME / CUDA_PATH
    for env in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if not env:
            continue
        cand = os.path.join(env, "bin", exe)
        if os.path.exists(cand):
            return cand
        if os.name == "nt":
            cand_exe = cand + ".exe"
            if os.path.exists(cand_exe):
                return cand_exe
    # Common Linux locations
    candidates = []
    # /usr/local/cuda, versions, and Nsight standalone installs
    candidates.extend(glob.glob("/usr/local/cuda*/bin/" + exe))
    candidates.append(f"/opt/nvidia/nsight-compute/{exe}")
    candidates.append(f"/opt/nvidia/nsight-systems/{exe}")
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def have_nsys() -> bool:
    return _which("nsys") is not None


def have_ncu() -> bool:
    return _which("ncu") is not None


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def _list_nsys_reports(nsys_path: str) -> List[str]:
    """Return available report names from `nsys stats` output (best-effort)."""
    # Some versions support `nsys stats --list report`, others only show in --help
    for args in (["--list", "report"], ["--help"]):
        code, out, err = run_cmd([nsys_path, "stats", *args])
        text = (out or "") + "\n" + (err or "")
        lines = [l.strip() for l in text.splitlines()]
        candidates = []
        for l in lines:
            # Heuristic: report names are single tokens without spaces or are shown like "- report <name>"
            # Collect tokens that look like names
            for tok in l.replace(",", " ").split():
                t = tok.strip().lower()
                if len(t) < 3:
                    continue
                # Ignore flags and placeholders
                if t.startswith("-") or t.startswith("--"):
                    continue
                # Keep alnum/dash tokens
                ok = all(ch.isalnum() or ch in ("-", "_") for ch in t)
                if ok:
                    candidates.append(t)
        # Dedup and return something reasonable
        uniq = sorted(set(candidates))
        # Filter to plausible reports
        plaus = [r for r in uniq if any(k in r for k in ["sum", "summary", "kern", "gpu", "api", "nvtx", "mem"])]
        if plaus:
            return plaus
    return []


def run_with_nsys(base_cmd: List[str], out_base: str, *, trace: str = "cuda,nvtx,osrt") -> Dict[str, Any]:
    """
    Wraps base_cmd with Nsight Systems. Returns paths to .qdrep and .sqlite if produced.
    """
    nsys = _which("nsys")
    if not nsys:
        return {"ok": False, "error": "nsys not found in PATH"}
    out_dir = os.path.dirname(out_base)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        nsys, "profile",
        "--force-overwrite=true",
        "--sample=cpu",
        "--trace", trace,
        "--stats=true",
        "-o", out_base,
        *base_cmd,
    ]
    code, out, err = run_cmd(cmd)
    # Nsight Systems has used .qdrep historically; newer versions default to .nsys-rep
    qdrep = out_base + ".qdrep"
    nsys_rep = out_base + ".nsys-rep"
    rep_path = qdrep if os.path.exists(qdrep) else (nsys_rep if os.path.exists(nsys_rep) else None)
    # Proactively export a full timeline SQLite from the .rep to ensure kernel tables exist
    sqlite = out_base + ".sqlite"
    if rep_path and not os.path.exists(sqlite):
        # Try multiple export syntaxes to handle version differences
        export_cmds = [
            [nsys, "export", "--sqlite", "true", "--force-overwrite=true", "-o", out_base, rep_path],
            [nsys, "export", "--type", "sqlite", "--force-overwrite=true", "-o", out_base, rep_path],
            [nsys, "export", "-t", "sqlite", "--force-overwrite=true", "-o", out_base, rep_path],
        ]
        for ec in export_cmds:
            try:
                run_cmd(ec)
                if os.path.exists(sqlite):
                    break
            except Exception:
                continue
    return {
        "ok": code == 0,
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "qdrep": qdrep if os.path.exists(qdrep) else None,
        "nsys_rep": nsys_rep if os.path.exists(nsys_rep) else None,
        "sqlite": sqlite if os.path.exists(sqlite) else None,
    }


def export_nsys_stats(rep_path: str, out_base: str) -> Dict[str, Any]:
    """
    Exports CSV stats from a .qdrep into several CSVs via `nsys stats`.
    Returns a dict with produced CSV paths.
    """
    nsys = _which("nsys")
    if not nsys or not os.path.exists(rep_path):
        return {"ok": False, "error": "nsys or report missing"}
    out_dir = os.path.dirname(out_base)
    os.makedirs(out_dir, exist_ok=True)

    # Detect available report names and map to canonical buckets
    avail = _list_nsys_reports(nsys)
    avail_set = set(avail)
    def pick(predicate) -> Optional[str]:
        # Choose the first available report matching the predicate
        for r in avail:
            if predicate(r):
                return r
        return None

    def has(name: str) -> bool:
        return name in avail_set

    # Map canonical -> actual
    mapping: Dict[str, Optional[str]] = {
        # The report names vary by Nsight Systems version. Try to detect; if not,
        # fall back through several known aliases (dash, underscore, legacy).
        "summary": (
            pick(lambda r: "summary" in r and "api" not in r and "mem" not in r and "nvtx" not in r)
            or ("summary" if has("summary") else None)
        ),
        "gpu-kern-summary": (
            pick(lambda r: "kern" in r and "gpu" in r)
            or ("cuda_gpu_kern_sum" if has("cuda_gpu_kern_sum") else None)
            or ("gpukernsum" if has("gpukernsum") else None)
        ),
        "gpu-activities": (
            pick(lambda r: "gpu" in r and ("activit" in r))
            or ("gpu_activities" if has("gpu_activities") else None)
            or ("gpu-activities" if has("gpu-activities") else None)
        ),
        "cuda-apisum": (
            pick(lambda r: "api" in r and "cuda" in r)
            or ("cuda_api_sum" if has("cuda_api_sum") else None)
            or ("cuda-apisum" if has("cuda-apisum") else ("cudaapisum" if has("cudaapisum") else None))
        ),
        "cuda-memory": (
            pick(lambda r: "mem" in r and "cuda" in r)
            or ("cuda_gpu_mem_size_sum" if has("cuda_gpu_mem_size_sum") else None)
            or ("cuda-memory" if has("cuda-memory") else ("cudamemory" if has("cudamemory") else None))
        ),
        "nvtxsum": (
            pick(lambda r: "nvtx" in r and ("sum" in r or "summary" in r))
            or ("nvtx_sum" if has("nvtx_sum") else None)
            or ("nvtxsum" if has("nvtxsum") else None)
        ),
    }

    produced: Dict[str, str] = {}
    last_out = ""; last_err = ""; last_code = 0
    # Export each mapped report individually with distinct -o basename
    for canon, actual in mapping.items():
        if not actual:
            continue
        outb = f"{out_base}.{canon}"
        cmd = [
            nsys, "stats",
            "--report", actual,
            "--format", "csv",
            "--force-overwrite=true",
            "-o", outb,
            rep_path,
        ]
        code, out, err = run_cmd(cmd)
        last_code, last_out, last_err = code, out, err
        p = f"{outb}.{actual}.csv"  # exact naming by nsys
        if not os.path.exists(p):
            # Some versions name as outb.<canon>.csv even if report differs
            alt = f"{outb}.{canon}.csv"
            if os.path.exists(alt):
                p = alt
        if os.path.exists(p):
            produced[canon] = p

    if produced:
        return {"ok": True, "returncode": 0, "stdout": last_out, "stderr": last_err, "csv": produced}
    # Last‑ditch: scan directory for any CSVs produced with unexpected names
    try:
        base_dir = os.path.dirname(out_base) or "."
        base_name = os.path.basename(out_base)
        found = {}
        for fn in os.listdir(base_dir):
            if not fn.endswith(".csv"):
                continue
            if not fn.startswith(base_name):
                continue
            fpath = os.path.join(base_dir, fn)
            lname = fn.lower()
            key = None
            if "kern" in lname and "gpu" in lname:
                key = "gpu-kern-summary"
            elif "activit" in lname and "gpu" in lname:
                key = "gpu-activities"
            elif "api" in lname and "cuda" in lname:
                key = "cuda-apisum"
            elif "mem" in lname and "cuda" in lname:
                key = "cuda-memory"
            elif "nvtx" in lname:
                key = "nvtxsum"
            elif "summary" in lname:
                key = "summary"
            if key and key not in found:
                found[key] = fpath
        if found:
            return {"ok": True, "returncode": last_code, "stdout": last_out, "stderr": last_err, "csv": found}
    except Exception:
        pass
    return {"ok": False, "returncode": last_code, "stdout": last_out, "stderr": last_err, "csv": {}}


def parse_nsys_csvs(csvs: Dict[str, str]) -> Dict[str, Any]:
    """
    Best-effort parser for a few high-level metrics:
    - total_gpu_time_ms
    - memcpy_time_ms, memset_time_ms, transfer_overhead_pct
    - num_unique_kernels
    - wall_time_ms (from summary)
    
    CSV schemas vary across versions; this parser is defensive.
    """
    import csv

    total_gpu_time_ms = 0.0
    memcpy_time_ms = 0.0
    memset_time_ms = 0.0
    wall_time_ms = None
    unique_kernels = set()
    kernels: List[Dict[str, Any]] = []
    cuda_api: List[Dict[str, Any]] = []
    transfers_map: Dict[str, Dict[str, Any]] = {}

    # gpu-activities: sum times and memcpy/memset buckets; also fallback kernel aggregation
    act_path = csvs.get("gpu-activities")
    kern_agg: Dict[str, Dict[str, Any]] = {}
    if act_path and os.path.exists(act_path):
        with open(act_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Find time column (varies: "Time (ns)" or "timeNs" etc.)
                time_ns = None
                for k in row.keys():
                    lk = k.lower()
                    if "time" in lk and "ns" in lk:
                        try:
                            time_ns = float(row[k])
                            break
                        except Exception:
                            continue
                if time_ns is None:
                    continue
                total_gpu_time_ms += time_ns / 1e6
                # Try common name columns
                name = (
                    row.get("Name")
                    or row.get("name")
                    or row.get("Activity Name")
                    or row.get("Activity" )
                    or ""
                )
                name = str(name)
                lname = name.lower()
                # Transfers buckets
                if "memcpy" in lname:
                    memcpy_time_ms += time_ns / 1e6
                    d = transfers_map.setdefault(name, {"name": name, "time_ms": 0.0, "calls": 0})
                    d["time_ms"] += time_ns / 1e6
                    d["calls"] += 1
                    continue
                if "memset" in lname:
                    memset_time_ms += time_ns / 1e6
                    d = transfers_map.setdefault(name, {"name": name, "time_ms": 0.0, "calls": 0})
                    d["time_ms"] += time_ns / 1e6
                    d["calls"] += 1
                    continue
                # Fallback kernel aggregation from activities when a dedicated kernel summary isn't available
                if name:
                    d = kern_agg.setdefault(name, {"name": name, "time_ns": 0.0, "calls": 0})
                    d["time_ns"] += time_ns
                    d["calls"] += 1

    # gpu-kern-summary: count unique kernel names
    kern_path = csvs.get("gpu-kern-summary")
    if kern_path and os.path.exists(kern_path):
        with open(kern_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name") or row.get("name") or row.get("Kernel Name")
                if name:
                    unique_kernels.add(name)
                    # total/avg times and calls
                    calls = None
                    tot_ns = None
                    avg_ns = None
                    for k, v in row.items():
                        lk = (k or "").lower()
                        try:
                            if calls is None and ("calls" in lk or "invocations" in lk):
                                calls = int(float(v))
                            if tot_ns is None and ("time" in lk and "ns" in lk and ("total" in lk or "sum" in lk or lk.endswith("time (ns)"))):
                                tot_ns = float(v)
                            if avg_ns is None and ("avg" in lk and "ns" in lk):
                                avg_ns = float(v)
                        except Exception:
                            pass
                    kernels.append({
                        "name": name,
                        "calls": calls,
                        "time_ms": (tot_ns / 1e6) if tot_ns is not None else None,
                        "avg_ms": (avg_ns / 1e6) if avg_ns is not None else None,
                    })
    # If we didn't get a dedicated kernel summary, synthesize kernels from activities
    if not kernels and kern_agg:
        for name, d in kern_agg.items():
            unique_kernels.add(name)
            t_ms = d["time_ns"] / 1e6
            c = int(d.get("calls") or 0)
            kernels.append({
                "name": name,
                "calls": c or None,
                "time_ms": t_ms,
                "avg_ms": (t_ms / c) if c else None,
            })

    # summary: wall clock time
    sum_path = csvs.get("summary")
    if sum_path and os.path.exists(sum_path):
        with open(sum_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    lk = (k or "").lower()
                    if "elapsed" in lk or "duration" in lk:
                        try:
                            wall_time_ns = float(v)
                            wall_time_ms = wall_time_ns / 1e6
                            break
                        except Exception:
                            pass

    transfer_overhead_pct = None
    gpu_busy_pct = None
    if total_gpu_time_ms > 0:
        transfer_overhead_pct = 100.0 * (memcpy_time_ms + memset_time_ms) / total_gpu_time_ms
    if wall_time_ms and wall_time_ms > 0:
        gpu_busy_pct = 100.0 * (total_gpu_time_ms / wall_time_ms)

    # CUDA API summary
    api_path = csvs.get("cuda-apisum")
    if api_path and os.path.exists(api_path):
        with open(api_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name") or row.get("API Name") or row.get("name")
                if not name:
                    continue
                calls = None
                tot_ns = None
                avg_ns = None
                for k, v in row.items():
                    lk = (k or "").lower()
                    try:
                        if calls is None and ("calls" in lk or "invocations" in lk):
                            calls = int(float(v))
                        if tot_ns is None and ("time" in lk and "ns" in lk and ("total" in lk or "sum" in lk or lk.endswith("time (ns)"))):
                            tot_ns = float(v)
                        if avg_ns is None and ("avg" in lk and "ns" in lk):
                            avg_ns = float(v)
                    except Exception:
                        pass
                cuda_api.append({
                    "name": name,
                    "calls": calls,
                    "time_ms": (tot_ns / 1e6) if tot_ns is not None else None,
                    "avg_ms": (avg_ns / 1e6) if avg_ns is not None else None,
                })

    # If we didn't get GPU kernel/mem activity totals, approximate total time
    # from CUDA API sums to avoid empty summary cards.
    if (not total_gpu_time_ms) and cuda_api:
        try:
            total_api_time_ms = sum((r.get("time_ms") or 0.0) for r in cuda_api)
            if total_api_time_ms:
                total_gpu_time_ms = total_api_time_ms
                if (memcpy_time_ms or memset_time_ms):
                    transfer_overhead_pct = 100.0 * (memcpy_time_ms + memset_time_ms) / total_gpu_time_ms
        except Exception:
            pass

    # post-process kernels with pct of total GPU time
    kernels_proc: List[Dict[str, Any]] = []
    for k in kernels:
        pct_gpu = None
        if total_gpu_time_ms and k.get("time_ms"):
            pct_gpu = 100.0 * (k["time_ms"] / total_gpu_time_ms)
        kernels_proc.append({**k, "pct_gpu_time": pct_gpu})
    # transfers list from map
    transfers = sorted(transfers_map.values(), key=lambda d: d.get("time_ms", 0.0), reverse=True)

    # Prefer showing 0 kernels instead of null if none were found
    num_unique_kernels = len(unique_kernels) if unique_kernels else (0 if not kernels_proc else len({k["name"] for k in kernels_proc if k.get("name")}))

    return {
        "gpu_busy_pct": gpu_busy_pct,
        "total_gpu_time_ms": total_gpu_time_ms or None,
        "memcpy_time_ms": memcpy_time_ms or None,
        "memset_time_ms": memset_time_ms or None,
        "transfer_overhead_pct": transfer_overhead_pct,
        "num_unique_kernels": num_unique_kernels,
        "kernels": sorted(kernels_proc, key=lambda d: d.get("time_ms", 0.0) or 0.0, reverse=True)[:30],
        "cuda_api": sorted(cuda_api, key=lambda d: d.get("time_ms", 0.0) or 0.0, reverse=True)[:30],
        "transfers": transfers[:30],
    }


def parse_nsys_sqlite(sqlite_path: str) -> Optional[Dict[str, Any]]:
    """
    Fallback parser that reads key metrics directly from Nsight Systems .sqlite export.
    This avoids relying on `nsys stats --report ...` names which vary across versions.

    Best-effort extraction (will return partials when certain tables are missing):
      - cuda_api: aggregated by API name from CUPTI_ACTIVITY_KIND_RUNTIME
      - kernels: aggregated by kernel name from any *KERNEL* activity table, if present
      - memcpy/memset: aggregated from CUPTI_ACTIVITY_KIND_MEMCPY / _MEMSET, if present
      - total_gpu_time_ms: sum of kernel + memcpy + memset times (approximate)
      - gpu_busy_pct: total_gpu_time_ms / wall_time_ms (approximate)
    """
    import sqlite3

    if not os.path.exists(sqlite_path):
        return None

    def table_exists(cur, name: str) -> bool:
        cur.execute("select 1 from sqlite_master where type='table' and name=?", (name,))
        return cur.fetchone() is not None

    def list_tables(cur) -> List[str]:
        cur.execute("select name from sqlite_master where type='table'")
        return [r[0] for r in cur.fetchall()]

    try:
        con = sqlite3.connect(sqlite_path)
        cur = con.cursor()
    except Exception:
        return None

    # CUDA API summary from runtime table
    cuda_api: List[Dict[str, Any]] = []
    wall_start = None
    wall_end = None
    try:
        if table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"):
            # Aggregate by function name
            q = (
                "select s.value as name, count(*) as calls, "
                "sum(r.end - r.start) as total_ns, avg(r.end - r.start) as avg_ns, "
                "min(r.start) as min_start, max(r.end) as max_end "
                "from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId "
                "group by r.nameId order by total_ns desc"
            )
            try:
                for name, calls, total_ns, avg_ns, min_start, max_end in cur.execute(q):
                    if wall_start is None or (min_start is not None and min_start < wall_start):
                        wall_start = min_start
                    if wall_end is None or (max_end is not None and max_end > wall_end):
                        wall_end = max_end
                    cuda_api.append({
                        "name": name,
                        "calls": int(calls) if calls is not None else None,
                        "time_ms": (total_ns / 1e6) if total_ns is not None else None,
                        "avg_ms": (avg_ns / 1e6) if avg_ns is not None else None,
                    })
            except Exception:
                # If heavy aggregate fails, fall back to a lighter pass
                try:
                    q2 = "select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId"
                    agg: Dict[str, Dict[str, Any]] = {}
                    for start, end, name in cur.execute(q2):
                        if wall_start is None or (start is not None and start < wall_start):
                            wall_start = start
                        if wall_end is None or (end is not None and end > wall_end):
                            wall_end = end
                        d = agg.setdefault(name, {"name": name, "calls": 0, "time_ns": 0.0})
                        d["calls"] += 1
                        if start is not None and end is not None:
                            d["time_ns"] += float(end - start)
                    for v in agg.values():
                        cuda_api.append({
                            "name": v["name"],
                            "calls": v["calls"],
                            "time_ms": v["time_ns"] / 1e6,
                            "avg_ms": (v["time_ns"] / v["calls"]) / 1e6 if v["calls"] else None,
                        })
                except Exception:
                    pass
    except Exception:
        pass

    # Kernel, memcpy, memset aggregation
    kernels: List[Dict[str, Any]] = []
    memcpy_time_ms = 0.0
    memset_time_ms = 0.0
    total_gpu_time_ms = 0.0
    num_unique_kernels: Optional[int] = None

    try:
        tbls = list_tables(cur)
        # Aggregate across any plausible kernel tables; different versions use different names
        kern_agg: Dict[str, Dict[str, Any]] = {}
        for t in tbls:
            tl = t.lower()
            if ("kernel" not in tl) or not ("cupti" in tl or "cuda" in tl or "gpu" in tl):
                continue
            # Inspect columns
            try:
                cur.execute(f"pragma table_info('{t}')")
                cols = [r[1] for r in cur.fetchall()]
            except Exception:
                continue
            cols_l = [c.lower() for c in cols]
            if not ("start" in cols_l and "end" in cols_l):
                continue
            # Try to find an identifier for the kernel name
            name_id_col = None
            name_str_col = None
            # Prefer demangledNameId, but accept any *nameId column
            if "demanglednameid" in cols_l:
                name_id_col = cols[cols_l.index("demanglednameid")]
            else:
                # fallback: any column ending with 'nameid'
                for c in cols:
                    cl = c.lower()
                    if cl.endswith("nameid"):
                        name_id_col = c
                        break
            # If no id, try direct string columns like demangledName/name/shortName
            if not name_id_col:
                for pref in ("demangledname", "name", "shortname"):
                    if pref in cols_l:
                        name_str_col = cols[cols_l.index(pref)]
                        break
                if not name_str_col:
                    # any column ending with 'name'
                    for c in cols:
                        if c.lower().endswith("name"):
                            name_str_col = c
                            break
            # Build and execute aggregation query for this table
            try:
                if name_id_col:
                    qk = (
                        f"select s.value as name, count(*) as calls, "
                        f"sum(k.end - k.start) as total_ns, avg(k.end - k.start) as avg_ns "
                        f"from {t} k join StringIds s on s.id = k.{name_id_col} "
                        f"group by k.{name_id_col}"
                    )
                elif name_str_col:
                    qk = (
                        f"select k.{name_str_col} as name, count(*) as calls, "
                        f"sum(k.end - k.start) as total_ns, avg(k.end - k.start) as avg_ns "
                        f"from {t} k group by k.{name_str_col}"
                    )
                else:
                    continue
                for name, calls, total_ns, avg_ns in cur.execute(qk):
                    if not name:
                        continue
                    d = kern_agg.setdefault(name, {"name": name, "calls": 0, "total_ns": 0.0, "sum_avg_ns": 0.0, "avg_cnt": 0})
                    d["calls"] += int(calls) if calls is not None else 0
                    if total_ns is not None:
                        d["total_ns"] += float(total_ns)
                    if avg_ns is not None:
                        d["sum_avg_ns"] += float(avg_ns)
                        d["avg_cnt"] += 1
            except Exception:
                # best effort on each table
                continue
        if kern_agg:
            for d in kern_agg.values():
                t_ms = d["total_ns"] / 1e6 if d.get("total_ns") else None
                avg_ns = (d["sum_avg_ns"] / d["avg_cnt"]) if d.get("avg_cnt") else None
                kernels.append({
                    "name": d["name"],
                    "calls": d["calls"] or None,
                    "time_ms": t_ms,
                    "avg_ms": (avg_ns / 1e6) if avg_ns is not None else None,
                })
            # unique names across all tables
            num_unique_kernels = len(kern_agg)
            total_gpu_time_ms += sum((k.get("time_ms") or 0.0) for k in kernels)
    except Exception:
        pass

    # memcpy/memset tables
    try:
        for t in list_tables(cur):
            tl = t.lower()
            if "memcpy" in tl or "memset" in tl:
                cur.execute(f"pragma table_info('{t}')")
                cols = [r[1].lower() for r in cur.fetchall()]
                if not ("start" in cols and "end" in cols):
                    continue
                q = f"select start, end from {t}"
                acc_ms = 0.0
                try:
                    for start, end in cur.execute(q):
                        if start is None or end is None:
                            continue
                        acc_ms += (float(end - start) / 1e6)
                except Exception:
                    continue
                if "memcpy" in tl:
                    memcpy_time_ms += acc_ms
                if "memset" in tl:
                    memset_time_ms += acc_ms
    except Exception:
        pass

    total_gpu_time_ms += memcpy_time_ms + memset_time_ms

    # Compute GPU busy if we have wall time
    wall_time_ms = None
    if wall_start is not None and wall_end is not None and wall_end > wall_start:
        wall_time_ms = (float(wall_end - wall_start) / 1e6)
    gpu_busy_pct = None
    if wall_time_ms and wall_time_ms > 0:
        gpu_busy_pct = 100.0 * (total_gpu_time_ms / wall_time_ms) if total_gpu_time_ms else 0.0

    # Add % GPU for kernels if total time known
    kernels_proc: List[Dict[str, Any]] = []
    for k in kernels:
        pct_gpu = None
        if total_gpu_time_ms and k.get("time_ms"):
            pct_gpu = 100.0 * (k["time_ms"] / total_gpu_time_ms)
        kernels_proc.append({**k, "pct_gpu_time": pct_gpu})

    # Sort outputs
    cuda_api_sorted = sorted(cuda_api, key=lambda d: d.get("time_ms", 0.0) or 0.0, reverse=True)[:30]
    kernels_sorted = sorted(kernels_proc, key=lambda d: d.get("time_ms", 0.0) or 0.0, reverse=True)[:30]

    # Build transfers list from CUDA API calls as a fallback
    transfers: List[Dict[str, Any]] = []
    if not transfers and cuda_api_sorted:
        agg: Dict[str, Dict[str, Any]] = {}
        for r in cuda_api_sorted:
            nm = (r.get("name") or "").lower()
            if "memcpy" in nm or "memset" in nm:
                d = agg.setdefault(r["name"], {"name": r["name"], "calls": 0, "time_ms": 0.0})
                d["calls"] += int(r.get("calls") or 0)
                if isinstance(r.get("time_ms"), (int, float)):
                    d["time_ms"] += float(r["time_ms"])
        transfers = sorted(agg.values(), key=lambda d: d.get("time_ms", 0.0), reverse=True)

        # If GPU-side memcpy/memset tables were missing, use API times as a rough proxy
        if memcpy_time_ms == 0.0 and any("memcpy" in (k.lower()) for k in agg.keys()):
            memcpy_time_ms = sum(v.get("time_ms") or 0.0 for k, v in agg.items() if "memcpy" in k.lower())
        if memset_time_ms == 0.0 and any("memset" in (k.lower()) for k in agg.keys()):
            memset_time_ms = sum(v.get("time_ms") or 0.0 for k, v in agg.items() if "memset" in k.lower())

    # As a last resort, if we have no kernel/mem GPU time, approximate total time from CUDA API total
    if not total_gpu_time_ms and cuda_api_sorted:
        total_api_time_ms = sum((r.get("time_ms") or 0.0) for r in cuda_api_sorted)
        if total_api_time_ms:
            total_gpu_time_ms = total_api_time_ms

    # If we have no kernel list at all, prefer showing 0 instead of null to avoid UI dashes
    if num_unique_kernels is None and not kernels_sorted:
        num_unique_kernels = 0

    out = {
        "gpu_busy_pct": gpu_busy_pct,
        "total_gpu_time_ms": total_gpu_time_ms or None,
        "memcpy_time_ms": memcpy_time_ms or None,
        "memset_time_ms": memset_time_ms or None,
        "transfer_overhead_pct": (100.0 * (memcpy_time_ms + memset_time_ms) / total_gpu_time_ms) if total_gpu_time_ms else None,
        "num_unique_kernels": num_unique_kernels,
        "kernels": kernels_sorted,
        "cuda_api": cuda_api_sorted,
        "transfers": transfers[:30],
    }

    # If we got nothing meaningful, return None
    if not (out["cuda_api"] or out["kernels"] or out["total_gpu_time_ms"]):
        return None
    return out


def run_with_ncu(
    base_cmd: List[str],
    out_base: str,
    *,
    profile_set: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs Nsight Compute to profile kernels. Produces .ncu-rep at out_base.

    Be defensive across environments (e.g., WSL) where some counter sets are
    restricted and cause NCU to fail without producing a report. We try a
    fallback sequence of lighter profiling sets.
    """
    ncu = _which("ncu")
    if not ncu:
        return {"ok": False, "error": "ncu not found in PATH"}
    out_dir = os.path.dirname(out_base)
    os.makedirs(out_dir, exist_ok=True)

    # Ensure we don't fail due to an existing report file
    rep = out_base + ".ncu-rep"
    try:
        if os.path.exists(rep):
            os.remove(rep)
    except Exception:
        pass

    tried: List[Tuple[List[str], Tuple[int, str, str]]] = []

    def attempt(set_name: Optional[str], *, tproc: str = "all") -> Tuple[bool, Tuple[int, str, str]]:
        cmd = [
            ncu,
            "--target-processes", tproc,
            "--kernel-name-base", "demangled",
            "-f",  # force-overwrite
            "-o", out_base,
        ]
        if set_name:
            cmd += ["--set", set_name]
        # Separate NCU options from application command explicitly
        cmd += ["--", *base_cmd]
        code, out, err = run_cmd(cmd)
        return (os.path.exists(rep) and code == 0), (code, out, err)

    # Try default (no set) first for broader compatibility (e.g., WSL counters),
    # then progressively add lightweight sets. If caller requested a set, try it first.
    sets_to_try: List[Optional[str]] = []
    if profile_set not in (None, "<default>"):
        sets_to_try.append(profile_set)
    for alt in (None, "launch-and-kernel", "launch", "speed-of-light"):
        if alt not in sets_to_try:
            sets_to_try.append(alt)

    ok = False
    last_res: Tuple[int, str, str] = (0, "", "")
    for sname in sets_to_try:
        ok, last_res = attempt(sname, tproc="all")
        tried.append((["--set", (sname or "<default>")], last_res))
        if ok:
            break
        # If the previous attempt failed and produced a partial/empty file, remove it before retry
        try:
            if os.path.exists(rep):
                os.remove(rep)
        except Exception:
            pass

    # As a last attempt, try application-only targeting (rarely helps with launchers)
    if not ok:
        try:
            ok, last_res = attempt(sets_to_try[-1], tproc="application-only")
            tried.append((["--target-processes", "application-only"], last_res))
        except Exception:
            pass

    code, out, err = last_res
    return {
        "ok": ok and code == 0,
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "ncu_rep": rep if os.path.exists(rep) else None,
        "tried_sets": [flags for flags, _ in tried],
    }


def parse_ncu_raw(rep_path: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort parser for Nsight Compute .ncu-rep using CSV "raw" page.

    Returns a dict like:
      {
        "kernels": [ { name, calls?, time_ms?, avg_ms?, pct_gpu_time? }, ...],
        "num_unique_kernels": int
      }

    Notes:
      - NCU page availability varies; we intentionally use "--page raw" because it is
        widely present (even when "summary" pages are not).
      - The CSV content is not a single fixed schema. We heuristically look for rows
        that introduce a kernel (e.g., "Kernel Name", or a single-cell "Kernel: <name>")
        and then collect nearby metrics like Duration/Time and Invocations/Calls.
      - We sum durations across passes/instances and aggregate calls.
    """
    ncu = _which("ncu")
    if not ncu or not os.path.exists(rep_path):
        return None

    # Try to import with the raw page first
    code, csv_text, err = run_cmd([ncu, "--import", rep_path, "--csv", "--page", "raw"])
    if code != 0 or not csv_text:
        # As a fallback, try without explicit page (older builds dump a default table)
        code2, out2, err2 = run_cmd([ncu, "--import", rep_path, "--csv"])  # best-effort
        if code2 != 0 or not out2:
            return None
        csv_text = out2

    import csv
    rdr = csv.reader(io.StringIO(csv_text))

    agg: Dict[str, Dict[str, Any]] = {}
    current: Optional[str] = None

    def ensure(name: str) -> Dict[str, Any]:
        d = agg.get(name)
        if not d:
            d = {"name": name, "calls": 0, "time_ms": 0.0, "_avg_ms_acc": [], "_avg_ms_cnt": 0}
            agg[name] = d
        return d

    def parse_number(s: str) -> Optional[float]:
        s = (s or "").strip()
        if not s:
            return None
        # Strip common decorations
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return None

    def pick_unit(row: List[str], label: str) -> Optional[str]:
        lab = (label or "").lower()
        if "(ns)" in lab or re.search(r"\bns\b", lab):
            return "ns"
        if "(ms)" in lab or re.search(r"\bms\b", lab):
            return "ms"
        if "(s)" in lab or re.search(r"\bs(ec|econd|)s?\b", lab):
            return "s"
        for tok in row[1:4]:
            t = (tok or "").strip().lower()
            if t in ("ns", "ms", "s", "sec", "second", "seconds"):
                return "s" if t.startswith("s") else t
        return None

    for row in rdr:
        if not row:
            continue
        # Normalize
        cells = [c.strip() for c in row]
        c0 = (cells[0] if cells else "").strip()
        c0l = c0.lower()

        # Kernel introduction lines
        name: Optional[str] = None
        if len(cells) >= 2 and c0l == "kernel name":
            name = cells[1]
        elif re.match(r"(?i)^kernel:\s*", c0):
            name = re.sub(r"(?i)^kernel:\s*", "", c0).strip()

        if name:
            current = name
            ensure(name)
            continue

        if not current:
            continue

        # Metric rows under a kernel
        label = c0l
        # Find first numeric value in the next few cells
        val: Optional[float] = None
        for idx in range(1, min(len(cells), 6)):
            v = parse_number(cells[idx])
            if v is not None:
                val = v
                break

        if val is None:
            continue

        # Duration / Time aggregation
        if ("duration" in label) or ("time" in label and "kernel" not in label and "%" not in label):
            unit = pick_unit(cells, c0)
            ms = None
            if unit == "ns":
                ms = val / 1e6
            elif unit == "ms":
                ms = val
            elif unit == "s":
                ms = val * 1000.0
            else:
                # Unknown; pick a sane heuristic
                if val > 1e6:
                    ms = val / 1e6
                elif val > 1000:
                    ms = val / 1000.0
                elif val < 100:
                    ms = val * 1000.0
                else:
                    ms = val
            ensure(current)["time_ms"] += ms
            continue

        # Calls / Invocations
        if ("invocations" in label) or ("calls" in label) or ("launches" in label) or ("instances" in label):
            try:
                ensure(current)["calls"] += int(round(val))
            except Exception:
                pass
            continue

        # Avg duration
        if "avg" in label and ("ns" in label or "ms" in label or "s" in label):
            unit = pick_unit(cells, c0)
            ms = None
            if unit == "ns":
                ms = val / 1e6
            elif unit == "ms":
                ms = val
            elif unit == "s":
                ms = val * 1000.0
            if ms is not None:
                d = ensure(current)
                d["_avg_ms_acc"].append(ms)
                d["_avg_ms_cnt"] += 1

    # If the above row-wise heuristic failed (common for NCU "raw" page formats
    # where data is provided in a flat table with column names), try a
    # column-driven parse like the user's working AWK example.
    if not agg:
        try:
            dr = csv.DictReader(io.StringIO(csv_text))
            key_kernel = None
            key_calls = None
            key_time_ns = None
            key_avg_ns = None

            # Build lowercase header map
            headers = [h for h in (dr.fieldnames or [])]
            hl = [h.lower() for h in headers]

            def find_col(preds):
                for i, h in enumerate(hl):
                    if preds(h):
                        return headers[i]
                return None

            key_kernel = find_col(lambda h: ("kernel" in h and "name" in h)) or "Kernel Name"
            key_calls = find_col(lambda h: ("calls" in h) or ("invocations" in h) or ("instances" in h) or ("launches" in h))
            key_time_ns = find_col(lambda h: ("time" in h and "ns" in h) or ("duration" in h and "ns" in h))
            key_avg_ns = find_col(lambda h: ("avg" in h and "ns" in h) or ("average" in h and "ns" in h))

            agg2: Dict[str, Dict[str, Any]] = {}
            for row in dr:
                name = (row.get(key_kernel) or "").strip() if key_kernel else ""
                if not name:
                    continue
                d = agg2.setdefault(name, {"name": name, "calls": 0, "time_ns": 0.0, "avg_ns_acc": 0.0, "avg_cnt": 0})
                # calls
                if key_calls:
                    try:
                        c = row.get(key_calls)
                        if c is not None and str(c).strip() != "":
                            d["calls"] += int(float(str(c).replace(",", "")))
                    except Exception:
                        pass
                # total time ns
                if key_time_ns:
                    try:
                        v = row.get(key_time_ns)
                        if v is not None and str(v).strip() != "":
                            d["time_ns"] += float(str(v).replace(",", ""))
                    except Exception:
                        pass
                # avg ns (optional)
                if key_avg_ns:
                    try:
                        v = row.get(key_avg_ns)
                        if v is not None and str(v).strip() != "":
                            d["avg_ns_acc"] += float(str(v).replace(",", ""))
                            d["avg_cnt"] += 1
                    except Exception:
                        pass

            if agg2:
                kernels: List[Dict[str, Any]] = []
                for d in agg2.values():
                    avg_ms = None
                    if d["avg_cnt"] > 0:
                        avg_ms = (d["avg_ns_acc"] / d["avg_cnt"]) / 1e6
                    elif d.get("calls") and d.get("time_ns"):
                        try:
                            c = int(d["calls"]) or 0
                            avg_ms = ((d["time_ns"] / 1e6) / c) if c else None
                        except Exception:
                            avg_ms = None
                    kernels.append({
                        "name": d["name"],
                        "calls": d.get("calls") or None,
                        "time_ms": (d.get("time_ns") or 0.0) / 1e6,
                        "avg_ms": avg_ms,
                        "pct_gpu_time": None,
                    })
                kernels_sorted = sorted(kernels, key=lambda r: r.get("time_ms", 0.0) or 0.0, reverse=True)[:30]
                return {"kernels": kernels_sorted, "num_unique_kernels": len(agg2)}
        except Exception:
            pass

        return None

    kernels: List[Dict[str, Any]] = []
    total_time_ms = 0.0
    for k in agg.values():
        avg_ms: Optional[float] = None
        if k["_avg_ms_cnt"] > 0:
            avg_ms = sum(k["_avg_ms_acc"]) / float(k["_avg_ms_cnt"])
        elif k.get("calls") and k.get("time_ms"):
            try:
                c = int(k["calls"]) or 0
                avg_ms = (k["time_ms"] / c) if c else None
            except Exception:
                avg_ms = None
        total_time_ms += float(k.get("time_ms") or 0.0)
        kernels.append({
            "name": k["name"],
            "calls": (k.get("calls") or None),
            "time_ms": (k.get("time_ms") or None),
            "avg_ms": avg_ms,
            "pct_gpu_time": None,  # unknown without a robust total
        })

    kernels_sorted = sorted(kernels, key=lambda d: d.get("time_ms", 0.0) or 0.0, reverse=True)[:30]

    out = {
        "kernels": kernels_sorted,
        "num_unique_kernels": len(agg),
        # We intentionally omit total time and pct fields; those remain None in UI
    }
    return out


def run_cuda_kernel_test() -> Dict[str, Any]:
    """
    Runs a short dummy CUDA workload under Nsight Compute to verify that device
    kernels are captured. Returns a detailed diagnostics dict with environment
    info, kernel list, and raw CSV preview when available.

    This does NOT associate with any run; it writes artifacts under project/tmp.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "env": {},
        "ncu": {"found": False, "version": None, "path": None},
        "nsys": {"found": have_nsys()},
        "python": sys.executable,
        "torch_cuda_available": None,
        "artifact": None,
        "kernels": [],
        "num_unique_kernels": 0,
        "tried_sets": [],
        "stdout": None,
        "stderr": None,
        "raw_preview": None,
        "notes": [],
    }

    # Env flags
    try:
        rel = __import__("platform").release().lower()
        if "microsoft" in rel or os.environ.get("WSL_DISTRO_NAME"):
            out["env"]["is_wsl"] = True
    except Exception:
        pass

    ncu = _which("ncu")
    if ncu:
        out["ncu"]["found"] = True
        out["ncu"]["path"] = ncu
        try:
            code, vout, _ = run_cmd([ncu, "--version"])  # prints version info
            if vout:
                out["ncu"]["version"] = vout.strip().splitlines()[0].strip()
        except Exception:
            pass

    # Prepare temp paths under project/tmp
    root = os.path.dirname(__file__)
    tmp_dir = os.path.join(root, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    script_path = os.path.join(tmp_dir, f"cuda_sanity_{stamp}.py")
    out_base = os.path.join(tmp_dir, f"ncu_cuda_sanity_{stamp}")
    rep = out_base + ".ncu-rep"

    # Emit small CUDA workload script
    script = """
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    # Larger matmul to ensure measurable kernels across devices
    a = torch.randn(8192, 8192, device="cuda")
    b = torch.randn(8192, 8192, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
print("done")
"""
    try:
        with open(script_path, "w") as f:
            f.write(script)
    except Exception as e:
        out["stderr"] = f"failed to write script: {e}"
        return out

    # Baseline: run without NCU to report torch CUDA availability
    try:
        code, pout, perr = run_cmd([sys.executable, script_path])
        out["stdout"] = pout
        out["stderr"] = perr or out.get("stderr")
        if pout and "cuda available:" in pout:
            avail = pout.split("cuda available:")[-1].strip().splitlines()[0].strip()
            out["torch_cuda_available"] = (avail.lower().startswith("true"))
    except Exception:
        pass

    # If NCU is not present, return early with baseline info
    if not ncu:
        out["ok"] = bool(out.get("torch_cuda_available"))
        out["notes"].append("Nsight Compute (ncu) not found in PATH/known locations.")
        return out

    # Attempt profiling with progressive sets
    tried: List[List[str]] = []
    def attempt(set_name: Optional[str], *, tproc: str = "all") -> Tuple[bool, Tuple[int, str, str]]:
        cmd = [
            ncu,
            "--target-processes", tproc,
            "--kernel-name-base", "demangled",
            "-f",
            "-o", out_base,
        ]
        if set_name:
            cmd += ["--set", set_name]
        cmd += ["--", sys.executable, script_path]
        code, o, e = run_cmd(cmd)
        tried.append([a for a in cmd if a.startswith("--") or a in ("-f", "-o")])
        return (os.path.exists(rep) and code == 0), (code, o, e)

    ok = False
    last: Tuple[int, str, str] = (0, "", "")
    for s in (None, "launch-and-kernel", "launch", "speed-of-light"):
        ok, last = attempt(s, tproc="all")
        if ok:
            break
        try:
            if os.path.exists(rep):
                os.remove(rep)
        except Exception:
            pass
    if not ok:
        ok, last = attempt(None, tproc="application-only")

    out["tried_sets"] = tried
    out["artifact"] = rep if os.path.exists(rep) else None

    # Parse kernels if present
    if out["artifact"]:
        parsed = parse_ncu_raw(rep)
        if parsed and parsed.get("kernels"):
            out["kernels"] = parsed["kernels"]
            out["num_unique_kernels"] = int(parsed.get("num_unique_kernels") or len(parsed["kernels"]))
            out["ok"] = True
            # Raw CSV preview
            try:
                code, csv_raw, _ = run_cmd([ncu, "--import", rep, "--csv", "--page", "raw"])
                if csv_raw:
                    lines = csv_raw.splitlines()
                    out["raw_preview"] = "\n".join(lines[:80])
            except Exception:
                pass
    else:
        out["stdout"] = (out.get("stdout") or "") + (last[1] or "")
        out["stderr"] = (out.get("stderr") or "") + (last[2] or "")
        out["ok"] = False
        if out.get("env", {}).get("is_wsl"):
            out["notes"].append("WSL detected: ensure GPU performance counters are enabled in NVIDIA Control Panel → Developer → Manage GPU Performance Counters → Allow access to All Users.")

    return out
