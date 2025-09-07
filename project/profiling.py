import os
import json
import shutil
import subprocess
from typing import Dict, Any, Optional, List, Tuple


def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)


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
    sqlite = out_base + ".sqlite"
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

    # gpu-activities: sum times and memcpy/memset buckets
    act_path = csvs.get("gpu-activities")
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
                name = "".join([row.get("Name") or row.get("name") or row.get("Activity Name") or ""])  # type: ignore
                lname = name.lower()
                if "memcpy" in lname:
                    memcpy_time_ms += time_ns / 1e6
                    d = transfers_map.setdefault(name, {"name": name, "time_ms": 0.0, "calls": 0})
                    d["time_ms"] += time_ns / 1e6
                    d["calls"] += 1
                if "memset" in lname:
                    memset_time_ms += time_ns / 1e6
                    d = transfers_map.setdefault(name, {"name": name, "time_ms": 0.0, "calls": 0})
                    d["time_ms"] += time_ns / 1e6
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
    num_unique_kernels = len(unique_kernels) if unique_kernels else (0 if not kernels_proc else len(kernels_proc))

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
        # Find a plausible kernel table
        kernel_tbl = None
        kernel_name_col = None
        for t in tbls:
            tl = t.lower()
            if "kernel" in tl and ("cupti" in tl or "cuda" in tl or "gpu" in tl):
                # Inspect columns
                cur.execute(f"pragma table_info('{t}')")
                cols = [r[1] for r in cur.fetchall()]
                cols_l = [c.lower() for c in cols]
                if ("start" in cols_l and "end" in cols_l):
                    # Prefer demangled name if present
                    if "demanglednameid" in cols_l:
                        kernel_name_col = "demangledNameId"
                    elif "nameid" in cols_l:
                        kernel_name_col = "nameId"
                    elif "name" in cols_l:
                        kernel_name_col = "name"
                    else:
                        continue
                    kernel_tbl = t
                    break
        if kernel_tbl and kernel_name_col:
            if kernel_name_col.lower().endswith("id"):
                # Join through StringIds
                qk = (
                    f"select s.value as name, count(*) as calls, "
                    f"sum(k.end - k.start) as total_ns, avg(k.end - k.start) as avg_ns "
                    f"from {kernel_tbl} k join StringIds s on s.id = k.{kernel_name_col} "
                    f"group by k.{kernel_name_col} order by total_ns desc"
                )
            else:
                qk = (
                    f"select k.{kernel_name_col} as name, count(*) as calls, "
                    f"sum(k.end - k.start) as total_ns, avg(k.end - k.start) as avg_ns "
                    f"from {kernel_tbl} k group by k.{kernel_name_col} order by total_ns desc"
                )
            try:
                for name, calls, total_ns, avg_ns in cur.execute(qk):
                    ms = (total_ns / 1e6) if total_ns is not None else None
                    kernels.append({
                        "name": name,
                        "calls": int(calls) if calls is not None else None,
                        "time_ms": ms,
                        "avg_ms": (avg_ns / 1e6) if avg_ns is not None else None,
                    })
                num_unique_kernels = len(kernels) if kernels else None
                total_gpu_time_ms += sum((k.get("time_ms") or 0.0) for k in kernels)
            except Exception:
                pass
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


def run_with_ncu(base_cmd: List[str], out_base: str, *, kernel_regex: str = ".*(gemm|matmul|attention|ScaledDotProduct).*", profile_set: str = "speed-of-light") -> Dict[str, Any]:
    """
    Runs Nsight Compute to profile hot kernels. Produces .ncu-rep at out_base.
    """
    ncu = _which("ncu")
    if not ncu:
        return {"ok": False, "error": "ncu not found in PATH"}
    out_dir = os.path.dirname(out_base)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        ncu,
        "--target-processes", "all",
        "--kernel-name-base", "demangled",
        "--kernel-regex", kernel_regex,
        "--set", profile_set,
        "-o", out_base,
        *base_cmd,
    ]
    code, out, err = run_cmd(cmd)
    rep = out_base + ".ncu-rep"
    return {
        "ok": code == 0,
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "ncu_rep": rep if os.path.exists(rep) else None,
    }
