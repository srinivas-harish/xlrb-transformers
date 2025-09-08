#!/usr/bin/env bash
set -euo pipefail

# profile_run.sh <run_id>
# Runs Nsight Systems and Nsight Compute for an existing or new run,
# producing a timeline .sqlite and a .ncu-rep with kernel names.

if [[ ${#} -lt 1 ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 1
fi

RUN_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${SCRIPT_DIR}"

# Optional venv activation if present
if [[ -f "${PROJ_DIR}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${PROJ_DIR}/.venv/bin/activate"
fi

PY="$(command -v python)"
if [[ -z "${PY}" ]]; then
  echo "python not found in PATH" >&2
  exit 2
fi

NSYS_BIN="$(command -v nsys || true)"
NCU_BIN="$(command -v ncu || true)"
if [[ -z "${NSYS_BIN}" ]]; then
  # Try common CUDA locations
  for c in /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/nsys; do
    [[ -x "$c" ]] && NSYS_BIN="$c" && break
  done
fi
if [[ -z "${NCU_BIN}" ]]; then
  for c in /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/ncu; do
    [[ -x "$c" ]] && NCU_BIN="$c" && break
  done
fi

SAVE_DIR="${PROJ_DIR}/runs/${RUN_ID}"
NSYS_DIR="${SAVE_DIR}/nsys"
NCU_DIR="${SAVE_DIR}/ncu"
mkdir -p "${NSYS_DIR}" "${NCU_DIR}"

REQ_JSON="${SAVE_DIR}/run_req.json"
if [[ ! -f "${REQ_JSON}" ]]; then
  echo "Creating default run_req.json at ${REQ_JSON}" >&2
  RUN_ID="${RUN_ID}" REQ_JSON="${REQ_JSON}" "${PY}" - <<'PY'
import json, os, sys
rid = os.environ.get('RUN_ID')
req = {
  "ablation": None,
  "overrides": {},
  "epochs": 3,
  "batch_size": 8,
  "max_len": 128,
  "save_dir": None,
  "save_artifacts": False,
  "gradient_checkpointing": True,
}
out = os.environ['REQ_JSON']
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
  json.dump(req, f, indent=2)
print(out)
PY
fi

BASE_CMD=("${PY}" "${PROJ_DIR}/runner.py" --infile "${REQ_JSON}" --run-id "${RUN_ID}" --save-dir "${SAVE_DIR}" --progress-file "${SAVE_DIR}/progress.jsonl")

if [[ -n "${NSYS_BIN}" ]]; then
  echo "[nsys] Profiling training → ${NSYS_DIR}/${RUN_ID}"
  "${NSYS_BIN}" profile --trace=cuda,nvtx,osrt --sample=none --cuda-memory-usage=true \
    --force-overwrite=true -o "${NSYS_DIR}/${RUN_ID}" -- "${BASE_CMD[@]}"
  # Export to sqlite (handles .nsys-rep or .qdrep)
  REP_BASE="${NSYS_DIR}/${RUN_ID}"
  REP_FILE=""
  if [[ -f "${REP_BASE}.nsys-rep" ]]; then REP_FILE="${REP_BASE}.nsys-rep"; fi
  if [[ -z "${REP_FILE}" && -f "${REP_BASE}.qdrep" ]]; then REP_FILE="${REP_BASE}.qdrep"; fi
  if [[ -n "${REP_FILE}" ]]; then
    echo "[nsys] Exporting sqlite from ${REP_FILE}"
    "${NSYS_BIN}" export --type sqlite --force-overwrite=true -o "${REP_BASE}" "${REP_FILE}" || true
  else
    echo "[nsys] Warning: no .nsys-rep or .qdrep produced" >&2
  fi
else
  echo "[nsys] Not found; skipping Nsight Systems" >&2
fi

if [[ -n "${NCU_BIN}" ]]; then
  echo "[ncu] Profiling short-run kernel window → ${NCU_DIR}/${RUN_ID}"
  # Prepare a short-run variant of the request
  REQ_NCU_JSON="${SAVE_DIR}/run_req_ncu.json"
  REQ_JSON="${REQ_JSON}" REQ_NCU_JSON="${REQ_NCU_JSON}" "${PY}" - <<'PY'
import json, os
base = os.environ['REQ_JSON']
out = os.environ['REQ_NCU_JSON']
with open(base) as f:
  req = json.load(f)
req['epochs'] = 1
bs = int(req.get('batch_size', 8) or 8)
req['max_train_batches'] = int(min(30, max(5, bs)))
with open(out, 'w') as f:
  json.dump(req, f, indent=2)
print(out)
PY
  "${NCU_BIN}" --target-processes all -f -o "${NCU_DIR}/${RUN_ID}" -- \
    "${PY}" "${PROJ_DIR}/runner.py" --infile "${REQ_NCU_JSON}" --run-id "${RUN_ID}-ncu" --save-dir "${SAVE_DIR}"
else
  echo "[ncu] Not found; skipping Nsight Compute" >&2
fi

echo "Done. Artifacts under ${SAVE_DIR}/nsys and ${SAVE_DIR}/ncu"
