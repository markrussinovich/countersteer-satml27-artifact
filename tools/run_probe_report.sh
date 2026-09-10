#!/bin/bash
# Wait for a probe run to finish, then run the validation report + figures (STEP 2).
#
#   tools/run_probe_report.sh [RUN_DIR] [GPU] [BASE_RUN]
#
#   RUN_DIR   probe run to validate      (default runs/gpt-oss-20b-paper)
#   GPU       CUDA device index          (default 0)
#   BASE_RUN  source of unsteered base-XPIA completions for Q4's attack-success
#             labels                     (default runs/gpt-oss-20b-resid)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_DIR="${1:-$ROOT/runs/gpt-oss-20b-paper}"
GPU="${2:-0}"
BASE_RUN="${3:-$ROOT/runs/gpt-oss-20b-resid}"

echo "[report] waiting for $RUN_DIR/probe_report.json"
wait_for "$RUN_DIR/probe_report.json"
sleep 10   # let the writer flush

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" tools/probe_report.py "$RUN_DIR" "$BASE_RUN"
