#!/usr/bin/env bash
# AML-cluster driver: AgentDojo 180-cell grid (max_new 4096, FOUR-ARM battery) for the
# Qwen3-30B FRAMING-HELD-OUT refit cell `dim_no_override_achf` (fit: --centre-action
# --exclude-override firm on runs/override_slope_qwen.json; keys merged 2026-09-12 into
# runs/qwen3-30b-thinking/probe_L*.pkl alongside the untouched deployed keys).
#
# Same structure as tools/controls/dojo_soa_qwen_job.sh (single CounterSteer battery; the
# undefended/clean anchors are produced by the four-arm battery itself). The DOSE is a
# required positional argument, chosen by the pre-registered dev dose search
# (tmp/qwen_achf_prereg_DRAFT.json) — do not hardcode it here.
#
# SUBMIT (from a box whose runs/qwen3-30b-thinking/probe_L*.pkl carry the _achf keys —
# verify before staging; the snapshot ships the pickles):
#   bash cluster/submit_job.sh --mode run --display-name xpia-dojo-achf-qwen \
#     --timeout-seconds 43200 --no-clean \
#     --slmx-cmd 'bash tools/controls/dojo_achf_qwen_job.sh <DOSE>'
set -uo pipefail

DOSE="${1:?usage: dojo_achf_qwen_job.sh DOSE (alpha in own-sigma units, e.g. 12)}"

LOCAL=runs/dojo_achf_qwen
LOGD=logs_achf_qwen
OUT=${OUT:-outputs}
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sync_blob() { cp -f "$LOCAL"/* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) &
SYNC_PID=$!

B=tools/controls/dojo_baseline_mn4096.sh
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
M="Qwen/Qwen3-30B-A3B-Thinking-2507"
PD="runs/qwen3-30b-thinking"

# _achf key presence preflight: a snapshot staged from a stale box would silently run a
# missing-direction crash 30 min into model load. Fail in second one instead.
.venv/bin/python - "$PD" <<'EOF' || { echo "DOJO-ACHF-QWEN-DONE rc=2 (achf keys missing)"; exit 2; }
import pickle, sys
for L in (8, 20, 32):
    p = pickle.load(open(f"{sys.argv[1]}/probe_L{L}.pkl", "rb"))
    assert "dim_no_override_achf" in p["dirs"] and p["sigmas"]["dim_no_override_achf"] > 0, L
print("[preflight] dim_no_override_achf present at L8/20/32 with positive sigma")
EOF

echo "[achf-qwen] battery countersteer_qwen_achf@${DOSE} START $(date -u '+%F %T')"
bash "$B" --label "countersteer_qwen_achf_a${DOSE}" --gpus "$GPUS" --max-new 4096 \
     --model "$M" --outdir "$LOCAL" --logdir "$LOGD" \
     --direction dim_no_override_achf --alpha "$DOSE" --layers 8,20,32 \
     --match-sigma-to "" --probe-dir "$PD"
rc=$?
echo "[achf-qwen] battery EXIT rc=$rc $(date -u '+%F %T')"

kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "DOJO-ACHF-QWEN-DONE rc=$rc"
exit "$rc"
