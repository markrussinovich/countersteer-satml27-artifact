#!/usr/bin/env bash
# Singularity driver: detector-filter comparison on GLM-4.5-Air (owner order 2026-09-10:
# "should we try piguard on glm? ... proceed") — the third CAPABLE-model point for the
# detector-cost-scales-with-capability question (gpt-oss fp 56% / Qwen fp 31% benign
# deletion vs near-free on the low-utility Llama; GLM has the program's HIGHEST AgentDojo
# benign utility, so a false-positive-prone filter has the most to destroy here).
#
# Three FOUR-arm 180-cell AgentDojo batteries at GLM's own dojo convention (max_new 8192,
# device auto, 2 shards x 4 GPUs — the agentdyn_grid_job.sh glm stage pattern):
#   1 countersteer_glm    the certified cell (dim_no_override_actioncentred @8 L20/24/28,
#                         sigma-matched to dim_no_override_both) — fresh same-process
#                         anchor so the filter rows compare within-run, not to the old row
#   2 pi_piguard_glm      PIGuard classifier (XPIA_MODEL_STORE flat dir; trust_remote_code
#                         pinned in PI_DETECTORS)
#   3 pi_promptguard_glm  PromptGuard-2-86M (gated; store-resolved)
#
# SUBMIT:
#   bash singularity/submit_job.sh --mode run --display-name xpia-dojo-filters-glm \
#     --timeout-seconds 79200 --no-clean --slmx-cmd 'bash tools/controls/dojo_filters_glm_job.sh'
set -uo pipefail

bash singularity/seed_model.sh --require zai-org/GLM-4.5-Air

LOCAL=runs/dojo_filters_glm
LOGD=logs_filters_glm
OUT=${OUT:-outputs}
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sync_blob() { cp -f "$LOCAL"/* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) &
SYNC_PID=$!

B=tools/controls/dojo_baseline_mn4096.sh
M="zai-org/GLM-4.5-Air"
PD="runs/glm45-air"

run_batt() { # label [battery args...]
  local label="$1"; shift
  echo "[filters-glm] battery $label START $(date -u '+%F %T')"
  bash "$B" --label "$label" --gpus 0-1-2-3,4-5-6-7 --device auto --max-new 8192 \
       --model "$M" --outdir "$LOCAL" --logdir "$LOGD" "$@"
  local rc=$?
  echo "[filters-glm] battery $label EXIT rc=$rc $(date -u '+%F %T')"
  sync_blob
  return "$rc"
}

CS=(--direction dim_no_override_actioncentred --alpha 8 --layers 20,24,28 \
    --match-sigma-to dim_no_override_both --probe-dir "$PD")

rc=0
run_batt countersteer_glm    "${CS[@]}"                                       || rc=1
run_batt pi_piguard_glm      --defense pi_detector_piguard     --probe-dir "$PD" || rc=1
run_batt pi_promptguard_glm  --defense pi_detector_promptguard --probe-dir "$PD" || rc=1

kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "DOJO-FILTERS-GLM-DONE rc=$rc"
exit "$rc"
