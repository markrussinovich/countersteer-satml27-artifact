#!/usr/bin/env bash
# Singularity driver: DETECTOR filters on AgentDyn-180 (owner-approved program,
# 2026-09-10) — the open-ended-benchmark gap in the detector picture: PIGuard +
# PromptGuard-2 have never been measured outside static AgentDojo. Four 4-arm batteries
# (2 models x 2 detectors), the SAME 180-stratified cells and mn4096/system-yaml
# convention as the CachePrune/reminder rival batteries (§25h/§26.10), so rows land in
# the existing tables with same-process anchors.
#
# SUBMIT (after xpia-dojo-filters-glm frees the node):
#   bash singularity/submit_job.sh --mode run --display-name xpia-agentdyn-detectors \
#     --timeout-seconds 79200 --no-clean --slmx-cmd 'bash tools/controls/agentdyn_detectors_job.sh'
set -uo pipefail

LOCAL=runs/agentdyn_rivals
LOGD=logs_agentdyn_detectors
OUT=${OUT:-outputs}
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sync_blob() { cp -f "$LOCAL"/* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) &
SYNC_PID=$!

B=tools/controls/dojo_baseline_mn4096.sh
rc=0

run_batt() { # model cells label defense
  echo "[agentdyn-detectors] battery $3 START $(date -u '+%F %T')"
  bash "$B" --model "$1" --cells "$2" --defense "$4" --label "$3" \
       --max-new 4096 --system yaml --gpus 0,1,2,3,4,5,6,7 \
       --outdir "$LOCAL" --logdir "$LOGD"
  local r=$?
  echo "[agentdyn-detectors] battery $3 EXIT rc=$r $(date -u '+%F %T')"
  sync_blob
  return "$r"
}

bash singularity/seed_model.sh --require openai/gpt-oss-20b || { echo "AGENTDYN-DETECTORS-DONE rc=3"; exit 3; }
run_batt openai/gpt-oss-20b runs/agentdyn_cells180.gptoss.json \
         agentdyn180_gptoss_piguard     pi_detector_piguard     || rc=1
run_batt openai/gpt-oss-20b runs/agentdyn_cells180.gptoss.json \
         agentdyn180_gptoss_promptguard pi_detector_promptguard || rc=1

bash singularity/seed_model.sh --require Qwen/Qwen3-30B-A3B-Thinking-2507 || { echo "AGENTDYN-DETECTORS-DONE rc=3"; exit 3; }
run_batt Qwen/Qwen3-30B-A3B-Thinking-2507 runs/agentdyn_cells180.qwen.json \
         agentdyn180_qwen_piguard       pi_detector_piguard     || rc=1
run_batt Qwen/Qwen3-30B-A3B-Thinking-2507 runs/agentdyn_cells180.qwen.json \
         agentdyn180_qwen_promptguard   pi_detector_promptguard || rc=1

# SecAlign on AgentDyn-180 (owner "yes", 2026-09-10): the strongest measured rival's
# open-ended-benchmark row. One battery, §22b arm semantics: the model IS the defense —
# attacked arm = SecAlign alone, defended arm = SecAlign+CounterSteer hybrid (the same
# steering config as the §22b AgentDojo battery: combo_ovr8_pat1 @8.06 sigma-matched).
# The 39G merged checkpoint is a flat dir in the store; blobfuse mmap-loading is
# pathological (singularity/README.md), so copy node-local first (sequential read, fine).
if [[ -d "${XPIA_MODEL_STORE:-/nonexistent}/secalign-dpo-gptoss/merged" ]]; then
  SA_SRC="$XPIA_MODEL_STORE/secalign-dpo-gptoss/merged"
elif [[ -d "${XPIA_MODEL_STORE:-/nonexistent}/secalign-dpo-gptoss" ]]; then
  SA_SRC="$XPIA_MODEL_STORE/secalign-dpo-gptoss"
else
  SA_SRC=""
fi
if [[ -n "$SA_SRC" ]]; then
  echo "[agentdyn-detectors] copying SecAlign checkpoint node-local from $SA_SRC ..."
  cp -r "$SA_SRC" /tmp/secalign-merged && \
  bash "$B" --model /tmp/secalign-merged --cells runs/agentdyn_cells180.gptoss.json \
       --direction combo_ovr8_pat1 --alpha 8.06 --layers 12,16,20 \
       --match-sigma-to dim_no_override --probe-dir runs/gpt-oss-20b-userabl \
       --label agentdyn180_gptoss_secalign \
       --max-new 4096 --system yaml --gpus 0,1,2,3,4,5,6,7 \
       --outdir "$LOCAL" --logdir "$LOGD" || rc=1
  sync_blob
else
  echo "[agentdyn-detectors] SecAlign checkpoint NOT in store — battery SKIPPED (stage it first)"
fi

kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "AGENTDYN-DETECTORS-DONE rc=$rc"
exit "$rc"
