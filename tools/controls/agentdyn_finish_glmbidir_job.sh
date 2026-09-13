#!/usr/bin/env bash
# AML-cluster continuation (2026-09-11): (1) the 10 SecAlign AgentDyn cells the
# preempted xpia-agentdyn-detectors-r2 job left unfinished (subset file
# runs/agentdyn_cells_secalign_missing.gptoss.json, all shopping suite), same
# §22b arm semantics and convention as tools/controls/agentdyn_detectors_job.sh;
# then (2) the GLM increase-side causal gate at the model's 8192 budget
# (tools/controls/glm_bidir_job.sh — the budget-conforming certification rerun;
# the 4096 preliminary is recorded unsigned, FINDINGS §26.35 addendum).
set -uo pipefail
export PYTHONPATH="$PWD/reference/agentdyn/src${PYTHONPATH:+:$PYTHONPATH}"
OUT=${OUT:-outputs}
LOCAL=runs/agentdyn_rivals
LOGD=logs_agentdyn_finish
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sync_blob() { cp -f "$LOCAL"/* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; cp -f logs_bidir/* "$OUT"/ 2>/dev/null || true; cp -f runs/glm45-air/results_add-dim-no-override-actioncentred-*.json "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) & SYNC_PID=$!
rc=0

if [[ -d "${XPIA_MODEL_STORE:-/nonexistent}/secalign-dpo-gptoss/merged" ]]; then SA_SRC="$XPIA_MODEL_STORE/secalign-dpo-gptoss/merged"
elif [[ -d "${XPIA_MODEL_STORE:-/nonexistent}/secalign-dpo-gptoss" ]]; then SA_SRC="$XPIA_MODEL_STORE/secalign-dpo-gptoss"
else SA_SRC=""; fi
if [[ -n "$SA_SRC" ]]; then
  echo "[finish] copying SecAlign checkpoint node-local from $SA_SRC ..."
  cp -r "$SA_SRC" /tmp/secalign-merged && \
  bash tools/controls/dojo_baseline_mn4096.sh --model /tmp/secalign-merged \
       --cells runs/agentdyn_cells_secalign_missing.gptoss.json \
       --direction combo_ovr8_pat1 --alpha 8.06 --layers 12,16,20 \
       --match-sigma-to dim_no_override --probe-dir runs/gpt-oss-20b-userabl \
       --label agentdyn180_gptoss_secalign_missing \
       --max-new 4096 --system yaml --gpus 0,1,2,3,4,5,6,7 \
       --outdir "$LOCAL" --logdir "$LOGD" || rc=1
  sync_blob
else
  echo "[finish] SecAlign checkpoint NOT in store — SKIPPED"; rc=1
fi

# GLM increase-side gate, budget-conforming (8192), deployed direction/layers.
bash tools/controls/glm_bidir_job.sh || rc=1
sync_blob
kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "AGENTDYN-FINISH-GLMBIDIR-DONE rc=$rc"
exit "$rc"
