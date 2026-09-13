#!/usr/bin/env bash
# AML-cluster: SecAlign AgentDojo battery at mn4096 (reviewer-flagged budget-parity gap:
# tab:soa runs at 4096 while the §22b SecAlign battery ran at the legacy 768 budget;
# owner order 2026-09-11 "Run secalign at mn4096"). Same §22b arm semantics: the model
# IS the defense — attacked arm = SecAlign alone, defended = SecAlign+CounterSteer
# hybrid at the deployed gpt-oss cell. Same 180-case grid, four arms, one process per
# shard. NOTE (review 2026-09-12): this job ran system=yaml while the tab:soa
# batteries ran the short SYSTEM constant — a disclosed deviation; measured not to
# carry the clean-utility deficit (FINDINGS §26.38 addendum) but the row is NOT
# byte-comparable to tab:soa.
set -uo pipefail
OUT=${OUT:-outputs}
LOCAL=runs/dojo_soa_gptoss
LOGD=logs_secalign_mn4096
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sync_blob() { cp -f "$LOCAL"/secalign_mn4096* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) & SYNC_PID=$!
rc=0
if [[ -d "${XPIA_MODEL_STORE:-/nonexistent}/secalign-dpo-gptoss/merged" ]]; then SA_SRC="$XPIA_MODEL_STORE/secalign-dpo-gptoss/merged"
elif [[ -d "${XPIA_MODEL_STORE:-/nonexistent}/secalign-dpo-gptoss" ]]; then SA_SRC="$XPIA_MODEL_STORE/secalign-dpo-gptoss"
else SA_SRC=""; fi
if [[ -n "$SA_SRC" ]]; then
  echo "[secalign-mn4096] copying checkpoint node-local from $SA_SRC ..."
  cp -r "$SA_SRC" /tmp/secalign-merged && \
  bash tools/controls/dojo_baseline_mn4096.sh --model /tmp/secalign-merged \
       --direction combo_ovr8_pat1 --alpha 8.06 --layers 12,16,20 \
       --match-sigma-to dim_no_override --probe-dir runs/gpt-oss-20b-userabl \
       --label secalign_mn4096 \
       --max-new 4096 --system yaml --gpus 0,1,2,3,4,5,6,7 \
       --outdir "$LOCAL" --logdir "$LOGD" || rc=1
else
  echo "[secalign-mn4096] checkpoint NOT in store — ABORT"; rc=1
fi
sync_blob
kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "SECALIGN-MN4096-DONE rc=$rc"
exit "$rc"
