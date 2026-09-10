#!/usr/bin/env bash
# Singularity driver: the STATE-OF-THE-ART DEFENSE COMPARISON on gpt-oss-20b — the full
# 180-cell AgentDojo grid at max_new 4096 (the un-censored budget, FINDINGS 21c), one
# FOUR-ARM battery per defense (clean / cleanplus / attacked / defended via
# tools/controls/dojo_baseline_mn4096.sh). Four arms per battery is deliberate and is the
# 21c standard: every defense gets its OWN same-process undefended anchor and its own
# within-run benign pairing, and the truncation audit stays gateable (it needs the clean
# arm's attacker_calls).
#
# Batteries, in priority order (owner directive 2026-09-04 — partial completion still
# delivers the flagship budget-clean table):
#   1 countersteer      deployed cell combo_ovr8_pat1 @ 8.06 sigma-matched L12/16/20
#                       (fresh same-code battery; mn4096 remains the cross-check)
#   2 spotlighting      AgentDojo inbuilt spotlighting_with_delimiting
#   3 prompt sandwich   AgentDojo inbuilt repeat_user_prompt
#   4 cacheprune        KV-mask reimplementation (arXiv:2504.21228), runs/cacheprune_mask.json
#   5 reminder          AutoDojo prompt-level reminder (arXiv:2606.15057)
#   6-8 filter family   ProtectAI / PromptGuard-2-86M / PIGuard classifiers in AgentDojo's
#                       own transformers_pi_detector wiring (PromptGuard resolves from the
#                       ro model-store mount — gated checkpoint, jobs carry no HF token)
#   9 stacked           CounterSteer + spotlighting in ONE arm (--stack-dojo): composition
#  10 tool_filter       the 768 table's capability-guard-failure row, re-run at 4096
#
# SUBMIT (from a box with the AML_* env exported; see singularity/README.md):
#   bash singularity/submit_job.sh --mode run --display-name xpia-dojo-soa-gptoss \
#     --timeout-seconds 72000 --no-clean --slmx-cmd 'bash tools/controls/dojo_soa_gptoss_job.sh'
# Score with tools/controls/score_dojo_soa.py over the harvested shards.
set -uo pipefail

LOCAL=runs/dojo_soa_gptoss
LOGD=logs_soa
OUT=${OUT:-outputs}
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# local writes + periodic blob sync (blobfuse uploads on close; the ENOENT lesson from
# dojo-baselines-20260830 — never point incremental writers at the blob mount directly)
sync_blob() { cp -f "$LOCAL"/* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) &
SYNC_PID=$!

B=tools/controls/dojo_baseline_mn4096.sh
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

run_batt() { # label [battery args...]
  local label="$1"; shift
  echo "[soa] battery $label START $(date -u '+%F %T')"
  bash "$B" --label "$label" --gpus "$GPUS" --max-new 4096 \
       --outdir "$LOCAL" --logdir "$LOGD" "$@"
  local rc=$?
  echo "[soa] battery $label EXIT rc=$rc $(date -u '+%F %T')"
  sync_blob
  return "$rc"
}

rc=0
run_batt countersteer  --direction combo_ovr8_pat1 --alpha 8.06                  || rc=1
run_batt spotlighting_with_delimiting --defense spotlighting_with_delimiting     || rc=1
run_batt repeat_user_prompt           --defense repeat_user_prompt               || rc=1
run_batt cacheprune    --kv-mask runs/cacheprune_mask.json                       || rc=1
run_batt reminder                     --defense reminder                         || rc=1
run_batt pi_protectai                 --defense pi_detector                      || rc=1
run_batt pi_promptguard               --defense pi_detector_promptguard          || rc=1
run_batt pi_piguard                   --defense pi_detector_piguard              || rc=1
run_batt stacked_cs_spotlight --direction combo_ovr8_pat1 --alpha 8.06 \
                              --stack-dojo spotlighting_with_delimiting          || rc=1
run_batt tool_filter                  --defense tool_filter                      || rc=1

kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "DOJO-SOA-GPTOSS-DONE rc=$rc"
exit "$rc"
