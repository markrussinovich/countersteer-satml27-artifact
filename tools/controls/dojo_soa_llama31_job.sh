#!/usr/bin/env bash
# Singularity driver: SoA-defense comparison on Llama-3.1-8B-Instruct (FLAGSHIP, promoted
# by owner 2026-09-10) — the full 180-cell AgentDojo grid at max_new 4096, FOUR-ARM
# batteries (design rationale in dojo_soa_gptoss_job.sh; identical structure, third
# flagship).
#
# Llama certified cell (EVAL_MATRIX row, FINDINGS §26.29):
#   dim_no_override_achf @ alpha 5, layers 12,16,20, sigma-matched to itself.
#
# NOT here: spotlighting_with_delimiting and repeat_user_prompt (running on the .9/.11
# A100s, 2026-09-10) and cacheprune (mask fit in flight; runs when
# runs/cacheprune_mask_llama31-8b.json exists — a later submission or A100 battery).
#
# SUBMIT (weights must be staged first — seed_model.sh --require is fatal-on-missing):
#   bash singularity/submit_job.sh --mode run --display-name xpia-dojo-soa-llama31 \
#     --timeout-seconds 43200 --no-clean --slmx-cmd 'bash tools/controls/dojo_soa_llama31_job.sh'
# (8B model: the 4-arm 180-cell battery measured 14-47 min per 2-GPU box on A100s;
#  6 batteries on 8 H100 shards each should clear well inside 12 h.)
set -uo pipefail

# Seed BOTH hub-resolved models into the node-local HF cache, then load OFFLINE.
# Llama-3.1 is GATED and jobs carry no HF token: an ONLINE from_pretrained hard-401s at
# the hub resolve step even with a seeded cache (measured: xpia-dojo-soa-llama31 first
# submission, 2026-09-10 — all 56 shards died identically in AutoTokenizer). PromptGuard
# and PIGuard resolve from the flat XPIA_MODEL_STORE mount (_pi_model_path) and are
# unaffected by HF_HUB_OFFLINE.
bash singularity/seed_model.sh --require meta-llama/Llama-3.1-8B-Instruct \
  || { echo "DOJO-SOA-LLAMA31-DONE rc=3 (seed llama failed)"; exit 3; }
bash singularity/seed_model.sh --require protectai/deberta-v3-base-prompt-injection-v2 \
  || { echo "DOJO-SOA-LLAMA31-DONE rc=3 (seed deberta failed)"; exit 3; }
export HF_HUB_OFFLINE=1

LOCAL=runs/dojo_soa_llama31
LOGD=logs_soa_llama31
OUT=${OUT:-outputs}
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sync_blob() { cp -f "$LOCAL"/* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) &
SYNC_PID=$!

B=tools/controls/dojo_baseline_mn4096.sh
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
M="meta-llama/Llama-3.1-8B-Instruct"
PD="runs/llama31-8b"

run_batt() { # label [battery args...]
  local label="$1"; shift
  echo "[soa-llama31] battery $label START $(date -u '+%F %T')"
  bash "$B" --label "$label" --gpus "$GPUS" --max-new 4096 --model "$M" \
       --outdir "$LOCAL" --logdir "$LOGD" "$@"
  local rc=$?
  echo "[soa-llama31] battery $label EXIT rc=$rc $(date -u '+%F %T')"
  sync_blob
  return "$rc"
}

CS=(--direction dim_no_override_achf --alpha 5 --layers 12,16,20 \
    --match-sigma-to dim_no_override_achf --probe-dir "$PD")

rc=0
run_batt countersteer_llama31   "${CS[@]}"                                           || rc=1
run_batt reminder_llama31         --defense reminder      --probe-dir "$PD"          || rc=1
run_batt pi_protectai_llama31     --defense pi_detector   --probe-dir "$PD"          || rc=1
run_batt pi_promptguard_llama31   --defense pi_detector_promptguard --probe-dir "$PD" || rc=1
run_batt pi_piguard_llama31       --defense pi_detector_piguard     --probe-dir "$PD" || rc=1
run_batt tool_filter_llama31      --defense tool_filter   --probe-dir "$PD"          || rc=1
run_batt stacked_llama31        "${CS[@]}" --stack-dojo spotlighting_with_delimiting || rc=1

kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "DOJO-SOA-LLAMA31-DONE rc=$rc"
exit "$rc"
