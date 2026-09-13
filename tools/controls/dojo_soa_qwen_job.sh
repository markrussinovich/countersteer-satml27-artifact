#!/usr/bin/env bash
# AML-cluster driver: SoA-defense comparison WAVE 2 on Qwen3-30B-A3B-Thinking-2507 — the
# full 180-cell AgentDojo grid at max_new 4096, FOUR-ARM batteries (see
# dojo_soa_gptoss_job.sh for the design rationale; identical structure, second anchor model).
#
# What this adds over the existing Qwen tables (tab:baselines-qwen, dojo-qwen-baselines-
# 20260830): (a) a four-arm CounterSteer battery at the deployed 12-sigma cell — the paper
# notes the current Qwen CounterSteer AgentDojo row has NO same-process undefended anchor;
# this battery repairs that; (b) the AutoDojo reminder + filter-classifier arms; (c) the
# stacked CounterSteer+spotlighting composition arm. spotlighting / sandwich / tool_filter
# Qwen rows already exist at 4096 with their own same-process undefended anchor and are NOT
# re-run.
#
# Qwen deployed cell (tab:qwen-dose, runs/qwen_ad_a12_def.shard*.json config):
#   dim_no_override_both @ alpha 12, layers 8,20,32, OWN sigma (--match-sigma-to '').
#
# SUBMIT (after the gpt-oss SoA job frees the node). TIMEOUT: the 2026-09-04 adversarial
# review measured the Qwen four-arm battery at ~3.5-4.5 h (170-190 s/episode precedent +
# shard skew + steering think-length), so six batteries need >=100000 s — do NOT submit
# with the 22 h default:
#   bash cluster/submit_job.sh --mode run --display-name xpia-dojo-soa-qwen \
#     --timeout-seconds 115200 --no-clean --slmx-cmd 'bash tools/controls/dojo_soa_qwen_job.sh'
set -uo pipefail

LOCAL=runs/dojo_soa_qwen
LOGD=logs_soa_qwen
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

run_batt() { # label [battery args...]
  local label="$1"; shift
  echo "[soa-qwen] battery $label START $(date -u '+%F %T')"
  bash "$B" --label "$label" --gpus "$GPUS" --max-new 4096 --model "$M" \
       --outdir "$LOCAL" --logdir "$LOGD" "$@"
  local rc=$?
  echo "[soa-qwen] battery $label EXIT rc=$rc $(date -u '+%F %T')"
  sync_blob
  return "$rc"
}

CS=(--direction dim_no_override_both --alpha 12 --layers 8,20,32 --match-sigma-to "" \
    --probe-dir "$PD")

rc=0
run_batt countersteer_qwen  "${CS[@]}"                                            || rc=1
run_batt reminder_qwen        --defense reminder      --probe-dir "$PD"           || rc=1
run_batt pi_protectai_qwen    --defense pi_detector   --probe-dir "$PD"           || rc=1
run_batt pi_promptguard_qwen  --defense pi_detector_promptguard --probe-dir "$PD" || rc=1
run_batt pi_piguard_qwen      --defense pi_detector_piguard     --probe-dir "$PD" || rc=1
run_batt stacked_qwen       "${CS[@]}" --stack-dojo spotlighting_with_delimiting  || rc=1

kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "DOJO-SOA-QWEN-DONE rc=$rc"
exit "$rc"
