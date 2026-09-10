#!/usr/bin/env bash
# LLMail-Inject replay, smoke-gated, single-GPU — parameterized launcher for the .7 box
# (Phi-3-medium and Gemma-4-31B rows of EVAL_MATRIX; modeled on the frozen GLM launcher
# tools/.frozen_glm_llmail_replay.sh, whose smoke rung + arm_flags gate it inherits).
#
# Invocation 0 = n=24 dev SMOKE, then tools/llmail_smoke_gate.py gates on
#   (a) arm_flags: no aborted arm, trunc <= 0.15 (GLM's llmail smoke died here), AND
#   (b) SCOREABILITY: clean arm produces a non-empty user-visible answer on >= half its
#       samples (`answered`, not tool calls — clean tool calls are structurally 0 on
#       llmail, anchors included; see the gate's docstring).
# A gate failure emits <PREFIX>_LLMAIL_SMOKE-GATE-FAILED and STOPS — the diagnosis
# (unmeasurable-closure vs budget change) goes back to the fleet coordinator, never
# baked in here.
#
# Then anchor-parity replay (mirrors runs/llmail_replay/*_completions.json _meta):
# corpus llmail, n-eval 0, steer-span payload, max_new 1024, stage sweep (dev 1537)
# then stage confirm (test 515 — a fixed external attack set, not a tuned-on split).
#
# Usage (all args required, in order):
#   tools/llmail_replay_p7.sh MODEL OUTDIR LAYERS DIRECTION ALPHA BATCH GPU PREFIX
# e.g.
#   tools/llmail_replay_p7.sh microsoft/Phi-3-medium-128k-instruct \
#       runs/phi3-medium-128k 8,12,16 dim_no_override_bal 6 12 1 PHI3
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-/datadrive/huggingface/}"

MODEL="$1"; OUTDIR="$2"; LAYERS="$3"; DIRECTION="$4"; ALPHA="$5"; BATCH="$6"
GPU="$7"; PREFIX="$8"
# PCI_BUS_ID pins CUDA's device order to nvidia-smi's, so the occupancy guard below and
# the CUDA runtime agree on which physical GPU $GPU names (adversarial review, 2026-09-05
# — this box's adjacent GPUs hold foreign 75GB residents; do not rely on default order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

# The assigned GPU must be FREE at start (foreign residents hold GPUs 0 and 4 on .7;
# 5-7 are another agent's). Abort rather than queue or spill.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "${used:-99999}" -ge 5000 ]; then
  echo "${PREFIX}_LLMAIL_REPLAY_DONE rc_dev=75 rc_test=75 GPU-${GPU}-OCCUPIED (${used} MiB); NOT starting"
  exit 75
fi

run() {
  "$ROOT/.venv/bin/python" -u "$ROOT/xpia_defense.py" --model "$MODEL" \
    --outdir "$ROOT/$OUTDIR" --corpus llmail --steer-span payload \
    --steer-layers "$LAYERS" --scale sigma --mode add --steer-clean \
    --max-new 1024 --batch "$BATCH" --device auto --directions "$DIRECTION" \
    --match-sigma-to dim_no_override_both --alphas "$ALPHA" "$@"
}

# Invocation 0: n=24 dev smoke (mandatory ladder rung).
run --stage sweep --n-eval 24
rc0=$?
echo "${PREFIX}_LLMAIL_SMOKE_DONE rc=$rc0"

"$ROOT/.venv/bin/python" "$ROOT/tools/llmail_smoke_gate.py" \
    --outdir "$ROOT/$OUTDIR" --rc "$rc0"
if [ $? -ne 0 ]; then
  echo "${PREFIX}_LLMAIL_SMOKE-GATE-FAILED"
  echo "${PREFIX}_LLMAIL_REPLAY_DONE rc_dev=98 rc_test=98 SMOKE-GATE-FAILED"
  exit 98
fi

# Invocation 1: DEV portion (stage sweep -> runs/llmail_dataset.dev.json, 1537).
run --stage sweep --n-eval 0
rc1=$?
echo "${PREFIX}_LLMAIL_DEV_DONE rc=$rc1"

# Invocation 2: TEST portion (stage confirm -> runs/llmail_dataset.test.json, 515).
run --stage confirm --n-eval 0
rc2=$?
echo "${PREFIX}_LLMAIL_REPLAY_DONE rc_dev=$rc1 rc_test=$rc2"
