#!/usr/bin/env bash
# Wave 3: push the boundary rule harder, now that it is the only cell that isolates
# DIRECTION-specific effect from MAGNITUDE effect.
#
# Wave 1 result, parameter abuse / fit wording, scoreable 65:
#   attacked                        44/65
#   direction + boundary rule       24/65   vs attacked  23 fixed / 3 broke  p=8.8e-5
#   random    + boundary rule       48/65   vs attacked   3 fixed / 7 broke  p=0.34  (NOTHING)
#   direction vs random, boundary   24 fixed / 0 broke    p<1e-6
#
# That is the control problem resolved from the opposite side. Under the FIXED rule the
# random control reduces ASR by sheer magnitude, which masks whatever the direction
# contributes (fixed-rule steered-vs-random: p=0.14 fit, 0.34 heldout_c). Under the boundary
# rule the step is gated on the direction's OWN projection, so a random axis -- whose
# projection carries no information -- produces no useful step and no effect. The direction
# halving ASR under that rule cannot be a magnitude artifact.
#
# The cost: raw ASR is much worse than the fixed rule (0.369 vs 0.138). The boundary rule
# only moves tokens above the 95th percentile of the LEGITIMATE distribution, and only just
# past it. This wave asks whether pushing it harder buys fixed-rule ASR while KEEPING the
# direction-specific credit -- which is the whole question for the project.
#
#   margin   push each over-boundary token further past the boundary
#   mirror   reflect instead of land (2x the deficit)
#
# Every cell carries the same-rule random control, because the entire point is the contrast.
#
# Usage: tools/run_boundary_strength_wave.sh [WAIT_PIDS...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$ROOT/runs/gpt-oss-20b-userabl"
COMMON=(--model openai/gpt-oss-20b --stage sweep --outdir "$OUT"
        --corpus param_abuse --match-sigma-to dim_no_override
        --steer-layers 12,16,20 --alphas 0 --scale sigma
        --directions dim_no_override_both,random --steer-clean)

for p in "$@"; do
    while [ -e "/proc/$p/exe" ]; do sleep 60; done
    echo "[wave3] pid $p exited $(date -Is)"
done

launch() {
    local gpu="$1" name="$2"; shift 2
    HF_HOME="${HF_HOME:-/datadrive/huggingface/}" CUDA_VISIBLE_DEVICES="$gpu" \
        nohup "$PY" -u "$ROOT/xpia_defense.py" "${COMMON[@]}" "$@" \
        > "$ROOT/logs/$name.log" 2>&1 &
    echo "[wave3] launched $name on GPU$gpu pid $!"
}

launch 0 bnd_m1_fit  --template fit       --step-rule boundary --step-margin 1.0
launch 1 bnd_m2_fit  --template fit       --step-rule boundary --step-margin 2.0
launch 2 bnd_mir_fit --template fit       --step-rule mirror
# does the direction-specific effect survive on wording the direction never saw?
launch 3 bnd_m1_hc   --template heldout_c --step-rule boundary --step-margin 1.0

sleep 90
for n in bnd_m1_fit bnd_m2_fit bnd_mir_fit bnd_m1_hc; do
    echo "--- $n ---"; tail -3 "$ROOT/logs/$n.log"
done
