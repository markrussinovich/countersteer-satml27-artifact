#!/usr/bin/env bash
# Wave 1 of the step-rule experiment: does making the step proportional to a token's own
# override-ness recover the correctness that fixed-magnitude span-wide steering costs?
#
# The question this answers, stated so a null is as informative as a win: the entire
# correctness cost is parameter drift, it is IDENTICAL with no attacker present (25 vs 25 on
# shipped), and it is identical under a magnitude-matched random direction -- so it is caused
# by perturbation magnitude, not by the direction. Fixed alpha*sigma hits all ~148 payload
# tokens while the injection is only ~33% of them. If CLEAN+ correctness does NOT rise
# materially above 0.636 under a proportional rule, the gating/magnitude family is dead and
# the lever is the operator (ablation / norm-preserving rotation) instead.
#
# Waits for any running xpia_defense jobs to exit first, by PID, verified through
# /proc/PID/exe -- never a pgrep -f pattern match, which matches this script's own argv.
#
# Usage: tools/run_step_rule_wave.sh [WAIT_PIDS...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$ROOT/runs/gpt-oss-20b-userabl"
MODEL="openai/gpt-oss-20b"
COMMON=(--model "$MODEL" --stage sweep --outdir "$OUT"
        --directions dim_no_override_both,random --match-sigma-to dim_no_override
        --steer-layers 12,16,20 --scale sigma --steer-clean)

for p in "$@"; do
    while [ -e "/proc/$p/exe" ]; do sleep 60; done
    echo "[wave] pid $p exited $(date -Is)"
done

launch() {  # launch GPU LOGNAME ARGS...
    local gpu="$1" name="$2"; shift 2
    HF_HOME="${HF_HOME:-/datadrive/huggingface/}" CUDA_VISIBLE_DEVICES="$gpu" \
        nohup "$PY" -u "$ROOT/xpia_defense.py" "${COMMON[@]}" "$@" \
        > "$ROOT/logs/$name.log" 2>&1 &
    echo "[wave] launched $name on GPU$gpu pid $!"
}

# GPU0 -- the knee of the EXISTING curve. alpha has never been swept for this direction;
#         8.0 was inherited from the superseded cell.
launch 0 step_fixed_sweep   --corpus shipped --n-eval 24 --alphas 2 4 6 8 --step-rule fixed
# GPU1/2 -- the two proportional rules. They set their own magnitude and ignore --alphas,
#           so the grid is a single placeholder entry.
launch 1 step_boundary_ship --corpus shipped --n-eval 24 --alphas 0 --step-rule boundary
launch 2 step_mirror_ship   --corpus shipped --n-eval 24 --alphas 0 --step-rule mirror
# GPU3 -- the same boundary rule on the corpus where the defense is weakest (ASR 0.138).
launch 3 step_boundary_par  --corpus param_abuse --alphas 0 --step-rule boundary

sleep 90
for n in step_fixed_sweep step_boundary_ship step_mirror_ship step_boundary_par; do
    echo "--- $n ---"; tail -3 "$ROOT/logs/$n.log"
done
