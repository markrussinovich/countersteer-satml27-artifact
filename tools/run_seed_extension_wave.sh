#!/usr/bin/env bash
# Wave 7: eight more random draws, because EVERY remaining weakness reduces to
# "the null has only 7 draws".
#
# Adversarial review of wave 5 established that the surviving claim is not a capability
# frontier -- it is simply that dim_no_override_both@8.0 has the lowest ASR of all eight
# arms, and that the six controls whose no_action is at or below ours carry 4.6-7.6x our ASR
# on the all-samples denominator. Against those six as a null sample our ASR is 4.7 SD below
# (new-observation t = -4.73, df=5, one-sided p = 0.003).
#
# But the exact non-parametric bound is rank 1 of 8 -> permutation p = 1/8 = 0.125. The
# p = 0.003 figure requires modelling seven draws as a normal sample. Eight more draws take
# the rank bound to 1/16 = 0.0625 and convert the question into a direct estimate of
# P(random draw <= our ASR at comparable capability) with a usable interval.
#
# CLEAN+ arms are deliberately omitted: the clean-payload cost is already established as
# magnitude-driven (CLEAN+ corr across the existing null spans 0.434-0.658 with ours
# mid-pack, and random5 ties ours EXACTLY at 0.658 with McNemar p=1.00). Dropping them
# halves the generation per process.
#
# Usage: tools/run_seed_extension_wave.sh [WAIT_PIDS...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

COMMON=(--model openai/gpt-oss-20b --stage sweep --outdir "$ROOT/runs/gpt-oss-20b-userabl"
        --corpus shipped --n-eval 96 --match-sigma-to dim_no_override
        --steer-layers 12,16,20 --alphas 8.0 --scale sigma)

for p in "$@"; do
    while [ -e "/proc/$p/exe" ]; do sleep 60; done
    echo "[wave7] pid $p exited $(date -Is)"
done

launch() {
    local gpu="$1" name="$2" dirs="$3"
    HF_HOME="${HF_HOME:-/datadrive/huggingface/}" CUDA_VISIBLE_DEVICES="$gpu" \
        nohup "$PY" -u "$ROOT/xpia_defense.py" "${COMMON[@]}" --directions "$dirs" \
        > "$ROOT/logs/$name.log" 2>&1 &
    echo "[wave7] launched $name on GPU$gpu ($dirs) pid $!"
}

launch 0 seedext_A random8,random9
launch 1 seedext_B random10,random11
launch 2 seedext_C random12,random13
launch 3 seedext_D random14,random15

sleep 90
for n in seedext_A seedext_B seedext_C seedext_D; do
    echo "--- $n ---"; tail -2 "$ROOT/logs/$n.log"
done
