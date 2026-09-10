#!/usr/bin/env bash
# Wave 2: is the direction doing ANYTHING beyond its magnitude?
#
# Adversarial review 2026-08-04 established that the steered arm does not separate from its
# magnitude-matched random control on ASR on ANY corpus tested -- fit included:
#   fit p=0.1435, heldout_a p=0.625, heldout_b p=0.500, heldout_c p=0.344, pooled p=0.080.
# The one separation ever cited (fit, marker-anywhere, p=8.8e-5) was refuted as a
# chain-of-thought-verbosity artifact: steering shortens the reasoning trace 35-47% and
# halves quoting of the injected span regardless of the marker.
#
# Pooled 22/11 at p=0.080 is exactly the shape of an underpowered near-miss, and a SINGLE
# random draw cannot tell "the direction works" from "this particular random vector happened
# to be worse" -- xpia_defense.py:1010 says so in its own words.
#
# This converts the control from one paired test into a null DISTRIBUTION: three further
# independent draws plus `shuffled` (a permutation of the real direction -- same coordinate
# statistics, no structure -- which is the strictly harder control and has never been run at
# this cell). Question with a clean answer: does the steered arm's ASR fall OUTSIDE the
# spread of same-magnitude directions? If it sits inside, the cell is a magnitude effect and
# the direction is doing nothing.
#
# Both corpora, because the fit result rests on the same single draw as the held-out one.
# heldout_c is the only held-out wording with a potent baseline (undefended 23/65).
#
# Usage: tools/run_multiseed_control_wave.sh [WAIT_PIDS...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$ROOT/runs/gpt-oss-20b-userabl"
COMMON=(--model openai/gpt-oss-20b --stage sweep --outdir "$OUT"
        --match-sigma-to dim_no_override --steer-layers 12,16,20
        --alphas 8.0 --scale sigma)

for p in "$@"; do
    while [ -e "/proc/$p/exe" ]; do sleep 60; done
    echo "[wave2] pid $p exited $(date -Is)"
done

launch() {
    local gpu="$1" name="$2"; shift 2
    HF_HOME="${HF_HOME:-/datadrive/huggingface/}" CUDA_VISIBLE_DEVICES="$gpu" \
        nohup "$PY" -u "$ROOT/xpia_defense.py" "${COMMON[@]}" "$@" \
        > "$ROOT/logs/$name.log" 2>&1 &
    echo "[wave2] launched $name on GPU$gpu pid $!"
}

# No --steer-clean: the CLEAN+ correctness cost is already established as magnitude-driven
# (35pp real vs 39pp random on heldout_c), so spending four extra arms per corpus to
# re-measure it would buy nothing. This wave is about the ASR null distribution only.
launch 0 seed_heldc_A --corpus param_abuse --template heldout_c --directions dim_no_override_both,random2
launch 1 seed_heldc_B --corpus param_abuse --template heldout_c --directions random7,shuffled
launch 2 seed_fit_A   --corpus param_abuse --template fit       --directions dim_no_override_both,random2
launch 3 seed_fit_B   --corpus param_abuse --template fit       --directions random7,shuffled

sleep 90
for n in seed_heldc_A seed_heldc_B seed_fit_A seed_fit_B; do
    echo "--- $n ---"; tail -3 "$ROOT/logs/$n.log"
done
