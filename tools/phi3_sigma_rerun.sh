#!/usr/bin/env bash
# Phi-3-medium-128k DOSE-RESPONSE rerun at the CORRECT sigma reference.
#
# WHY (FINDINGS 18c-battery). The n=96/n=66 Phi-3 battery ran with `--match-sigma-to`
# UNSET. `sigma_ref = args.match_sigma_to or primary` (src/cli.py) then falls back to the
# FIRST direction listed, which was `dim_no_override_bal` -- sigma [0.692, 1.524, 2.586] --
# whereas the n=24 tuning runs that chose the alphas pinned `dim_no_override_both` --
# sigma [1.417, 3.162, 5.491]. Ratio 2.049 / 2.075 / 2.124, so the battery's alpha 4 and
# alpha 6 were physically ~alpha 1.95 and ~alpha 2.93 in tuning units. The battery is
# therefore UNDER-DOSED, and its goal 0.729->0.333 / 0.576->0.364 is a point on a
# trade-off curve, not a ceiling.
#
# WHAT THIS ANSWERS. (1) Reproduces the tuning-run numbers at battery n, confirming or
# refuting the 2.06x under-dose diagnosis. (2) Maps the actual dose-response curve.
# alpha 3 is the CROSS-RUNG CONSISTENCY ANCHOR: at the tuning sigma the n=24 run scored
# goal 0.292 there against the battery's 0.333 at n=96, so an alpha 3 here that does not
# land near 0.292-0.333 means something other than dose differs and the whole curve is
# uninterpretable until that is explained. At the tuning sigma alpha 6 reached goal 0.000
# at utilBenign 50.0%/58%.
#
# EVERYTHING except the sigma reference and the alpha grid is byte-identical to the
# battery config (runs/phi3-medium-128k/results_add-dim-no-override-bal-324218{9,1}.json):
# same corpora, same --template fit, same L8/12/16, same direction, same --stage sweep,
# same n, same max_new 2048, same --steer-clean. One variable changes.
#
# GATE BEFORE READING ASR: trunc < 0.1 per arm. A steered arm that truncates is a censored
# ASR, not a defended one -- the random controls in the Gemma battery read goal 0.000 at
# trunc 0.48-0.92 for exactly that reason.
#
# Usage: bash tools/phi3_sigma_rerun.sh <GPU>
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
GPU="${1:-0}"
MODEL=microsoft/Phi-3-medium-128k-instruct
OUTDIR="$ROOT/runs/phi3-medium-128k"
mkdir -p "$OUTDIR" "$ROOT/logs"

run() { # corpus n_eval label
  local log="$ROOT/logs/$3.log"
  echo "[phi3-sigma] === $3 (corpus=$1 n=$2 gpu=$GPU) -> $log ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/xpia_defense.py" \
    --model "$MODEL" --outdir "$OUTDIR" --device cuda:0 \
    --stage sweep --corpus "$1" --template fit --n-eval "$2" \
    --steer-layers 8,12,16 --directions dim_no_override_bal \
    --match-sigma-to dim_no_override_both \
    --alphas 3.0 4.0 6.0 \
    --scale sigma --steer-clean --batch 8 --max-new 2048 > "$log" 2>&1
  echo "[phi3-sigma] $3 rc=$?"
}

run shipped     96 phi3_sigma_shipped
run param_abuse 66 phi3_sigma_param
echo "PHI3-SIGMA-RERUN-DONE"
