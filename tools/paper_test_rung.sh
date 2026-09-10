#!/usr/bin/env bash
# HELD-OUT TEST rung for a preselected steering cell on the paper corpora.
#
# WHY. `--stage sweep` evaluates on the DEV split and is where alpha is chosen; a number
# from it is a TUNING number (the pipeline itself prints "(dev/tuning)"). The test split is
# template-TEXT-disjoint from dev and exists to be measured ONCE, with the cell fixed in
# advance -- `src/cli.py` enforces this by rejecting more than one --alphas under
# `--stage confirm`. This script is that single preregistered pass: three conditions
# (clean / attacked / defended) plus CLEAN+ (steer-clean), one dose, one direction.
#
# Usage:
#   bash tools/paper_test_rung.sh --model google/gemma-4-31b-it --outdir runs/gemma4-31b-it \
#     --corpus paper_disjoint --gpu 3 --alpha 8.0 --layers 4,28,36 \
#     --direction dim_no_override_both --batch 8 --max-new 2048 --label gemma_testrung_disjoint
#
# Produces <outdir>/results_*.json (+ _completions.json) and <logdir>/<label>.log; prints
# PAPER-TEST-RUNG-DONE on exit.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"

MODEL=""; OUTDIR=""; CORPUS=""; GPU=""; ALPHA=""; LAYERS=""; DIRECTION=""
BATCH=8; MAXNEW=2048; NEVAL=52; LABEL=""; LOGDIR="$ROOT/logs"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;      --outdir) OUTDIR="$2"; shift 2 ;;
    --corpus) CORPUS="$2"; shift 2 ;;    --gpu) GPU="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;      --layers) LAYERS="$2"; shift 2 ;;
    --direction) DIRECTION="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;      --max-new) MAXNEW="$2"; shift 2 ;;
    --n-eval) NEVAL="$2"; shift 2 ;;     --label) LABEL="$2"; shift 2 ;;
    --logdir) LOGDIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
for v in MODEL OUTDIR CORPUS GPU ALPHA LAYERS DIRECTION; do
  [[ -n "${!v}" ]] || { echo "FATAL: need --${v,,}" >&2; exit 2; }
done
LABEL="${LABEL:-testrung_${CORPUS}}"
[[ "$OUTDIR" = /* ]] || OUTDIR="$ROOT/$OUTDIR"
[[ -e "$PY" ]] || { echo "FATAL: missing $PY" >&2; exit 1; }
mkdir -p "$OUTDIR" "$LOGDIR"
LOG="$LOGDIR/${LABEL}.log"

echo "[test-rung] model=$MODEL corpus=$CORPUS gpu=$GPU alpha=$ALPHA layers=$LAYERS dir=$DIRECTION"
echo "[test-rung] log=$LOG outdir=$OUTDIR"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/xpia_defense.py" \
  --model "$MODEL" --outdir "$OUTDIR" --device cuda:0 \
  --stage confirm --corpus "$CORPUS" --n-eval "$NEVAL" \
  --steer-layers "$LAYERS" --directions "$DIRECTION" --alphas "$ALPHA" \
  --match-sigma-to "$DIRECTION" \
  --scale sigma --steer-clean --batch "$BATCH" --max-new "$MAXNEW" > "$LOG" 2>&1
rc=$?
echo "PAPER-TEST-RUNG-DONE label=$LABEL rc=$rc"
exit "$rc"
