#!/usr/bin/env bash
# Sequential HELD-OUT TEST rung queue for Gemma-4-31B on one GPU.
#
# Runs, in order, after the paper_disjoint alpha-8 pass already in flight:
#   1. paper_param  @ alpha 8  -- the PREREGISTERED primary cell (selected on dev)
#   2. paper_param  @ alpha 6  -- prespecified DOSE-RESPONSE control, not a selection
#      candidate: dev read 0.500 at alpha 6 vs 0.019 at alpha 8, so a test-split alpha 6
#      that does NOT land near 0.500 means the split or the scorer differs and alpha 8
#      must not be interpreted until that is explained.
#   3. paper_disjoint @ alpha 6 -- the same control on the other corpus, for symmetry.
#
# `--stage confirm` rejects more than one --alphas, so the dose-response control has to be
# a separate invocation; it is declared here BEFORE any of its results exist.
#
# --batch 4: every arm of the n=52 dev battery hit `OOM -> sub-batch 4` and the backoff is
# re-discovered per arm (src/arms.py sub_carry is a run_arm local), so batch 8 buys ten
# wasted failed prefills and nothing else.
# --match-sigma-to explicitly: `sigma_ref = args.match_sigma_to or primary` (src/cli.py)
# makes the physical step depend on ARGUMENT ORDER, which under-dosed the Phi-3 battery by
# ~2.06x. A preregistered run must not depend on argument ordering.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-3}"
MODEL=google/gemma-4-31b-it
OUTDIR="$ROOT/runs/gemma4-31b-it"
LAYERS=4,28,36
DIR=dim_no_override_both

run() { # corpus alpha label
  echo "[chain] === $3 (corpus=$1 alpha=$2 gpu=$GPU) ==="
  bash "$ROOT/tools/paper_test_rung.sh" --model "$MODEL" --outdir "$OUTDIR" \
    --corpus "$1" --gpu "$GPU" --alpha "$2" --layers "$LAYERS" --direction "$DIR" \
    --batch 4 --max-new 2048 --label "$3"
  echo "[chain] $3 rc=$?"
}

run paper_param    8.0 gemma_testrung_param_a8
run paper_param    6.0 gemma_testrung_param_a6
run paper_disjoint 6.0 gemma_testrung_disjoint_a6
echo "GEMMA-TESTRUNG-CHAIN-DONE"
