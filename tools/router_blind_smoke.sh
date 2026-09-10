#!/usr/bin/env bash
# ROUTER-BLIND / ROUTER-NULL smoke ladder (FINDINGS section 23 follow-up).
#
# The section-23 diagnosis: on Qwen3-Next-80B and GLM-4.5-Air the steering DIRECTION is fine
# (fired-vs-not separation 0.8-1.0 sigma, same as the two models where a cell works) but
# every behaviourally effective dose moves 55-90% of the top-k MoE routing mass, and the
# degeneration that follows (rumination, repetition loops, tool-call spam, truncation) is the
# phenotype of computing whole spans on the wrong experts. The working gpt-oss cell leaves
# ~90% of routing intact. So the hypothesis under test here is: KEEP THE EDIT, HIDE IT FROM
# THE ROUTERS.
#
# Three lanes, each a separate remedy, each emitting the full three-condition matrix plus
# CLEAN+ in ONE process so every defended arm is paired sample-for-sample against the same
# clean and base-XPIA arms:
#
#   BLIND     --router-blind accum : the residual edit is subtracted back out of the input of
#             every downstream MoE router, in PRE-NORM space with the model's own norm module
#             re-applied. Routers route as if unsteered; attention and the experts still see
#             the steered stream. `accum` is the cheap single-pass estimate.
#   EXACT     --router-blind clean : the same thing with an extra unsteered prefill supplying
#             the literal clean router input. EXACT, ~2x prefill cost. Run this only where
#             BLIND shows signal, or where --router-blind-report says `accum` is too loose.
#   RNULL     ordinary steering along a direction that has been projected out of the top-r
#             right singular subspace of the stacked downstream router gates
#             (tools/controls/build_routernull_direction.py). No new hooks, no extra pass.
#
# READING RULE (owner directive, FINDINGS section 18f): an arm's ASR is quotable only where
# its trunc < 0.1. --max-new is therefore a required argument.
#
# Usage:
#   tools/router_blind_smoke.sh --model M --tag T --layers 28,32,40 \
#       --corpus shipped [--corpus-2 paper_disjoint] --alphas "8 12" --max-new 8192 \
#       [--lanes blind,exact,rnull] [--direction dim_no_override_both] \
#       [--rnull-direction dim_no_override_both_rnull128] \
#       [--n-eval 24] [--batch 4] [--report]
set -uo pipefail

MODEL=""; TAG=""; LAYERS=""; CORPUS=""; CORPUS_2=""; ALPHAS=""; MAX_NEW=""
LANES="blind,rnull"; DIRECTION="dim_no_override_both"; RNULL_DIRECTION=""
N_EVAL=24; BATCH=4; REPORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --corpus) CORPUS="$2"; shift 2 ;;
    --corpus-2) CORPUS_2="$2"; shift 2 ;;
    --alphas) ALPHAS="$2"; shift 2 ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --lanes) LANES="$2"; shift 2 ;;
    --direction) DIRECTION="$2"; shift 2 ;;
    --rnull-direction) RNULL_DIRECTION="$2"; shift 2 ;;
    --n-eval) N_EVAL="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --report) REPORT="--router-blind-report"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
for req in MODEL TAG LAYERS CORPUS ALPHAS MAX_NEW; do
  [[ -n "${!req}" ]] || { echo "FATAL: missing required argument for $req" >&2; exit 2; }
done

PY="${PY:-$(command -v python)}"
OUTD="outputs/$TAG"
mkdir -p "$OUTD"
harvest() { cp -f "runs/$TAG"/results_*.json "$OUTD/" 2>/dev/null || true; }
trap harvest EXIT

# PREFLIGHT. A missing sigma does NOT crash the additive path -- probes.build_dirs returns
# 0.0 via .get(name, 0.0), which makes step = alpha*0 = 0: a no-op arm that reports as a
# defense. Check every direction any lane will ask for, before spending the node.
CHECK_DIRS="$DIRECTION"
[[ ",$LANES," == *",rnull,"* ]] && CHECK_DIRS="$CHECK_DIRS,${RNULL_DIRECTION:-}"
TAG="$TAG" LAYERS="$LAYERS" CHECK_DIRS="$CHECK_DIRS" $PY - <<'PYCHK' || { echo "FATAL: staged pickles unusable"; exit 1; }
import os, pickle, sys
sys.path.insert(0, "tools/controls")
import _probe_eval as E
import __main__ as m
if not hasattr(m, "TorchLogReg"): m.TorchLogReg = E.X.TorchLogReg
tag = os.environ["TAG"]
names = [d for d in os.environ["CHECK_DIRS"].split(",") if d]
if not names:
    sys.exit("no direction to check -- --lanes rnull needs --rnull-direction")
for L in [int(x) for x in os.environ["LAYERS"].split(",")]:
    d = pickle.load(open(f"runs/{tag}/probe_L{L}.pkl", "rb"))
    for n in names:
        assert n in d.get("dirs", {}), f"L{L} missing direction {n}"
        assert d.get("sigmas", {}).get(n), f"L{L} missing sigma for {n}"
print(f"[preflight] {names} dir+sigma present at L{os.environ['LAYERS']}")
PYCHK

sweep() {  # DIRECTION CORPUS EXTRA GPUS LOGNAME
  local dirname=$1 corpus=$2 extra=$3 gpus=$4 logname=$5
  echo "=== [$TAG] $(date -u +%H:%M:%S) dir=$dirname corpus=$corpus extra=[$extra] gpus=$gpus"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpus $PY xpia_defense.py --model "$MODEL" --outdir "runs/$TAG" \
      --device auto --stage sweep --corpus "$corpus" --n-eval "$N_EVAL" \
      --directions "$dirname" --alphas $ALPHAS --scale sigma --mode add \
      --steer-layers "$LAYERS" --steer-clean --batch "$BATCH" --max-new "$MAX_NEW" \
      $extra 2>&1 | tee "$OUTD/$logname"
  return "${PIPESTATUS[0]}"
}

run_lane() {  # LANE GPUS
  local lane=$1 gpus=$2 extra="" dirname="$DIRECTION"
  case "$lane" in
    blind) extra="--router-blind accum $REPORT" ;;
    exact) extra="--router-blind clean $REPORT" ;;
    rnull) dirname="$RNULL_DIRECTION"
           [[ -n "$dirname" ]] || { echo "FATAL: --lanes rnull needs --rnull-direction" >&2; return 2; } ;;
    *) echo "unknown lane: $lane" >&2; return 2 ;;
  esac
  sweep "$dirname" "$CORPUS" "$extra" "$gpus" "rblind_${lane}_${CORPUS}.log" || return 1
  if [[ -n "$CORPUS_2" ]]; then
    sweep "$dirname" "$CORPUS_2" "$extra" "$gpus" "rblind_${lane}_${CORPUS_2}.log" || return 1
  fi
}

rc=0; pids=(); names=(); gpu=0
IFS=',' read -ra LANE_LIST <<< "$LANES"
for lane in "${LANE_LIST[@]}"; do
  [[ -n "$lane" ]] || continue
  # one 4-GPU lane per entry on an 8-GPU node; more than two lanes are serialised by the
  # scheduler rather than oversubscribed
  g="$gpu,$((gpu+1)),$((gpu+2)),$((gpu+3))"
  run_lane "$lane" "$g" &
  pids+=($!); names+=("$lane"); gpu=$((gpu+4))
  [[ $gpu -ge 8 ]] && { for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "*** lane ${names[$i]} FAILED"; rc=1; }; done; pids=(); names=(); gpu=0; }
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "*** lane ${names[$i]} FAILED"; rc=1; }; done
harvest
echo "ROUTER-BLIND-SMOKE-DONE tag=$TAG rc=$rc"
exit $rc
