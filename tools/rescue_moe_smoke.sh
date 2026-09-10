#!/usr/bin/env bash
# MoE RESCUE smoke ladder (FINDINGS §23 follow-up): two zero-new-code interventions run as
# two parallel 4-GPU lanes on one 8-GPU node, for any model whose probe pickles carry the
# steered direction.
#
#   lane DOSE    additive steering (mode=add) over an explicit alpha grid -- the
#                "angle-matched re-dose" remedy. §23a finding 1: our sigma unit is not
#                norm-comparable across models, so the useful unit is delivered ROTATION
#                and the grid must straddle the model's own behavioural threshold rather
#                than reuse the 1/4/16/64 bracket.
#   lane ABL     projection ABLATION (alpha ignored, dose-free by construction, so it
#                cannot overshoot), in one of two flavours selected by --abl-mode:
#                  ablate     h <- h - (h.d)d on every span token. §23d pre-screen: on
#                             Qwen3-Next it costs 8-14 deg of residual rotation and keeps
#                             71-75% of top-k MoE routing, vs 42-45% kept at the first
#                             additive dose with any behavioural effect.
#                  ablate_mp  MEAN-PRESERVING: h <- h - ((h-mu).d)d. Removes the SAME
#                             coordinate while CANCELLING that net displacement.
#                             §23k found plain `ablate` is NOT a neutral axis removal: the
#                             coordinate has a large non-zero mean, so deleting it applies a
#                             net push -- -2.83 sigma on Qwen3-Next L28/32/40 (additive
#                             alpha ~ -1.6) in the ATTACK-favouring direction, +0.75 sigma
#                             on GLM-4.5-Air L20/24/28. That confounds the ablation null.
#                             mu comes from --mu-source (default probe_grand, the probe
#                             pickles' own corpus mean); see src/probes.build_means. The
#                             cancellation is EXACT only for --mu-source span, whose mu is
#                             taken at the steer site; a stored mu is a PROBE-site mean
#                             applied at the BLOCK-OUTPUT site, so a residual push of
#                             unmeasured size survives on these models. The two are a PAIR
#                             (probe_grand deletes the span-level offset but leaves a
#                             residual push; span leaves the offset but has no residual),
#                             so run both lanes rather than choosing.
#
# Both lanes emit the full three-condition matrix plus CLEAN+ in ONE process per lane, so
# every defended arm is paired sample-for-sample against the same clean and base-XPIA arms
# (src/cli.py threads the clean completions in as the correctness reference).
#
# READING RULE (owner directive 2026-09-01, FINDINGS §18f): an arm's ASR is quotable only
# where its trunc < 0.1. Truncation is why the §18f/§18g brackets could not be read at the
# doses that moved goal; --max-new is a required argument here for that reason.
#
# Usage:
#   tools/rescue_moe_smoke.sh --model M --tag T --layers 20,24,28 \
#       --dose-corpus paper_param --dose-alphas "6 8 10 12" \
#       --abl-corpus paper_param [--abl-corpus-2 shipped] \
#       [--abl-mode ablate|ablate_mp] [--mu-source probe_grand|role_mean|span|capture:PATH] \
#       --max-new 8192 [--n-eval 24] [--batch 4] [--direction dim_no_override_both]
#
# The paired control for an ablation lane is a NEGATIVE additive dose, which isolates the
# mean shift from the deletion; it runs in the DOSE lane and argparse accepts it as written:
#       --dose-alphas "-1"          (equivalently --alphas=-1 when calling xpia_defense.py)
set -uo pipefail

MODEL=""; TAG=""; LAYERS=""; DOSE_CORPUS=""; DOSE_ALPHAS=""; ABL_CORPUS=""; ABL_CORPUS_2=""
MAX_NEW=""; N_EVAL=24; BATCH=4; DIRECTION="dim_no_override_both"
ABL_MODE="ablate"; MU_SOURCE=""
DOSE_GPUS="0,1,2,3"; ABL_GPUS="4,5,6,7"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --dose-corpus) DOSE_CORPUS="$2"; shift 2 ;;
    --dose-alphas) DOSE_ALPHAS="$2"; shift 2 ;;
    --abl-corpus) ABL_CORPUS="$2"; shift 2 ;;
    --abl-corpus-2) ABL_CORPUS_2="$2"; shift 2 ;;
    --abl-mode) ABL_MODE="$2"; shift 2 ;;
    --mu-source) MU_SOURCE="$2"; shift 2 ;;
    --dose-gpus) DOSE_GPUS="$2"; shift 2 ;;
    --abl-gpus) ABL_GPUS="$2"; shift 2 ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --n-eval) N_EVAL="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --direction) DIRECTION="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
# `--dose-alphas none` / `--abl-corpus none` run ONE lane on a shared node, so two models
# (or a re-run of just one lane) can occupy the two halves of a single 8-GPU node. NOTE:
# with the DOSE lane disabled AND --abl-corpus-2 given, the second corpus takes over
# $DOSE_GPUS and runs in PARALLEL rather than chained -- otherwise half the node idles for
# hours (see the lane launch below).
[[ "$DOSE_ALPHAS" == "none" ]] && DOSE_CORPUS="none"
for req in MODEL TAG LAYERS DOSE_CORPUS DOSE_ALPHAS ABL_CORPUS MAX_NEW; do
  [[ -n "${!req}" ]] || { echo "FATAL: missing required argument for $req" >&2; exit 2; }
done
[[ "$DOSE_ALPHAS" == "none" && "$ABL_CORPUS" == "none" ]] && {
  echo "FATAL: both lanes disabled -- nothing to run" >&2; exit 2; }
# --mu-source is meaningful ONLY for a mean-preserving lane. Accepting it silently under
# --abl-mode ablate would run PLAIN ablation -- the operator ablate_mp exists to
# de-confound -- from a command line that reads as mean-preserving. src/cli.py refuses the
# same combination; refuse it here too, before the node is claimed.
if [[ -n "$MU_SOURCE" && "$ABL_MODE" != ablate_mp* ]]; then
  echo "FATAL: --mu-source $MU_SOURCE given but --abl-mode is $ABL_MODE, which reads no mean" >&2
  exit 2
fi

PY="${PY:-$(command -v python)}"
OUTD="outputs/$TAG"
# LANE LOGS GO TO LOCAL SCRATCH, NOT STRAIGHT TO THE BLOB MOUNT. `outputs/` is a blobfuse
# mount, and blobfuse does not publish a file that is still OPEN: a lane's `tee` holds its
# log open for the whole sweep, so the log reads as ZERO BYTES from outside the job until
# that sweep ends. That produced a live-monitoring false alarm on 2026-09-01 -- a long dose
# lane looked identical to a lane that had never started, while a short ablate lane that had
# already FINISHED (and so closed its file) was fully readable. Note this is NOT python-side
# buffering: the startup banner at src/cli.py:403 is printed with flush=True, so a running
# lane has always emitted its config header within seconds. Writing locally and copying on
# an interval republishes a CLOSED file each time, which blobfuse does publish.
LANELOG="${TMPDIR:-/tmp}/lanelogs-$$"
mkdir -p "$OUTD" "$LANELOG"
sync_logs() { cp -f "$LANELOG"/*.log "$OUTD/" 2>/dev/null || true; }
harvest() {
  cp -f "runs/$TAG"/results_*.json "$OUTD/" 2>/dev/null || true
  sync_logs
  [[ -n "${SYNC_PID:-}" ]] && kill "$SYNC_PID" 2>/dev/null
  return 0
}
trap harvest EXIT
# republish every lane log once a minute so an in-flight run is observable from outside
( while true; do sleep 60; sync_logs; done ) &
SYNC_PID=$!

# PREFLIGHT. A missing sigma does NOT crash the additive path -- probes.py returns 0.0 via
# .get(name, 0.0), which makes step = alpha*0 = 0: a no-op arm that reports as a defense.
# Check before spending the node (pre-submission review, 2026-09-01).
TAG="$TAG" LAYERS="$LAYERS" DIRECTION="$DIRECTION" ABL_MODE="$ABL_MODE" \
MU_SOURCE="$MU_SOURCE" MODEL="$MODEL" $PY - <<'PYCHK' || { echo "FATAL: staged pickles unusable"; exit 1; }
import os, pickle, sys
sys.path.insert(0, "tools/controls")
import _probe_eval as E
import __main__ as m
if not hasattr(m, "TorchLogReg"): m.TorchLogReg = E.X.TorchLogReg
tag, dirname = os.environ["TAG"], os.environ["DIRECTION"]
for L in [int(x) for x in os.environ["LAYERS"].split(",")]:
    d = pickle.load(open(f"runs/{tag}/probe_L{L}.pkl", "rb"))
    assert dirname in d.get("dirs", {}), f"L{L} missing direction {dirname}"
    assert d.get("sigmas", {}).get(dirname), f"L{L} missing sigma for {dirname}"
print(f"[preflight] {dirname} dir+sigma present at L{os.environ['LAYERS']}")
# MEAN-PRESERVING ABLATION needs a mu as well, and a missing mu is exactly the
# silent-fallback class of bug that turns a defended arm into an undefended one under a
# defended name -- so resolve it HERE, before the node is spent. build_means raises with the
# missing artifact named; it never returns zeros.
# ABLATE_MODES, NOT MODES. `add` is a legal --mode but it is NOT dose-free, and this lane
# always passes `--alphas 0`: run_arm's guard would build NO Steer and the "defended" arm
# would run COMPLETELY UNDEFENDED under an ablation lane's name and log file. That is
# FINDINGS 23e exactly, and accepting `add` here would re-open it (review, 2026-09-02).
mode = os.environ["ABL_MODE"]
if mode not in E.X.ABLATE_MODES:
    raise SystemExit(f"--abl-mode {mode} is not an ablation mode; this lane runs at "
                     f"--alphas 0, where only {list(E.X.ABLATE_MODES)} build a Steer")
if mode in E.X.MEAN_PRESERVING_MODES:
    layers = [int(x) for x in os.environ["LAYERS"].split(",")]
    srcname = os.environ["MU_SOURCE"] or E.X.MU_DEFAULT
    if srcname == "span":
        print("[preflight] mu source `span` is computed per input; nothing to stage")
    else:
        mus = E.X.build_means(f"runs/{tag}", layers, "cpu", srcname,
                              model_id=os.environ["MODEL"])
        # the SAME helper the sweep prints from, off the SAME build_dirs output -- two
        # implementations of this number would disagree the moment a sigma is substituted
        dirs, sig, _ = E.X.build_dirs(f"runs/{tag}", layers, dirname, "cpu")
        mp = E.X.mu_projection(mus, dirs, sig)
        for L, v in zip(layers, mp):
            print(f"[preflight] L{L} mu.d_hat = {v:+.3f} sigma")
        print(f"[preflight] plain `ablate` would apply {-sum(mp):+.3f} sigma NET along d "
              f"over L{os.environ['LAYERS']} (additive-equivalent alpha "
              f"{-sum(mp) / len(layers) ** 0.5:+.2f}); `{mode}` applies 0 by construction")
PYCHK

# THE WIRING REGRESSIONS, before the node is claimed. 6 CPU-seconds against the failure mode
# that has now cost two 8xH100 nodes: an arm with no hook reporting itself as a defense.
for v in verify_ablate_wiring verify_ablate_mp; do
  $PY "tools/controls/$v.py" >/dev/null || { echo "FATAL: tools/controls/$v.py FAILED" >&2; exit 1; }
done
echo "[preflight] steer-wiring + mean-preserving-ablation verifiers pass"

sweep() {  # MODE CORPUS ALPHAS GPUS LOGNAME
  local mode=$1 corpus=$2 alphas=$3 gpus=$4 logname=$5
  # --mu-source is read ONLY by the mean-preserving modes (src/cli.py builds mu just for
  # them), so it is passed only where it means something.
  local extra=()
  [[ -n "$MU_SOURCE" && "$mode" == ablate_mp* ]] && extra=(--mu-source "$MU_SOURCE")
  echo "=== [$TAG] $(date -u +%H:%M:%S) mode=$mode corpus=$corpus alphas=[$alphas] gpus=$gpus ${extra[*]-}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpus $PY xpia_defense.py --model "$MODEL" --outdir "runs/$TAG" \
      --device auto --stage sweep --corpus "$corpus" --n-eval "$N_EVAL" \
      --directions "$DIRECTION" --alphas $alphas --scale sigma --mode "$mode" \
      ${extra[@]+"${extra[@]}"} \
      --steer-layers "$LAYERS" --steer-clean --batch "$BATCH" --max-new "$MAX_NEW" \
      2>&1 | tee "$LANELOG/$logname"
  local rc="${PIPESTATUS[0]}"
  sync_logs          # publish this lane's completed log immediately
  return "$rc"
}

rc=0
pids=(); labels=()
if [[ "$DOSE_ALPHAS" != "none" ]]; then
  sweep add "$DOSE_CORPUS" "$DOSE_ALPHAS" "$DOSE_GPUS" "rescue_dose_${DOSE_CORPUS}.log" &
  pids+=($!); labels+=(dose)
fi
# lane ABL. alpha 0 is passed only to populate the sweep's alpha grid: ablation is
# DOSE-FREE (src/steering.py projects out d irrespective of alpha).
#
# TWO no-op traps sit on this path, one of which already fired. (1) Steer.prefill_off skips
# the hook at alpha 0 -- it is gated on mode=="add", so ablate is safe. (2) run_arm's
# steer-construction guard decides whether to build a Steer AT ALL, and until 2026-09-01 it
# had no `mode` term, so an ablate arm at alpha 0 got steer=None and ran COMPLETELY
# UNDEFENDED while reporting as a defense (caught by metrics bit-identical to base-XPIA on
# both large MoE models). Guarded by tools/controls/verify_ablate_wiring.py, which must go
# through run_arm -- a test that constructs Steer directly passes and sees nothing.
if [[ "$ABL_CORPUS" != "none" ]]; then
  # WHERE THE SECOND CORPUS RUNS. Chaining it behind the first left the DOSE half of the
  # node idle for the whole run whenever `--dose-alphas none` was passed -- and one corpus
  # is 2h41m on 4xA100 at max_new 4096, so the two-corpus layout was throwing away a
  # lane-day (review, 2026-09-02). With the DOSE lane disabled its GPUs are free, so the
  # second corpus becomes its OWN parallel lane there; with a DOSE lane running they are
  # not free and it stays chained behind the first, exactly as before.
  # ...BUT ONLY IF THE TWO GPU SETS ARE ACTUALLY DISJOINT. "The DOSE lane is off so its GPUs
  # are free" is false when the caller points both flags at the same cards, which is the
  # normal case on a 4-GPU box (--abl-gpus 0,1,2,3 with DOSE_GPUS still at its 0,1,2,3
  # default). Launching both lanes there put TWO 80B model loads on the same four cards and
  # OOM'd at load: "Tried to allocate 39.88 GiB ... Process 102603 has 1.17 GiB in use"
  # (2026-09-02). Overlap => run them CHAINED, which is correct everywhere and merely slower.
  _gpu_overlap() {
    local a b
    for a in ${1//,/ }; do for b in ${2//,/ }; do [[ "$a" == "$b" ]] && return 0; done; done
    return 1
  }
  if _gpu_overlap "$ABL_GPUS" "$DOSE_GPUS" && [[ -n "$ABL_CORPUS_2" && "$DOSE_ALPHAS" == "none" ]]; then
    echo "[lanes] ABL_GPUS=$ABL_GPUS overlaps DOSE_GPUS=$DOSE_GPUS -- running the two ablate"
    echo "        corpora CHAINED on $ABL_GPUS instead of in parallel (avoids a load-time OOM)"
  fi
  if [[ -n "$ABL_CORPUS_2" && "$DOSE_ALPHAS" == "none" ]] && ! _gpu_overlap "$ABL_GPUS" "$DOSE_GPUS"; then
    sweep "$ABL_MODE" "$ABL_CORPUS" "0" "$ABL_GPUS" "rescue_${ABL_MODE}_${ABL_CORPUS}.log" &
    pids+=($!); labels+=("$ABL_MODE:$ABL_CORPUS")
    sweep "$ABL_MODE" "$ABL_CORPUS_2" "0" "$DOSE_GPUS" "rescue_${ABL_MODE}_${ABL_CORPUS_2}.log" &
    pids+=($!); labels+=("$ABL_MODE:$ABL_CORPUS_2")
  else
    (
      sweep "$ABL_MODE" "$ABL_CORPUS" "0" "$ABL_GPUS" "rescue_${ABL_MODE}_${ABL_CORPUS}.log" || exit 1
      if [[ -n "$ABL_CORPUS_2" ]]; then
        sweep "$ABL_MODE" "$ABL_CORPUS_2" "0" "$ABL_GPUS" "rescue_${ABL_MODE}_${ABL_CORPUS_2}.log" || exit 1
      fi
    ) &
    pids+=($!); labels+=("$ABL_MODE:$ABL_CORPUS${ABL_CORPUS_2:++$ABL_CORPUS_2}")
  fi
fi
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "*** ${labels[$i]} lane FAILED"; rc=1; }
done
harvest
echo "RESCUE-MOE-SMOKE-DONE tag=$TAG rc=$rc"
exit $rc
