#!/usr/bin/env bash
# FULL-battery AgentDojo run for ONE inbuilt-defense baseline at a raised generation budget.
#
# WHY THIS EXISTS (FINDINGS section 21c). The baselines table
# (tools/controls/score_dojo_baselines.py) is built from runs made at max_new 768, and 768 is
# demonstrably a CENSORING budget on this benchmark: AgentDojo's `security` checker is
# OUTCOME-based, so a turn cut off mid-compliance scores as SECURE. The 4096 re-anchoring of
# our own steering cell moved the undefended arm 0.360 -> 0.475, and spotlighting alone had
# 52/175 truncated episodes at 768. Every baseline row is therefore a censored LOWER BOUND and
# cannot be quoted on the same footing as the re-anchored numbers. This script reruns a
# baseline at max_new 4096 so it can be.
#
# WHY ALL FOUR ARMS, NOT `--defended-only`. The cheap `--defended-only` mode is what the
# original 768 baselines used, and it is UNGATEABLE: `attacker_calls` is only populated when
# the clean arm runs (agentdojo_run.py), and without that ground truth
# tools/controls/agentdojo_trunc_audit.py cannot tell whether a truncated turn was composing
# the attacker's call -- it exits 2 (BLIND / verdict UNKNOWN). A gate that reports "clean"
# because it cannot see is worse than no gate, so the budget rerun must run the clean arm.
# Running all four also yields the within-run clean-vs-CLEAN+ benign pairing and a within-run
# undefended comparator, which is exactly the footing the re-anchored table is quoted on.
# Cost is contained: clean/cleanplus are cached per (suite,user_task) task group, so ~39
# groups back ~180 cells.
#
# SHARDING ACROSS HOSTS. agentdojo_run.py shards by TASK GROUP index mod nshard off the cells
# file, which is identical on every box at the same commit, so shards are disjoint and
# reproducible across machines. All four arms of a given cell always run in the SAME shard and
# therefore on the SAME host, so the paired benign delta is never computed across hosts (clean
# utility LEVELS drift ~5pp between hosts -- see score_dojo_baselines.py).
#
# Usage (one host; repeat per host with a different --shard-base):
#   bash tools/controls/dojo_baseline_mn4096.sh \
#     --defense spotlighting_with_delimiting --gpus 1,2,3,5,6,7 --nshard 6 --shard-base 0
#   bash tools/controls/dojo_baseline_mn4096.sh \
#     --kv-mask runs/cacheprune_mask.json --label cacheprune --gpus 0,1 --nshard 4 --shard-base 0
#
# Produces, per shard: <outdir>/<label>_mn4096.shard<N>.json (+ .transcripts.json), and
# <logdir>/<label>_mn4096.shard<N>.log. Prints DOJO-BASELINE-MN4096-DONE on exit.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"

DEFENSE=""; KVMASK=""; LABEL=""; GPUS=""; NSHARD=""; SHARD_BASE=0
DIRECTION=""; ALPHA=""; LAYERS=""; MATCH_SIGMA="__unset__"; STACK=""; PROBEDIR=""
MAXNEW=4096; MODEL="openai/gpt-oss-20b"; SYSTEM=""; DEVICE=""; ALPHAS=""
CELLS="$ROOT/runs/agentdojo_cells.json"
OUTDIR="$ROOT/runs/dojo_baselines_mn4096"
LOGDIR="$ROOT/logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --defense)    DEFENSE="$2"; shift 2 ;;
    --kv-mask)    KVMASK="$2";  shift 2 ;;
    --direction)  DIRECTION="$2"; shift 2 ;;   # steering battery (e.g. the deployed cell)
    --alpha)      ALPHA="$2";   shift 2 ;;
    --layers)     LAYERS="$2";  shift 2 ;;
    --match-sigma-to) MATCH_SIGMA="$2"; shift 2 ;;
    --probe-dir)  PROBEDIR="$2"; shift 2 ;;
    --stack-dojo) STACK="$2";   shift 2 ;;     # steering + prompt-level defense, one arm
    --label)      LABEL="$2";   shift 2 ;;
    --gpus)       GPUS="$2";    shift 2 ;;
    --nshard)     NSHARD="$2";  shift 2 ;;
    --shard-base) SHARD_BASE="$2"; shift 2 ;;
    --max-new)    MAXNEW="$2";  shift 2 ;;
    --system)     SYSTEM="$2";  shift 2 ;;   # 'yaml' for AgentDyn (benchmark's own message)
    --alphas)     ALPHAS="$2";  shift 2 ;;   # dose-frontier sweep (agentdojo_run --alphas)
    --device)     DEVICE="$2";  shift 2 ;;   # e.g. 'auto' for multi-GPU-per-shard models;
                                             # then --gpus items may be dash-joined groups
                                             # (0-1-2-3,4-5-6-7 = 2 shards x 4 GPUs)
    --model)      MODEL="$2";   shift 2 ;;
    --cells)      CELLS="$2";   shift 2 ;;
    --outdir)     OUTDIR="$2";  shift 2 ;;
    --logdir)     LOGDIR="$2";  shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$DEFENSE" || -n "$KVMASK" || -n "$DIRECTION" ]] \
  || { echo "FATAL: need --defense, --kv-mask, or --direction" >&2; exit 2; }
_n_modes=0
[[ -n "$DEFENSE" ]] && _n_modes=$((_n_modes+1))
[[ -n "$KVMASK"  ]] && _n_modes=$((_n_modes+1))
# --direction alone = steering battery; --direction + --stack-dojo = stacked battery
[[ -n "$STACK" && -z "$DIRECTION" ]] && { echo "FATAL: --stack-dojo needs --direction" >&2; exit 2; }
[[ -n "$DIRECTION" ]] && _n_modes=$((_n_modes+1))
[[ "$_n_modes" -gt 1 ]] && { echo "FATAL: --defense / --kv-mask / --direction are mutually exclusive" >&2; exit 2; }
[[ -n "$GPUS" ]] || { echo "FATAL: need --gpus" >&2; exit 2; }
LABEL="${LABEL:-$DEFENSE}"
[[ -n "$LABEL" ]] || { echo "FATAL: need --label for a non---defense battery" >&2; exit 2; }

IFS=',' read -r -a GPUARR <<< "$GPUS"
NSHARD="${NSHARD:-${#GPUARR[@]}}"

# PRE-FLIGHT. A long multi-GPU run must not discover a missing input an hour in.
for f in "$PY" "$CELLS"; do
  [[ -e "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done
if [[ -n "$KVMASK" ]]; then
  [[ "$KVMASK" = /* ]] || KVMASK="$ROOT/$KVMASK"
  [[ -e "$KVMASK" ]] || { echo "FATAL: missing kv-mask $KVMASK" >&2; exit 1; }
  "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$KVMASK" \
    || { echo "FATAL: kv-mask does not parse: $KVMASK" >&2; exit 1; }
fi
mkdir -p "$OUTDIR" "$LOGDIR"

echo "[dojo-mn4096] label=$LABEL defense=${DEFENSE:-none} kvmask=${KVMASK:-none}"
echo "[dojo-mn4096] gpus=$GPUS nshard=$NSHARD shard-base=$SHARD_BASE max_new=$MAXNEW"
echo "[dojo-mn4096] outdir=$OUTDIR"

pids=(); shards=()
for i in "${!GPUARR[@]}"; do
  gpu="${GPUARR[$i]//-/,}"   # dash-joined group -> CVD list (0-1-2-3 -> 0,1,2,3)
  shard=$((SHARD_BASE + i))
  out="$OUTDIR/${LABEL}_mn4096.shard${shard}.json"
  log="$LOGDIR/${LABEL}_mn4096.shard${shard}.log"
  args=(--model "$MODEL" --max-new "$MAXNEW" --cells "$CELLS"
        --shard "$shard" --nshard "$NSHARD" --no-adjudicate --out "$out")
  [[ -n "$SYSTEM" ]] && args+=(--system "$SYSTEM")
  [[ -n "$DEVICE" ]] && args+=(--device "$DEVICE")
  [[ -n "$ALPHAS" ]] && args+=(--alphas "$ALPHAS")
  [[ -n "$DEFENSE" ]] && args+=(--dojo-defense "$DEFENSE")
  [[ -n "$KVMASK"  ]] && args+=(--kv-mask "$KVMASK")
  [[ -n "$DIRECTION" ]] && args+=(--direction "$DIRECTION")
  [[ -n "$ALPHA"   ]] && args+=(--alpha "$ALPHA")
  [[ -n "$LAYERS"  ]] && args+=(--layers "$LAYERS")
  [[ -n "$PROBEDIR" ]] && args+=(--probe-dir "$PROBEDIR")
  [[ "$MATCH_SIGMA" != "__unset__" ]] && args+=(--match-sigma-to "$MATCH_SIGMA")
  [[ -n "$STACK"   ]] && args+=(--stack-dojo "$STACK")
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/tools/controls/agentdojo_run.py" "${args[@]}" \
    > "$log" 2>&1 &
  pids+=("$!"); shards+=("$shard")
  echo "[dojo-mn4096] launched shard $shard on GPU $gpu -> $out (log $log)"
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

# AN ARTIFACT IS NOT WRITTEN UNTIL IT PARSES (CLAUDE.md). json.dump streams, so a late
# serialisation error leaves a truncated file with a fresh mtime and a plausible size.
for s in "${shards[@]}"; do
  f="$OUTDIR/${LABEL}_mn4096.shard${s}.json"
  if "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$f" >/dev/null 2>&1; then
    echo "[dojo-mn4096] shard $s artifact OK: $f"
  else
    echo "[dojo-mn4096] shard $s ARTIFACT BAD OR MISSING: $f" >&2; rc=1
  fi
done

echo "DOJO-BASELINE-MN4096-DONE label=$LABEL rc=$rc"
exit "$rc"
