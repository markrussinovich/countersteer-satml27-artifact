#!/usr/bin/env bash
# THE ACTION-REMOVAL EVAL: does a model's defense survive removing the action-identity
# component from its steering direction?
#
# WHY (FINDINGS §23p.12). Three of six models carry a deployed `dim_no_override_both` that is
# heavily aligned with the tool-vs-param ACTION-IDENTITY axis -- glm45-air 0.950,
# qwen3next-80b 0.874, phi3-medium-128k 0.771 -- i.e. most of the direction's energy encodes
# "which attack class is this row", not "treat this as an instruction". The paper's mechanistic
# claim is that the recipe finds an instruction-treatment direction, so this is a claim-level
# risk on any model in that list, NOT a risk to the measured attack-success drop, which stands
# either way.
#
# THE INSTRUMENT. `dim_no_override_ac` (built by `build_override_direction.py --centre-action
# --key-suffix _ac`) differs from `dim_no_override_both` by EXACTLY the action term:
#       d_both = d_ac + w^T A,   w_i = (1/n1 + 1/n0)(ybar[sid_i,action_i] - ybar[sid_i])
# an identity verified to ~1e-15. Run both at MATCHED alpha*sigma and the only variable is the
# presence of that term.
#
# READING, fixed before any run:
#   defense SURVIVES on `_ac`  -> the action component was a PASSENGER; the mechanistic claim
#                                 holds on this model and no re-capture is needed
#   defense COLLAPSES on `_ac` -> the direction largely WAS the action axis on this model; the
#                                 paper must say that instead, and a re-capture is required
# This costs one eval sweep per model instead of one full factorial capture per model.
#
# Usage:
#   bash tools/action_removal_eval.sh MODEL_ID TAG LAYERS CORPUS N_EVAL [ALPHAS] [MAXNEW] [GATE_S]
# e.g.
#   bash tools/action_removal_eval.sh microsoft/Phi-3-medium-128k-instruct phi3-medium-128k \
#        8,12,16 shipped 96 "4 6" 2048 43200
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
# HF_HOME, EXPLICITLY. `tools/common.sh` exports it, but an `ssh HOST 'bash tools/foo.sh'`
# invocation never sources common.sh -- and transformers then downloads a SECOND copy of every
# model into ~/.cache/huggingface on the ROOT filesystem. That is not hypothetical: on
# 2026-09-02 this script pulled fresh 27 GB (Phi-3) and 49 GB (Qwen3-Next, partial) copies onto
# <FLEET_HOST_B>'s root fs even though both models were already present under /datadrive, and helped
# fill it to 100%, killing the run. Any script launchable over ssh must set this itself.
export HF_HOME="${HF_HOME:-/datadrive/huggingface/}"

MODEL="${1:?usage: action_removal_eval.sh MODEL TAG LAYERS CORPUS N_EVAL [ALPHAS] [MAXNEW] [GATE_S]}"
TAG="${2:?}"; LAYERS="${3:?}"; CORPUS="${4:?}"; NEVAL="${5:?}"
ALPHAS="${6:-4 6}"; MAXNEW="${7:-2048}"; GATE="${8:-43200}"

# PRE-FLIGHT. Both directions must exist WITH positive sigmas before the model is loaded.
# `--match-sigma-to dim_no_override_both` is pinned EXPLICITLY below and checked here: the
# §18c-battery incident under-dosed Phi-3 by 2.05x because `sigma_ref` silently fell back to
# `primary` (the first --directions entry), which is positional.
TAG="$TAG" LAYERS="$LAYERS" $PY - <<'PYCHK' || { echo "FATAL: directions unusable"; exit 1; }
import os, sys
sys.path.insert(0, "tools/controls")
import _probe_eval as E
tag = os.environ["TAG"]
for L in [int(x) for x in os.environ["LAYERS"].split(",")]:
    p = E.X.load_probe(f"runs/{tag}/probe_L{L}.pkl")
    for k in ("dim_no_override_both", "dim_no_override_ac"):
        assert k in p["dirs"], f"L{L}: `{k}` missing -- run build_override_direction --centre-action"
        s = p.get("sigmas", {}).get(k, 0.0)
        assert s and s > 0, f"L{L}: `{k}` sigma={s} -- would be a SILENT NO-OP under --scale sigma"
print("[preflight] both directions present with positive sigmas at " + os.environ["LAYERS"])
PYCHK

# GPU GATE. Wait rather than collide with another session; TIMEOUT-MARKER so an expiring gate
# is never misread as a completed run.
# HONOUR CUDA_VISIBLE_DEVICES. `nvidia-smi` reports every GPU on the box regardless of it, so
# a gate counting all of them would wait forever whenever this run is deliberately confined to
# a subset -- which is the normal case when another session owns the rest of the node.
VIS="${CUDA_VISIBLE_DEVICES:-}"
gpu_used() {
  if [ -n "$VIS" ]; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$VIS"
  else
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  fi
}
NEED=$(gpu_used | wc -l)
echo "[gate] watching ${NEED} GPU(s)${VIS:+ (CUDA_VISIBLE_DEVICES=$VIS)}"
waited=0
while [ "$GATE" -gt 0 ]; do
  free=$(gpu_used | awk '$1<5000{n++} END{print n+0}')
  [ "$free" -ge "$NEED" ] && { echo "[gate] all $NEED GPUs free after ${waited}s"; break; }
  if [ "$waited" -ge "$GATE" ]; then
    echo "[gate] TIMEOUT-MARKER: only $free/$NEED free after ${waited}s; NOT starting"
    echo "ACTION_REMOVAL_DONE rc=75 (gate timeout, nothing ran)"; exit 75
  fi
  [ $((waited % 600)) -eq 0 ] && echo "[gate] $free/$NEED GPUs free, waited ${waited}s"
  sleep 60; waited=$((waited+60))
done

echo "=== action-removal eval  $(date -u +%H:%M:%S)  $TAG / $CORPUS n=$NEVAL alphas=$ALPHAS ==="
$PY -u xpia_defense.py --model "$MODEL" --outdir "runs/$TAG" --stage sweep \
    --corpus "$CORPUS" --n-eval "$NEVAL" --steer-layers "$LAYERS" \
    --scale sigma --mode add --steer-clean --max-new "$MAXNEW" --device auto \
    --directions dim_no_override_both,dim_no_override_ac \
    --match-sigma-to dim_no_override_both --alphas $ALPHAS
rc=$?
echo "ACTION_REMOVAL_DONE rc=$rc tag=$TAG corpus=$CORPUS"
exit $rc
