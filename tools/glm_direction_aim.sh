#!/usr/bin/env bash
# GLM-4.5-Air: does AIMING the direction better beat DOSING the current one harder?
# (FINDINGS §23m/§23n theory 1; coordinator-approved 2026-09-02.)
#
# WHY THESE ARMS. §23n measured that GLM's deployed `dim_no_override_both` has ~84% of its
# variance OUTSIDE the span of its own two per-action obedience axes (gpt-oss: 19-26%,
# Gemma: ~10-20%), and that the equal-weight orthogonal composition `dim_no_override_bal`
# has sigma/||h|| 0.046 against the pooled direction's 0.102 -- i.e. it lands GLM inside the
# 0.039-0.040 band where the recipe works on gpt-oss and Qwen3-30B. This asks whether that
# geometry converts into tier-1/tier-2 behaviour.
#
# LANE A -- A CLEAN ANALYTIC ABLATION OF THE ACTION TERM (re-staged 2026-09-02 after the
# action-centring analysis). The centring result gives an EXACT identity, verified to ~1e-15
# at every layer:
#       d_both  =  d_actioncentred  +  w^T A          (w = firing-rate-weighted ACTION contrast)
# so `_both` vs `_actioncentred` at matched alpha*sigma differs by EXACTLY the action main
# effect and nothing else. That is a far better experiment than the original
# `_both`/`_bal`/`_param` triple, because `_bal` gives no such identity -- and because on GLM
# `_bal` and `_actioncentred` turn out to be the SAME direction (cos 0.995/0.998/0.999 at
# L20/24/28, sigma equal to within 3%), so running both would have measured noise. `_bal` is
# kept as ONE consistency arm at a single alpha, not as an independent condition; `_param`
# is dropped (it is a half-data fit, as-built split-half reliability 0.35-0.61).
# `--match-sigma-to dim_no_override_both` keeps alpha*sigma identical across arms, so the
# only variable is WHERE the step points. alpha4 is GLM's known no-effect dose for the pooled
# direction (goal 0.625 vs base 0.708), so an arm that moves tier 1 THERE cannot be dose.
#
# THE HONEST PRIOR, recorded before the run. Both `_actioncentred` and `_bal` FAIL GLM's own
# estimability gate (split-half reliability 0.31/0.33 against the >0.70 gate; the action-
# centred direction's held-out `firm` AUC in its own space is 0.40, ~2 null-sd BELOW chance),
# and `_both`'s healthy-looking 0.956/0.801 is largely carried by the action label itself.
# This is run anyway for one specific reason: **Gemma-4-31B's pooled gate printed "NO USABLE
# DIRECTION" (held-out AUC 0.569-0.640 at all 15 layers) and its cell is the best in the
# project** (goal 0.038 at utilBenign 100%). Gate failure has already been shown NOT to
# predict behavioural failure on this exact recipe, so the gate is not grounds to skip the
# measurement -- but a null here is the EXPECTED outcome and must be reported as plainly as a
# win.
#
# LANE B -- THE COMPOSED CELL, never built for GLM before today. Combo keys bake their own
# sigma (sigma_dno * ||a1*u1 + a2*u2||) and MUST run at --alphas 1.0 with NO
# --match-sigma-to; at a1=8 that reproduces the deployed arm's magnitude, which is why lane
# B's alpha-1 combo and lane A's alpha-8 pooled arm are magnitude-comparable by
# construction. Cross-lane comparison is admissible only if the two lanes' `clean_sha`
# agree (FINDINGS §23l) -- same box, same corpus, same max_new, greedy, so they should;
# CHECK IT, do not assume it.
#
# EFFICIENCY, standing practice for these MoE sweeps (owner directive 2026-09-02). This
# script runs ONE replica because GLM-4.5-Air is ~212 GB in bf16 and the local box has
# 4x80 GB = 320 GB -- a second replica does not fit. On an 8-GPU node it DOES: two 4-GPU
# replicas with the arm list split roughly halve wall-clock at identical cost, and the
# one-process pairing requirement (every dose against ONE clean and ONE base-XPIA arm) is
# preserved by keeping each COMPARISON whole inside a replica rather than by keeping every
# arm in one process. The sub-integer sweep spent ~53 min/arm on 8xH100 at ~33% memory
# utilisation for want of this. When running on 8 GPUs, split by LANE
# (`CUDA_VISIBLE_DEVICES=0,1,2,3 ... A` and `4,5,6,7 ... B`) and check `clean_sha` agrees
# across the two lanes before tabling them together (FINDINGS §23l).
#
# GPU GATE. Waits for the local GPUs to drain before loading 212 GB of weights, so this
# never collides with another session's job. `A job is not running until something says it
# is` cuts both ways: do not start one on top of somebody else's.
#
# Usage: bash tools/glm_direction_aim.sh [LANES] [GATE_SECONDS]
#        LANES = A, B, or AB (default AB); GATE_SECONDS default 43200, 0 = do not wait
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
LANES="${1:-AB}"
GATE="${2:-43200}"
TAG=glm45-air
MODEL=zai-org/GLM-4.5-Air
LAYERS=20,24,28
COMMON=(--model "$MODEL" --outdir "runs/$TAG" --stage sweep --corpus paper_param
        --n-eval 24 --steer-layers "$LAYERS" --scale sigma --mode add
        --max-new 8192 --batch 4 --device auto --steer-clean)

# PRE-FLIGHT: every direction this script names must exist WITH a usable sigma, checked
# before the 106B model is loaded. src/probes.build_dirs(require_sigma=True) now raises on a
# zero sigma (tools/controls/verify_sigma_preflight.py), but that fires after model load;
# this fires in milliseconds. A missing key here means the §23n refit was not synced to this
# box.
TAG="$TAG" LANES="$LANES" $PY - <<'PYCHK' || { echo "FATAL: staged directions unusable"; exit 1; }
import os, sys
sys.path.insert(0, "tools/controls")
import _probe_eval as E
need = {"A": ["dim_no_override_both", "dim_no_override_actioncentred",
              "dim_no_override_bal"],
        "B": ["combo_ovr8_pat1", "combo_ovr8_pat1_bal", "dim_no_override_both"]}
want = sorted({k for ln in os.environ["LANES"] for k in need.get(ln, [])})
for L in (20, 24, 28):
    p = E.X.load_probe(f"runs/{os.environ['TAG']}/probe_L{L}.pkl")
    for k in want:
        assert k in p["dirs"], f"L{L}: direction `{k}` missing -- sync the §23n refit"
        s = p.get("sigmas", {}).get(k, 0.0)
        assert s and s > 0, f"L{L}: `{k}` has sigma {s} -- would be a SILENT NO-OP"
print(f"[preflight] {want} present with positive sigmas at L20/24/28")
PYCHK

# Free = every visible GPU under 5 GB. Poll, with a hard timeout and a MARKER so an expiring
# gate can never be misread as the job having run (CLAUDE.md, 2026-08-27).
NEED_FREE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
waited=0
while [ "$GATE" -gt 0 ]; do
  free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
         | awk '$1 < 5000 {n++} END {print n+0}')
  [ "$free" -ge "$NEED_FREE" ] && { echo "[gate] all $NEED_FREE GPUs free after ${waited}s"; break; }
  if [ "$waited" -ge "$GATE" ]; then
    echo "[gate] TIMEOUT-MARKER: only $free/$NEED_FREE GPUs free after ${waited}s; NOT starting"
    echo "GLM_DIRECTION_AIM_DONE rc=75 (gate timeout, nothing ran)"
    exit 75
  fi
  [ $((waited % 600)) -eq 0 ] && echo "[gate] $free/$NEED_FREE GPUs free, waited ${waited}s"
  sleep 60; waited=$((waited + 60))
done

rc=0
if [[ "$LANES" == *A* ]]; then
  echo "=== LANE A  $(date -u +%H:%M:%S)  direction at MATCHED magnitude ==="
  # the ablation proper: identical alpha*sigma, differing by exactly the action term
  $PY -u xpia_defense.py "${COMMON[@]}" \
      --directions dim_no_override_both,dim_no_override_actioncentred \
      --match-sigma-to dim_no_override_both --alphas 4 8 \
      || { rc=$?; echo "LANE A ablation arm FAILED rc=$rc -- aborting: the "\
           "consistency arm alone has no paired baseline and would be a "\
           "misleading artifact"; echo "GLM_DIRECTION_AIM_DONE rc=$rc"; exit $rc; }
  # one consistency arm: `_bal` is cos 0.995-0.999 to `_actioncentred` on GLM, so this should
  # reproduce the action-centred alpha-8 result. If it does not, one of the two fits is wrong.
  $PY -u xpia_defense.py "${COMMON[@]}" \
      --directions dim_no_override_bal \
      --match-sigma-to dim_no_override_both --alphas 8 || rc=$?
  echo "LANE_A_RC=$rc"
fi
if [[ "$LANES" == *B* ]]; then
  echo "=== LANE B  $(date -u +%H:%M:%S)  composed override+role cells (own baked sigma) ==="
  $PY -u xpia_defense.py "${COMMON[@]}" \
      --directions combo_ovr8_pat1,combo_ovr8_pat1_bal --alphas 1.0 || rc=$?
  echo "LANE_B_RC=$rc"
fi
echo "GLM_DIRECTION_AIM_DONE rc=$rc"
exit $rc
