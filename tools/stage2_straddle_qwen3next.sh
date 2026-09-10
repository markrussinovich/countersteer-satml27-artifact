#!/usr/bin/env bash
# Stage-2 dose STRADDLE for Qwen3-Next-80B-A3B-Thinking (FINDINGS 18f follow-up).
#
# Why: stage-1 bracket found no live cell — alpha 4 does not defend (goal 0.429 vs base
# 0.476) and alpha 16 zeroes goal only while failing the capability guard (xunatt 3.67x)
# AND under-measured (trunc 0.46 at max_new 4096, the harness's own TRUNCATED warning).
# The candidate window, if any, is alpha 4-16 at a bigger budget.
#
# What: alphas {4, 6, 8, 12} (4 = re-anchor against the stage-1 point, now at the bigger
# budget), max_new 8192, dim_no_override_both at the factorial-gate layers 28,32,40
# (identical to the stage-1 bracket's steer set), n=24 sweeps on shipped AND
# paper_disjoint, three-condition matrix + CLEAN+ per corpus (xpia_defense emits clean /
# base-XPIA / defended / CLEAN+ arms in one sweep).
#
# READING RULE (owner directive 2026-09-01): an arm's ASR is quotable only where its
# trunc < 0.1; the per-arm trunc is printed in the sweep log and stored in the artifact.
#
# Runs on one 8x H100 node: 2 model instances x 4 GPUs (GPN=4, --device auto), one
# corpus per instance in parallel.
set -uo pipefail
MODEL="Qwen/Qwen3-Next-80B-A3B-Thinking"
TAG="qwen3next-80b"
PY="${PY:-$(command -v python)}"
OUTD="outputs/$TAG"
mkdir -p "$OUTD"

harvest() { cp -f "runs/$TAG"/results_*.json "$OUTD/" 2>/dev/null || true; }
trap harvest EXIT

ls "runs/$TAG"/probe_L*.pkl >/dev/null 2>&1 || { echo "FATAL: no probe pickles in runs/$TAG"; exit 1; }
$PY - <<'PYCHK' || { echo "FATAL: staged pickles lack dim_no_override_both"; exit 1; }
import pickle, sys
sys.path.insert(0, "tools/controls")
import _probe_eval as E
import __main__ as m
if not hasattr(m, "TorchLogReg"): m.TorchLogReg = E.X.TorchLogReg
for L in (28, 32, 40):
    d = pickle.load(open(f"runs/qwen3next-80b/probe_L{L}.pkl", "rb"))
    assert "dim_no_override_both" in d.get("dirs", {}), f"L{L} missing direction"
    # a missing sigma silently becomes step 0.0 (probes.py: .get(name, 0.0)) -- a no-op
    # defense that mismeasures instead of crashing (pre-submission review, 2026-09-01)
    assert d.get("sigmas", {}).get("dim_no_override_both"), f"L{L} missing sigma"
print("[preflight] dim_no_override_both dir+sigma present at L28/32/40")
PYCHK

run_corpus() {  # CORPUS GPUS
  local corpus=$1 gpus=$2
  echo "=== [$TAG] $(date -u +%H:%M:%S) straddle sweep corpus=$corpus gpus=$gpus"
  CUDA_VISIBLE_DEVICES=$gpus $PY xpia_defense.py --model "$MODEL" --outdir "runs/$TAG" \
      --device auto --stage sweep --corpus "$corpus" --n-eval 24 \
      --directions dim_no_override_both --alphas 4 6 8 12 --scale sigma \
      --steer-layers 28,32,40 --steer-clean --batch 4 --max-new 8192 \
      2>&1 | tee "$OUTD/straddle_${corpus}.log"
  return "${PIPESTATUS[0]}"
}

run_corpus shipped 0,1,2,3 &
P_SHIP=$!
run_corpus paper_disjoint 4,5,6,7 &
P_DISJ=$!
rc=0
wait "$P_SHIP" || { echo "*** shipped sweep FAILED"; rc=1; }
wait "$P_DISJ" || { echo "*** paper_disjoint sweep FAILED"; rc=1; }
harvest
echo "STRADDLE-DONE rc=$rc"
exit $rc
