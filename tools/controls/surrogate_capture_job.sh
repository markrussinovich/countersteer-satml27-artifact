#!/usr/bin/env bash
# AML-cluster job driver: the ATTACKER'S SURROGATE re-derivation capture (FINDINGS 10v
# addendum, surrogate-defense white-box transfer).
#
# Re-runs the factorial capture behind the deployed `dim_no_override_both` with
# attacker-side variation, writing every artifact under the `surrogate` tag:
#   --design locked24   the EXACT deployed design (4 override x 3 voice x 2 actions,
#                       no delegation -- verified against runs/override_slope.json)
#   --n 36              the deployed fit used the first 24 paired probe samples
#                       (paired_samples is deterministic); capturing 36 gives BOTH
#                       surrogates in one run: the exact-recipe attacker (first-24 sids,
#                       differing only by generation nondeterminism) and a
#                       different-draw variant (all 36). Fit both, report both cos
#                       (adversarial review 2026-08-30, defect 2).
#   --split probe       same split: the corpus and the code are public
#   8 shards            results are shard-assignment-invariant (each framing is its own
#                       run_arm call), so use the whole node; the deployed capture's
#                       nshard=4 was an A100-count artifact, not provenance.
#
# Usage (from the snapshot root, venv installed): bash tools/controls/surrogate_capture_job.sh
set -uo pipefail
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs}
mkdir -p "$OUT"

# one first-touch per cache before the fan-out: concurrent first-touch HF downloads race
# in the xet cache (model), and load_dataset races the same way (adversarial review
# 2026-08-30, defect 7)
"$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-20b')" \
  || { echo "MODEL PREDOWNLOAD FAILED"; exit 1; }
"$PY" -c "import xpia_defense as X; print('dataset ok:', len(X.build_dataset()))" \
  || { echo "DATASET PREDOWNLOAD FAILED"; exit 1; }

pids=()
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s "$PY" tools/controls/override_slope_experiment.py \
    --shard "$s" --nshard 8 --n 36 --split probe \
    --design locked24 --tag surrogate \
    --probe-run runs/gpt-oss-20b-paper --device cuda:0 \
    > "$OUT/override_slope_surrogate.shard${s}.log" 2>&1 &
  pids+=("$!")
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
cp runs/override_slope_surrogate.shard*.json "$OUT/" 2>/dev/null || rc=1
echo "JOB_DONE rc=$rc"
ls -la "$OUT"
exit "$rc"
