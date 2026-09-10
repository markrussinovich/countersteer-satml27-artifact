#!/usr/bin/env bash
# Singularity job driver: AgentDojo's OWN inbuilt defenses (single-model, no detector)
# measured on gpt-oss-20b over the same 180 cells as our steering runs.
#
#   GPUs 0-5  attacked arm only (--defended-only): defense ON + injection ON, AgentDojo's
#             own `security`/`utility` checkers. 3 defenses x 2 shards.
#   GPUs 6-7  benign pairing (--benign-only): clean (raw undefended) vs CLEAN+ (defense on,
#             no injection) over unique user tasks -- the deployment-cost number.
#             3 defenses x 2 shards, run sequentially per GPU.
#
# The clean comparator and the attacked-undefended comparator are NOT re-run here: they are
# arm-for-arm the existing runs/agentdojo_run.shard*.json (raw model, same cells file).
#
# Usage (from the snapshot root, venv installed): bash tools/controls/dojo_baselines_job.sh
set -uo pipefail
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs}
mkdir -p "$OUT"
DEFS=(spotlighting_with_delimiting repeat_user_prompt tool_filter)

pids=()
g=0
for d in "${DEFS[@]}"; do
  for s in 0 1; do
    CUDA_VISIBLE_DEVICES=$g "$PY" tools/controls/agentdojo_run.py \
      --dojo-defense "$d" --defended-only --no-adjudicate \
      --shard "$s" --nshard 2 \
      --out "$OUT/dojo_${d}_atk.shard${s}.json" \
      > "$OUT/log_${d}_atk${s}.log" 2>&1 &
    pids+=("$!")
    g=$((g + 1))
  done
done

for s in 0 1; do
  gpu=$((6 + s))
  (
    for d in "${DEFS[@]}"; do
      CUDA_VISIBLE_DEVICES=$gpu "$PY" tools/controls/agentdojo_run.py \
        --dojo-defense "$d" --benign-only --no-adjudicate \
        --shard "$s" --nshard 2 \
        --out "$OUT/dojo_${d}_benign.shard${s}.json" \
        > "$OUT/log_${d}_benign${s}.log" 2>&1
    done
  ) &
  pids+=("$!")
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "JOB_DONE rc=$rc"
ls -la "$OUT"
exit "$rc"
