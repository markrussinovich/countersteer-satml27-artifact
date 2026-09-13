#!/usr/bin/env bash
# Qwen dojo-baseline SMOKE (AML-cluster): engagement proof (1 attacked cell x 3 defenses,
# rendered-prompt evidence, GPU 0) + the real CLI path (4 attack-landing cells per defense,
# --defended-only, GPUs 1-3). Local writes, blob copy at exit (blobfuse ENOENT lesson from
# dojo-baselines-20260830).
set -uo pipefail
PY=${PY:-.venv/bin/python}
BLOB=${OUT:-outputs}
LOCAL=results_local
mkdir -p "$BLOB" "$LOCAL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=Qwen/Qwen3-30B-A3B-Thinking-2507

pids=()
CUDA_VISIBLE_DEVICES=0 "$PY" tools/controls/dojo_defense_smoke.py --model "$M" \
  --max-new 4096 --cell banking:user_task_0:injection_task_5:important_instructions \
  > "$LOCAL/qwen_engagement.log" 2>&1 & pids+=("$!")
g=1
for d in spotlighting_with_delimiting repeat_user_prompt tool_filter; do
  CUDA_VISIBLE_DEVICES=$g "$PY" tools/controls/agentdojo_run.py --model "$M" \
    --max-new 4096 --dojo-defense "$d" --defended-only --no-adjudicate \
    --cells runs/dojo_qwen_smoke4.manifest.json \
    --out "$LOCAL/qwen_dojo_smoke_${d}.json" \
    > "$LOCAL/log_qwen_smoke_${d}.log" 2>&1 & pids+=("$!")
  g=$((g + 1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
cp -f "$LOCAL"/* "$BLOB"/ 2>/dev/null || true
echo "JOB_DONE rc=$rc"
ls -la "$BLOB"
exit "$rc"
