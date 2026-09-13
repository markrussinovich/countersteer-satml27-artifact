#!/usr/bin/env bash
# AML-cluster job: AgentDojo inbuilt-defense baselines on Qwen3-30B-A3B-Thinking-2507,
# same 180 cells as runs/qwen_agentdojo_run.shard*. Adds a SAME-PROCESS undefended
# attacked comparator (--direction '' = raw pipeline, identical code path/host/hardware),
# so the Qwen table does not lean on the .7 A100 comparator across hosts.
#
#   GPUs 0-5  attacked arms (--defended-only, 2 shards each): undefended, spotlighting,
#             repeat_user_prompt; then the six benign pairings (2 shards x 3 defenses)
#   GPUs 6-7  tool_filter attacked (2 shards)
#
# Local writes + periodic blob sync (blobfuse ENOENT lesson, dojo-baselines-20260830);
# expandable_segments for the long workspace episodes' CUDA OOMs.
set -uo pipefail
PY=${PY:-.venv/bin/python}
BLOB=${OUT:-outputs}
LOCAL=results_local
mkdir -p "$BLOB" "$LOCAL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=Qwen/Qwen3-30B-A3B-Thinking-2507
R=tools/controls/agentdojo_run.py
SPOT=spotlighting_with_delimiting
REP=repeat_user_prompt
TF=tool_filter

sync_blob() { cp -f "$LOCAL"/* "$BLOB"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) &
SYNC_PID=$!

atk() { # gpu defense-or-'' shard outname
  local dflag=()
  if [ -n "$2" ]; then dflag=(--dojo-defense "$2"); else dflag=(--direction ""); fi
  CUDA_VISIBLE_DEVICES=$1 "$PY" "$R" --model "$M" --max-new 4096 "${dflag[@]}" \
    --defended-only --no-adjudicate --shard "$3" --nshard 2 \
    --out "$LOCAL/qwen_dojo_$4_atk.shard$3.json" \
    > "$LOCAL/log_qwen_$4_atk$3.log" 2>&1
}
ben() { # gpu defense shard
  CUDA_VISIBLE_DEVICES=$1 "$PY" "$R" --model "$M" --max-new 4096 --dojo-defense "$2" \
    --benign-only --no-adjudicate --shard "$3" --nshard 2 \
    --out "$LOCAL/qwen_dojo_$2_benign.shard$3.json" \
    > "$LOCAL/log_qwen_$2_benign$3.log" 2>&1
}

pids=()
( atk 0 ""      0 undefended; ben 0 "$SPOT" 0 ) & pids+=("$!")
( atk 1 ""      1 undefended; ben 1 "$SPOT" 1 ) & pids+=("$!")
( atk 2 "$SPOT" 0 "$SPOT";    ben 2 "$REP"  0 ) & pids+=("$!")
( atk 3 "$SPOT" 1 "$SPOT";    ben 3 "$REP"  1 ) & pids+=("$!")
( atk 4 "$REP"  0 "$REP";     ben 4 "$TF"   0 ) & pids+=("$!")
( atk 5 "$REP"  1 "$REP";     ben 5 "$TF"   1 ) & pids+=("$!")
( atk 6 "$TF"   0 "$TF" ) & pids+=("$!")
( atk 7 "$TF"   1 "$TF" ) & pids+=("$!")

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
kill "$SYNC_PID" 2>/dev/null
sync_blob
echo "JOB_DONE rc=$rc"
ls -la "$BLOB"
exit "$rc"
