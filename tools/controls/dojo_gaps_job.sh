#!/usr/bin/env bash
# Gap-filling resubmit for job dojo-baselines-20260830 (Failed: transient blobfuse ENOENT
# creating outputs/*.transcripts.json.tmp under 8 concurrent per-cell rewriters; compute was
# healthy -- one shard completed all 90 cells).
#
# FIXES APPLIED HERE:
#   - all artifacts + logs are written to LOCAL scratch (results_local/) and synced to the
#     blob mount every 3 min and at exit -- no run-critical open() ever touches blobfuse
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for the CUDA-OOM cells (7 recorded
#     OOMs on long workspace episodes; retried once here)
#
# Cell sets are the EXACT COMPLEMENT manifests runs/dojo_gaps_*.manifest.json (built from
# the partial artifacts; a cell is re-run iff it has no row or its arm errored).
# Gap outputs are named .shard2/.shard3 so the harvest glob dojo_*_[atk|benign].shard[0-9]
# picks up originals + gaps together.
set -uo pipefail
PY=${PY:-.venv/bin/python}
BLOB=${OUT:-outputs}
LOCAL=results_local
mkdir -p "$BLOB" "$LOCAL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sync_blob() { cp -f "$LOCAL"/* "$BLOB"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) &
SYNC_PID=$!

R=tools/controls/agentdojo_run.py
SPOT=spotlighting_with_delimiting
REP=repeat_user_prompt
TF=tool_filter
pids=()

atk() { # gpu defense shard nshard outshard
  CUDA_VISIBLE_DEVICES=$1 "$PY" "$R" --dojo-defense "$2" --defended-only --no-adjudicate \
    --cells "runs/dojo_gaps_$2_atk.manifest.json" --shard "$3" --nshard "$4" \
    --out "$LOCAL/dojo_$2_atk.shard$5.json" \
    > "$LOCAL/log_$2_atkgap$3.log" 2>&1
}
ben() { # gpu defense shard nshard outshard
  CUDA_VISIBLE_DEVICES=$1 "$PY" "$R" --dojo-defense "$2" --benign-only --no-adjudicate \
    --cells "runs/dojo_gaps_$2_benign.manifest.json" --shard "$3" --nshard "$4" \
    --out "$LOCAL/dojo_$2_benign.shard$5.json" \
    > "$LOCAL/log_$2_benigngap$3.log" 2>&1
}

atk 0 "$TF" 0 2 2 & pids+=("$!")
atk 1 "$TF" 1 2 3 & pids+=("$!")
atk 2 "$SPOT" 0 2 2 & pids+=("$!")
atk 3 "$SPOT" 1 2 3 & pids+=("$!")
atk 4 "$REP" 0 1 2 & pids+=("$!")
ben 5 "$SPOT" 0 2 2 & pids+=("$!")
ben 6 "$TF" 0 1 2 & pids+=("$!")
( ben 7 "$REP" 0 1 2; CUDA_VISIBLE_DEVICES=7 "$PY" "$R" --dojo-defense "$SPOT" \
    --benign-only --no-adjudicate --cells "runs/dojo_gaps_${SPOT}_benign.manifest.json" \
    --shard 1 --nshard 2 --out "$LOCAL/dojo_${SPOT}_benign.shard3.json" \
    > "$LOCAL/log_${SPOT}_benigngap1.log" 2>&1 ) & pids+=("$!")

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
kill "$SYNC_PID" 2>/dev/null
sync_blob
echo "JOB_DONE rc=$rc"
ls -la "$BLOB"
exit "$rc"
