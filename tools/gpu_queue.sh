#!/bin/bash
# Run an arbitrary command on the first GPU that frees up. Keeps all four cards busy without
# anyone sitting and watching for one to drain.
#
#   tools/gpu_queue.sh LOGFILE [--min-free-mb N] [--timeout-min N] -- CMD...
#
# CMD is run with CUDA_VISIBLE_DEVICES set to the free card and HF_HOME exported, from the
# repo root. `$PY` is available to CMD as the venv interpreter.
#
# WHY THIS EXISTS. tools/run_when_gpu_free.sh and tools/queue_next.sh do the same waiting, but
# each hardcodes ONE run directory, ONE model and ONE fixed argument list -- so a new
# experiment meant a new copy of the same loop. That is the pattern CLAUDE.md forbids
# ("scripts must be model-agnostic ... no absolute paths hardcoded"). This takes the command
# as arguments and derives every path from the repo root.
#
# Example:
#   tools/gpu_queue.sh logs/alphasteer_build.log -- \
#     "$PY" tools/controls/build_alphasteer.py --layers 12,16,20
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LOG="${1:?usage: gpu_queue.sh LOGFILE [opts] -- CMD...}"; shift
MIN_FREE_MB=5000
TIMEOUT_MIN=360
while [ $# -gt 0 ]; do
  case "$1" in
    --min-free-mb) MIN_FREE_MB="$2"; shift 2 ;;
    --timeout-min) TIMEOUT_MIN="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "FATAL: no command after --" >&2; exit 2; }

cd "$ROOT" || exit 1
LOCKDIR="$ROOT/tmp/gpu_locks"
mkdir -p "$LOCKDIR"
TRIES=$((TIMEOUT_MIN * 2))
for _ in $(seq 1 "$TRIES"); do
  # CLAIM THE CARD UNDER A LOCK. Two queue runners polling on the same 30s tick will both see
  # the same card free and both launch onto it -- two 20B bf16 models do not fit in 80GB, so
  # the second OOMs and takes an hour of the first one's work with it. flock makes the claim
  # atomic; the lock is held for the lifetime of the job, so a card is never double-booked.
  CLAIMED=""
  for G in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
             | awk -F', ' -v m="$MIN_FREE_MB" '$2 < m {print $1}'); do
    exec {LK}>"$LOCKDIR/gpu$G.lock"
    if flock -n "$LK"; then
      CLAIMED="$G"
      break
    fi
    exec {LK}>&-
  done
  if [ -n "$CLAIMED" ]; then
    # LOUD SEPARATOR. This log is APPENDED to, so a previous (possibly failed) run's output
    # sits above this line. Reading the top of the file and thinking it is the current run has
    # already caused one round of confusion.
    { echo; echo "================================================================"; } >> "$LOG"
    echo "[gpu_queue] claimed GPU$CLAIMED at $(date -Is); running: $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$CLAIMED" "$@" >> "$LOG" 2>&1
    rc=$?
    echo "[gpu_queue] exited rc=$rc at $(date -Is)" | tee -a "$LOG"
    exit $rc
  fi
  sleep 30
done
echo "[gpu_queue] FATAL: no GPU dropped below ${MIN_FREE_MB}MB in ${TIMEOUT_MIN}min" | tee -a "$LOG"
exit 1
