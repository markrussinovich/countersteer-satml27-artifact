#!/bin/bash
# Shared setup for every script in tools/. Source it, don't copy it:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Exports ROOT, PY, and HF_HOME. Never hardcode an absolute repo path in a script --
# (paths are derived from the repo root; hardcoded absolute paths break on relocation.)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
export HF_HOME="${HF_HOME:-/datadrive/huggingface/}"

[ -x "$PY" ] || { echo "FATAL: $PY missing or not executable (venv symlink broken?)" >&2; exit 1; }

# wait_for FILE [TRIES] [SLEEP] -- poll for a file, fail loudly rather than silently.
wait_for() {
  local f="$1" tries="${2:-240}" nap="${3:-20}"
  for _ in $(seq 1 "$tries"); do
    [ -f "$f" ] && return 0
    sleep "$nap"
  done
  echo "FATAL: $f never appeared after $((tries * nap))s" >&2
  return 1
}

# free_gpus -- print indices of GPUs with no compute processes, one per line.
free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 1000 {print $1}'
}

# alive PATTERN [EXE] -- count of live processes matching PATTERN whose executable is EXE
# (default python). Does NOT count the caller.
#
# `pgrep -f foo.py` matches the watcher's OWN command line, because that command line
# contains the pattern. So `until ! pgrep -f "foo.py"; do sleep 20; done` waits on itself
# and never returns -- and it looks exactly like a healthy long-running job. That burned 23
# minutes of idle GPU on 2026-08-03 while the run it was "waiting for" had never started.
#
# The usual `[f]oo` dodge does NOT fix this when the pattern is a variable: the caller's
# argv still contains the plain string, so it still self-matches. (Verified -- the first
# version of this function returned 1 for a job that did not exist.) Filtering by $$/$PPID
# is also not enough, because harness and wrapper shells that merely MENTION the pattern
# have their own PIDs.
#
# The reliable discriminator is the EXECUTABLE: a job is a python process, while every
# shell that merely quotes the pattern is bash. Check /proc/PID/exe, not the command line.
alive() {
  local p="$1" exe="${2:-python}" n=0 pid c
  for pid in $(pgrep -f "$p" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    c=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || continue
    case "$c" in *"$exe"*) n=$((n + 1)) ;; esac
  done
  echo "$n"
}

# require_alive PATTERN [WHAT] -- assert something is actually running, and say so if not.
# Use before any wait loop: waiting for a job that never started is indistinguishable from
# waiting for a slow one, and only one of them ever finishes.
require_alive() {
  local n; n=$(alive "$1")
  [ "$n" -gt 0 ] || { echo "FATAL: nothing matching '$1' is running (${2:-job never started})" >&2; return 1; }
  echo "[alive] $n process(es) matching '$1'"
}

# json_ok FILE -- the artifact must PARSE, not merely exist.
#
# json.dump() streams into the file handle, so a TypeError partway through (a set, a numpy
# scalar, a NaN) leaves a TRUNCATED file with a fresh mtime and a plausible size. On
# 2026-08-03 that silently destroyed a 28-minute 4-GPU generation run: every completion was
# computed, then lost at write time. Existence is not success.
json_ok() { "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" >/dev/null 2>&1; }

# await PATTERN [TIMEOUT_S] -- block until no process matches, failing loudly if none ever
# did. This is the ONLY sanctioned way to wait for a job in this repo.
await() {
  local p="$1" limit="${2:-14400}" waited=0
  require_alive "$p" || return 1
  while [ "$(alive "$p")" -gt 0 ]; do
    sleep 15; waited=$((waited + 15))
    if [ "$waited" -ge "$limit" ]; then
      echo "FATAL: '$p' still running after ${limit}s -- not waiting further" >&2
      return 1
    fi
  done
  echo "[await] '$p' finished after ${waited}s"
}

# kill_job PATTERN [EXE] -- kill jobs matching PATTERN whose EXECUTABLE is EXE (default python).
#
# `pkill -f PAT` and `ps|grep PAT|kill` BOTH match the CALLER'S OWN command line, because that
# command line contains the pattern. The bracket dodge (`[o]verride`) does not help: the shell's
# argv still holds the literal string. This has killed the calling shell three times in this
# project. Same discriminator as `alive`: /proc/PID/exe, never the command line.
kill_job() {
  local p="$1" exe="${2:-python}" pid c n=0
  for pid in $(pgrep -f "$p" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    c=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || continue
    case "$c" in *"$exe"*) kill "$pid" && n=$((n+1)) ;; esac
  done
  echo "[kill_job] signalled $n process(es) matching '$p'"
}
