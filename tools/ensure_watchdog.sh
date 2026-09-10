#!/usr/bin/env bash
# Make sure a fleet watchdog is running. Idempotent, detached, safe to call liberally.
#
# WHY THIS EXISTS. Layer-2 coverage (CLAUDE.md RULE ZERO) keeps being lost two ways:
#
#   1. `tools/fleet_watchdog.sh` EXITS on the first job that completes cleanly ("exiting to
#      notify on completion of ..."), dropping coverage for every OTHER row in the manifest.
#      Observed 2026-09-02: it exited at 00:50 and the fleet ran ~14 h with four active rows
#      and no watchdog.
#   2. A watchdog started as a TRACKED background command in a Claude session gets KILLED.
#      That happened twice in one session on 2026-09-02 (an AML completion watch, then the
#      watchdog itself, ~5 min after being restarted).
#
# So: never start the watchdog as a tracked background command. Start it through this, which
# `setsid`-detaches it from the session's process group. The same lesson as
# `tools/harvest_and_relay.sh`: anything that MUST survive goes in a detached process, and the
# tracked process is only allowed to carry a notification.
#
# This does NOT fix defect (1) -- that needs a change inside fleet_watchdog.sh to report a
# completion and keep watching the rest (filed in todo/02-efficiency-backlog.md). Until then,
# call this after every job completion, or from a cron/loop.
#
# Usage: bash tools/ensure_watchdog.sh [LOG]      (default log: logs/fleet_watchdog_main.log)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
LOG="${1:-logs/fleet_watchdog_main.log}"

# Self-match-proof: `pgrep -f` matches any shell whose argv merely QUOTES the pattern,
# including this script's own caller. The `[f]` trick keeps this invocation from counting
# itself (CLAUDE.md: "a job is not running until something says it is" -- and the converse,
# a false positive here silently leaves the fleet uncovered).
n=$(pgrep -f '[f]leet_watchdog\.sh' 2>/dev/null | wc -l)
if [ "$n" -gt 0 ]; then
  echo "[ensure-watchdog] already running ($n proc); nothing to do"
  tail -1 "$LOG" 2>/dev/null
  exit 0
fi

if [ ! -s tmp/fleet_jobs.tsv ]; then
  echo "[ensure-watchdog] REFUSING: tmp/fleet_jobs.tsv missing or empty -- a watchdog with an" >&2
  echo "                  empty manifest reports 'all done' and exits, which looks like health." >&2
  exit 1
fi
rows=$(grep -cvE '^\s*(#|$)' tmp/fleet_jobs.tsv)
echo "[ensure-watchdog] starting detached (KEEP_WATCHING=${KEEP_WATCHING:-1}); manifest has $rows job row(s)"
# KEEP_WATCHING defaults to 1 HERE (the watchdog itself still defaults to 0, preserving the
# owner rule for anyone invoking it directly). Rationale: this fleet routinely has jobs
# completing while others run, and exit-on-first-completion then drops coverage for every
# other row -- observed three times on 2026-09-02, twice on a FALSE completion detected in the
# window between a manifest row being added and its job starting. Completions still notify
# exactly once, via the durable ledger.
KEEP_WATCHING="${KEEP_WATCHING:-1}" setsid nohup bash tools/fleet_watchdog.sh >> "$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true

# POSITIVE liveness check before claiming success -- never report a job as running from the
# absence of an error.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  n=$(pgrep -f '[f]leet_watchdog\.sh' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && { echo "[ensure-watchdog] alive (pid $(pgrep -f '[f]leet_watchdog\.sh' | head -1))"; exit 0; }
done
echo "[ensure-watchdog] FAILED to start -- check $LOG" >&2
exit 1
