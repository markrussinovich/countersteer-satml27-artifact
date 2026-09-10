#!/usr/bin/env bash
# Completion watch for a job running on ANOTHER fleet box (CLAUDE.md RULE ZERO: every
# launched job gets a watch armed in the SAME turn as the launch).
#
# `await` in tools/common.sh only inspects the LOCAL process table, so it cannot cover a job
# on .7/.9/.11; tools/await_aml_job.sh only covers Singularity jobs. This fills that gap.
#
# It reuses the SAME discriminator as tools/common.sh:alive -- /proc/PID/exe, never the
# command line. `pgrep -f PAT` matches any shell that merely QUOTES the pattern, and over ssh
# the remote shell running this very probe carries the pattern in its own argv, so a naive
# `pgrep -fc` counts the probe itself and the watch never returns. Filtering on the executable
# (python vs bash) is what makes the count honest.
#
# Prints exactly one terminal line and exits: 0 when no matching process remains, 1 if nothing
# matched at the start (waiting on a job that never launched is indistinguishable from waiting
# on a slow one, and only one of them ends), 2 on TIMEOUT. The timeout line carries
# TIMEOUT-MARKER so an expiring watch can never be misread as the job finishing.
#
# Usage: tools/await_remote.sh HOST PATTERN [TIMEOUT_SECONDS] [POLL_SECONDS] [EXE_SUBSTRING]
set -uo pipefail

HOST="${1:?usage: await_remote.sh HOST PATTERN [TIMEOUT] [POLL] [EXE]}"
PAT="${2:?usage: await_remote.sh HOST PATTERN [TIMEOUT] [POLL] [EXE]}"
TIMEOUT="${3:-43200}"
POLL="${4:-60}"
EXE="${5:-python}"

# Count remote processes matching PAT whose EXECUTABLE contains EXE. Empty output means the
# ssh probe itself failed, which is UNKNOWN -- not "process gone".
count_of() {
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
    "n=0; for p in \$(pgrep -f '$PAT' 2>/dev/null); do e=\$(readlink -f /proc/\$p/exe 2>/dev/null); case \"\$e\" in *$EXE*) n=\$((n+1));; esac; done; echo \$n" 2>/dev/null
}

n="$(count_of)"
if [[ -z "$n" ]]; then
  echo "[await-remote] $HOST '$PAT': ssh probe failed at arm time -- NOT waiting"; exit 1
fi
if (( n == 0 )); then
  echo "[await-remote] $HOST '$PAT': nothing matching is running -- job never started"; exit 1
fi
echo "[await-remote] $HOST '$PAT': watching $n process(es) (timeout ${TIMEOUT}s, poll ${POLL}s)"

waited=0
unknown=0
while (( waited < TIMEOUT )); do
  sleep "$POLL"
  waited=$(( waited + POLL ))
  n="$(count_of)"
  if [[ -z "$n" ]]; then
    # a network blip must not be read as completion -- the fleet watchdog learned this the
    # hard way (announced a healthy remote job as finished after a 15s blip)
    unknown=$(( unknown + 1 ))
    (( unknown >= 5 )) && { echo "[await-remote] $HOST '$PAT': unreachable for 5 consecutive polls"; exit 1; }
    continue
  fi
  unknown=0
  if (( n == 0 )); then
    echo "[await-remote] $HOST '$PAT': FINISHED after ${waited}s -- harvest and report now"; exit 0
  fi
done
echo "[await-remote] TIMEOUT-MARKER $HOST '$PAT': watch expired after ${waited}s with $n still running -- THE JOB DID NOT FINISH; re-arm"
exit 2
