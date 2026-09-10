#!/usr/bin/env bash
# Wait for an Azure ML job, harvest its completions artifacts, then score them AND run the
# reasoning-relay decomposition -- all in one detached process.
#
# WHY THIS EXISTS. `todo/02-efficiency-backlog.md` records that tracked background watches in
# a Claude session have been killed 5+ times mid-flight, twice leaving a job with ZERO
# coverage; it happened again to this job's watch on 2026-09-02. The documented mitigation is
# a `setsid`-detached watch (survives) PLUS a tracked one (delivers the notification). This
# goes one better: the DETACHED copy does the actual work, so if the tracked watch dies the
# analysis still lands on disk and can be picked up from the marker file. A watch that only
# watches is worth nothing once it is killed.
#
# RULE ZERO is why the scoring is in here at all: a harvested-but-unscored artifact is a
# result that does not exist.
#
# Usage: tools/harvest_and_relay.sh JOB_NAME TAG OUTDIR [TIMEOUT_S]
#   JOB_NAME  Azure ML job name (e.g. xpia-glm-subinteger-v2)
#   TAG       run tag / blob subdirectory (e.g. glm45-air)
#   OUTDIR    where to put the harvest + analysis (e.g. tmp/glm_floor_analysis/subint)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
# HF_HOME, EXPLICITLY. `tools/common.sh` exports it, but an `ssh HOST 'bash tools/foo.sh'`
# invocation never sources common.sh -- and transformers then downloads a SECOND copy of every
# model into ~/.cache/huggingface on the ROOT filesystem. That is not hypothetical: on
# 2026-09-02 this script pulled fresh 27 GB (Phi-3) and 49 GB (Qwen3-Next, partial) copies onto
# <FLEET_HOST_B>'s root fs even though both models were already present under /datadrive, and helped
# fill it to 100%, killing the run. Any script launchable over ssh must set this itself.
export HF_HOME="${HF_HOME:-/datadrive/huggingface/}"

JOB="${1:?usage: harvest_and_relay.sh JOB_NAME TAG OUTDIR [TIMEOUT_S]}"
TAG="${2:?}"
OUTDIR="${3:?}"
TIMEOUT="${4:-43200}"
mkdir -p "$OUTDIR"
MARK="$OUTDIR/HARVEST_RELAY_STATUS"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUTDIR/harvest_and_relay.log"; }
: > "$OUTDIR/harvest_and_relay.log"
echo "RUNNING" > "$MARK"

ACCOUNT="${AML_BLOB_ACCOUNT:-<BLOB_ACCOUNT>}"
CONTAINER="${AML_BLOB_CONTAINER:-<BLOB_CONTAINER>}"
PREFIX="xpia_steering_experiments/$JOB/$TAG/"

say "waiting on $JOB (timeout ${TIMEOUT}s)"
if ! bash tools/await_aml_job.sh "$JOB" "$TIMEOUT" 120 >> "$OUTDIR/harvest_and_relay.log" 2>&1; then
  # await prints its own TIMEOUT-MARKER; distinguish "watch expired" from "job failed"
  say "await returned non-zero -- job Failed/Canceled, or the WATCH timed out. NOT harvesting."
  echo "AWAIT_NONZERO" > "$MARK"
  exit 1
fi
say "job reached a terminal state; harvesting from $ACCOUNT/$CONTAINER/$PREFIX"

mapfile -t BLOBS < <(az storage blob list --auth-mode login --account-name "$ACCOUNT" \
  -c "$CONTAINER" --prefix "$PREFIX" --query "[?contains(name,'_completions.json')].name" -o tsv 2>/dev/null)
if [ "${#BLOBS[@]}" -eq 0 ]; then
  say "NO completions blobs under $PREFIX -- the job wrote none (check its log)."
  echo "NO_ARTIFACT" > "$MARK"
  exit 2
fi

got=()
for b in "${BLOBS[@]}"; do
  f="$OUTDIR/$(basename "$b")"
  az storage blob download --auth-mode login --account-name "$ACCOUNT" -c "$CONTAINER" \
     -n "$b" -f "$f" --overwrite --no-progress >/dev/null 2>&1
  # AN ARTIFACT IS NOT WRITTEN UNTIL IT PARSES (CLAUDE.md; json.dump streams, so a
  # serialisation error leaves a truncated file with a plausible size and a fresh mtime).
  if $PY -c "import json,sys; json.load(open(sys.argv[1]))" "$f" >/dev/null 2>&1; then
    got+=("$f"); say "harvested + parses: $f"
  else
    say "TRUNCATED OR UNPARSEABLE, skipping: $f"
  fi
done
[ "${#got[@]}" -eq 0 ] && { say "nothing parseable"; echo "ALL_UNPARSEABLE" > "$MARK"; exit 3; }

say "=== canonical scoring (tools/controls/score_table.py) ==="
# --no-adjudicate: the drift adjudicator appends to a SHARED cache (runs/drift_adjudication.json)
# and this may run concurrently with another session's scoring.
$PY tools/controls/score_table.py "${got[@]}" --no-adjudicate >> "$OUTDIR/harvest_and_relay.log" 2>&1

say "=== reasoning-relay decomposition (the free measurement) ==="
$PY tmp/glm_floor_analysis/relay.py "${got[@]}" >> "$OUTDIR/harvest_and_relay.log" 2>&1

say "DONE -- results in $OUTDIR/harvest_and_relay.log"
echo "DONE" > "$MARK"
