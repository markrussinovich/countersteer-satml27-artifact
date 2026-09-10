#!/usr/bin/env bash
# SMOKE for the SoA-defense comparison job (dojo_soa_gptoss_job.sh): every battery
# configuration, ONE task-group shard each (--nshard 45 --shard 0, ~4-6 cells), all four
# arms, max_new 4096 — config-identical to the full job except the shard count. One battery
# per GPU, in parallel. Purpose (smoke-first ladder, CLAUDE.md): prove on the CLUSTER env,
# before the ~11 h full job, that (a) the three classifier checkpoints load there — incl.
# the gated PromptGuard from the ro model-store mount — (b) the reminder/stacked wirings
# run end-to-end through agentdojo_run.py, (c) the CachePrune mask stages and parses, and
# (d) engagement columns (pi_checked/pi_flagged, steered_tokens) are populated.
set -uo pipefail

LOCAL=runs/dojo_soa_smoke
LOGD=logs_soa_smoke
OUT=${OUT:-outputs}
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv/bin/python}
R=tools/controls/agentdojo_run.py

sync_blob() { cp -f "$LOCAL"/* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; }

# ARMS: comma-list of battery labels to smoke (default: all eight). Lets a fix to one
# defense family be re-smoked alone (e.g. ARMS=pi_protectai,pi_promptguard,pi_piguard
# after the 2026-09-04 chunked-detector correction) without repeating the rest.
ARMS=${ARMS:-countersteer,cacheprune,reminder,pi_protectai,pi_promptguard,pi_piguard,stacked,tool_filter}

want() { case ",$ARMS," in *",$1,"*) return 0;; *) return 1;; esac; }

run1() { # gpu label extra-args...
  local gpu="$1" label="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$R" \
    --model openai/gpt-oss-20b --max-new 4096 --no-adjudicate \
    --shard 0 --nshard 45 --out "$LOCAL/soa_smoke_${label}.json" "$@" \
    > "$LOGD/soa_smoke_${label}.log" 2>&1
}

pids=(); g=0
launch() { # label extra-args...
  local label="$1"; shift
  want "$label" || return 0
  run1 "$g" "$label" "$@" & pids+=($!)
  g=$((g + 1))
}
launch countersteer  --direction combo_ovr8_pat1 --alpha 8.06
launch cacheprune    --kv-mask runs/cacheprune_mask.json
launch reminder      --dojo-defense reminder
launch pi_protectai  --dojo-defense pi_detector
launch pi_promptguard --dojo-defense pi_detector_promptguard
launch pi_piguard    --dojo-defense pi_detector_piguard
launch stacked       --direction combo_ovr8_pat1 --alpha 8.06 \
                     --stack-dojo spotlighting_with_delimiting
launch tool_filter   --dojo-defense tool_filter

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
sync_blob
for f in "$LOCAL"/soa_smoke_*.json; do
  "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$f" >/dev/null 2>&1 \
    || { echo "SMOKE ARTIFACT BAD: $f" >&2; rc=1; }
done
echo "DOJO-SOA-SMOKE-DONE rc=$rc"
exit "$rc"
