#!/usr/bin/env bash
# Singularity driver: the §24f pre-registered SoA-defense benign composition-fidelity
# GENERATION — four §24d cells, defense-on-clean arms only (no injection anywhere), via
# tools/controls/soa_fidelity.py. Judging happens OFF-cluster afterwards
# (judge_utility.py needs the Azure judge endpoint; this job only generates completions).
#
# One cell per GPU, arms sequential within a cell. gpt-oss cells carry the cacheprune arm
# (mask staged as runs/cacheprune_mask.json); Qwen has no fitted mask (recorded gap).
#
# SUBMIT (queues behind the SoA batteries; do not displace them):
#   bash singularity/submit_job.sh --mode run --display-name xpia-soa-fidelity \
#     --timeout-seconds 28800 --no-clean --slmx-cmd 'bash tools/controls/soa_fidelity_job.sh'
set -uo pipefail

LOCAL=runs/soa_fidelity
LOGD=logs_fidelity
OUT=${OUT:-outputs}
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv/bin/python}
R=tools/controls/soa_fidelity.py

GPT_DEF="spotlight,sandwich,reminder,cacheprune,pi_protectai,pi_promptguard,pi_piguard"
QWEN_DEF="spotlight,sandwich,reminder,pi_protectai,pi_promptguard,pi_piguard"

sync_blob() { cp -f "$LOCAL"/* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) &
SYNC_PID=$!

run1() { # gpu cell model defenses extra...
  local gpu="$1" cell="$2" model="$3" defs="$4"; shift 4
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$R" \
    --source "runs/soa_fidelity_source_${cell}.json" --model "$model" \
    --defenses "$defs" --out "$LOCAL/soa_fidelity_${cell}.json" "$@" \
    > "$LOGD/soa_fidelity_${cell}.log" 2>&1
}

# SELF-SMOKE GATE (smoke-first ladder): 4 samples, two representative arms (one render
# transform + one classifier filter), on GPU 0, before any full cell runs. A runner bug
# costs ~3 minutes here instead of four wasted cell runs.
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$R" \
    --source runs/soa_fidelity_source_gptoss_webparam.json --model openai/gpt-oss-20b \
    --defenses "spotlight,pi_protectai" --n 4 \
    --out runs/soa_fidelity/soa_fidelity_smoke.json \
    > "$LOGD/soa_fidelity_smoke.log" 2>&1 \
  || { echo "SOA-FIDELITY-SMOKE-FAILED"; tail -5 "$LOGD/soa_fidelity_smoke.log"; sync_blob; exit 1; }
"$PY" - <<'PYEOF' || { echo "SOA-FIDELITY-SMOKE-BAD-ARTIFACT"; sync_blob; exit 1; }
import json
d = json.load(open("runs/soa_fidelity/soa_fidelity_smoke.json"))
arms = [k for k in d if k != "_meta"]
assert "CLEAN+spotlight" in arms and "CLEAN+pi_protectai" in arms, arms
assert all(len(d[a]) == 4 for a in arms), {a: len(d[a]) for a in arms}
assert any(d["CLEAN+spotlight"][i] for i in range(4)), "empty completions"
print("[fidelity-smoke] PASS", arms)
PYEOF
cp -f runs/soa_fidelity/soa_fidelity_smoke.json "$OUT"/ 2>/dev/null || true
# rename the source symlink trick is unnecessary: the smoke used the webparam source
mv runs/soa_fidelity/soa_fidelity_smoke.json runs/soa_fidelity/soa_fidelity_smoke.done.json 2>/dev/null || true

pids=()
run1 0 gptoss_webparam   openai/gpt-oss-20b "$GPT_DEF" --kv-mask runs/cacheprune_mask.json & pids+=($!)
run1 1 gptoss_jsonhijack openai/gpt-oss-20b "$GPT_DEF" --kv-mask runs/cacheprune_mask.json & pids+=($!)
run1 2 qwen_webparam     Qwen/Qwen3-30B-A3B-Thinking-2507 "$QWEN_DEF"                      & pids+=($!)
run1 3 qwen_jsonhijack   Qwen/Qwen3-30B-A3B-Thinking-2507 "$QWEN_DEF"                      & pids+=($!)

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
sync_blob
for f in "$LOCAL"/soa_fidelity_*.json; do
  "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$f" >/dev/null 2>&1 \
    || { echo "FIDELITY ARTIFACT BAD: $f" >&2; rc=1; }
done
kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "SOA-FIDELITY-DONE rc=$rc"
exit "$rc"
