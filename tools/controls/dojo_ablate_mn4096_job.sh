#!/usr/bin/env bash
# AML-cluster: gpt-oss AgentDojo battery with the PROJECTION operator (mode=ablate,
# dose-free; owner operator program 2026-09-11, FINDINGS §26.37: dev52 shows projection
# is a near-perfect tool-hijack defense at better utility than subtraction, but not a
# param defense — this battery answers the AgentDojo benign+security question at tab:soa
# conventions: 180 cells, four arms, mn4096, DEFAULT short system message, one process
# per shard). Direction combo_ovr8_pat1 L12/16/20, alpha 0 (ablate ignores dose).
set -uo pipefail
OUT=${OUT:-outputs}
LOCAL=runs/dojo_soa_gptoss
LOGD=logs_ablate_mn4096
mkdir -p "$OUT" "$LOCAL" "$LOGD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sync_blob() { cp -f "$LOCAL"/ablate_mn4096* "$OUT"/ 2>/dev/null || true; cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) & SYNC_PID=$!
rc=0
bash cluster/seed_model.sh --require openai/gpt-oss-20b || { echo "ABLATE-MN4096-DONE rc=3"; exit 3; }
bash tools/controls/dojo_baseline_mn4096.sh \
     --direction combo_ovr8_pat1 --alpha 0 --steer-mode ablate --layers 12,16,20 \
     --match-sigma-to dim_no_override --probe-dir runs/gpt-oss-20b-userabl \
     --label ablate_mn4096 --max-new 4096 --gpus 0,1,2,3,4,5,6,7 \
     --outdir "$LOCAL" --logdir "$LOGD" || rc=1
# §23e wiring assertion on the PRODUCTION artifacts: every shard's defended rows must
# carry steered tokens (an ablate arm identical-to-attacked is the silent no-op class).
python3 - <<'EOF' || rc=1
import json, glob, sys
bad=[]
for p in sorted(glob.glob("runs/dojo_soa_gptoss/ablate_mn4096_mn4096.shard*.json")):
    if ".transcripts." in p: continue
    d=json.load(open(p))
    for r in d.get("results", []):
        de=r.get("defended") or {}
        # exemptions (2026-09-12): error rows (all fields None) and episodes where NO
        # arm makes a tool call (nothing to steer) are not no-ops.
        if de.get("llm_calls") is None: continue
        if not de.get("n_calls_made") and not (r.get("clean") or {}).get("n_calls_made"):
            continue
        if not de.get("steered_tokens", 0):
            bad.append((p.split("/")[-1], r.get("suite"), r.get("user_task"), r.get("injection_task")))
print(f"[wiring] defended rows with steered_tokens==0: {len(bad)}")
for b in bad[:10]: print("  ", b)
sys.exit(1 if bad else 0)
EOF
sync_blob
kill "$SYNC_PID" 2>/dev/null || true
sync_blob
echo "ABLATE-MN4096-DONE rc=$rc"
exit "$rc"
