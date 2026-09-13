#!/usr/bin/env bash
# AML-cluster job driver: SURROGATE-DEFENSE WHITE-BOX TRANSFER GCG (FINDINGS 10v addendum).
#
# The attacker GCG-optimizes injection suffixes with THEIR OWN re-derived steering defense
# in the gradient loop (SURR_KEY from runs/gpt-oss-20b-surrogate, run at the deployed
# magnitude convention: alpha 8.06, --match-sigma-to dim_no_override); verdicts replay
# every optimized string against the DEPLOYED locked cell (combo_ovr8_pat1 @ alpha 1.0,
# own sigma, runs/gpt-oss-20b-userabl) plus undefended, static, and the surrogate cell
# itself (did the attack at least beat the model it was optimized against?).
#
# Smoke gate first (n=1, 2 steps): finite loss, BOTH cells resolved distinctly (the
# defended_surrogate verdict must exist -- adversarial review 2026-08-30, defect 3),
# cos(attack, verdict) persisted in the artifact, BEFORE the 8-shard fan-out.
#
# Artifacts are written to LOCAL scratch and synced to blob by a background loop --
# os.replace rewrites on the blobfuse mount have already killed a job with transient
# ENOENT (dojo-baselines-20260830; adversarial review 2026-08-30, defect 1).
#
# n=16 dev samples (seed 0: the seeded pool's first 16, a superset of the naive-transfer
# run's 8), 250 steps, k=20, 8 shards x 2 samples -- one shard per H100.
#
# Usage (from the snapshot root, venv installed): bash tools/controls/surrogate_gcg_job.sh
set -uo pipefail
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs}
SCRATCH=${SCRATCH:-runs/gcg_scratch}
SURR_KEY=${SURR_KEY:-combo_ovr8_pat1_surr}
mkdir -p "$OUT" "$SCRATCH"

ATK=(--attack-probe-dir runs/gpt-oss-20b-surrogate
     --attack-direction "$SURR_KEY"
     --attack-alpha 8.06
     --attack-match-sigma-to dim_no_override)

"$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-20b')" \
  || { echo "MODEL PREDOWNLOAD FAILED"; exit 1; }
"$PY" -c "import xpia_defense as X; print('dataset ok:', len(X.build_dataset()))" \
  || { echo "DATASET PREDOWNLOAD FAILED"; exit 1; }

# ── smoke gate ──────────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 "$PY" tools/controls/adaptive_gcg.py \
  --n 1 --steps 2 --cand 8 --micro 4 --k 20 "${ATK[@]}" \
  --out "$SCRATCH/adaptive_surrogate_smoke.json" \
  > "$OUT/adaptive_surrogate_smoke.log" 2>&1
rc=$?
"$PY" - "$SCRATCH/adaptive_surrogate_smoke.json" <<'PYEOF'
import json, math, sys
d = json.load(open(sys.argv[1]))
meta = d.get("meta") or {}
cos = meta.get("cos_attack_verdict")
assert cos and all(c < 0.9999 for c in cos), \
    f"attack cell resolved EQUAL to the verdict cell (cos={cos}) -- not a surrogate run"
assert meta["attack_cell"]["direction"] != meta["verdict_cell"]["direction"]
rows = [r for r in d["results"] if "final_ce" in r]
assert rows, f"smoke produced no optimized sample: {d['results']}"
assert all(math.isfinite(r["final_ce"]) for r in rows), f"non-finite CE: {rows}"
assert rows[0].get("defended_surrogate") is not None, \
    "defended_surrogate verdict missing -- the surrogate replay arm did not run"
print(f"[smoke-gate] OK: cos(attack,verdict)={[round(c,4) for c in cos]} "
      f"final_ce={rows[0]['final_ce']:.3f} static={rows[0]['defended_static']} "
      f"adaptive={rows[0]['defended_adaptive']} surr={rows[0]['defended_surrogate']} "
      f"replay_ids_match={rows[0].get('replay_ids_match')}")
PYEOF
gate=$?
cp "$SCRATCH/adaptive_surrogate_smoke.json" "$OUT/" 2>/dev/null
if [[ "$rc" -ne 0 || "$gate" -ne 0 ]]; then
  echo "SMOKE GATE FAILED (run rc=$rc, artifact check rc=$gate) -- full run NOT started"
  tail -40 "$OUT/adaptive_surrogate_smoke.log"
  exit 1
fi
grep -E '^\[cell\]|^\[smoke-gate\]' "$OUT/adaptive_surrogate_smoke.log" || true

# ── full run: 8 shards x 2 samples, artifacts on local scratch, synced to blob ─────────
pids=()
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s "$PY" tools/controls/adaptive_gcg.py \
    --n 16 --steps 250 --k 20 --shard "$s" --nshard 8 "${ATK[@]}" \
    --out "$SCRATCH/adaptive_surrogate.shard${s}.json" \
    > "$OUT/adaptive_surrogate.shard${s}.log" 2>&1 &
  pids+=("$!")
done

# periodic checkpoint sync (local scratch -> blob), so partial results survive a node loss
(
  while :; do
    sleep 300
    cp "$SCRATCH"/adaptive_surrogate.shard*.json "$OUT/" 2>/dev/null
  done
) &
sync_pid=$!

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
kill "$sync_pid" 2>/dev/null
cp "$SCRATCH"/adaptive_surrogate.shard*.json "$OUT/" 2>/dev/null || rc=1
echo "JOB_DONE rc=$rc"
ls -la "$OUT"
exit "$rc"
