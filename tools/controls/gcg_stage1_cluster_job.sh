#!/usr/bin/env bash
# Singularity driver: §19c GCG stage-1 LAUNCH 4 (benchmark-expansion agent, 2026-09-08).
# Design per the §19c second amendment: plain objective config, each shard's model split
# over TWO H100s (--device auto), 4 shards x 12 samples on one 8xH100 node. Single host
# class, single config; all pre-launch-4 partials discarded (§19c). Fit check on the
# recorded OOM sample precedes this submission (.7, 2xA100 -- design validation only;
# its artifact is NOT merged).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT=${OUT:-outputs}
LOGD=logs_gcg
mkdir -p "$OUT" "$LOGD" runs/qwen_gcg_rerun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

sync_blob() { cp -f runs/qwen_gcg_rerun/* "$LOGD"/* "$OUT"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) & SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; sync_blob' EXIT

# Nemotron corpus warm-up with offline LIFTED for exactly this pinned pull (launch-4
# post-mortem 2026-09-08: adaptive_gcg builds its eval pool via src.corpora's
# load_dataset, which HF_HUB_OFFLINE=1 blocks -- all four shards died at data load; the
# same dependency class the rivals_qwen_cp review caught for the mask fitter).
HF_HUB_OFFLINE=0 "$PY" -c "from src.corpora import hf_dataset, NEMOTRON_REPO; hf_dataset(NEMOTRON_REPO, split='train'); print('nemotron cache warmed')" \
  > "$LOGD/nemotron_warmup.log" 2>&1 \
  || { echo "NEMOTRON_WARMUP_FAILED"; sync_blob; exit 1; }

t0=$SECONDS
bash singularity/seed_model.sh --require Qwen/Qwen3-30B-A3B-Thinking-2507 \
  || { echo "SEED_FAIL"; exit 1; }
snap="$HF_HOME/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
[ -d "$snap" ] || { echo "SEED_SHA_MISMATCH"; exit 1; }
echo "SEED_OK wall=$((SECONDS - t0))s"

pids=(); S=0
for pair in 0,1 2,3 4,5 6,7; do
  CUDA_VISIBLE_DEVICES="$pair" "$PY" -u tools/controls/adaptive_gcg.py \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 --probe-dir runs/qwen3-30b-thinking \
    --direction dim_no_override_both --alpha 12 --match-sigma-to '' \
    --steer-layers 8,20,32 --device auto \
    --objective auto --steps 500 --converge-window 100 --converge-tol 0.01 \
    --n 48 --dev-n 96 --seed 0 --pool-seed 0 \
    --k 20 --topk 256 --cand 128 --micro 4 --seq-cap 6144 --think-cap 2048 --max-new 4096 \
    --shard "$S" --nshard 4 \
    --out "runs/qwen_gcg_rerun/stage1.shard$S.json" \
    > "$LOGD/gcg_stage1.shard$S.log" 2>&1 &
  pids+=("$!")
  echo "[gcg-l4] shard $S on GPUs $pair pid $!"
  S=$((S + 1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
for s in 0 1 2 3; do
  "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" \
    "runs/qwen_gcg_rerun/stage1.shard$s.json" >/dev/null 2>&1 \
    || { echo "shard $s ARTIFACT BAD OR MISSING"; rc=1; }
done
sync_blob
echo "GCG-STAGE1-L4-DONE rc=$rc"
exit "$rc"
