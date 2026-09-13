#!/usr/bin/env bash
# H100 delta-variance job (pre-registered as FINDINGS §26.23): rerun ONLY the 9 tasks
# where the §25e H100 grid disagrees with the §26.22 A100 replicate majority, clean +
# CLEAN+ arms, published full-span recipe, N independent replicates (fresh process +
# model load each) on one H100 GPU per replicate.
#
# Purpose: adjudicate MACHINE vs DATE vs regeneration-chaos for the §25e-vs-§26.22
# benign-utility discrepancy. Config mirrors tmp/leafsel_v2batt/run_var_7.sh (the A100
# variance run) exactly; code/fork/model equivalence verified before submission
# (agentdojo_run.py + agentdojo_bridge.py md5-identical to the staged v2 tree the A100
# run executed; reference/agentdyn/src hash f124ce3a == the A100 run's agentdyn_src;
# model snapshot 6cee5e81 on both fabrics; torch 2.7.1+cu126 / transformers 5.14.1 /
# tokenizers 0.22.2 pinned identically).
#
# Usage (inside a AML-cluster run-mode job, from the snapshot root):
#   bash tools/controls/h100_delta_var_job.sh [N_REPS]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT=${OUT:-outputs}
NREPS="${1:-5}"
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/reference/agentdyn/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$ROOT/reference/agentdyn/src/agentdojo" ] \
  || { echo "FATAL: reference/agentdyn/src not staged"; echo "H100_DELTA_VAR_DONE rc=1"; exit 1; }

# Seed + pin-assert the exact snapshot both prior measurements used.
bash cluster/seed_model.sh --require openai/gpt-oss-20b \
  || { echo "SEED_FAIL openai/gpt-oss-20b"; echo "H100_DELTA_VAR_DONE rc=1"; exit 1; }
SNAPDIR="$HF_HOME/hub/models--openai--gpt-oss-20b/snapshots"
if [ ! -d "$SNAPDIR/6cee5e81ee83917806bbde320786a8fb61efebee" ]; then
  echo "SEED_SHA_MISMATCH: staged $(ls "$SNAPDIR" 2>/dev/null) != expected 6cee5e81..."
  echo "H100_DELTA_VAR_DONE rc=1"; exit 1
fi
echo "SEED_OK openai/gpt-oss-20b sha=6cee5e81ee83917806bbde320786a8fb61efebee"

# Environment fingerprint for the comparability record.
"$PY" - <<'PYEOF'
import torch, transformers, tokenizers, platform
print(f"[env] torch={torch.__version__} transformers={transformers.__version__} "
      f"tokenizers={tokenizers.__version__} python={platform.python_version()} "
      f"gpu={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
PYEOF

COMMON=(--cells runs/agentdyn_cells_h100delta.json --benign-only
        --system yaml --max-new 4096 --alpha 8.0
        --direction dim_no_override_both --layers 12,16,20
        --match-sigma-to dim_no_override --no-adjudicate --device cuda:0
        --probe-dir runs/gpt-oss-20b-userabl)

pids=(); rc=0
for r in $(seq 0 $((NREPS-1))); do
  CUDA_VISIBLE_DEVICES=$r "$PY" -u tools/controls/agentdojo_run.py \
      "${COMMON[@]}" --out "$OUT/h100delta.rep$r.json" \
      > "$OUT/h100delta.rep$r.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
for r in $(seq 0 $((NREPS-1))); do
  "$PY" -c "import json;json.load(open('$OUT/h100delta.rep$r.json'))" \
    || { echo "BADART $OUT/h100delta.rep$r.json does not parse"; rc=1; }
done
echo "H100_DELTA_VAR_DONE rc=$rc"
