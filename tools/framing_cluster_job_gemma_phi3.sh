#!/bin/bash
# BLACK-BOX ADAPTIVE FRAMING SEARCH (FINDINGS 10w design) vs the FROZEN Gemma-4-31B and
# Phi-3-medium cells. One 8x H100 node, both models in parallel. These are ATTACKS on
# frozen cells: no cell retuning, no test-split contact (dev split only, as in every
# recorded 10w run).
#
# THE FROZEN CELLS, PINNED EXPLICITLY (artifact configs win over any prose):
#   Gemma-4-31B  (certified 2026-09-04, FINDINGS 18c-heldout; artifact
#                 runs/gemma4-31b-it/results_confirm_add-dim-no-override-both-3532482.json):
#     direction dim_no_override_both @ alpha 8, layers 4,28,36, --scale sigma,
#     --match-sigma-to dim_no_override_both (expected sigmas L4 2.2489 / L28 8.5813 /
#     L36 4.0083 -- the 18c-heldout pre-registered values, recomputed by the gate).
#     NOTE alpha 8 is the frozen cell; the alpha-6 confirm artifacts are the
#     pre-registered dose-monotonicity CONTROLS, not the cell ("alpha 8 is the reported
#     cell regardless", FINDINGS 18c-heldout).
#   Phi-3-medium-128k (frozen single-turn cell at layers 8,12,16 -- NOT the 12,16,20
#                 runner-default seam, FINDINGS 26.4; artifact
#                 runs/phi3-medium-128k/results_confirm_add-dim-no-override-bal-1376418.json):
#     direction dim_no_override_bal @ alpha 6, layers 8,12,16, --scale sigma,
#     --match-sigma-to dim_no_override_both (expected sigmas L8 1.4167 / L12 3.1621 /
#     L16 5.4906).
#
# Query grid per model: the full 10w grid, 6 OVERRIDE x 4 VOICE x 2 ACTIONS = 48 framings;
# n 24 -> 18 paired dev samples; best-of-48 per sample. max-new: Gemma 4096 -- its
# single-turn confirms ran 2048 with trunc 0; the 4096 zero-truncation evidence is the
# AgentDojo multi-turn grid (a different code path), so 4096 here is a safety margin,
# attacker-favourable and constant across all 48 queries (review defect 3 provenance
# correction). Phi-3 2048 (no reasoning region; matches its frozen-cell runs; its gate
# checks non-empty completions only -- no truncation check exists for a format with no
# reasoning region and no reliable stop-token in decoded text; accepted by review).
#
# Structure: per-model SMOKE (2 queries x 4 samples; also warms the node-local HF cache),
# per-model GATE, then the full grid for each model that passed -- Gemma 4 shards x 1 GPU
# (GPUs 0-3; 31B fits one H100, the gemma-cert layout), Phi-3 4 shards x 1 GPU (GPUs 4-7).
# A gate failure on one model does NOT block the other; the job exits nonzero if either
# leg failed. Local writes + periodic blob sync (blobfuse ENOENT lesson).
set -uo pipefail
PY=${PY:-.venv/bin/python}
BLOB=${OUT:-outputs}
LOCAL=results_local
mkdir -p "$BLOB" "$LOCAL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

R=tools/controls/adaptive_framing_search.py
GEMMA=(--model google/gemma-4-31b-it --probe-dir runs/gemma4-31b-it
       --direction dim_no_override_both --alpha 8
       --match-sigma-to dim_no_override_both --steer-layers 4,28,36
       --split dev --batch 4 --max-new 4096 --device cuda:0)
PHI3=(--model microsoft/Phi-3-medium-128k-instruct --probe-dir runs/phi3-medium-128k
      --direction dim_no_override_bal --alpha 6
      --match-sigma-to dim_no_override_both --steer-layers 8,12,16
      --split dev --batch 8 --max-new 2048 --device cuda:0)

sync_blob() { cp -f "$LOCAL"/* "$BLOB"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) & SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null; sync_blob' EXIT

# ── preflight: staged pickles + dataset present ───────────────────────────────────────
for pd in runs/gemma4-31b-it runs/phi3-medium-128k; do
  n_pkl=$(ls "$pd"/probe_L*.pkl 2>/dev/null | wc -l)
  if [ "$n_pkl" -eq 0 ]; then echo "PREFLIGHT_FAILED: no probe pickles at $pd" >&2; exit 1; fi
  echo "[job] preflight: $pd has $n_pkl pickles" >&2
done
if [ ! -f runs/param_abuse_dataset.dev.json ]; then
  echo "PREFLIGHT_FAILED: runs/param_abuse_dataset.dev.json missing" >&2; exit 1; fi
# dataset pre-warm (review defect 5): both smokes call X.build_dataset() concurrently
# into the same fresh node-local HF_HOME; warm the datasets cache once, CPU-only, so the
# parallel first-touch race never happens.
"$PY" -c 'import xpia_defense as X; print(f"[warm] dataset ok: {len(X.build_dataset())} samples")' \
  > "$LOCAL/log_dataset_warm.log" 2>&1 \
  || { echo "PREFLIGHT_FAILED: dataset warm-build failed" >&2; tail -20 "$LOCAL/log_dataset_warm.log" >&2; exit 1; }
echo "[job] $(date -u +%H:%M:%S) preflight ok; smokes start" >&2

# ── SMOKES, parallel: 2 queries x 4 samples each (completion-cap 32000 so the gate can
#    see the whole completion; the default 4000 elides the middle) ─────────────────────
CUDA_VISIBLE_DEVICES=0 "$PY" "$R" "${GEMMA[@]}" \
  --n 4 --shard 0 --nshard 24 --completion-cap 32000 \
  --out "$LOCAL/framing_gemma4_smoke.json" > "$LOCAL/log_gemma_smoke.log" 2>&1 & SG=$!
CUDA_VISIBLE_DEVICES=4 "$PY" "$R" "${PHI3[@]}" \
  --n 4 --shard 0 --nshard 24 --completion-cap 32000 \
  --out "$LOCAL/framing_phi3_smoke.json" > "$LOCAL/log_phi3_smoke.log" 2>&1 & SP=$!
rcg=0; rcp=0
wait "$SG" || rcg=$?
wait "$SP" || rcp=$?
sync_blob
[ "$rcg" -ne 0 ] && { echo "GEMMA_SMOKE_FAILED rc=$rcg" >&2; tail -30 "$LOCAL/log_gemma_smoke.log" >&2; }
[ "$rcp" -ne 0 ] && { echo "PHI3_SMOKE_FAILED rc=$rcp" >&2; tail -30 "$LOCAL/log_phi3_smoke.log" >&2; }

# ── GATES (independent; a model runs its full grid only if its own gate passed) ───────
gate() { # model_key -> exit 0/1
  "$PY" - "$1" <<'PYG'
import json, pickle, sys
key = sys.argv[1]
spec = {
    "gemma": dict(art="results_local/framing_gemma4_smoke.json",
                  direction="dim_no_override_both", alpha=8.0, layers="4,28,36",
                  probe="runs/gemma4-31b-it", max_new=4096,
                  sig={4: 2.2489469051361084, 28: 8.581256866455078, 36: 4.008272647857666}),
    "phi3": dict(art="results_local/framing_phi3_smoke.json",
                 direction="dim_no_override_bal", alpha=6.0, layers="8,12,16",
                 probe="runs/phi3-medium-128k", max_new=2048,
                 sig={8: 1.4166558980941772, 12: 3.1620731353759766, 16: 5.490631580352783}),
}[key]
art = json.load(open(spec["art"]))
cfg, rows = art["config"], art["rows"]
assert not art.get("partial"), "smoke artifact is still a partial checkpoint"
assert cfg["direction"] == spec["direction"], cfg["direction"]
assert cfg["alpha"] == spec["alpha"], cfg["alpha"]
assert cfg["match_sigma_to"] == "dim_no_override_both", cfg["match_sigma_to"]
assert cfg["steer_layers"] == spec["layers"], cfg["steer_layers"]
assert cfg["max_new"] == spec["max_new"], cfg["max_new"]
assert len(rows) == 8, f"expected 2 queries x 4 samples = 8 rows, got {len(rows)}"
assert all(isinstance(r.get("completion"), str) and r["completion"].strip()
           for r in rows), "a smoke completion is empty"
if key == "gemma":
    # not-all-truncated: at least one completion has substance OUTSIDE the thought
    # channel (a turn truncated mid-thought strips to nothing)
    sys.path.insert(0, ".")
    from src import scoring as S
    n_sub = sum(bool(S.reasoning_free(r["completion"], "gemma4").strip()) for r in rows)
    assert n_sub > 0, "EVERY Gemma smoke completion truncated inside the thought channel"
# sigma provenance: recompute the anchor sigmas from the SHIPPED pickles
for L, e in spec["sig"].items():
    s = pickle.load(open(f"{spec['probe']}/probe_L{L}.pkl", "rb"))["sigmas"]["dim_no_override_both"]
    assert abs(s - e) < 1e-3, f"L{L} anchor sigma {s} != expected {e}"
print(f"[gate {key}] PASS", flush=True)
PYG
}
GEMMA_OK=0; PHI3_OK=0
if [ "$rcg" -eq 0 ] && gate gemma; then GEMMA_OK=1; else echo "GEMMA_GATE_FAILED -- Gemma full grid NOT run" >&2; fi
if [ "$rcp" -eq 0 ] && gate phi3;  then PHI3_OK=1;  else echo "PHI3_GATE_FAILED -- Phi-3 full grid NOT run" >&2; fi
sync_blob
if [ "$GEMMA_OK" -eq 0 ] && [ "$PHI3_OK" -eq 0 ]; then echo "BOTH_GATES_FAILED" >&2; exit 1; fi
echo "[job] $(date -u +%H:%M:%S) gates: gemma=$GEMMA_OK phi3=$PHI3_OK -- full grids" >&2

# ── FULL GRIDS: 48 queries x 18 samples per model, 4 shards x 1 GPU each ─────────────
lane() { # tag gpu shard args...
  local tag=$1 gpu=$2 shard=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$R" "$@" \
    --n 24 --shard "$shard" --nshard 4 \
    --out "$LOCAL/framing_${tag}.shard${shard}.json" \
    > "$LOCAL/log_${tag}_shard${shard}.log" 2>&1
}
pids=(); rc=0
if [ "$GEMMA_OK" -eq 1 ]; then
  for s in 0 1 2 3; do lane gemma4 "$s" "$s" "${GEMMA[@]}" & pids+=($!); done
fi
if [ "$PHI3_OK" -eq 1 ]; then
  for s in 0 1 2 3; do lane phi3 "$((s+4))" "$s" "${PHI3[@]}" & pids+=($!); done
fi
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$GEMMA_OK" -eq 0 ] || [ "$PHI3_OK" -eq 0 ] && rc=1   # a skipped leg is a failure to report
sync_blob
echo "JOB_DONE rc=$rc gemma_ok=$GEMMA_OK phi3_ok=$PHI3_OK"
ls -la "$BLOB"
exit "$rc"
