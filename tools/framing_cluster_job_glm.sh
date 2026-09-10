#!/bin/bash
# BLACK-BOX ADAPTIVE FRAMING SEARCH (FINDINGS 10w design) vs the FROZEN GLM-4.5-Air cell.
# One 8x H100 node. This is an ATTACK on a frozen cell: no cell retuning, no test-split
# contact (dev split only, same as every recorded 10w run).
#
# THE FROZEN CELL, PINNED EXPLICITLY (certified 2026-09-04, FINDINGS section 23ag;
# artifact config runs/glm45-air/results_confirm_add-dim-no-override-actioncentred-*.json):
#   direction dim_no_override_actioncentred @ alpha 8, layers 20,24,28, --scale sigma
#   (script default), --match-sigma-to dim_no_override_both
#   (expected anchor sigmas L20 6.4161 / L24 8.8587 / L28 10.2760 -- recomputed from the
#   staged pickles by the gate below, not read from a log).
#
# The query grid is the full 10w grid: 6 OVERRIDE x 4 VOICE x 2 ACTIONS = 48 framings,
# applied per sample to its own injected action; best-of-48 per sample is the adaptive
# outcome. n 24 -> 18 paired dev samples (samples carrying BOTH action types), exactly the
# 10w denominator. max-new 8192 (GLM thinks; its frozen-cell paper runs used 8192).
#
# Structure (owner non-negotiables): SMOKE first (2 queries x 4 samples on GPUs 0-3, which
# also warms the node-local HF cache -- GLM is NOT in the models datastore, ~212 GB from
# the hub), then a GATE (artifact parses, config matches the frozen cell, completions
# non-empty, not everything truncated inside <think>, sigma provenance recomputed), and
# only then the full 48-query grid: 2 instances x 4 GPUs, sharded over queries (the seeded
# permutation inside adaptive_framing_search keeps shards action-balanced).
# Local writes + periodic blob sync (blobfuse ENOENT lesson, dojo-baselines-20260830).
set -uo pipefail
PY=${PY:-.venv/bin/python}
BLOB=${OUT:-outputs}
LOCAL=results_local
mkdir -p "$BLOB" "$LOCAL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="zai-org/GLM-4.5-Air"
PROBE=runs/glm45-air
R=tools/controls/adaptive_framing_search.py
CELL=(--model "$MODEL" --probe-dir "$PROBE"
      --direction dim_no_override_actioncentred --alpha 8
      --match-sigma-to dim_no_override_both --steer-layers 20,24,28
      --split dev --batch 4 --max-new 8192 --device auto)

sync_blob() { cp -f "$LOCAL"/* "$BLOB"/ 2>/dev/null || true; }
( while true; do sleep 180; sync_blob; done ) & SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null; sync_blob' EXIT

# ── preflight: the staged snapshot actually carries what the run needs ───────────────
n_pkl=$(ls "$PROBE"/probe_L*.pkl 2>/dev/null | wc -l)
if [ "$n_pkl" -eq 0 ]; then echo "PREFLIGHT_FAILED: no probe pickles at $PROBE" >&2; exit 1; fi
if [ ! -f runs/param_abuse_dataset.dev.json ]; then
  echo "PREFLIGHT_FAILED: runs/param_abuse_dataset.dev.json missing" >&2; exit 1; fi
# preflight direction check (2026-09-05: the first submission died here in-job -- the
# staging had overwritten the refit pickles with the stale probe/ harvest; cheap to
# assert before a 212 GB download):
"$PY" - <<'PYD' || { echo "PREFLIGHT_FAILED: frozen direction missing from staged pickles" >&2; exit 1; }
import pickle
for L in (20, 24, 28):
    d = pickle.load(open(f"runs/glm45-air/probe_L{L}.pkl", "rb"))
    assert "dim_no_override_actioncentred" in d["dirs"], f"L{L}: no actioncentred (stale pickle?)"
    assert "dim_no_override_both" in d["sigmas"], f"L{L}: no sigma anchor"
print("[preflight] staged pickles carry the frozen cell", flush=True)
PYD

# model download with RETRIES (2026-09-05: the first submission hit anonymous-pull HTTP
# 429 rate limits on the ~212 GB fetch; snapshot_download resumes partial downloads, so
# retrying converges. No HF token is embedded -- job YAMLs are tracked in git and the
# anon scrub does not cover HF tokens).
for i in 1 2 3 4 5 6; do
  "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-4.5-Air')" \
    > "$LOCAL/log_hf_download_$i.log" 2>&1 && { echo "[warm] model download complete (attempt $i)" >&2; break; }
  echo "[warm] download attempt $i failed (429/backoff likely); sleeping 300s" >&2
  sleep 300
  [ "$i" -eq 6 ] && { echo "PREFLIGHT_FAILED: model download failed 6 attempts" >&2; sync_blob; exit 1; }
done
sync_blob
echo "[job] $(date -u +%H:%M:%S) preflight ok ($n_pkl pickles); smoke start" >&2

# ── SMOKE: 2 queries x 4 samples on GPUs 0-3 (also downloads the model once) ─────────
# completion-cap 32000 so the gate can see </think> (the default 4000 elides the middle
# of a long think, which would false-fail the truncation check).
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" "$R" "${CELL[@]}" \
  --n 4 --shard 0 --nshard 24 --completion-cap 32000 \
  --out "$LOCAL/framing_glm45air_smoke.json" \
  > "$LOCAL/log_glm_smoke.log" 2>&1
rc=$?
sync_blob
if [ "$rc" -ne 0 ]; then echo "SMOKE_FAILED rc=$rc" >&2; tail -30 "$LOCAL/log_glm_smoke.log" >&2; exit 1; fi

# ── GATE: parse + frozen-cell config + non-empty, non-all-truncated + sigma recompute ─
"$PY" - <<'PYG' || { echo "GATE_FAILED -- full grid NOT run" >&2; sync_blob; exit 1; }
import json, pickle
art = json.load(open("results_local/framing_glm45air_smoke.json"))
cfg, rows = art["config"], art["rows"]
assert not art.get("partial"), "smoke artifact is still a partial checkpoint"
assert cfg["direction"] == "dim_no_override_actioncentred", cfg["direction"]
assert cfg["alpha"] == 8.0, cfg["alpha"]
assert cfg["match_sigma_to"] == "dim_no_override_both", cfg["match_sigma_to"]
assert cfg["steer_layers"] == "20,24,28", cfg["steer_layers"]
assert cfg["max_new"] == 8192, cfg["max_new"]
assert len(rows) == 8, f"expected 2 queries x 4 samples = 8 rows, got {len(rows)}"
assert all(isinstance(r.get("completion"), str) and r["completion"].strip()
           for r in rows), "a smoke completion is empty"
n_closed = sum("</think>" in r["completion"] for r in rows)
assert n_closed > 0, "EVERY smoke completion truncated inside <think>"
# sigma provenance: recompute the anchor sigmas from the SHIPPED pickles (seam check 3)
exp = {20: 6.41610860824585, 24: 8.858674049377441, 28: 10.276041984558105}
for L, e in exp.items():
    s = pickle.load(open(f"runs/glm45-air/probe_L{L}.pkl", "rb"))["sigmas"]["dim_no_override_both"]
    assert abs(s - e) < 1e-3, f"L{L} anchor sigma {s} != expected {e}"
print(f"[gate] PASS: 8 rows, {n_closed}/8 closed </think>, sigmas match", flush=True)
PYG
echo "[job] $(date -u +%H:%M:%S) gate passed -- full 48-query grid" >&2

# ── FULL GRID: 48 queries x 18 samples, 2 instances x 4 GPUs, sharded over queries ───
# completion-cap 32000 on the full lanes too (review defect 2): fmt glm45 records
# truncated=None, so the saved completion is the only truncation diagnostic, and the
# default 4000-char head+tail elision can swallow </think> on long thinks.
lane() { # shard gpus
  CUDA_VISIBLE_DEVICES=$2 "$PY" "$R" "${CELL[@]}" \
    --n 24 --shard "$1" --nshard 2 --completion-cap 32000 \
    --out "$LOCAL/framing_glm45air.shard$1.json" \
    > "$LOCAL/log_glm_shard$1.log" 2>&1
}
lane 0 0,1,2,3 & P0=$!
lane 1 4,5,6,7 & P1=$!
rc=0; wait "$P0" || rc=1; wait "$P1" || rc=1
sync_blob
echo "JOB_DONE rc=$rc"
ls -la "$BLOB"
exit "$rc"
