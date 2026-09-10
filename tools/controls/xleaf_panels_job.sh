#!/usr/bin/env bash
# Singularity driver: xleaf cross-model content-leaf transfer panels, GLM-4.5-Air +
# Qwen3-30B on ONE 8xH100 node (pre-registered: tmp/xleaf/PREREG.md, 2026-09-09;
# adversarially signed off with corrections C1/C2 applied; mirrors the gpt-oss
# §26.13/§26.15 ladder with the EXISTING v1 selector -- transfer evidence, v1 caveat on
# every leaf row).
#
# PLACEMENT NOTE (deviation from the reviewed .7 plan, disclosed): the leafsel v2/180
# battery (fleet-top) claimed .7 GPUs 1/2/5/7 + .9 GPU0 + .11 GPU0 before launch; the
# cluster is measured free (az ml job list: none running). Both models therefore run
# here, one node, all driver flags byte-identical to the reviewed lanes.
#
# Stages, in order:
#   glm panel  14 unique tasks x (clean + cleanplus@{full,leaf,random})  2 shards x 4 GPU
#   glm sec    20 firing cells x (attacked + defended@{full,leaf,random}) 2 shards x 4 GPU
#   qwen panel 13 unique tasks x (clean + cleanplus@{full,leaf,random})  4 shards x 1 GPU
#   qwen sec   20 firing cells x (attacked + defended@{full,leaf,random}) 4 shards x 1 GPU
#   (qwen panel and sec run CONCURRENTLY: panel on GPUs 0-3, sec on GPUs 4-7)
# random on the sec stages = review1 C2, the §26.16 coverage-ablation transfer arm.
# SIGMA GATES (post-stage, review1 C6): every shard log's [steer] line must carry the
# shipped grid's announced sigma, rebuilt from the staged pkls before submission.
#
# SUBMIT (AML_* env exported; see singularity/README.md):
#   bash singularity/submit_job.sh --mode run --display-name xpia-xleaf-panels \
#     --timeout-seconds 57600 --no-clean \
#     --slmx-cmd 'bash tools/controls/xleaf_panels_job.sh'
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT=${OUT:-outputs}
LOGD=logs_xleaf
mkdir -p "$OUT" "$LOGD" runs/xleaf_panels
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/reference/agentdyn/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$ROOT/reference/agentdyn/src/agentdojo" ] \
  || { echo "FATAL: reference/agentdyn/src not staged"; exit 1; }

# review1 C1: the submission snapshot must carry EXACTLY the reviewed bytes -- the v2
# agent edits the live tree, and submit_job.sh rsyncs it at submission time. Refuse to
# run on drifted code rather than measure an unreviewed selector.
echo "3fb5a8872ee2f57b5693e1e53caf71c8  tools/controls/agentdojo_bridge.py
1afdd8930a733291186f6010b84cd8bc  tools/controls/agentdojo_run.py" | md5sum -c - \
  || { echo "XLEAF_CODE_DRIFT: staged bridge/driver differ from the reviewed bytes"; exit 1; }

GLM_SIGMA='sigma=[6.4161, 8.8587, 10.276]'
QWEN_SIGMA='sigma=[0.2917, 0.593, 1.0736]'

sync_blob() {
  cp -f runs/xleaf_panels/* "$OUT"/ 2>/dev/null || true
  cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true
}
( while true; do sleep 180; sync_blob; done ) & SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; sync_blob' EXIT

seed() { # ORG/NAME EXPECTED_SHA
  local model="$1" sha="$2" t0=$SECONDS
  bash singularity/seed_model.sh --require "$model" || { echo "SEED_FAIL $model"; return 1; }
  local snapdir="$HF_HOME/hub/models--${model//\//--}/snapshots"
  [ -d "$snapdir/$sha" ] \
    || { echo "SEED_SHA_MISMATCH $model: staged $(ls "$snapdir" 2>/dev/null) != $sha"; return 1; }
  echo "SEED_OK $model sha=$sha wall=$((SECONDS - t0))s"
}

seed zai-org/GLM-4.5-Air a24ceef6ce4f3536971efe9b778bdaa1bab18daa || exit 1
seed Qwen/Qwen3-30B-A3B-Thinking-2507 144afc2f379b542fdd4e85a1fcd5e1f79112d95d || exit 1
export HF_HUB_OFFLINE=1

GLM_COMMON=(--model zai-org/GLM-4.5-Air --system yaml --max-new 8192
            --alpha 8.0 --direction dim_no_override_actioncentred --layers 20,24,28
            --match-sigma-to dim_no_override_both --probe-dir runs/glm45-air
            --no-adjudicate --device auto)
QWEN_COMMON=(--model Qwen/Qwen3-30B-A3B-Thinking-2507 --system yaml --max-new 4096
             --alpha 12.0 --direction dim_no_override_both --layers 8,20,32
             --match-sigma-to '' --probe-dir runs/qwen3-30b-thinking
             --no-adjudicate --device cuda:0)

sigma_gate() { # KEY NSHARD EXPECTED  -- post-stage: every shard log's [steer] line
  local key="$1" nshard="$2" expected="$3" rc=0 i
  for i in $(seq 0 $((nshard - 1))); do
    if grep -q '\[steer\]' "$LOGD/xleaf_${key}.shard$i.log"; then
      grep -qF "$expected" "$LOGD/xleaf_${key}.shard$i.log" \
        || { echo "SIGMA_MISMATCH ${key}.shard$i"; rc=1; }
    else
      echo "NO_STEER_LINE ${key}.shard$i"; rc=1
    fi
  done
  return "$rc"
}

run_shard() { # KEY SHARD NSHARD CVD COMMON_NAME EXTRA...
  local key="$1" i="$2" nshard="$3" cvd="$4" cname="$5[@]"; shift 5
  CUDA_VISIBLE_DEVICES="$cvd" "$PY" -u tools/controls/agentdojo_run.py "${!cname}" "$@" \
    --shard "$i" --nshard "$nshard" \
    --out "runs/xleaf_panels/xleaf_${key}.shard$i.json" \
    > "$LOGD/xleaf_${key}.shard$i.log" 2>&1
}

rc=0

# ── GLM stages (4-GPU device-auto shards) ────────────────────────────────────────────
glm_stage() { # KEY CELLS EXTRA...
  local key="$1" cells="$2"; shift 2
  local pids=() src=0
  run_shard "$key" 0 2 0,1,2,3 GLM_COMMON --cells "$cells" "$@" & pids+=("$!")
  run_shard "$key" 1 2 4,5,6,7 GLM_COMMON --cells "$cells" "$@" & pids+=("$!")
  for p in "${pids[@]}"; do wait "$p" || src=1; done
  sigma_gate "$key" 2 "$GLM_SIGMA" || src=1
  sync_blob
  echo "XLEAF_STAGE_${key^^}_DONE rc=$src"
  return "$src"
}
glm_stage glm_panel runs/agentdyn_cells.xleaf_panel_glm.json \
  --benign-only --span-select full,leaf,random || rc=1
glm_stage glm_sec runs/agentdyn_cells.xleaf_sec_glm.json \
  --security-only --span-select full,leaf,random || rc=1

# ── Qwen stages (1-GPU shards; panel on GPUs 0-3, sec on GPUs 4-7, concurrent) ───────
qpids=()
for i in 0 1 2 3; do
  run_shard qwen_panel "$i" 4 "$i" QWEN_COMMON \
    --cells runs/agentdyn_cells.xleaf_panel_qwen.json \
    --benign-only --span-select full,leaf,random & qpids+=("$!")
done
for i in 0 1 2 3; do
  run_shard qwen_sec "$i" 4 "$((i + 4))" QWEN_COMMON \
    --cells runs/agentdyn_cells.xleaf_sec_qwen.json \
    --security-only --span-select full,leaf,random & qpids+=("$!")
done
qrc=0; for p in "${qpids[@]}"; do wait "$p" || qrc=1; done
sigma_gate qwen_panel 4 "$QWEN_SIGMA" || qrc=1
sigma_gate qwen_sec 4 "$QWEN_SIGMA" || qrc=1
sync_blob
echo "XLEAF_STAGE_QWEN_DONE rc=$qrc"
[ "$qrc" = 0 ] || rc=1

sync_blob
echo "XLEAF_PANELS_JOB_DONE rc=$rc"
exit "$rc"
