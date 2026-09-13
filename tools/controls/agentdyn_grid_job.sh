#!/usr/bin/env bash
# AML-cluster driver: the AgentDyn FULL-GRID program (benchmark-expansion agent,
# 2026-09-05; smoke gate PASSED on gpt-oss — FINDINGS 25a). One invocation runs one or
# more stages IN ORDER on one 8xH100 node; partial completion still delivers whole stages
# (the SoA priority-order pattern).
#
# Stages (args, in the order given):
#   gptoss  gpt-oss-20b  FULL AgentDyn grid (all solvable x all injection tasks),
#           4-arm battery via dojo_baseline_mn4096.sh, deployed cell
#           dim_no_override_both@8 L12/16/20 sigma-matched, max_new 4096, 8x1-GPU shards.
#   qwen    Qwen3-30B-A3B-Thinking FULL grid, champion cell dim_no_override_both@12
#           OWN sigma L8/20/32, max_new 4096, 8x1-GPU shards.
#   glm     GLM-4.5-Air SAMPLED-180 grid (seed 0, stratified over the 3 suites — the
#           precedent of its local 180-cell AgentDojo grid; a full 560-cell GLM grid does
#           not fit a node-day). Frozen cell dim_no_override_actioncentred@8 L20/24/28
#           sigma-matched to dim_no_override_both, max_new 8192 (its dojo-grid config,
#           runs/glm45-air/glm_dojo_full.json), device auto, 2 shards x 4 GPUs.
#   ipi     IPI Arena replay vs Qwen3-30B (FINDINGS 25b): 4 arms, 8x1-GPU shards,
#           attacks from the staged pinned dump (HF-offline).
#   ipi_gptoss / ipi_phi3 / ipi_gemma / ipi_glm
#           IPI Arena replay as a CORE EVAL on the other deployed cells (owner
#           directive 2026-09-06). Same runner/attack dump as `ipi`; each stage runs
#           its model's DEPLOYED cell (verified against BEST_DEFENSE.md/EVAL_MATRIX.md
#           and the certified run metas):
#             gptoss  combo_ovr8_pat1 @8.06 sigma-matched-to-dim_no_override L12/16/20,
#                     max_new 4096 (the dojo/AutoDojo/SoA convention), 8x1-GPU shards
#                     under attn_impl=flex_attention (SERVING-PATH CHANGE, preflight-
#                     gated + equivalence-measured). Eager cannot run this corpus on
#                     80GB at any device-map split: gpt-oss eager prefill holds THREE
#                     simultaneous (64 x s^2) bf16 tensors (matmul + sink-concat +
#                     max-subtract), ~96 GiB transient at the longest IPI prompts
#                     (~16k tok: delete-tmp, nimbus-llm-exfil) on the single GPU
#                     hosting the layer. Both xpia-ipi-core4 (8x1-GPU) and
#                     xpia-ipi-gptoss2 (4x2-GPU auto) OOM'd exactly there. gpt-oss has
#                     no sdpa/flash path in transformers 5.14.1; flex_attention is
#                     blockwise (sinks handled via LSE renorm in
#                     transformers/integrations/flex_attention.py).
#             phi3    dim_no_override_bal @6 sigma-matched-to-dim_no_override_both,
#                     FROZEN single-turn layers L8/12/16, max_new 4096 (its frozen-layer
#                     dojo budget, FINDINGS 26.4), 8x1-GPU shards
#             gemma   dim_no_override_both @8 OWN sigma L4/28/36, max_new 4096,
#                     8x1-GPU shards (62GB bf16 fits one H100; relay precedent on .7)
#             glm     dim_no_override_actioncentred @8 sigma-matched-to-
#                     dim_no_override_both L20/24/28, max_new 8192 (its dojo-grid
#                     budget), 2 shards x 4 GPUs, device auto
#
# Every stage seeds its model from the blob store with --require and asserts the staged
# snapshot SHA — the first gptoss seed under a real job closes the §23ao staging-smoke
# condition (README caveat: blobfuse/image seam previously uncertified in-job).
#
# SUBMIT (AML_* env exported; see cluster/README.md):
#   bash cluster/submit_job.sh --mode run --display-name xpia-agentdyn-gptoss-glm \
#     --timeout-seconds 84000 --no-clean \
#     --slmx-cmd 'bash tools/controls/agentdyn_grid_job.sh gptoss glm'
#   bash cluster/submit_job.sh --mode run --display-name xpia-agentdyn-qwen-ipi \
#     --timeout-seconds 72000 --no-clean \
#     --slmx-cmd 'bash tools/controls/agentdyn_grid_job.sh qwen ipi'
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT=${OUT:-outputs}
LOGD=logs_agentdyn
mkdir -p "$OUT" "$LOGD" runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Load OFFLINE after blob seeding (cluster/README.md caveat: an online
# from_pretrained re-resolves the hub's refs/main and can drift off the staged pin).
# Everything these stages load resolves locally: seeded snapshots, staged probe pkls,
# the staged pinned IPI attacks dump.
export HF_HUB_OFFLINE=1
# AgentDyn = the vendored agentdojo fork, selected by PYTHONPATH (never installed);
# artifacts stamp agentdojo_path so fork/upstream runs cannot be conflated.
export PYTHONPATH="$ROOT/reference/agentdyn/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$ROOT/reference/agentdyn/src/agentdojo" ] \
  || { echo "FATAL: reference/agentdyn/src not staged"; exit 1; }

sync_blob() {
  cp -f runs/agentdyn_screen.* runs/agentdyn_cells.* "$OUT"/ 2>/dev/null || true
  cp -f runs/agentdyn_grid_*/* runs/agentdyn_rivals/* runs/agentdyn_dose/* "$OUT"/ 2>/dev/null || true
  cp -f runs/ipi_arena/* runs/cacheprune_mask_qwen.json "$OUT"/ 2>/dev/null || true
  cp -f "$LOGD"/* "$OUT"/ 2>/dev/null || true
}
( while true; do sleep 180; sync_blob; done ) & SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; sync_blob' EXIT

seed() { # ORG/NAME EXPECTED_SHA
  local model="$1" sha="$2" t0=$SECONDS
  bash cluster/seed_model.sh --require "$model" || { echo "SEED_FAIL $model"; return 1; }
  local snapdir="$HF_HOME/hub/models--${model//\//--}/snapshots"
  if [ ! -d "$snapdir/$sha" ]; then
    echo "SEED_SHA_MISMATCH $model: staged $(ls "$snapdir" 2>/dev/null) != expected $sha"
    return 1
  fi
  echo "SEED_OK $model sha=$sha wall=$((SECONDS - t0))s"
}

screen_stage() { # KEY MODEL CVD... (one shard per CVD string; nshard = number of CVDs)
  local key="$1" model="$2"; shift 2
  local i=0 pids=()
  for cvd in "$@"; do
    CUDA_VISIBLE_DEVICES="$cvd" "$PY" -u tools/controls/agentdojo_smoke.py \
      --model "$model" --device "$( [[ "$cvd" == *,* ]] && echo auto || echo cuda:0 )" \
      --suite shopping,github,dailylife --system yaml --max-new "${SCREEN_MAXNEW:-4096}" \
      --screen --shard "$i" --nshard "$#" \
      --out "runs/agentdyn_screen.$key.shard$i.json" \
      > "$LOGD/screen_$key.shard$i.log" 2>&1 &
    pids+=("$!"); i=$((i + 1))
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  "$PY" -u tools/controls/agentdojo_pairs.py \
    --screen "runs/agentdyn_screen.$key.shard*.json" \
    --only important_instructions --n-main "${NCELLS:-2000}" --seed 0 \
    --out "runs/agentdyn_cells.$key.json" >> "$LOGD/pairs_$key.log" 2>&1 || rc=1
  sync_blob
  return "$rc"
}

ipi_stage() { # KEY MODEL PROBEDIR DIRECTION ALPHA LAYERS MATCH_SIGMA MAX_NEW ATTN CVD...
  # One IPI Arena replay over the staged pinned attack dump; one shard per CVD string
  # (a comma-joined CVD group shards the model over those GPUs via --device auto).
  # ATTN: "" = the default (certified) attention resolution; non-empty (e.g.
  # flex_attention) changes the serving path -- gate it with attn_impl_preflight.py
  # first, and the artifact carries a per-path label.
  local key="$1" model="$2" probedir="$3" dir="$4" alpha="$5" layers="$6" \
        msig="$7" mnew="$8" attn="$9"; shift 9
  mkdir -p runs/ipi_arena
  local i=0 pids=() nsh=$#
  for cvd in "$@"; do
    CUDA_VISIBLE_DEVICES="$cvd" "$PY" -u tools/controls/ipi_arena_replay.py \
      --model "$model" \
      --device "$( [[ "$cvd" == *,* ]] && echo auto || echo cuda:0 )" \
      --probe-dir "$probedir" --direction "$dir" --alpha "$alpha" \
      --layers "$layers" --match-sigma-to "$msig" --max-new "$mnew" \
      --attn-impl "$attn" \
      --shard "$i" --nshard "$nsh" \
      --attacks-json runs/ipi_arena_attacks_dataset.json \
      --out "runs/ipi_arena/$key.shard$i.json" \
      > "$LOGD/ipi_$key.shard$i.log" 2>&1 &
    pids+=("$!"); i=$((i + 1))
  done
  local rc=0 p s
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  for ((s = 0; s < nsh; s++)); do
    "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" \
      "runs/ipi_arena/$key.shard$s.json" >/dev/null 2>&1 \
      || { echo "IPI $key shard $s ARTIFACT BAD OR MISSING"; rc=1; }
  done
  sync_blob
  return "$rc"
}

rc=0
for stage in "$@"; do
  echo "[agentdyn-grid] stage $stage START $(date -u '+%F %T')"
  case "$stage" in
    gptoss)
      seed openai/gpt-oss-20b 6cee5e81ee83917806bbde320786a8fb61efebee || { rc=1; continue; }
      NCELLS=2000 screen_stage gptoss openai/gpt-oss-20b 0 1 2 3 4 5 6 7 || rc=1
      bash tools/controls/dojo_baseline_mn4096.sh \
        --model openai/gpt-oss-20b --cells runs/agentdyn_cells.gptoss.json \
        --direction dim_no_override_both --alpha 8 --layers 12,16,20 \
        --match-sigma-to dim_no_override --probe-dir runs/gpt-oss-20b-userabl \
        --max-new 4096 --system yaml --label agentdyn_gptoss \
        --gpus 0,1,2,3,4,5,6,7 \
        --outdir runs/agentdyn_grid_gptoss --logdir "$LOGD" || rc=1
      ;;
    qwen)
      seed Qwen/Qwen3-30B-A3B-Thinking-2507 144afc2f379b542fdd4e85a1fcd5e1f79112d95d \
        || { rc=1; continue; }
      NCELLS=2000 screen_stage qwen Qwen/Qwen3-30B-A3B-Thinking-2507 0 1 2 3 4 5 6 7 || rc=1
      bash tools/controls/dojo_baseline_mn4096.sh \
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 --cells runs/agentdyn_cells.qwen.json \
        --direction dim_no_override_both --alpha 12 --layers 8,20,32 \
        --match-sigma-to '' --probe-dir runs/qwen3-30b-thinking \
        --max-new 4096 --system yaml --label agentdyn_qwen \
        --gpus 0,1,2,3,4,5,6,7 \
        --outdir runs/agentdyn_grid_qwen --logdir "$LOGD" || rc=1
      ;;
    glm)
      seed zai-org/GLM-4.5-Air a24ceef6ce4f3536971efe9b778bdaa1bab18daa || { rc=1; continue; }
      NCELLS=180 SCREEN_MAXNEW=8192 \
        screen_stage glm zai-org/GLM-4.5-Air 0,1,2,3 4,5,6,7 || rc=1
      bash tools/controls/dojo_baseline_mn4096.sh \
        --model zai-org/GLM-4.5-Air --cells runs/agentdyn_cells.glm.json \
        --direction dim_no_override_actioncentred --alpha 8 --layers 20,24,28 \
        --match-sigma-to dim_no_override_both --probe-dir runs/glm45-air \
        --max-new 8192 --system yaml --device auto --label agentdyn_glm \
        --gpus 0-1-2-3,4-5-6-7 \
        --outdir runs/agentdyn_grid_glm --logdir "$LOGD" || rc=1
      ;;
    ipi)
      # model already seeded by the qwen stage when chained; seed defensively anyway
      seed Qwen/Qwen3-30B-A3B-Thinking-2507 144afc2f379b542fdd4e85a1fcd5e1f79112d95d \
        || { rc=1; continue; }
      mkdir -p runs/ipi_arena
      pids=()
      for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES="$i" "$PY" -u tools/controls/ipi_arena_replay.py \
          --shard "$i" --nshard 8 \
          --attacks-json runs/ipi_arena_attacks_dataset.json \
          --out "runs/ipi_arena/qwen.shard$i.json" \
          > "$LOGD/ipi_qwen.shard$i.log" 2>&1 &
        pids+=("$!")
      done
      for p in "${pids[@]}"; do wait "$p" || rc=1; done
      for i in 0 1 2 3 4 5 6 7; do
        "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" \
          "runs/ipi_arena/qwen.shard$i.json" >/dev/null 2>&1 \
          || { echo "IPI shard $i ARTIFACT BAD OR MISSING"; rc=1; }
      done
      ;;
    ipi_gptoss)
      seed openai/gpt-oss-20b 6cee5e81ee83917806bbde320786a8fb61efebee || { rc=1; continue; }
      # flex_attention is REQUIRED here (a serving-path change, gated + measured):
      # eager gpt-oss prefill holds 3 simultaneous (64 x s^2) bf16 tensors -- ~96 GiB
      # transient at the corpus's longest prompts (~16k tok), un-hostable on any 80GB
      # device-map split (both xpia-ipi-core4 and the 2-GPU xpia-ipi-gptoss2 OOM'd).
      # Preflight gates the stage: long-prompt feasibility under flex + measured
      # eager-vs-flex greedy equivalence (reported into the stage log).
      CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/controls/attn_impl_preflight.py \
        --model openai/gpt-oss-20b --attn-impl flex_attention --long-tokens 17000 \
        > "$LOGD/ipi_gptoss.preflight.log" 2>&1 \
        || { echo "IPI gptoss PREFLIGHT FAIL (see $LOGD/ipi_gptoss.preflight.log)"
             sync_blob; rc=1; continue; }
      sync_blob
      ipi_stage gptoss openai/gpt-oss-20b runs/gpt-oss-20b-userabl \
        combo_ovr8_pat1 8.06 12,16,20 dim_no_override 4096 flex_attention \
        0 1 2 3 4 5 6 7 || rc=1
      ;;
    ipi_phi3)
      seed microsoft/Phi-3-medium-128k-instruct a088b37c71d441ab6d862bb3fcfe6165b3014702 \
        || { rc=1; continue; }
      ipi_stage phi3 microsoft/Phi-3-medium-128k-instruct runs/phi3-medium-128k \
        dim_no_override_bal 6.0 8,12,16 dim_no_override_both 4096 "" \
        0 1 2 3 4 5 6 7 || rc=1
      ;;
    ipi_gemma)
      seed google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475 || { rc=1; continue; }
      ipi_stage gemma google/gemma-4-31B-it runs/gemma4-31b-it \
        dim_no_override_both 8.0 4,28,36 "" 4096 "" \
        0 1 2 3 4 5 6 7 || rc=1
      ;;
    ipi_glm)
      seed zai-org/GLM-4.5-Air a24ceef6ce4f3536971efe9b778bdaa1bab18daa || { rc=1; continue; }
      ipi_stage glm zai-org/GLM-4.5-Air runs/glm45-air \
        dim_no_override_actioncentred 8.0 20,24,28 dim_no_override_both 8192 "" \
        0,1,2,3 4,5,6,7 || rc=1
      ;;
    rivals_gptoss)
      # RIVAL-DEFENSE batteries (owner order 2026-09-07): CachePrune + reminder on the
      # flagship, on BOTH benchmarks. IPI: 4 arms per rival, 8x1-GPU shards (clean/attacked
      # comparators re-run in-battery -- the SoA same-process convention). AgentDyn:
      # 180-stratified cells (runs/agentdyn_cells180.gptoss.json, seed 0 from the FULL-grid
      # screens -- the GLM precedent), 4-arm battery via dojo_baseline_mn4096.sh.
      seed openai/gpt-oss-20b 6cee5e81ee83917806bbde320786a8fb61efebee || { rc=1; continue; }
      # flex_attention REQUIRED + preflight-gated for gpt-oss on this corpus (~16k-token
      # prompts; eager prefill OOM'd every xpia-ipi-core4 shard) -- same gate as
      # ipi_gptoss. Comparability: the gpt-oss steering IPI rows are flex-path too.
      CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/controls/attn_impl_preflight.py \
        --model openai/gpt-oss-20b --attn-impl flex_attention --long-tokens 17000 \
        > "$LOGD/rivals_gptoss.preflight.log" 2>&1 \
        || { echo "rivals_gptoss PREFLIGHT FAIL (see $LOGD/rivals_gptoss.preflight.log)"
             sync_blob; rc=1; continue; }
      mkdir -p runs/ipi_arena
      for rival in cacheprune reminder; do
        rargs=(--rival "$rival"); [ "$rival" = cacheprune ] && rargs+=(--kv-mask runs/cacheprune_mask.json)
        pids=()
        for i in 0 1 2 3 4 5 6 7; do
          # NOTE: no --direction/--alpha/--match-sigma-to -- steering is OFF in rival
          # arms by construction, and inert-but-recorded steering config in _meta trips
          # cross-artifact config diffs (review 2026-09-07, defect 4). config.rival is
          # the arm label.
          CUDA_VISIBLE_DEVICES="$i" "$PY" -u tools/controls/ipi_arena_replay.py \
            --model openai/gpt-oss-20b --probe-dir runs/gpt-oss-20b-userabl \
            --max-new 4096 --attn-impl flex_attention \
            "${rargs[@]}" --shard "$i" --nshard 8 \
            --attacks-json runs/ipi_arena_attacks_dataset.json \
            --out "runs/ipi_arena/gptoss_${rival}.shard$i.json" \
            > "$LOGD/ipi_gptoss_${rival}.shard$i.log" 2>&1 &
          pids+=("$!")
        done
        for p in "${pids[@]}"; do wait "$p" || rc=1; done
        sync_blob
      done
      bash tools/controls/dojo_baseline_mn4096.sh \
        --model openai/gpt-oss-20b --cells runs/agentdyn_cells180.gptoss.json \
        --kv-mask runs/cacheprune_mask.json --label agentdyn180_gptoss_cacheprune \
        --max-new 4096 --system yaml --gpus 0,1,2,3,4,5,6,7 \
        --outdir runs/agentdyn_rivals --logdir "$LOGD" || rc=1
      sync_blob
      bash tools/controls/dojo_baseline_mn4096.sh \
        --model openai/gpt-oss-20b --cells runs/agentdyn_cells180.gptoss.json \
        --defense reminder --label agentdyn180_gptoss_reminder \
        --max-new 4096 --system yaml --gpus 0,1,2,3,4,5,6,7 \
        --outdir runs/agentdyn_rivals --logdir "$LOGD" || rc=1
      ;;
    rivals_qwen)
      seed Qwen/Qwen3-30B-A3B-Thinking-2507 144afc2f379b542fdd4e85a1fcd5e1f79112d95d \
        || { rc=1; continue; }
      # CachePrune masks are PER-MODEL (coordinate space = layers x kv_heads x head_dim);
      # fit Qwen's own mask first (paper defaults, same recipe as the gpt-oss mask).
      # The fit needs the Nemotron corpus, which the offline job has no cache for
      # (review 2026-09-07, defect 2): warm the datasets cache with offline lifted for
      # exactly that pull (revision PINNED in src/corpora.py, so no drift), then fit
      # offline from cache. 2-GPU device auto: a grad-retained forward over a 61GB model
      # has no measured single-GPU precedent (defect 5).
      QWEN_MASK_OK=1
      if [ ! -f runs/cacheprune_mask_qwen.json ]; then
        HF_HUB_OFFLINE=0 "$PY" -c "from src.corpora import hf_dataset, NEMOTRON_REPO; hf_dataset(NEMOTRON_REPO, split='train'); print('nemotron cache warmed')" \
          >> "$LOGD/cacheprune_mask_qwen.log" 2>&1 \
          || { echo "QWEN_MASK_NEMOTRON_WARMUP_FAILED"; QWEN_MASK_OK=0; rc=1; }
        if [ "$QWEN_MASK_OK" = 1 ]; then
          CUDA_VISIBLE_DEVICES=0,1 "$PY" -u tools/controls/build_cacheprune_mask.py \
            --model Qwen/Qwen3-30B-A3B-Thinking-2507 --device auto \
            --out runs/cacheprune_mask_qwen.json >> "$LOGD/cacheprune_mask_qwen.log" 2>&1 \
            || { echo "QWEN_MASK_FIT_FAILED"; QWEN_MASK_OK=0; rc=1; }
        fi
      fi
      sync_blob
      mkdir -p runs/ipi_arena
      # a mask-fit failure loses only the CACHEPRUNE half; reminder needs no mask
      # (review 2026-09-07, defect 2 second part)
      RIVALS="cacheprune reminder"
      [ "$QWEN_MASK_OK" = 1 ] || RIVALS="reminder"
      for rival in $RIVALS; do
        rargs=(--rival "$rival"); [ "$rival" = cacheprune ] && rargs+=(--kv-mask runs/cacheprune_mask_qwen.json)
        pids=()
        for i in 0 1 2 3 4 5 6 7; do
          CUDA_VISIBLE_DEVICES="$i" "$PY" -u tools/controls/ipi_arena_replay.py \
            --shard "$i" --nshard 8 "${rargs[@]}" \
            --attacks-json runs/ipi_arena_attacks_dataset.json \
            --out "runs/ipi_arena/qwen_${rival}.shard$i.json" \
            > "$LOGD/ipi_qwen_${rival}.shard$i.log" 2>&1 &
          pids+=("$!")
        done
        for p in "${pids[@]}"; do wait "$p" || rc=1; done
        sync_blob
      done
      if [ "$QWEN_MASK_OK" = 1 ]; then
        bash tools/controls/dojo_baseline_mn4096.sh \
          --model Qwen/Qwen3-30B-A3B-Thinking-2507 --cells runs/agentdyn_cells180.qwen.json \
          --kv-mask runs/cacheprune_mask_qwen.json --label agentdyn180_qwen_cacheprune \
          --max-new 4096 --system yaml --gpus 0,1,2,3,4,5,6,7 \
          --outdir runs/agentdyn_rivals --logdir "$LOGD" || rc=1
      else
        echo "SKIP agentdyn180_qwen_cacheprune (mask fit failed)"
      fi
      sync_blob
      bash tools/controls/dojo_baseline_mn4096.sh \
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 --cells runs/agentdyn_cells180.qwen.json \
        --defense reminder --label agentdyn180_qwen_reminder \
        --max-new 4096 --system yaml --gpus 0,1,2,3,4,5,6,7 \
        --outdir runs/agentdyn_rivals --logdir "$LOGD" || rc=1
      ;;
    rivals_qwen_cp)
      # COMPLETION of the xpia-rivals-qwen Failed job's cacheprune half (FINDINGS 25i):
      # mask fit with the cross-device fix, then IPI cacheprune, then AgentDyn-180
      # cacheprune. The reminder half completed in the original job and is NOT re-run.
      seed Qwen/Qwen3-30B-A3B-Thinking-2507 144afc2f379b542fdd4e85a1fcd5e1f79112d95d \
        || { rc=1; continue; }
      # a pre-existing mask must PARSE before the fit is skipped (review D6)
      if [ -f runs/cacheprune_mask_qwen.json ] && \
         ! "$PY" -c "import json;json.load(open('runs/cacheprune_mask_qwen.json'))" 2>/dev/null; then
        echo "stale/truncated qwen mask -- refitting"; rm -f runs/cacheprune_mask_qwen.json
      fi
      if [ ! -f runs/cacheprune_mask_qwen.json ]; then
        HF_HUB_OFFLINE=0 "$PY" -c "from src.corpora import hf_dataset, NEMOTRON_REPO; hf_dataset(NEMOTRON_REPO, split='train'); print('nemotron cache warmed')" \
          >> "$LOGD/cacheprune_mask_qwen.log" 2>&1 \
          || { echo "QWEN_MASK_NEMOTRON_WARMUP_FAILED"; rc=1; continue; }
        CUDA_VISIBLE_DEVICES=0,1 "$PY" -u tools/controls/build_cacheprune_mask.py \
          --model Qwen/Qwen3-30B-A3B-Thinking-2507 --device auto \
          --out runs/cacheprune_mask_qwen.json >> "$LOGD/cacheprune_mask_qwen.log" 2>&1 \
          || { echo "QWEN_MASK_FIT_FAILED"; rc=1; continue; }
      fi
      sync_blob
      mkdir -p runs/ipi_arena
      pids=()
      for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES="$i" "$PY" -u tools/controls/ipi_arena_replay.py \
          --rival cacheprune --kv-mask runs/cacheprune_mask_qwen.json \
          --shard "$i" --nshard 8 \
          --attacks-json runs/ipi_arena_attacks_dataset.json \
          --out "runs/ipi_arena/qwen_cacheprune.shard$i.json" \
          > "$LOGD/ipi_qwen_cacheprune.shard$i.log" 2>&1 &
        pids+=("$!")
      done
      for p in "${pids[@]}"; do wait "$p" || rc=1; done
      sync_blob
      bash tools/controls/dojo_baseline_mn4096.sh \
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 --cells runs/agentdyn_cells180.qwen.json \
        --kv-mask runs/cacheprune_mask_qwen.json --label agentdyn180_qwen_cacheprune \
        --max-new 4096 --system yaml --gpus 0,1,2,3,4,5,6,7 \
        --outdir runs/agentdyn_rivals --logdir "$LOGD" || rc=1
      ;;
    dose_frontier_gptoss)
      # §26.5 item 1 (owner program 2026-09-08; pre-registration in FINDINGS): the
      # AgentDyn dose frontier on the deployed gpt-oss direction, alphas {5.5,6.7,8.06},
      # ONE process per cell with shared clean/attacked arms (agentdojo_run --alphas).
      seed openai/gpt-oss-20b 6cee5e81ee83917806bbde320786a8fb61efebee || { rc=1; continue; }
      bash tools/controls/dojo_baseline_mn4096.sh \
        --model openai/gpt-oss-20b --cells runs/agentdyn_cells180.gptoss.json \
        --direction dim_no_override_both --alphas 5.5,6.7,8 --alpha 8 \
        --layers 12,16,20 --match-sigma-to dim_no_override \
        --probe-dir runs/gpt-oss-20b-userabl \
        --max-new 4096 --system yaml --label agentdyn180_gptoss_dosefrontier \
        --gpus 0,1,2,3,4,5,6,7 \
        --outdir runs/agentdyn_dose --logdir "$LOGD" || rc=1
      cp -f runs/agentdyn_dose/* "$OUT"/ 2>/dev/null || true
      ;;
    traj_norm_gptoss)
      # §26.5 item 2 / §26.9 (pre-registered; review8 SHIP-CLEARED): the trajectory-
      # normalized-exposure CONTROLLED battery at alpha* = 8 (§26.10: the frontier found
      # no cliff, so alpha* defaults to the deployed dose). Six arms per cell in ONE
      # process: shared clean+attacked, {fixed, energy-norm} x {CLEAN+, defended}.
      seed openai/gpt-oss-20b 6cee5e81ee83917806bbde320786a8fb61efebee || { rc=1; continue; }
      mkdir -p runs/agentdyn_dose
      pids=()
      for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES="$i" "$PY" -u tools/controls/agentdojo_run.py \
          --model openai/gpt-oss-20b --cells runs/agentdyn_cells180.gptoss.json \
          --direction dim_no_override_both --alpha 8 --layers 12,16,20 \
          --match-sigma-to dim_no_override --probe-dir runs/gpt-oss-20b-userabl \
          --steer-schedule fixed,energy-norm \
          --max-new 4096 --system yaml --no-adjudicate \
          --shard "$i" --nshard 8 \
          --out "runs/agentdyn_dose/trajnorm_gptoss.shard$i.json" \
          > "$LOGD/trajnorm_gptoss.shard$i.log" 2>&1 &
        pids+=("$!")
      done
      for p in "${pids[@]}"; do wait "$p" || rc=1; done
      for i in 0 1 2 3 4 5 6 7; do
        "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" \
          "runs/agentdyn_dose/trajnorm_gptoss.shard$i.json" >/dev/null 2>&1 \
          || { echo "trajnorm shard $i ARTIFACT BAD OR MISSING"; rc=1; }
      done
      sync_blob
      ;;
    *) echo "unknown stage: $stage"; rc=1 ;;
  esac
  echo "[agentdyn-grid] stage $stage EXIT rc=$rc $(date -u '+%F %T')"
  sync_blob
done

echo "AGENTDYN-GRID-JOB-DONE rc=$rc"
exit "$rc"
