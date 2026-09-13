#!/bin/bash
# Gemma-4-31B CERTIFICATION job -- one 8x H100 node, everything GPU-heavy in the Gemma
# certification program (owner addendum 2026-09-04: put the one-shot test pass and the
# AgentDojo full grid on AML-cluster):
#
#   GPU 0  ONE-SHOT held-out test pass, paper_param_heldout    (confirm, 4 arms, n=52, a8)
#          then the PRESPECIFIED a6 dose-monotonicity control (separate confirm invocation;
#          a8 is the reported cell REGARDLESS of how a6 reads -- see the pre-registration)
#          then the 6-sample T*-REPLICA equivalence probe: paper_param (ORIGINAL corpus)
#          --n-eval 6, config byte-matched to the A100 T* run 3532482, to MEASURE (not
#          assume) A100-vs-H100 equivalence on this model (review defect C-2)
#   GPU 1  ONE-SHOT held-out test pass, paper_disjoint_heldout (a8), then its a6 control
#   GPU 2-5  AgentDojo FULL GRID, 180 cells x 4 arms, 4 shards
#   GPU 6-7  AgentDojo paired BENIGN arms (clean + CLEAN+ over unique tasks), 2 shards
#
# THE FROZEN CELL, PINNED EXPLICITLY EVERYWHERE (the Phi-3 grid silently ran runner-default
# layers -- that seam is exactly what the explicit --layers 4,28,36 here closes):
#   direction dim_no_override_both @ alpha 8, layers 4,28,36, --scale sigma,
#   --match-sigma-to dim_no_override_both (expected effective sigmas 2.2489/8.5813/4.0083,
#   the §18c-testrung numbers -- verify the [steer]/[preflight] lines reprint them).
#
# PRE-REGISTERED in FINDINGS §18c-heldout (written BEFORE this launch; adversarial review
# 2026-09-04 defect C-1 closed): one reported dose (a8), the a6 arms are dose-monotonicity
# controls and NOT selection candidates, bars stated there. The heldout corpora were built
# doc+wording+template-disjoint from every Gemma dev/tuning input
# (build_paper_heldout_dataset.py asserts it at build time; adversarially recomputed OK).
#
# TIMEOUT GROUNDING (review defect C-3): the 6-cell feasibility smoke on a .7 A100
# measured 503s wall for 6 cells x 4 arms (avg 84s/cell, worst cell 192s, zero truncation
# at max_new 4096). Worst-case shard = 45 cells x 192s = 2.4h; benign shards are smaller;
# confirm legs measured 3330-3645s each on A100. Submit timeout 28800s (8h) is >2.5x the
# worst chain (a8 confirm + a6 control + replica probe ~= 2.2h).
#
# LABELLING (review C-5): --no-adjudicate matches the Phi-3 grid, NOT the Qwen grid
# (adjudicated). AgentDojo tiers use the benchmark's own checkers either way; when gemma
# `ours_*` columns sit beside Qwen's, say the adjudication settings differ.
#
# SUBMIT (from the /datadrive/xpia-gemmacert clone on <FLEET_HOST_B>, where az login lives):
#   bash cluster/submit_job.sh --mode run --display-name xpia-gemma-cert \
#     --timeout-seconds 28800 --no-clean --slmx-cmd 'bash tools/gemma_cert_cluster_job.sh'
# Then arm tools/await_aml_job.sh (env in ITS OWN process) + a fleet manifest row.
set -uo pipefail
PY="${PY:-$(command -v python)}"
MODEL=google/gemma-4-31B-it
PROBE=runs/gemma4-31b-it
DIR=dim_no_override_both
LAYERS_CLI=4,28,36
OUTD="outputs/gemma_cert"
mkdir -p "$OUTD" runs/gemma4-31b-it
harvest() {
  cp -f runs/gemma4-31b-it/results_confirm_*.json "$OUTD/" 2>/dev/null || true
  cp -f runs/gemma4-31b-it/gemma_dojo_*.json "$OUTD/" 2>/dev/null || true
}
trap harvest EXIT
pids=()

confirm() { # gpu corpus alpha n label
  CUDA_VISIBLE_DEVICES=$1 $PY -u xpia_defense.py \
    --model "$MODEL" --outdir runs/gemma4-31b-it --device cuda:0 \
    --stage confirm --corpus "$2" --n-eval "$4" \
    --steer-layers "$LAYERS_CLI" --directions "$DIR" --alphas "$3" \
    --match-sigma-to "$DIR" --scale sigma --steer-clean \
    --batch 4 --max-new 2048 > "$OUTD/$5.log" 2>&1
  echo "[chain] $5 rc=$?" >> "$OUTD/chain.log"
  harvest
}

# GPU 0: param heldout a8 (THE one-shot) -> a6 control -> T*-replica equivalence probe
(
  confirm 0 paper_param_heldout    8.0 52 confirm_param_heldout_a8
  confirm 0 paper_param_heldout    6.0 52 confirm_param_heldout_a6
  confirm 0 paper_param            8.0  6 confirm_param_TSTARREPLICA_a8_n6
) & pids+=($!)
# GPU 1: disjoint heldout a8 (THE one-shot) -> a6 control
(
  confirm 1 paper_disjoint_heldout 8.0 52 confirm_disjoint_heldout_a8
  confirm 1 paper_disjoint_heldout 6.0 52 confirm_disjoint_heldout_a6
) & pids+=($!)

# GPU 2-5: the 180-cell grid, 4 shards, all four arms per cell.
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((2 + i)) $PY -u tools/controls/agentdojo_run.py \
    --model "$MODEL" --device cuda:0 --probe-dir "$PROBE" \
    --direction "$DIR" --match-sigma-to "$DIR" --alpha 8.0 --layers "$LAYERS_CLI" \
    --max-new 4096 --no-adjudicate --cells runs/agentdojo_cells.json \
    --shard "$i" --nshard 4 \
    --out "runs/gemma4-31b-it/gemma_dojo_full.shard$i.json" \
    > "$OUTD/dojo_full_shard$i.log" 2>&1 &
  pids+=($!)
done

# GPU 6-7: paired benign arms (clean + CLEAN+ over the unique user tasks).
for j in 0 1; do
  CUDA_VISIBLE_DEVICES=$((6 + j)) $PY -u tools/controls/agentdojo_run.py \
    --model "$MODEL" --device cuda:0 --probe-dir "$PROBE" \
    --direction "$DIR" --match-sigma-to "$DIR" --alpha 8.0 --layers "$LAYERS_CLI" \
    --max-new 4096 --no-adjudicate --cells runs/agentdojo_cells.json \
    --benign-only --shard "$j" --nshard 2 \
    --out "runs/gemma4-31b-it/gemma_dojo_benign.shard$j.json" \
    > "$OUTD/dojo_benign_shard$j.log" 2>&1 &
  pids+=($!)
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
harvest
cat "$OUTD/chain.log" 2>/dev/null
echo "GEMMA-CERT-JOB-DONE rc=$rc"
exit $rc
