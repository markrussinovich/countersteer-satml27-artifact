#!/usr/bin/env bash
# GLM-4.5-Air increase-side causal gate (the one gate its deployed direction has not
# run; sec7/limitations names it). One dev sweep at alpha = -8 on its firing corpus
# (paper_param, n=52, its own 8192 budget, device auto over 4 GPUs). Expected if the
# direction is causally bidirectional: attacked goal INCREASES above the ~0.75 dev base
# (headroom 0.25; saturation caveat rides the readout as with Gemma's qualified pass).
set -uo pipefail
bash cluster/seed_model.sh --require zai-org/GLM-4.5-Air || { echo "GLM-BIDIR-DONE rc=3"; exit 3; }
OUT=${OUT:-outputs}
mkdir -p "$OUT" runs/glm45-air logs_bidir
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python xpia_defense.py \
  --model zai-org/GLM-4.5-Air --stage sweep --device auto \
  --outdir runs/glm45-air --corpus paper_param --n-eval 52 --max-new 8192 \
  --directions dim_no_override_actioncentred --alphas -8 \
  --steer-layers 20,24,28 --match-sigma-to dim_no_override_both --steer-clean \
  > logs_bidir/glm_bidir_am8.log 2>&1
rc=$?
grep -E "^  \[" logs_bidir/glm_bidir_am8.log || true
cp -f logs_bidir/* "$OUT"/ 2>/dev/null || true
cp -f runs/glm45-air/results_add-dim-no-override-actioncentred-*.json "$OUT"/ 2>/dev/null || true
echo "GLM-BIDIR-DONE rc=$rc"
exit "$rc"
