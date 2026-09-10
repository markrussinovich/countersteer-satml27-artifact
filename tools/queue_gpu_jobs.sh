#!/bin/bash
# Dispatch queued jobs onto GPUs as they free, one job per freed GPU, no contention.
#
# WHY: the matched head-to-head runs 6-arm processes on GPU0/2 and 4-arm on GPU1/3, so
# GPU1 frees ~61 min and GPU3 ~49 min before the critical path (measured: 152 GPU-min idle,
# 29% of the pool). This fills that tail. A 5th process CANNOT share a busy card -- each
# holds 51.3 GiB and the weights alone are 39 GiB -- so waiting for a genuinely free GPU is
# the only option, not conservatism.
#
# NOTE build_param_abuse_dataset.py hardcodes cuda:0 and has no --device switch, so
# CUDA_VISIBLE_DEVICES is load-bearing: without it a queued build lands on a BUSY card.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$ROOT"
# Ordered by value. The test-split corpus is on the ladder's critical path -- `--stage
# confirm --corpus param_abuse` SystemExits without it -- and the judge is stubbed during
# corpus construction, so building it reveals nothing about the held-out split.
JOBS=(
  "heldout_a|dev 0 1 openai/gpt-oss-20b --n-eval 96 --template heldout_a"
  "heldout_b|dev 0 1 openai/gpt-oss-20b --n-eval 96 --template heldout_b"
  "testsplit|test 0 1 openai/gpt-oss-20b"
  "heldout_c|dev 0 1 openai/gpt-oss-20b --n-eval 96 --template heldout_c"
)
for J in "${JOBS[@]}"; do
  NAME="${J%%|*}"; ARGS="${J#*|}"
  G=""
  for _ in $(seq 1 900); do
    G=$(free_gpus | head -1); [ -n "$G" ] && break; sleep 20
  done
  [ -n "$G" ] || { echo "[queue] no GPU freed for $NAME"; exit 1; }
  echo "[queue] $(date +%H:%M:%S) starting $NAME on GPU $G"
  CUDA_VISIBLE_DEVICES=$G setsid nohup "$PY" -u \
    tools/controls/build_param_abuse_dataset.py $ARGS \
    > "$ROOT/logs/queued_${NAME}.log" 2>&1 < /dev/null &
  sleep 120
done
echo "[queue] all dispatched"
