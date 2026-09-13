#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TODAY="$(date -u +%Y-%m-%d)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

PYTHON="${QWEN3_5_0_8B_PYTHON:-$ROOT_DIR/.venv/bin/python}"
MODEL_PATH="${QWEN3_5_0_8B_MODEL:-$ROOT_DIR/models/Qwen/Qwen3.5-0.8B}"
MODEL_ID="${QWEN3_5_0_8B_MODEL_ID:-Qwen/Qwen3.5-0.8B}"
MODE="${QWEN3_5_0_8B_MODE:-plan}"
RESULTS_DIR="${QWEN3_5_0_8B_RESULTS_DIR:-$ROOT_DIR/results/examples/qwen3_5_0_8b}"
RUN_NAME="${QWEN3_5_0_8B_RUN_NAME:-qwen3_5_0_8b_${MODE}_${TIMESTAMP}}"
RUN_ROOT="$RESULTS_DIR/$RUN_NAME"
START_STAGE="${QWEN3_5_0_8B_START_STAGE:-check-servers}"
STATE_DIR="${QWEN3_5_0_8B_PIPELINE_STATE_DIR:-$ROOT_DIR/.local_state/examples/qwen3_5_0_8b/pipeline}"
LOG_DIR="${QWEN3_5_0_8B_LOG_DIR:-$ROOT_DIR/logs/$TODAY/examples/qwen3_5_0_8b}"
CACHE_DIR="${QWEN3_5_0_8B_CACHE_DIR:-$ROOT_DIR/.local_state/examples/qwen3_5_0_8b/cache}"
TMP_DIR="${QWEN3_5_0_8B_TMPDIR:-$ROOT_DIR/.local_state/tmp/qwen3_5_0_8b}"
PID_FILE="$STATE_DIR/run.pid"
PGID_FILE="$STATE_DIR/run.pgid"
STAGE_FILE="$STATE_DIR/stage"
RUN_NAME_FILE="$STATE_DIR/run_name"
PORTS_CSV="${QWEN3_5_0_8B_PORTS:-18080,18081,18082,18083,18084,18085,18086,18087}"
GPUS_CSV="${QWEN3_5_0_8B_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
LD_PRELOAD_PATH="${QWEN3_5_0_8B_LD_PRELOAD:-}"
REFERENCE_CONDA_ENV="${QWEN3_5_0_8B_REFERENCE_CONDA_ENV:-}"
if [[ -z "$LD_PRELOAD_PATH" && -n "$REFERENCE_CONDA_ENV" && -f "$REFERENCE_CONDA_ENV/lib/libstdc++.so.6" ]]; then
  LD_PRELOAD_PATH="$REFERENCE_CONDA_ENV/lib/libstdc++.so.6"
fi

LABELING_PROTOCOL="${QWEN3_5_0_8B_LABELING_PROTOCOL:-risk_faced}"
DATASET_NAME="${QWEN3_5_0_8B_DATASET:-}"
FEATURE_NAME="${QWEN3_5_0_8B_FEATURE_NAME:-default}"
MODEL_FAMILY="${QWEN3_5_0_8B_MODEL_FAMILY:-qwen3.5}"
FEAT_BACKEND="${QWEN3_5_0_8B_FEAT_BACKEND:-transformers_hook}"
COLLECT_PROCESSES="${QWEN3_5_0_8B_COLLECT_PROCESSES:-8}"
WORKERS_PER_SERVER="${QWEN3_5_0_8B_WORKERS_PER_SERVER:-256}"
COLLECT_WORKERS="${QWEN3_5_0_8B_COLLECT_WORKERS:-$((COLLECT_PROCESSES * WORKERS_PER_SERVER))}"
if [[ "$COLLECT_WORKERS" -ne "$((COLLECT_PROCESSES * WORKERS_PER_SERVER))" ]]; then
  echo "QWEN3_5_0_8B_COLLECT_WORKERS must equal COLLECT_PROCESSES * WORKERS_PER_SERVER" >&2
  exit 2
fi
REQUEST_TIMEOUT="${QWEN3_5_0_8B_REQUEST_TIMEOUT:-600}"
TEMPERATURE="${QWEN3_5_0_8B_TEMPERATURE:-1}"
EPOCHS="${QWEN3_5_0_8B_EPOCHS:-3}"
FEAT_BATCH_SIZE="${QWEN3_5_0_8B_FEAT_BATCH_SIZE:-8}"
FEAT_MAX_PROMPT_TOKENS="${QWEN3_5_0_8B_FEAT_MAX_PROMPT_TOKENS:-}"
TRAIN_BATCH_SIZE="${QWEN3_5_0_8B_TRAIN_BATCH_SIZE:-256}"
TRAIN_EPOCHS="${QWEN3_5_0_8B_TRAIN_EPOCHS:-5}"
LEARNING_RATE="${QWEN3_5_0_8B_LEARNING_RATE:-0.0003}"
LAYER_INDICES="${QWEN3_5_0_8B_LAYER_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}"

mkdir -p \
  "$TMP_DIR" \
  "$CACHE_DIR/home" \
  "$CACHE_DIR/xdg" \
  "$CACHE_DIR/hf" \
  "$CACHE_DIR/modelscope" \
  "$CACHE_DIR/torch" \
  "$CACHE_DIR/torchinductor" \
  "$CACHE_DIR/triton" \
  "$CACHE_DIR/uv" \
  "$CACHE_DIR/vllm" \
  "$CACHE_DIR/vllm-config"
export HOME="$CACHE_DIR/home"
export TMPDIR="$TMP_DIR"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/hf/transformers"
export MODELSCOPE_CACHE="$CACHE_DIR/modelscope"
export TORCH_HOME="$CACHE_DIR/torch"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export UV_CACHE_DIR="$CACHE_DIR/uv"
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
export VLLM_CONFIG_ROOT="$CACHE_DIR/vllm-config"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$MODE" in
  plan)
    MAX_CASES="${QWEN3_5_0_8B_MAX_CASES:-1}"
    PLAN_ONLY=1
    ;;
  smoke)
    MAX_CASES="${QWEN3_5_0_8B_MAX_CASES:-}"
    EPOCHS="${QWEN3_5_0_8B_EPOCHS:-1}"
    PLAN_ONLY=0
    ;;
  full)
    MAX_CASES="${QWEN3_5_0_8B_MAX_CASES:-}"
    PLAN_ONLY=0
    ;;
  *)
    echo "Unsupported QWEN3_5_0_8B_MODE=$MODE; use plan, smoke, or full" >&2
    exit 2
    ;;
esac

if [[ -z "$DATASET_NAME" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    DATASET_NAME="qwen3_5_0_8b_smoke_pair"
  else
    DATASET_NAME="broad"
  fi
fi

is_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    tr -d '[:space:]' <"$path"
  fi
  return 0
}

set_stage() {
  mkdir -p "$STATE_DIR"
  printf '%s\n' "$1" >"$STAGE_FILE"
}

stage_index() {
  case "$1" in
    check-servers) echo 0 ;;
    collect) echo 1 ;;
    label) echo 2 ;;
    featurize) echo 3 ;;
    partition) echo 4 ;;
    train) echo 5 ;;
    eval) echo 6 ;;
    completed|completed-plan) echo 7 ;;
    *)
      echo "Unsupported QWEN3_5_0_8B_START_STAGE=$1; use check-servers, collect, label, featurize, partition, train, or eval" >&2
      exit 2
      ;;
  esac
}

should_run_stage() {
  local stage="$1"
  [[ "$(stage_index "$stage")" -ge "$(stage_index "$START_STAGE")" ]]
}

run_cli() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    IPI_AWARE_UPSTREAM_API_KEY=EMPTY \
    NO_PROXY=127.0.0.1,localhost \
    no_proxy=127.0.0.1,localhost \
    "$PYTHON" -m ipi_aware.probes.cli "$@"
}

check_servers_ready() {
  IFS=',' read -r -a ports <<<"$PORTS_CSV"
  local port
  for port in "${ports[@]}"; do
    curl --fail --silent --max-time 10 "http://127.0.0.1:$port/v1/models" >/dev/null
  done
}

split_layers_for_shard() {
  "$PYTHON" - "$LAYER_INDICES" "$1" "$2" <<'PY'
import sys

layers = [int(item) for item in sys.argv[1].split(",") if item.strip()]
index = int(sys.argv[2])
total = int(sys.argv[3])
print(",".join(str(layer) for offset, layer in enumerate(layers) if offset % total == index))
PY
}

write_train_config() {
  local config_path="$1"
  local output_dir="$2"
  local layers_csv="$3"
  mkdir -p "$RUN_ROOT/configs" "$RUN_ROOT/probes"
  "$PYTHON" - "$config_path" "$RUN_ROOT" "$DATASET_NAME" "$output_dir" "$LABELING_PROTOCOL" "$FEATURE_NAME" "$TRAIN_BATCH_SIZE" "$LEARNING_RATE" "$TRAIN_EPOCHS" "$layers_csv" <<'PY'
import json
import sys

(
    path,
    root,
    dataset_name,
    output_dir,
    labeling_protocol,
    feature_name,
    batch_size,
    learning_rate,
    epochs,
    layers_csv,
) = sys.argv[1:11]
layers = [int(item) for item in layers_csv.split(",") if item.strip()]
payload = {
    "root": root,
    "dataset": dataset_name,
    "output_dir": output_dir,
    "labeling_protocol": labeling_protocol,
    "feature_name": feature_name,
    "batch_size": int(batch_size),
    "threshold": 0.5,
    "random_seed": 42,
    "grid": {
        "layer_index": layers,
        "probe_architecture": ["linear"],
        "feature_composition": ["single_layer"],
        "learning_rate": [float(learning_rate)],
        "epochs": [int(epochs)],
        "feature_normalization": ["standard"],
    },
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
print(path)
PY
}

write_eval_config() {
  local config_path="$RUN_ROOT/configs/eval_${LABELING_PROTOCOL}.yaml"
  mkdir -p "$RUN_ROOT/configs" "$RUN_ROOT/eval"
  "$PYTHON" - "$config_path" "$RUN_ROOT" "$FEATURE_NAME" "$DATASET_NAME" "$LABELING_PROTOCOL" "$@" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
feature_name = sys.argv[3]
dataset_name = sys.argv[4]
labeling_protocol = sys.argv[5]
probe_dirs = sys.argv[6:]
output_dir = run_root / "eval" / f"{dataset_name}_{labeling_protocol}"
lines = [
    f"root: {run_root}",
    f"feature_name: {feature_name}",
    "eval_batch_size: 4096",
    f"output_dir: {output_dir}",
    "discover_from:",
]
for probe_dir in probe_dirs:
    lines.extend([
        f"  - dir: {probe_dir}",
        "    all_layers: false",
    ])
lines.append("")
path.write_text("\n".join(lines), encoding="utf-8")
print(path)
PY
}

discover_train_dirs() {
  local train_parent_root="$RUN_ROOT/probes/${DATASET_NAME}_${LABELING_PROTOCOL}"
  if [[ -d "$train_parent_root" ]]; then
    find "$train_parent_root" -mindepth 1 -maxdepth 1 -type d | sort
  fi
}

collect_args() {
  local -a cmd=(
    collect
    --results-dir "$RESULTS_DIR"
    --run-name "$RUN_NAME"
    --model vllm_parsed
    --model-id "$MODEL_ID"
    --benchmark-version v1.2.2
    --temperature "$TEMPERATURE"
    --epochs "$EPOCHS"
    --max-workers "$COLLECT_WORKERS"
    --num-processes "$COLLECT_PROCESSES"
    --upstream-ports "$PORTS_CSV"
    --continue-on-error
    --request-timeout "$REQUEST_TIMEOUT"
    --log-interval 100
  )
  if [[ "$MODE" == "smoke" ]]; then
    cmd+=(
      --suite workspace
      --user-task user_task_39
      --injection-task injection_task_0
      --attack direct
      --attack important_instructions
      --attack long_horizon_important_instructions
      --system-message-name default
    )
  else
    cmd+=(
      --suite banking
      --suite slack
      --suite travel
      --suite workspace
      --attack direct
      --attack ignore_previous
      --attack important_instructions
      --attack long_horizon_important_instructions
      --attack system_message
      --attack tool_knowledge
      --system-message-name default
      --system-message-name default_tool_careful
      --system-message-name safety_reminder_balanced
      --system-message-name safety_reminder_explicit
    )
  fi
  if [[ "$PLAN_ONLY" == "1" ]]; then
    cmd+=(--plan-only)
  fi
  if [[ -n "$MAX_CASES" ]]; then
    cmd+=(--max-cases "$MAX_CASES")
  fi
  printf '%s\0' "${cmd[@]}"
}

run_pipeline() {
  cd "$ROOT_DIR"
  mkdir -p "$STATE_DIR" "$LOG_DIR" "$RESULTS_DIR"
  printf '%s\n' "$RUN_NAME" >"$RUN_NAME_FILE"
  stage_index "$START_STAGE" >/dev/null
  if [[ ! -x "$PYTHON" ]]; then
    echo "Python is not executable: $PYTHON" >&2
    exit 1
  fi
  if [[ "$PLAN_ONLY" != "1" && ! -d "$MODEL_PATH" ]]; then
    echo "model path does not exist: $MODEL_PATH" >&2
    exit 1
  fi
  IFS=',' read -r -a gpus <<<"$GPUS_CSV"

  if [[ "$PLAN_ONLY" != "1" ]] && should_run_stage check-servers; then
    set_stage check-servers
    check_servers_ready
  fi

  if should_run_stage collect; then
    set_stage collect
    mapfile -d '' -t cmd < <(collect_args)
    run_cli "${cmd[@]}" >"$LOG_DIR/${RUN_NAME}.collect.log" 2>&1
  fi

  if [[ "$PLAN_ONLY" == "1" ]]; then
    set_stage completed-plan
    return 0
  fi

  if should_run_stage label; then
    set_stage label
    run_cli label --root "$RUN_ROOT" --labeling-protocol "$LABELING_PROTOCOL" --verbose \
      >"$LOG_DIR/${RUN_NAME}.label.log" 2>&1
  fi

  local index gpu log_file child_pid rc failed=0
  local -a child_pids=()
  if should_run_stage featurize; then
    set_stage featurize
    if [[ "$FEAT_BACKEND" == "vllm_direct" ]]; then
      "$PYTHON" "$ROOT_DIR/scripts/apply_vllm_ipi_aware_patch.py" --check \
        >"$LOG_DIR/${RUN_NAME}.vllm_patch_check.log" 2>&1
    fi
    child_pids=()
    failed=0
    for index in "${!gpus[@]}"; do
      gpu="${gpus[$index]}"
      log_file="$LOG_DIR/${RUN_NAME}.featurize.gpu${gpu}.shard${index}of${#gpus[@]}.log"
      local -a feat_cmd=(
        "$PYTHON" -m ipi_aware.probes.cli featurize
        --root "$RUN_ROOT"
        --model "$MODEL_PATH"
        --backend "$FEAT_BACKEND"
        --model-family "$MODEL_FAMILY"
        --feature-name "$FEATURE_NAME"
        --selected-position -1
        --batch-size "$FEAT_BATCH_SIZE"
        --device-map auto
        --bucket-by-prompt-length
        --longest-first
        --enable-split-saving
        --shard "$index/${#gpus[@]}"
        --verbose
      )
      if [[ -n "$FEAT_MAX_PROMPT_TOKENS" ]]; then
        feat_cmd+=(--max-prompt-tokens "$FEAT_MAX_PROMPT_TOKENS")
      fi
      local -a feat_env=(env -u LD_LINK -u LD_LIBRARY_PATH -u LD_PRELOAD "CUDA_VISIBLE_DEVICES=$gpu")
      if [[ -n "$LD_PRELOAD_PATH" && -f "$LD_PRELOAD_PATH" ]]; then
        feat_env+=("LD_PRELOAD=$LD_PRELOAD_PATH")
      fi
      "${feat_env[@]}" "${feat_cmd[@]}" >"$log_file" 2>&1 &
      child_pid="$!"
      child_pids+=("$child_pid")
    done
    set +e
    for child_pid in "${child_pids[@]}"; do
      wait "$child_pid"
      rc=$?
      (( rc == 0 )) || failed=1
    done
    set -e
    (( failed == 0 ))
  fi

  local train_parent_root="$RUN_ROOT/probes/${DATASET_NAME}_${LABELING_PROTOCOL}"
  local layers_csv shard_name train_config train_output_dir
  local -a train_parent_dirs=()
  if should_run_stage partition; then
    set_stage partition
    if [[ "$MODE" == "smoke" ]]; then
      run_cli partition --root "$RUN_ROOT" --dataset "$DATASET_NAME" \
        --train-grid-point workspace__default__clean \
        --train-grid-point workspace__default__direct \
        --train-grid-point workspace__default__important_instructions \
        --train-grid-point workspace__default__long_horizon_important_instructions \
        --labeling-protocol "$LABELING_PROTOCOL" --partition-granularity decision-points --verbose \
        >"$LOG_DIR/${RUN_NAME}.partition.log" 2>&1
    else
      run_cli partition --root "$RUN_ROOT" --dataset "$DATASET_NAME" \
        --labeling-protocol "$LABELING_PROTOCOL" --partition-granularity traces --verbose \
        >"$LOG_DIR/${RUN_NAME}.partition.log" 2>&1
    fi
  fi

  if should_run_stage train; then
    set_stage train
    child_pids=()
    failed=0
    for index in "${!gpus[@]}"; do
      gpu="${gpus[$index]}"
      layers_csv="$(split_layers_for_shard "$index" "${#gpus[@]}")"
      [[ -n "$layers_csv" ]] || continue
      shard_name="gpu${gpu}_layers_${layers_csv//,/_}"
      train_config="$RUN_ROOT/configs/probe_train_${LABELING_PROTOCOL}_${shard_name}.json"
      train_output_dir="$train_parent_root/$shard_name"
      write_train_config "$train_config" "$train_output_dir" "$layers_csv" >/dev/null
      train_parent_dirs+=("$train_output_dir")
      env -u LD_LINK -u LD_LIBRARY_PATH -u LD_PRELOAD \
        CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON" -m ipi_aware.probes.cli train --config "$train_config" --device cuda:0 --verbose \
        >"$LOG_DIR/${RUN_NAME}.train.${shard_name}.log" 2>&1 &
      child_pid="$!"
      child_pids+=("$child_pid")
    done
    set +e
    for child_pid in "${child_pids[@]}"; do
      wait "$child_pid"
      rc=$?
      (( rc == 0 )) || failed=1
    done
    set -e
    (( failed == 0 ))
  else
    mapfile -t train_parent_dirs < <(discover_train_dirs)
  fi

  if should_run_stage eval; then
    set_stage eval
    if [[ "${#train_parent_dirs[@]}" -eq 0 ]]; then
      echo "No probe directories found under $train_parent_root for eval." >&2
      exit 1
    fi
    eval_config="$(write_eval_config "${train_parent_dirs[@]}")"
    env -u LD_LINK -u LD_LIBRARY_PATH -u LD_PRELOAD \
      CUDA_VISIBLE_DEVICES="${gpus[0]}" \
      "$PYTHON" -m ipi_aware.probes.cli eval-groups --config "$eval_config" --device cuda:0 --verbose \
      >"$LOG_DIR/${RUN_NAME}.eval.log" 2>&1
  fi

  set_stage completed
}

start_pipeline() {
  mkdir -p "$STATE_DIR" "$LOG_DIR"
  if [[ -f "$PID_FILE" ]] && is_alive "$(<"$PID_FILE")"; then
    echo "Refusing duplicate start: controller PID $(<"$PID_FILE") is alive."
    exit 1
  fi
  rm -f "$PID_FILE" "$PGID_FILE" "$STAGE_FILE"
  printf '\n[%s] starting Qwen3.5-0.8B pipeline mode=%s run=%s start_stage=%s\n' "$(date -u +%FT%TZ)" "$MODE" "$RUN_NAME" "$START_STAGE" >>"$LOG_DIR/${RUN_NAME}.controller.log"
  nohup setsid "$0" run >>"$LOG_DIR/${RUN_NAME}.controller.log" 2>&1 < /dev/null &
  local pid=$! pgid
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  printf '%s\n' "$pid" >"$PID_FILE"
  printf '%s\n' "$pgid" >"$PGID_FILE"
  printf '%s\n' "$RUN_NAME" >"$RUN_NAME_FILE"
  echo "started Qwen3.5-0.8B pipeline pid=$pid pgid=$pgid"
  echo "run root: $RUN_ROOT"
  echo "start stage: $START_STAGE"
  echo "controller log: $LOG_DIR/${RUN_NAME}.controller.log"
  echo "status: $0 status"
  echo "stop: $0 stop"
}

status_pipeline() {
  local pid pgid stage current_run
  pid="$(read_file "$PID_FILE")"
  pgid="$(read_file "$PGID_FILE")"
  stage="$(read_file "$STAGE_FILE")"
  current_run="$(read_file "$RUN_NAME_FILE")"
  if is_alive "$pid"; then
    echo "running pid=$pid pgid=${pgid:-unknown}"
  elif [[ -n "$pgid" ]] && kill -0 -- "-$pgid" 2>/dev/null; then
    echo "running process group pgid=$pgid"
  else
    echo "stopped"
  fi
  echo "stage: ${stage:-unknown}"
  echo "run: ${current_run:-$RUN_NAME}"
  echo "run root: $RESULTS_DIR/${current_run:-$RUN_NAME}"
  echo "log dir: $LOG_DIR"
  echo "stop: $0 stop"
}

stop_pipeline() {
  local pid pgid
  pid="$(read_file "$PID_FILE")"
  pgid="$(read_file "$PGID_FILE")"
  if [[ -n "$pgid" ]] && kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    sleep 2
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  elif is_alive "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE" "$PGID_FILE"
  echo "stopped Qwen3.5-0.8B pipeline"
}

case "${1:-}" in
  run) run_pipeline ;;
  start) start_pipeline ;;
  status) status_pipeline ;;
  stop) stop_pipeline ;;
  *) echo "Usage: $0 {start|status|stop|run}"; exit 2 ;;
esac
