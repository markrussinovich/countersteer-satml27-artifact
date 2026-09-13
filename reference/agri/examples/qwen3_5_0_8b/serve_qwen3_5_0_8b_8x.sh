#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TODAY="$(date -u +%Y-%m-%d)"

MODEL_PATH="${QWEN3_5_0_8B_MODEL:-$ROOT_DIR/models/Qwen/Qwen3.5-0.8B}"
SERVED_MODEL_NAME="${QWEN3_5_0_8B_SERVED_MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
CUDA_DEVICES_CSV="${QWEN3_5_0_8B_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PORTS_CSV="${QWEN3_5_0_8B_PORTS:-18080,18081,18082,18083,18084,18085,18086,18087}"
STATE_DIR="${QWEN3_5_0_8B_SERVER_STATE_DIR:-$ROOT_DIR/.local_state/examples/qwen3_5_0_8b/server_8x}"
LOG_DIR="${QWEN3_5_0_8B_LOG_DIR:-$ROOT_DIR/logs/$TODAY/examples/qwen3_5_0_8b}"
CACHE_DIR="${QWEN3_5_0_8B_CACHE_DIR:-$ROOT_DIR/.local_state/examples/qwen3_5_0_8b/cache}"
TMP_DIR="${QWEN3_5_0_8B_TMPDIR:-$ROOT_DIR/.local_state/tmp/qwen3_5_0_8b}"
REFERENCE_CONDA_ENV="${QWEN3_5_0_8B_REFERENCE_CONDA_ENV:-}"
DEFAULT_VLLM_BIN="$ROOT_DIR/.venv/bin/vllm"
if [[ ! -x "$DEFAULT_VLLM_BIN" && -n "$REFERENCE_CONDA_ENV" && -x "$REFERENCE_CONDA_ENV/bin/vllm" ]]; then
  DEFAULT_VLLM_BIN="$REFERENCE_CONDA_ENV/bin/vllm"
fi
VLLM_BIN="${QWEN3_5_0_8B_VLLM_BIN:-$DEFAULT_VLLM_BIN}"
HOST="${QWEN3_5_0_8B_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${QWEN3_5_0_8B_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${QWEN3_5_0_8B_MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${QWEN3_5_0_8B_MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${QWEN3_5_0_8B_GPU_MEMORY_UTILIZATION:-0.80}"
ENABLE_AUTO_TOOL_CHOICE="${QWEN3_5_0_8B_ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${QWEN3_5_0_8B_TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${QWEN3_5_0_8B_REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${QWEN3_5_0_8B_LANGUAGE_MODEL_ONLY:-1}"
ENABLE_PREFIX_CACHING="${QWEN3_5_0_8B_ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${QWEN3_5_0_8B_ENABLE_CHUNKED_PREFILL:-1}"
LD_PRELOAD_PATH="${QWEN3_5_0_8B_LD_PRELOAD:-}"
if [[ -z "$LD_PRELOAD_PATH" && -n "$REFERENCE_CONDA_ENV" && -f "$REFERENCE_CONDA_ENV/lib/libstdc++.so.6" ]]; then
  LD_PRELOAD_PATH="$REFERENCE_CONDA_ENV/lib/libstdc++.so.6"
fi
PID_DIR="$STATE_DIR/pids"

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
export VLLM_NO_USAGE_STATS=1

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<<"$raw"
}

is_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

server_log() {
  local gpu="$1" port="$2"
  printf '%s/vllm_gpu%s_port%s.log\n' "$LOG_DIR" "$gpu" "$port"
}

start_one() {
  local gpu="$1" port="$2" log_file="$3"
  local -a cmd=(
    env -u LD_LINK -u LD_LIBRARY_PATH -u LD_PRELOAD
  )
  if [[ -n "$LD_PRELOAD_PATH" && -f "$LD_PRELOAD_PATH" ]]; then
    cmd+=("LD_PRELOAD=$LD_PRELOAD_PATH")
  fi
  cmd+=(
    "CUDA_VISIBLE_DEVICES=$gpu"
    "$VLLM_BIN" serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$port"
    --reasoning-parser "$REASONING_PARSER"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --tensor-parallel-size 1
  )
  if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
    cmd+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
  fi
  if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
    cmd+=(--language-model-only)
  fi
  if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    cmd+=(--enable-prefix-caching)
  fi
  if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    cmd+=(--enable-chunked-prefill)
  fi

  printf '\n[%s] starting Qwen3.5-0.8B server gpu=%s port=%s\n' "$(date -u +%FT%TZ)" "$gpu" "$port" >>"$log_file"
  printf 'cmd:' >>"$log_file"
  printf ' %q' "${cmd[@]}" >>"$log_file"
  printf '\n' >>"$log_file"
  nohup setsid "${cmd[@]}" >>"$log_file" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$PID_DIR/port${port}.pid"
}

start_servers() {
  local -a gpus ports
  split_csv "$CUDA_DEVICES_CSV" gpus
  split_csv "$PORTS_CSV" ports
  if [[ "${#gpus[@]}" -ne "${#ports[@]}" ]]; then
    echo "QWEN3_5_0_8B_CUDA_VISIBLE_DEVICES and QWEN3_5_0_8B_PORTS must have the same length" >&2
    exit 2
  fi
  if [[ ! -x "$VLLM_BIN" ]]; then
    echo "vLLM binary is not executable: $VLLM_BIN" >&2
    exit 1
  fi
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model path does not exist: $MODEL_PATH" >&2
    exit 1
  fi
  mkdir -p "$PID_DIR" "$LOG_DIR"
  local index gpu port pid_file
  for index in "${!gpus[@]}"; do
    gpu="${gpus[$index]}"
    port="${ports[$index]}"
    pid_file="$PID_DIR/port${port}.pid"
    if [[ -f "$pid_file" ]] && is_alive "$(<"$pid_file")"; then
      echo "port $port already has managed PID $(<"$pid_file")"
      continue
    fi
    start_one "$gpu" "$port" "$(server_log "$gpu" "$port")"
    echo "started gpu=$gpu port=$port pid=$(<"$pid_file") log=$(server_log "$gpu" "$port")"
  done
  echo "topology: ${#gpus[@]} independent TP=1 servers"
  echo "vllm binary: $VLLM_BIN"
  echo "cache dir: $CACHE_DIR"
  echo "tmp dir: $TMP_DIR"
  echo "reference env: $REFERENCE_CONDA_ENV"
  echo "health: $0 status"
  echo "stop: $0 stop"
}

status_servers() {
  local -a gpus ports
  split_csv "$CUDA_DEVICES_CSV" gpus
  split_csv "$PORTS_CSV" ports
  local index gpu port pid_file pid state
  for index in "${!ports[@]}"; do
    gpu="${gpus[$index]}"
    port="${ports[$index]}"
    pid_file="$PID_DIR/port${port}.pid"
    pid=""
    [[ -f "$pid_file" ]] && pid="$(<"$pid_file")"
    if is_alive "$pid"; then
      if curl --fail --silent --max-time 5 "http://127.0.0.1:$port/v1/models" >/dev/null; then
        state="running-ready"
      else
        state="running-not-ready"
      fi
    else
      state="stopped"
    fi
    echo "gpu=$gpu port=$port pid=${pid:-none} state=$state vllm=$VLLM_BIN log=$(server_log "$gpu" "$port")"
  done
}

stop_servers() {
  local -a ports
  split_csv "$PORTS_CSV" ports
  local port pid_file pid
  for port in "${ports[@]}"; do
    pid_file="$PID_DIR/port${port}.pid"
    [[ -f "$pid_file" ]] || continue
    pid="$(<"$pid_file")"
    if is_alive "$pid"; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for port in "${ports[@]}"; do
    pid_file="$PID_DIR/port${port}.pid"
    [[ -f "$pid_file" ]] || continue
    pid="$(<"$pid_file")"
    if is_alive "$pid"; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  done
  echo "stopped Qwen3.5-0.8B managed servers"
}

case "${1:-}" in
  start) start_servers ;;
  status) status_servers ;;
  stop) stop_servers ;;
  *) echo "Usage: $0 {start|status|stop}"; exit 2 ;;
esac
