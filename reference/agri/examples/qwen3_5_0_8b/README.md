# Qwen3.5-0.8B Probe Pipeline Example

This example runs the released `ipi-signal-probe` workflow against
`Qwen/Qwen3.5-0.8B`. It is intended both as a small-model reproduction recipe
and as a usability test for the open-source artifact.

The scripts default to a checkpoint path inside this repository checkout:

```bash
models/Qwen/Qwen3.5-0.8B
```

Download or symlink `Qwen/Qwen3.5-0.8B` there, or override
`QWEN3_5_0_8B_MODEL` if the model lives elsewhere.

## Setup

From the release root:

```bash
uv sync --extra dev --extra agentdojo --extra models
```

The release also defines a `vllm` extra pinned to vLLM 0.19.0 for the direct
capture overlay. Some Linux environments may need to install vLLM separately if
locked transitive wheels are unavailable. In that case, point the launcher at a
known-good vLLM binary:

```bash
export QWEN3_5_0_8B_VLLM_BIN=/path/to/env/bin/vllm
```

If you keep a separate reference environment, record it explicitly:

```bash
export QWEN3_5_0_8B_REFERENCE_CONDA_ENV=/path/to/reference/env
```

The server launcher first tries the release `.venv/bin/vllm`; if that is not
installed and `QWEN3_5_0_8B_REFERENCE_CONDA_ENV` is set, it falls back to that
environment's `bin/vllm` and records the choice in server logs/status.

## Run

Start eight one-GPU vLLM servers. Each server is tensor-parallel size 1, and
the collector pins one process to each server:

```bash
examples/qwen3_5_0_8b/serve_qwen3_5_0_8b_8x.sh start
```

Run a small end-to-end smoke:

```bash
QWEN3_5_0_8B_MODE=smoke examples/qwen3_5_0_8b/run_probe_pipeline.sh start
```

Plan the collection grid without requiring vLLM servers:

```bash
QWEN3_5_0_8B_MODE=plan examples/qwen3_5_0_8b/run_probe_pipeline.sh start
```

Run the full 4-suite, 4-prompt, 6-attack collection:

```bash
QWEN3_5_0_8B_MODE=full examples/qwen3_5_0_8b/run_probe_pipeline.sh start
```

The default featurization backend is `transformers_hook`. To exercise the
direct vLLM capture backend, first apply the vLLM 0.19.0 overlay documented in
`docs/vllm_patches_v0.19.0.md`, then run with:

```bash
QWEN3_5_0_8B_FEAT_BACKEND=vllm_direct \
QWEN3_5_0_8B_MODE=full \
examples/qwen3_5_0_8b/run_probe_pipeline.sh start
```

Resume an existing run from a later stage:

```bash
QWEN3_5_0_8B_MODE=full \
QWEN3_5_0_8B_RUN_NAME=qwen3_5_0_8b_full_20260731_reference \
QWEN3_5_0_8B_START_STAGE=featurize \
examples/qwen3_5_0_8b/run_probe_pipeline.sh start
```

`QWEN3_5_0_8B_START_STAGE` accepts `check-servers`, `collect`, `label`,
`featurize`, `partition`, `train`, or `eval`. This is useful after a machine
restart, an interrupted GPU job, or a code fix to one downstream stage. The
featurizer skips grid points whose split feature directory already exists.

Check progress:

```bash
examples/qwen3_5_0_8b/serve_qwen3_5_0_8b_8x.sh status
examples/qwen3_5_0_8b/run_probe_pipeline.sh status
```

Stop managed processes:

```bash
examples/qwen3_5_0_8b/run_probe_pipeline.sh stop
examples/qwen3_5_0_8b/serve_qwen3_5_0_8b_8x.sh stop
```

## Defaults

- GPUs: `0,1,2,3,4,5,6,7`
- Ports: `18080,18081,18082,18083,18084,18085,18086,18087`
- Serving topology: 8 independent TP=1 vLLM servers
- Collection topology: 8 collector processes, 256 workers per process/server,
  2048 total workers
- Featurization topology: 8 shards, one per GPU
- Featurization backend: `transformers_hook` by default; set
  `QWEN3_5_0_8B_FEAT_BACKEND=vllm_direct` after applying the vLLM overlay
- Featurization batch size: `8` by default, with progressive halving on CUDA OOM
- Probe-training topology: layer grid split across 8 one-GPU training shards
- Results root: `results/examples/qwen3_5_0_8b`
- Run name: `qwen3_5_0_8b_<mode>_<timestamp>`
- Model family: `qwen3.5`
- Labeling protocol: `risk_faced`
  (`eval-groups` in this release evaluates `risk_faced`)
- Training dataset: `broad`
- Probe layers: `0..23`, matching the checkpoint's 24 hidden layers

Use `QWEN3_5_0_8B_*` environment variables in the scripts to override these
defaults.

## Cache And Temp Files

The launchers redirect runtime caches away from the root filesystem by default:

- `HOME`, `XDG_CACHE_HOME`, Hugging Face, ModelScope, Torch, Triton, uv, and
  vLLM cache/config roots go under
  `.local_state/examples/qwen3_5_0_8b/cache`.
- `TMPDIR` defaults to `.local_state/tmp/qwen3_5_0_8b`.
- vLLM usage/config writes are disabled or redirected by the server launcher.

Override `QWEN3_5_0_8B_CACHE_DIR` and `QWEN3_5_0_8B_TMPDIR` when running on a
different machine. Keep these on a data/workspace filesystem for full runs.
