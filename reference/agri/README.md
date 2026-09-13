<div align="center" style="margin-top:0; padding-top:0;">
  <h1 style="margin-top:0; padding-top:0;">IPI Exposure Signal</h1>

  <h4>
    <a href="https://arxiv.org/abs/2608.02657"><img src="https://img.shields.io/badge/arXiv-2608.02657-b31b1b.svg" alt="arXiv"></a>
  </h4>
</div>

This is the code repository for our paper: `Your Agentic LLMs Secretly Encode
Latent Signals of Indirect Prompt-Injection Exposure`.

ArXiv version and paper link: https://arxiv.org/abs/2608.02657

This repository implements the probing pipeline for latent IPI-exposure signals:
AgentDojo trace collection, IPI risk labeling, hidden-state featurization, and
IPI-exposure probe training/evaluation.

## Citation

```bibtex
@misc{dong2026agenticllmssecretlyencode,
  title={Your Agentic LLMs Secretly Encode Latent Signals of Indirect Prompt-Injection Exposure},
  author={Jianshuo Dong and Yiming Liu and Maosen Zhang and Nan Deng and Xu Peng and Xiaoping Zhang and Tianwei Zhang and Jie Zhang and Han Qiu},
  year={2026},
  eprint={2608.02657},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2608.02657},
}
```

## Quick Start

Get started with the CPU-friendly checks first:

### 1. Clone and Setup

```bash
git clone https://github.com/jianshuod/IPI-exposure-signal.git
cd IPI-exposure-signal

# Install uv if needed.
curl -LsSf https://astral.sh/uv/install.sh | sh

# This repository targets Python 3.12.
uv python install 3.12
uv sync --extra dev
```

### 2. Run Tests

```bash
uv run pytest
```

The test suite is designed to run without GPU resources. It checks core message
serialization, AgentDojo long-horizon attack registration, labeling logic,
probe dataset construction, training metrics, token accounting, and vLLM patch
packaging.

### 3. Configure a Model Endpoint

For collection, point the collector at an OpenAI-compatible model server:

```bash
export IPI_AWARE_UPSTREAM_BASE_URL=http://127.0.0.1:8000/v1
export IPI_AWARE_UPSTREAM_API_KEY=EMPTY
```

For hidden-state extraction and probe training, install model dependencies and
the PyTorch build that matches your CUDA environment:

```bash
uv sync --extra dev --extra models
```

## Pipeline Overview

The main workflow is exposed through `ipi-signal-probe`:

Command | Purpose
--- | ---
`collect` | Run AgentDojo tasks and save traces plus decision points
`label` | Assign IPI risk labels to collected decision points
`featurize` | Extract model hidden states for probe examples
`partition` | Build train/validation/evaluation splits
`train` | Train lightweight probes over hidden-state features
`eval` | Evaluate trained probes on configured splits
`eval-groups` | Run grouped evaluation for labeling protocols and held-out settings

Example:

```bash
uv run ipi-signal-probe collect \
  --suite workspace \
  --attack direct \
  --injected \
  --max-cases 4 \
  --results-dir results/probe_traces/v2

uv run ipi-signal-probe label \
  --root results/probe_traces/v2/<run-name> \
  --labeling-protocol risk_faced

uv run ipi-signal-probe featurize \
  --root results/probe_traces/v2/<run-name> \
  --model Qwen/Qwen3-8B \
  --backend transformers_hook \
  --model-family qwen3 \
  --selected-position -1

uv run ipi-signal-probe partition \
  --root results/probe_traces/v2/<run-name> \
  --dataset broad

# Training consumes a JSON config. The Qwen example below generates the
# train/eval configs used for its full run.
```

Supported labeling protocols are `risk_faced`, `risk_visible`, and
`risk_actual`.

## Reproducible Qwen3.5-0.8B Example

The `examples/qwen3_5_0_8b/` directory contains a runnable small-model
reproduction and open-source usability test for `Qwen/Qwen3.5-0.8B`.

Start eight one-GPU vLLM servers:

```bash
examples/qwen3_5_0_8b/serve_qwen3_5_0_8b_8x.sh start
```

Run a smoke test:

```bash
QWEN3_5_0_8B_MODE=smoke \
examples/qwen3_5_0_8b/run_probe_pipeline.sh start
```

Run the full example:

```bash
QWEN3_5_0_8B_MODE=full \
examples/qwen3_5_0_8b/run_probe_pipeline.sh start
```

The full example uses 8 independent TP=1 vLLM servers, 8 collector processes,
256 workers per process/server, 8 featurization shards, and one-GPU probe
training shards. See `examples/qwen3_5_0_8b/README.md` for all environment
variables, resume modes, cache settings, and status/stop commands.

## vLLM Direct Capture

The default featurization backend is `transformers_hook`. The direct vLLM
hidden-state capture backend requires the checked vLLM 0.19.0 overlay:

```bash
uv sync --frozen --extra dev --extra vllm
uv run python scripts/apply_vllm_ipi_aware_patch.py --check
```

Follow `docs/vllm_patches_v0.19.0.md` before using:

```bash
uv run ipi-signal-probe featurize \
  --root results/probe_traces/v2/<run-name> \
  --model Qwen/Qwen3-8B \
  --backend vllm_direct \
  --model-family qwen3 \
  --selected-position -1
```

## Long-Horizon AgentDojo Attacks

This repository vendors a patched AgentDojo checkout under `vendor/agentdojo`.
Upstream AgentDojo does not natively expose the long-horizon attacks required by
the paper, so collection commands should be run from this source checkout, or
with `vendor/agentdojo/src` placed ahead of any installed upstream AgentDojo on
`PYTHONPATH`.

## Directory Structure

```text
IPI-exposure-signal/
├── ipi_aware/
│   ├── data_collection/      # AgentDojo traces and decision points
│   ├── message_content.py    # Chat-message normalization used by collection/features
│   └── probes/               # labels, features, partitions, training, eval
├── examples/
│   └── qwen3_5_0_8b/         # 8-GPU Qwen/Qwen3.5-0.8B reproduction scripts
├── patches/                  # vLLM 0.19.0 hidden-state capture overlay
├── vendor/agentdojo/         # patched AgentDojo source for long-horizon attacks
├── docs/                     # vLLM patch notes
├── scripts/                  # utility scripts
└── tests/                    # CPU-friendly regression tests
```

## Environment Configuration

Variable | Purpose
--- | ---
`IPI_AWARE_UPSTREAM_BASE_URL` | OpenAI-compatible model endpoint used by collection
`IPI_AWARE_UPSTREAM_API_KEY` | API key for the upstream endpoint, or `EMPTY` for local vLLM
`PYTHONPATH` | Put `vendor/agentdojo/src` first when using the patched AgentDojo source outside this checkout
`QWEN3_5_0_8B_MODEL` | Override the default Qwen/Qwen3.5-0.8B checkpoint path in the example scripts
`QWEN3_5_0_8B_CACHE_DIR` | Redirect model and local caches to a data filesystem
`QWEN3_5_0_8B_FEAT_BACKEND` | Set to `vllm_direct` after applying the vLLM overlay

## License

This project is licensed under Apache-2.0. See `LICENSE`.
