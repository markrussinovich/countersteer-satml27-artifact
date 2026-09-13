# vLLM 0.19.0 IPI-Aware capture overlay

This release supports exactly **vLLM 0.19.0**, upstream tag `v0.19.0`, commit
`2a69949bdadf0e8942b7a1619b229cb475beef20`. The overlay uses whole-file
replacement, so other vLLM versions are intentionally rejected.

## Reproducible installation

Install the locked project environment and the exact vLLM extra:

```bash
uv sync --frozen --extra dev --extra vllm
```

Validate that the installed package is an unmodified vLLM 0.19.0 baseline:

```bash
mkdir -p logs/$(date -u +%F)
uv run python scripts/apply_vllm_ipi_aware_patch.py --check --report logs/$(date -u +%F)/vllm_patch_before.json
```

The report must contain `"state": "upstream"`. Apply the overlay, persist the
exact hashes that were installed, and verify the idempotent patched state:

```bash
uv run python scripts/apply_vllm_ipi_aware_patch.py --report logs/$(date -u +%F)/vllm_patch_apply.json
uv run python scripts/apply_vllm_ipi_aware_patch.py --check --report logs/$(date -u +%F)/vllm_patch_after.json
```

The final report must contain `"state": "patched"`. `--target` may be used to
select a different installed package explicitly:

```bash
uv run python scripts/apply_vllm_ipi_aware_patch.py --target /path/to/site-packages/vllm --check
```

The installer verifies:

- the installed version is exactly `0.19.0`;
- every replaced file matches either the official upstream SHA-256 or this
  release's patched SHA-256;
- both new helper modules are absent or already match this release;
- the patch directory contains every manifest entry.

Unknown hashes are rejected. This protects local vLLM modifications from being
silently overwritten. To restore an official baseline, reinstall vLLM 0.19.0
in the selected environment and rerun `--check` before applying the overlay.

## Overlay contents

Only files that differ from official vLLM 0.19.0 are shipped and applied.

| File | Purpose |
| --- | --- |
| `entrypoints/openai/api_server.py` | Auxiliary registration endpoints retained by the overlay; the public release workflow uses durable capture files. |
| `model_executor/models/_ipi_aware_capture.py` | Shared layer-output capture helpers. |
| `model_executor/models/_ipi_aware_sender.py` | Auxiliary callback sender retained by the overlay. |
| `model_executor/models/qwen2.py` | Capture hooks; Qwen3 inherits this model path. |
| `model_executor/models/qwen3_next.py` | Capture hooks for Qwen3-Next and inherited Qwen3.5 forward paths. |
| `model_executor/models/qwen3_5.py` | Capture initialization for the Qwen3.5 override. |
| `model_executor/models/gemma4.py` | Gemma4 capture hooks. |
| `model_executor/models/gpt_oss.py` | GPT-OSS capture hooks. |
| `v1/executor/multiproc_executor.py` | Single pipeline-batch scheduling while capture is active. |
| `v1/worker/gpu_model_runner.py` | Position resolution, eager capture, and persistence. |

Qwen3.5 hybrid-cache, ExtractHiddenStates, Gemma4 EAGLE3, and mixed-page-size
changes previously described as local patches are already present in official
vLLM 0.19.0. They are therefore not duplicated or overwritten by this release.

## Capture-position contract

For vLLM 0.19.0, `InputBatch.num_computed_tokens_cpu` is the number of tokens
computed before the current scheduled chunk. The overlay consequently maps a
request only against this interval:

```text
[num_computed_tokens, num_computed_tokens + num_scheduled_tokens)
```

It does not guess a post-advance interval. This prevents duplicate or shifted
hidden-state rows under chunked prefill.

## Runtime configuration

- `IPI_AWARE_CAPTURE_LAYERS`: comma-separated layer indices.
- `IPI_AWARE_SELECTED_POSITIONS`: comma-separated prompt-relative positions.
- `IPI_AWARE_CAPTURE_DIR`: durable output directory for multiprocess capture.
- `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1`: required for models
  with `attention_chunk_size` when applicable.

When launching vLLM in environments with inherited native-library paths, clear
`LD_LINK`, `LD_LIBRARY_PATH`, and `LD_PRELOAD`, then set the intended
`LD_PRELOAD` explicitly for that environment.
