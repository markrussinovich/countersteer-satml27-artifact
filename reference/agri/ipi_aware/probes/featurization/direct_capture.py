"""Direct hidden-state capture via two-tier extraction protocol.

The GPUModelRunner (orchestrator) resolves target token positions from the
batch layout and passes them to model.forward() (executor). The model gathers
hidden states only at those positions for the configured layers. After each
forward pass, the ModelRunner collects the small gathered tensors and
accumulates them across chunked-prefill passes.

For TP=1 (UniProcExecutor), all communication is via Python attributes.
For TP>1, falls back to file-based communication via safetensors.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .hidden_states import HiddenStateCapture


class DirectCaptureExtractor:
    """Extract hidden states via the two-tier extraction protocol."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        layer_ids: Sequence[int] | None = None,
        selected_positions: Sequence[int] | None = None,
        max_model_len: int | None = None,
        num_gpus: int = 1,
        gpu_memory_utilization: float = 0.92,
        enforce_eager: bool = False,
        enable_chunked_prefill: bool = False,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int | None = None,
        language_model_only: bool = True,
        dtype: str = "auto",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.layer_ids = tuple(layer_ids) if layer_ids is not None else None
        self.selected_positions = list(selected_positions) if selected_positions else [-1]
        self.max_model_len = max_model_len
        self.num_gpus = num_gpus
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.language_model_only = language_model_only
        self.dtype = dtype
        self._llm = None
        self._tmpdir = None
        self._resolved_layer_ids: tuple[int, ...] = ()
        self._is_single_gpu = num_gpus == 1

    def _get_runner(self):
        """Access the GPUModelRunner (single-GPU only)."""
        wrapper = self._llm.llm_engine.model_executor.driver_worker
        return wrapper.worker.model_runner

    def __enter__(self) -> DirectCaptureExtractor:
        from vllm import LLM

        if self.layer_ids is not None:
            self._resolved_layer_ids = self.layer_ids
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            tc = getattr(config, "text_config", config)
            n_layers = int(getattr(tc, "num_hidden_layers", 0))
            self._resolved_layer_ids = tuple(range(n_layers))

        os.environ["IPI_AWARE_SELECTED_POSITIONS"] = ",".join(str(p) for p in self.selected_positions)
        os.environ["IPI_AWARE_CAPTURE_LAYERS"] = ",".join(str(l) for l in self._resolved_layer_ids)

        if self._is_single_gpu:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        else:
            # Force spawn to avoid "Cannot re-initialize CUDA in forked subprocess"
            # when transformers imports silently init CUDA at the C++ level.
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            self._tmpdir = tempfile.mkdtemp(prefix="ipi_aware_capture_")
            os.environ["IPI_AWARE_CAPTURE_DIR"] = self._tmpdir

        llm_kwargs: dict[str, Any] = dict(
            model=self.model_name_or_path,
            enforce_eager=self.enforce_eager,
            enable_chunked_prefill=self.enable_chunked_prefill,
            language_model_only=self.language_model_only,
            enable_prefix_caching=False,
            max_num_seqs=self.max_num_seqs,
            dtype=self.dtype,
        )
        if self.num_gpus > 1:
            llm_kwargs["tensor_parallel_size"] = self.num_gpus
        if self.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        if self.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
        elif self.enable_chunked_prefill:
            llm_kwargs["max_num_batched_tokens"] = 131072

        self._llm = LLM(**llm_kwargs)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._llm is not None:
            try:
                self._llm.llm_engine.engine_core.shutdown(timeout=10)
            except Exception:
                pass
            self._llm = None
        os.environ.pop("IPI_AWARE_CAPTURE_DIR", None)
        os.environ.pop("IPI_AWARE_SELECTED_POSITIONS", None)
        os.environ.pop("IPI_AWARE_CAPTURE_LAYERS", None)
        os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
        return None

    def capture(
        self,
        prompts: Sequence[Any],
        *,
        selected_positions: Sequence[int] | None = None,
        timing_callback: Any | None = None,
    ) -> tuple[HiddenStateCapture, ...]:
        if not prompts:
            return ()

        positions = selected_positions or self.selected_positions
        generate_start = time.monotonic()

        if hasattr(prompts[0], "messages"):
            # ChatMessagesPrompt path — vLLM handles tokenization
            captures = self._capture_chat_messages(prompts, positions)
            prompt_tokens_total = None
        else:
            # ProbePrompt path — use pre-computed token IDs
            prompt_token_ids = [
                tuple(int(token) for token in prompt.token_ids[: prompt.decode_start_index])
                for prompt in prompts
            ]
            if self._is_single_gpu:
                captures = self._capture_batched_single(prompt_token_ids, positions)
            else:
                captures = self._capture_batched_multi(prompt_token_ids, positions)
            prompt_tokens_total = sum(len(t) for t in prompt_token_ids)

        generate_seconds = time.monotonic() - generate_start

        if callable(timing_callback):
            timing_callback({
                "batch_size": len(prompts),
                "prompt_tokens_total": prompt_tokens_total or 0,
                "completed_items": len(captures),
                "generate_seconds": generate_seconds,
                "postprocess_seconds": 0.0,
                "total_seconds": time.monotonic() - generate_start,
                "error": None,
            })

        return tuple(captures)

    def _capture_batched_single(
        self,
        prompt_token_ids: list[tuple[int, ...]],
        positions: list[int],
    ) -> list[HiddenStateCapture]:
        """Single-GPU path: ModelRunner orchestrates, no file I/O."""
        from vllm import SamplingParams as SP

        runner = self._get_runner()
        token_inputs = [
            {"prompt_token_ids": list(tids)}
            for tids in prompt_token_ids
        ]
        sizes = [len(t) for t in prompt_token_ids]

        runner._ipi_aware_prompt_sizes = sizes
        runner._ipi_aware_captured = None
        runner._ipi_aware_captured_req_indices = []

        self._llm.generate(token_inputs, SP(max_tokens=1))

        captured = runner._ipi_aware_captured
        req_indices = runner._ipi_aware_captured_req_indices
        runner._ipi_aware_captured = None
        runner._ipi_aware_captured_req_indices = []

        if captured is None:
            raise ValueError("No hidden states captured (single-GPU).")

        # Reorder rows to match submission order (chunked prefill may
        # capture in scheduling order, not submission order).
        if req_indices and len(req_indices) == captured.shape[0]:
            sort_order = sorted(range(len(req_indices)),
                                key=lambda i: req_indices[i])
            if sort_order != list(range(len(req_indices))):
                captured = captured[sort_order]

        return _tensor_to_captures(
            captured, prompt_token_ids, positions,
            layer_ids=list(self._resolved_layer_ids))

    def _capture_chat_messages(
        self,
        chat_prompts: Sequence[Any],
        positions: list[int],
    ) -> list[HiddenStateCapture]:
        """Chat-messages path: LLM.chat() handles tokenization and chat template."""
        from vllm import SamplingParams as SP

        runner = self._get_runner()
        runner._ipi_aware_captured = None
        runner._ipi_aware_captured_req_indices = []
        runner._ipi_aware_prompt_sizes = []  # sentinel: non-None to skip file-writing path

        conversations = [list(p.messages) for p in chat_prompts]
        tools = None
        for p in chat_prompts:
            if p.tools:
                tools = list(p.tools)
                break

        results = self._llm.chat(
            conversations,
            sampling_params=SP(max_tokens=1),
            add_generation_prompt=True,
            tools=tools,
            use_tqdm=False,
        )

        captured = runner._ipi_aware_captured
        req_indices = runner._ipi_aware_captured_req_indices
        runner._ipi_aware_captured = None
        runner._ipi_aware_captured_req_indices = []

        if captured is None:
            raise ValueError("No hidden states captured (chat-messages path).")

        if req_indices and len(req_indices) == captured.shape[0]:
            sort_order = sorted(range(len(req_indices)),
                                key=lambda i: req_indices[i])
            if sort_order != list(range(len(req_indices))):
                captured = captured[sort_order]
                results = [results[i] for i in sort_order]

        prompt_token_ids = [tuple(r.prompt_token_ids) for r in results]

        return _tensor_to_captures(
            captured, prompt_token_ids, positions,
            layer_ids=list(self._resolved_layer_ids))

    def _capture_batched_multi(
        self,
        prompt_token_ids: list[tuple[int, ...]],
        positions: list[int],
    ) -> list[HiddenStateCapture]:
        """Multi-GPU (TP>1) path: file-based communication via safetensors."""
        from vllm import SamplingParams as SP

        token_inputs = [
            {"prompt_token_ids": list(tids)}
            for tids in prompt_token_ids
        ]

        sizes_path = os.path.join(self._tmpdir, "ipi_aware_prompt_sizes.txt")
        with open(sizes_path, "w") as f:
            f.write(",".join(str(len(t)) for t in prompt_token_ids))
            f.flush()
            os.fsync(f.fileno())

        # Clean up previous capture files
        for fp in Path(self._tmpdir).glob("rank*.safetensors"):
            fp.unlink()

        # Signal workers to reset their _ipi_aware_captured accumulator.
        # Uses a monotonic batch ID so the worker detects batch changes
        # reliably even across multiple forward passes.
        batch_id_path = os.path.join(self._tmpdir, "ipi_aware_batch_id")
        self._batch_counter = getattr(self, "_batch_counter", 0) + 1
        with open(batch_id_path, "w") as f:
            f.write(str(self._batch_counter))
            f.flush()
            os.fsync(f.fileno())

        self._llm.generate(token_inputs, SP(max_tokens=1))

        capture_files = sorted(Path(self._tmpdir).glob("rank*.safetensors"))
        if not capture_files:
            raise ValueError("No hidden states captured (multi-GPU).")

        stacked, layer_ids = _read_capture_file(capture_files[0])
        return _tensor_to_captures(stacked, prompt_token_ids, positions, layer_ids)


def _read_capture_file(
    fp: Path,
) -> tuple[torch.Tensor, list[int] | None]:
    """Read safetensors capture from TP rank 0, reorder to prompt order."""
    import safetensors

    with safetensors.safe_open(fp, framework="pt") as t:
        stacked = t.get_tensor("hidden_states")
        layer_ids = t.get_tensor("layer_ids").tolist() if "layer_ids" in t.keys() else None
        ri = t.get_tensor("request_indices").tolist() if "request_indices" in t.keys() else None

    if ri is not None:
        # Reorder rows so they appear in prompt-index order.
        # Chunked prefill may capture rows in a different order than submission.
        sort_order = sorted(range(len(ri)), key=lambda i: ri[i])
        if sort_order != list(range(len(ri))):
            stacked = stacked[sort_order]

    return stacked, layer_ids


def _tensor_to_captures(
    stacked: torch.Tensor,
    batch_token_ids: list[tuple[int, ...]],
    positions: list[int],
    layer_ids: list[int] | None = None,
) -> list[HiddenStateCapture]:
    """Convert stacked capture tensor to per-prompt HiddenStateCapture list."""
    num_positions = len(positions)
    n_layers = stacked.shape[1]
    expected_rows = len(batch_token_ids) * num_positions

    if stacked.shape[0] != expected_rows:
        raise ValueError(
            f"Captured {stacked.shape[0]} rows but expected {expected_rows} "
            f"({len(batch_token_ids)} prompts × {num_positions} positions)"
        )

    captures: list[HiddenStateCapture] = []
    for j, token_ids in enumerate(batch_token_ids):
        prompt_stacked = stacked[j * num_positions:(j + 1) * num_positions]
        resolved_positions = _normalize_positions(positions, len(token_ids))

        layer_hidden = []
        layer_shapes = []
        for layer_idx in range(n_layers):
            layer_t = prompt_stacked[:, layer_idx, :]
            layer_hidden.append(layer_t.tolist())
            layer_shapes.append(tuple(layer_t.shape))

        captures.append(HiddenStateCapture(
            input_token_ids=token_ids,
            selected_positions=resolved_positions,
            layer_shapes=tuple(layer_shapes),
            hidden_states=tuple(layer_hidden),
            layer_ids=tuple(layer_ids) if layer_ids else None,
        ))

    return captures


def _normalize_positions(positions: Sequence[int], token_count: int) -> tuple[int, ...]:
    resolved = []
    for p in positions:
        abs_p = p if p >= 0 else token_count + p
        if abs_p < 0 or abs_p >= token_count:
            raise IndexError(f"position {p} out of range for {token_count} tokens")
        resolved.append(int(abs_p))
    return tuple(resolved)
