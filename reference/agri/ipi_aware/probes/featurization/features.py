from __future__ import annotations

import inspect
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .hidden_states import (
    HiddenStateCapture,
    capture_prefill_residual_stream,
    capture_prefill_residual_stream_batch,
    UnsupportedResidualStreamCaptureError,
)
from ..utils import default_layer_ids
from .direct_capture import DirectCaptureExtractor
from .gemma4_prompt_restore import (
    Gemma4PromptRestoreError,
    restore_gemma4_messages_from_prompt_token_ids,
)
from ipi_aware.message_content import normalize_chat_message
from ipi_aware.data_collection.schema import DecisionPointRecord


@dataclass(frozen=True, slots=True)
class ProbePrompt:
    text: str
    token_ids: tuple[int, ...]
    decode_start_index: int


@dataclass(frozen=True, slots=True)
class ChatMessagesPrompt:
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] | None
    decision_point_id: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ProbeExample:
    decision_point_id: str
    decision_point: DecisionPointRecord
    prompt: ProbePrompt | ChatMessagesPrompt
    hidden_state_capture: HiddenStateCapture
    metadata: dict[str, Any]


def build_probe_prompt(
    decision_point: DecisionPointRecord,
    *,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
    prefer_replay_request: bool = False,
) -> ProbePrompt:
    """Build ProbePrompt from a decision point.

    Uses stored prompt_token_ids directly when available (no tokenizer
    needed).  Falls back to loading a tokenizer and applying the chat
    template when prompt_token_ids is absent.
    """
    if decision_point.prompt_token_ids and not prefer_replay_request:
        tids = tuple(int(t) for t in decision_point.prompt_token_ids)
        return ProbePrompt(text="", token_ids=tids, decode_start_index=len(tids))
    loaded_tokenizer = tokenizer or _load_tokenizer(tokenizer_name_or_path)
    replay_request = _require_openai_chat_replay_request(decision_point)
    prompt_messages = _normalize_prompt_messages(replay_request.get("messages") or ())
    if not prompt_messages:
        raise ValueError(f"decision point {decision_point.decision_point_id} has no prompt messages")
    text = _apply_chat_template(
        loaded_tokenizer,
        prompt_messages,
        tools=replay_request.get("tools") or (),
    )
    token_ids = tuple(_encode_text(loaded_tokenizer, text))
    return ProbePrompt(text=text, token_ids=token_ids, decode_start_index=len(token_ids))


def build_chat_messages_prompt(
    decision_point: DecisionPointRecord,
    *,
    restore_from_prompt_token_ids: bool = False,
    source_tokenizer: Any | None = None,
) -> ChatMessagesPrompt:
    """Build ChatMessagesPrompt from a decision point's replay_request."""
    replay_request = _require_openai_chat_replay_request(decision_point)
    metadata: dict[str, Any] | None = None
    if restore_from_prompt_token_ids:
        try:
            restored = restore_gemma4_messages_from_prompt_token_ids(
                decision_point,
                tokenizer=source_tokenizer,
            )
            messages = tuple(_normalize_prompt_messages(restored.messages))
            metadata = {"prompt_restore_source": "restored_from_prompt_token_ids"}
        except Gemma4PromptRestoreError as exc:
            messages = tuple(_normalize_prompt_messages(replay_request.get("messages") or ()))
            metadata = {
                "prompt_restore_source": "replay_request_messages_fallback",
                "prompt_restore_error": str(exc),
            }
    else:
        messages = tuple(_normalize_prompt_messages(replay_request.get("messages") or ()))
    tools_raw = replay_request.get("tools")
    tools = tuple(tools_raw) if tools_raw else None
    return ChatMessagesPrompt(
        messages=messages,
        tools=tools,
        decision_point_id=decision_point.decision_point_id,
        metadata=metadata,
    )


# Legacy alias
build_qwen3_probe_prompt = build_probe_prompt


def describe_probe_prompt(
    decision_point: DecisionPointRecord,
    *,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
    selected_positions: Sequence[int] | None = None,
    preview_radius: int = 4,
) -> dict[str, Any]:
    loaded_tokenizer = tokenizer or _load_tokenizer(tokenizer_name_or_path)
    prompt = build_probe_prompt(
        decision_point,
        tokenizer=loaded_tokenizer,
    )
    # Compute text separately since build_probe_prompt skips it
    # when prompt_token_ids is available.
    replay_request = _require_openai_chat_replay_request(decision_point)
    prompt_messages = _normalize_prompt_messages(replay_request.get("messages") or ())
    if not prompt_messages:
        raise ValueError(f"decision point {decision_point.decision_point_id} has no prompt messages")
    prompt_text = prompt.text or _apply_chat_template(
        loaded_tokenizer,
        prompt_messages,
        tools=replay_request.get("tools") or (),
    )
    resolved_positions = _resolve_selected_positions(
        selected_positions or [-1],
        len(prompt.token_ids),
    )
    preview_indices: list[int] = []
    for position in resolved_positions:
        preview_indices.extend(
            range(max(0, position - preview_radius), min(len(prompt.token_ids), position + preview_radius + 1))
        )
    preview_indices = sorted(set(preview_indices))
    return {
        "decision_point_id": decision_point.decision_point_id,
        "normalized_messages": prompt_messages,
        "prompt_text": prompt_text,
        "prompt_token_ids": list(prompt.token_ids),
        "prompt_token_count": len(prompt.token_ids),
        "selected_positions": list(selected_positions or [-1]),
        "resolved_selected_positions": [
            {
                "selected_position": int(selected_position),
                "resolved_position": int(resolved_position),
                "token_id": int(prompt.token_ids[resolved_position]),
                "token_text": _decode_token_text(loaded_tokenizer, prompt.token_ids[resolved_position]),
            }
            for selected_position, resolved_position in zip(selected_positions or [-1], resolved_positions, strict=True)
        ],
        "token_preview": [
            {
                "position": int(index),
                "token_id": int(prompt.token_ids[index]),
                "token_text": _decode_token_text(loaded_tokenizer, prompt.token_ids[index]),
                "is_selected": bool(index in set(resolved_positions)),
            }
            for index in preview_indices
        ],
    }


def calibrate_qwen35_assistant_prefill_position(
    decision_point: DecisionPointRecord,
    *,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
) -> int:
    loaded_tokenizer = tokenizer or _load_tokenizer(tokenizer_name_or_path)
    replay_request = _require_openai_chat_replay_request(decision_point)
    prompt_messages = _normalize_prompt_messages(replay_request.get("messages") or ())
    if not prompt_messages:
        raise ValueError(f"decision point {decision_point.decision_point_id} has no prompt messages")
    if decision_point.prompt_token_ids is None:
        raise ValueError(
            f"decision point {decision_point.decision_point_id} is missing prompt_token_ids for qwen3.5 calibration"
        )

    stored_token_ids = tuple(int(token) for token in decision_point.prompt_token_ids)
    canonical_text = _apply_chat_template(
        loaded_tokenizer,
        prompt_messages,
        tools=replay_request.get("tools") or (),
    )
    canonical_token_ids = tuple(_encode_text(loaded_tokenizer, canonical_text))
    if not canonical_token_ids:
        raise ValueError(f"decision point {decision_point.decision_point_id} produced an empty canonical prompt")
    if len(canonical_token_ids) > len(stored_token_ids):
        raise ValueError(
            f"stored prompt_token_ids do not share the canonical qwen3.5 no-thinking prefix for {decision_point.decision_point_id}"
        )
    if stored_token_ids[: len(canonical_token_ids)] != canonical_token_ids:
        raise ValueError(
            f"stored prompt_token_ids do not share the canonical qwen3.5 no-thinking prefix for {decision_point.decision_point_id}"
        )
    assistant_absolute_position = len(canonical_token_ids) - 1
    return int(assistant_absolute_position - len(stored_token_ids))


def build_probe_examples_transformers_hook(
    decision_points: Sequence[DecisionPointRecord],
    *,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
    model: Any | None = None,
    model_name_or_path: str | None = None,
    selected_positions: Sequence[int] | None = None,
    batch_size: int = 1,
    error_callback: Callable[[DecisionPointRecord, Exception], None] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[ProbeExample, ...]:
    return _build_probe_examples_with_capture_functions(
        decision_points,
        tokenizer=tokenizer,
        tokenizer_name_or_path=tokenizer_name_or_path,
        model=model,
        model_name_or_path=model_name_or_path,
        selected_positions=selected_positions,
        batch_size=batch_size,
        error_callback=error_callback,
        progress_callback=progress_callback,
        capture_single=capture_prefill_residual_stream,
        capture_batch=capture_prefill_residual_stream_batch,
        fatal_capture_errors=(UnsupportedResidualStreamCaptureError,),
    )


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "oom" in msg
        or isinstance(exc, RuntimeError)
        and "cuda" in msg
        and "memory" in msg
    )


def _build_probe_examples_with_capture_functions(
    decision_points: Sequence[DecisionPointRecord],
    *,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
    model: Any | None = None,
    model_name_or_path: str | None = None,
    selected_positions: Sequence[int] | None = None,
    batch_size: int = 1,
    error_callback: Callable[[DecisionPointRecord, Exception], None] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    capture_single: Callable[..., HiddenStateCapture],
    capture_batch: Callable[..., tuple[HiddenStateCapture, ...]],
    fatal_capture_errors: tuple[type[Exception], ...],
) -> tuple[ProbeExample, ...]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    prepared_batches: list[tuple[DecisionPointRecord, ProbePrompt]] = []
    skipped_count = 0
    for decision_point in decision_points:
        try:
            prompt = build_probe_prompt(
                decision_point,
                tokenizer=tokenizer,
                tokenizer_name_or_path=tokenizer_name_or_path,
            )
        except Exception as exc:
            if callable(error_callback):
                error_callback(decision_point, exc)
            skipped_count += 1
            continue
        prepared_batches.append((decision_point, prompt))

    examples: list[ProbeExample] = []
    total_examples = len(prepared_batches)
    effective_batch_size = batch_size
    start = 0
    while start < len(prepared_batches):
        batch = prepared_batches[start : start + effective_batch_size]
        original_batch_size = len(batch)
        decision_point_batch = [item[0] for item in batch]
        prompt_batch = [item[1] for item in batch]
        successful_batch_items: list[tuple[DecisionPointRecord, ProbePrompt, HiddenStateCapture]] = []
        try:
            if len(prompt_batch) == 1:
                capture = capture_single(
                    prompt_batch[0],
                    model=model,
                    model_name_or_path=model_name_or_path,
                    selected_positions=selected_positions,
                )
                successful_batch_items.append(
                    (decision_point_batch[0], prompt_batch[0], capture)
                )
            else:
                capture_batch_results = capture_batch(
                    prompt_batch,
                    model=model,
                    model_name_or_path=model_name_or_path,
                    selected_positions=selected_positions,
                    tokenizer=tokenizer,
                )
                successful_batch_items.extend(
                    zip(decision_point_batch, prompt_batch, capture_batch_results, strict=True)
                )
        except fatal_capture_errors:
            raise
        except Exception as exc:
            import sys
            is_oom = _is_oom_error(exc)
            print(
                f"[featurize] batch of {len(prompt_batch)} failed ({type(exc).__name__}: {exc}), "
                f"{'OOM — ' if is_oom else ''}retrying with progressive halving",
                file=sys.stderr, flush=True,
            )
            if len(prompt_batch) == 1:
                if is_oom:
                    if callable(error_callback):
                        error_callback(decision_point_batch[0], exc)
                    skipped_count += 1
                    if callable(progress_callback):
                        progress_callback(len(examples), total_examples, skipped_count)
                    start += original_batch_size
                    continue
                raise

            # Progressive batch-size halving: split the failed batch into
            # smaller sub-batches (half, quarter, ...) until one succeeds.
            remaining = list(zip(decision_point_batch, prompt_batch, strict=True))
            sub_batch_size = max(1, len(prompt_batch) // 2)
            while remaining:
                chunk = remaining[:sub_batch_size]
                remaining = remaining[sub_batch_size:]
                chunk_dp = [item[0] for item in chunk]
                chunk_prompts = [item[1] for item in chunk]
                try:
                    if len(chunk_prompts) == 1:
                        capture = capture_single(
                            chunk_prompts[0],
                            model=model,
                            model_name_or_path=model_name_or_path,
                            selected_positions=selected_positions,
                        )
                        successful_batch_items.append(
                            (chunk_dp[0], chunk_prompts[0], capture)
                        )
                    else:
                        chunk_results = capture_batch(
                            chunk_prompts,
                            model=model,
                            model_name_or_path=model_name_or_path,
                            selected_positions=selected_positions,
                            tokenizer=tokenizer,
                        )
                        successful_batch_items.extend(
                            zip(chunk_dp, chunk_prompts, chunk_results, strict=True)
                        )
                    # First successful sub-batch size becomes the new cap
                    if sub_batch_size < effective_batch_size:
                        effective_batch_size = sub_batch_size
                        print(
                            f"[featurize] settled on effective_batch_size={effective_batch_size}",
                            file=sys.stderr, flush=True,
                        )
                except fatal_capture_errors:
                    raise
                except Exception as sub_exc:
                    if sub_batch_size <= 1 and _is_oom_error(sub_exc):
                        # OOM at size 1 — this prompt is too long, skip it
                        for dp in chunk_dp:
                            if callable(error_callback):
                                error_callback(dp, sub_exc)
                            skipped_count += 1
                        if callable(progress_callback):
                            progress_callback(len(examples), total_examples, skipped_count)
                        continue
                    if sub_batch_size <= 1:
                        raise
                    # Half again and re-queue
                    print(
                        f"[featurize] sub-batch of {len(chunk_prompts)} also failed, "
                        f"halving to {max(1, sub_batch_size // 2)}",
                        file=sys.stderr, flush=True,
                    )
                    remaining = list(zip(chunk_dp, chunk_prompts, strict=True)) + remaining
                    sub_batch_size = max(1, sub_batch_size // 2)

            if not successful_batch_items:
                if callable(progress_callback):
                    progress_callback(len(examples), total_examples, skipped_count)
                start += original_batch_size
                continue
        for decision_point, prompt, capture in successful_batch_items:
            examples.append(
                ProbeExample(
                    decision_point_id=decision_point.decision_point_id,
                    decision_point=decision_point,
                    prompt=prompt,
                    hidden_state_capture=capture,
                    metadata=_build_probe_example_metadata(decision_point),
                )
            )
        if callable(progress_callback):
            progress_callback(len(examples), total_examples, skipped_count)
        start += original_batch_size
    return tuple(examples)


def build_probe_examples_vllm_direct(
    decision_points: Sequence[DecisionPointRecord],
    *,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
    model_name_or_path: str,
    selected_positions: Sequence[int] | None = None,
    num_gpus: int = 1,
    gpu_memory_utilization: float = 0.92,
    enforce_eager: bool = False,
    enable_chunked_prefill: bool = False,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    layer_ids: Sequence[int] | None = None,
    batch_size: int = 0,
    error_callback: Callable[[DecisionPointRecord, Exception], None] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    timing_callback: Callable[[dict[str, Any]], None] | None = None,
    extractor: DirectCaptureExtractor | None = None,
    use_chat_messages: bool = False,
    restore_chat_messages_from_prompt_token_ids: bool = False,
    source_tokenizer: Any | None = None,
    rebuild_prompt_token_ids_from_replay: bool = False,
) -> tuple[ProbeExample, ...]:
    if layer_ids is None:
        layer_ids = default_layer_ids(model_name_or_path)

    prepared: list[tuple[DecisionPointRecord, ProbePrompt | ChatMessagesPrompt]] = []
    skipped_count = 0
    for decision_point in decision_points:
        try:
            if use_chat_messages:
                prompt = build_chat_messages_prompt(
                    decision_point,
                    restore_from_prompt_token_ids=restore_chat_messages_from_prompt_token_ids,
                    source_tokenizer=source_tokenizer,
                )
            else:
                prompt = build_probe_prompt(
                    decision_point,
                    tokenizer=tokenizer,
                    tokenizer_name_or_path=tokenizer_name_or_path,
                    prefer_replay_request=rebuild_prompt_token_ids_from_replay,
                )
        except Exception as exc:
            if callable(error_callback):
                error_callback(decision_point, exc)
            skipped_count += 1
            continue
        prepared.append((decision_point, prompt))

    total_examples = len(prepared)
    examples: list[ProbeExample] = []

    # Use provided extractor or create a temporary one
    if extractor is not None:
        _do_vllm_direct_capture(
            extractor=extractor,
            prepared=prepared,
            selected_positions=selected_positions,
            batch_size=batch_size,
            timing_callback=timing_callback,
            examples=examples,
            skipped_count=skipped_count,
            total_examples=total_examples,
            progress_callback=progress_callback,
        )
    else:
        with DirectCaptureExtractor(
            model_name_or_path=model_name_or_path,
            selected_positions=selected_positions,
            layer_ids=layer_ids,
            num_gpus=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_chunked_prefill=enable_chunked_prefill,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs or 256,
            max_num_batched_tokens=max_num_batched_tokens,
        ) as ext:
            _do_vllm_direct_capture(
                extractor=ext,
                prepared=prepared,
                selected_positions=selected_positions,
                batch_size=batch_size,
                timing_callback=timing_callback,
                examples=examples,
                skipped_count=skipped_count,
                total_examples=total_examples,
                progress_callback=progress_callback,
            )

    return tuple(examples)


def _do_vllm_direct_capture(
    *,
    extractor: DirectCaptureExtractor,
    prepared: list[tuple[DecisionPointRecord, ProbePrompt | ChatMessagesPrompt]],
    selected_positions: Sequence[int] | None,
    batch_size: int = 0,
    timing_callback: Callable[[dict[str, Any]], None] | None,
    examples: list[ProbeExample],
    skipped_count: int,
    total_examples: int,
    progress_callback: Callable[[int, int, int], None] | None,
) -> None:
    if batch_size < 1:
        # Single batch (original behavior)
        prompt_batch = [item[1] for item in prepared]
        capture_batch = extractor.capture(
            prompt_batch,
            selected_positions=selected_positions,
            timing_callback=timing_callback,
        )
        for (decision_point, prompt), capture in zip(prepared, capture_batch, strict=True):
            metadata = _build_probe_example_metadata(decision_point)
            if isinstance(prompt, ChatMessagesPrompt) and prompt.metadata:
                metadata.update(prompt.metadata)
            examples.append(
                ProbeExample(
                    decision_point_id=decision_point.decision_point_id,
                    decision_point=decision_point,
                    prompt=prompt,
                    hidden_state_capture=capture,
                    metadata=metadata,
                )
            )
        if callable(progress_callback):
            progress_callback(len(examples), total_examples, skipped_count)
        return

    # Chunked capture at the given batch_size
    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        prompt_batch = [item[1] for item in chunk]
        capture_batch = extractor.capture(
            prompt_batch,
            selected_positions=selected_positions,
            timing_callback=timing_callback,
        )
        for (decision_point, prompt), capture in zip(chunk, capture_batch, strict=True):
            metadata = _build_probe_example_metadata(decision_point)
            if isinstance(prompt, ChatMessagesPrompt) and prompt.metadata:
                metadata.update(prompt.metadata)
            examples.append(
                ProbeExample(
                    decision_point_id=decision_point.decision_point_id,
                    decision_point=decision_point,
                    prompt=prompt,
                    hidden_state_capture=capture,
                    metadata=metadata,
                )
            )
        if callable(progress_callback):
            progress_callback(len(examples), total_examples, skipped_count)


def write_probe_dataset(
    *,
    dataset_dir: str | Path,
    examples: Sequence[ProbeExample],
    backend_name: str,
    model_name_or_path: str,
) -> Path:
    dataset_root = Path(dataset_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    if _looks_like_grid_point_dir(dataset_root):
        feature_payload = _build_tensor_native_feature_payload(
            examples=examples,
            backend_name=backend_name,
            model_name_or_path=model_name_or_path,
            trace_root=dataset_root,
        )
        feature_path = dataset_root / "features.pt"
        torch = _import_torch()
        torch.save(feature_payload, feature_path)
        manifest = {
            "dataset_dir": str(dataset_root),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_examples": len(examples),
            "feature_format": "tensor_native",
            "storage_format": "tensor_native_v1",
            "format_version": 3,
            "feature_path": "features.pt",
            "backend": backend_name,
            "model_name_or_path": model_name_or_path,
        }
        dtype_summary = feature_payload.get("dtype_summary") or []
        if dtype_summary:
            manifest["feature_dtypes"] = dtype_summary
        (dataset_root / "feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        _update_dataset_manifest_feature_paths(dataset_root, feature_path="features.pt")
        return dataset_root
    feature_cells_dir = dataset_root / "feature_cells"
    feature_cells_dir.mkdir(parents=True, exist_ok=True)
    torch = _import_torch()
    grouped_examples = _group_examples_by_feature_cell(examples)
    cell_manifest: dict[str, dict[str, Any]] = {}
    dtype_summary: list[str] = []
    seen_dtypes: set[str] = set()
    total_examples = 0
    for cell_key, cell_examples in sorted(grouped_examples.items()):
        feature_payload = _build_tensor_native_feature_payload(
            examples=cell_examples,
            backend_name=backend_name,
            model_name_or_path=model_name_or_path,
            trace_root=dataset_root,
        )
        cell_path = feature_cells_dir / f"{cell_key}.pt"
        torch.save(feature_payload, cell_path)
        total_examples += len(cell_examples)
        for dtype_name in feature_payload["dtype_summary"]:
            if dtype_name not in seen_dtypes:
                dtype_summary.append(dtype_name)
                seen_dtypes.add(dtype_name)
        suite_name, system_prompt_key, attack_name = _split_feature_cell_key(cell_key)
        cell_manifest[cell_key] = {
            "feature_path": str(Path("feature_cells") / f"{cell_key}.pt"),
            "example_count": len(cell_examples),
            "suite_name": suite_name,
            "system_prompt_key": system_prompt_key,
            "attack_name": attack_name,
        }
    manifest = {
        "dataset_dir": str(dataset_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_examples": total_examples,
        "feature_format": "tensor_native_cells",
        "storage_format": "tensor_native_cells_v1",
        "format_version": 3,
        "feature_path": "feature_cells",
        "feature_cell_count": len(cell_manifest),
        "cells": cell_manifest,
        "backend": backend_name,
        "model_name_or_path": model_name_or_path,
    }
    if dtype_summary:
        manifest["feature_dtypes"] = dtype_summary
    (dataset_root / "feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _update_dataset_manifest_feature_paths(dataset_root)
    return dataset_root


def _group_examples_by_feature_cell(examples: Sequence[ProbeExample]) -> dict[str, list[ProbeExample]]:
    grouped: dict[str, list[ProbeExample]] = defaultdict(list)
    for example in examples:
        grouped[_feature_cell_key_from_metadata(example.metadata)].append(example)
    return dict(grouped)


def _feature_cell_key_from_metadata(metadata: Mapping[str, Any]) -> str:
    suite_name = str(metadata.get("suite_name") or "unknown")
    system_prompt_key = str(metadata.get("system_prompt_key") or "unknown")
    attack_name = metadata.get("attack_name")
    canonical_attack_name = "clean" if attack_name in (None, "") else str(attack_name)
    return f"{suite_name}__{system_prompt_key}__{canonical_attack_name}"


def _split_feature_cell_key(cell_key: str) -> tuple[str, str, str]:
    suite_name, system_prompt_key, attack_name = cell_key.split("__", 2)
    return suite_name, system_prompt_key, attack_name


def _update_dataset_manifest_feature_paths(dataset_root: Path, *, feature_path: str = "feature_cells") -> None:
    manifest_path = dataset_root / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "dataset_id": dataset_root.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    manifest["feature_path"] = feature_path
    manifest["feature_manifest_path"] = "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def load_decision_points_from_trace_artifact(
    path: str | Path,
    *,
    progress_callback: Any | None = None,
    max_points: int | None = None,
) -> tuple[DecisionPointRecord, ...]:
    input_path = Path(path)
    if input_path.is_dir() and (input_path / "grid_points").is_dir():
        raise ValueError(
            f"run root {input_path} contains grid_points; scan grid point subdirectories explicitly"
        )
    decision_points_path = input_path / "decision_points.jsonl" if input_path.is_dir() else input_path
    if decision_points_path.name != "decision_points.jsonl":
        raise ValueError(
            f"expected a trace run directory or decision_points.jsonl path, got {input_path}"
        )
    if not decision_points_path.exists():
        raise FileNotFoundError(f"decision_points.jsonl not found at {decision_points_path}")
    points: list[DecisionPointRecord] = []
    total_bytes = decision_points_path.stat().st_size
    completed_bytes = 0
    with decision_points_path.open(encoding="utf-8") as handle:
        for line in handle:
            completed_bytes += len(line.encode("utf-8"))
            if not line.strip():
                if callable(progress_callback):
                    progress_callback(completed_bytes, total_bytes, decision_points_path)
                continue
            points.append(DecisionPointRecord.from_dict(json.loads(line)))
            if callable(progress_callback):
                progress_callback(completed_bytes, total_bytes, decision_points_path)
            if max_points is not None and len(points) >= max_points:
                break
    return tuple(points)


def iter_grid_point_dirs(root: str | Path) -> tuple[Path, ...]:
    input_path = Path(root)
    grid_points_dir = input_path / "grid_points"
    if not grid_points_dir.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(grid_points_dir.iterdir())
        if path.is_dir() and (path / "decision_points.jsonl").exists()
    )


def _looks_like_grid_point_dir(path: Path) -> bool:
    return path.parent.name == "grid_points" and (path / "decision_points.jsonl").exists()


def _example_to_row(example: ProbeExample) -> dict[str, Any]:
    return {
        "decision_point_id": example.decision_point.decision_point_id,
        "trace_id": example.decision_point.trace_id,
        "prompt_text": example.prompt.text,
        "prompt_token_ids": list(example.prompt.token_ids),
        "selected_positions": list(example.hidden_state_capture.selected_positions),
        "layer_shapes": [list(shape) for shape in example.hidden_state_capture.layer_shapes],
        "layer_features": [
            _flatten_hidden_state_capture_layer(layer_hidden)
            for layer_hidden in example.hidden_state_capture.hidden_states
        ],
        "metadata": example.metadata,
    }


def _build_tensor_native_feature_payload(
    *,
    examples: Sequence[ProbeExample],
    backend_name: str,
    model_name_or_path: str,
    trace_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    dtype_summary: list[str] = []
    seen_dtypes: set[str] = set()
    for example_index, example in enumerate(examples):
        row, row_dtypes = _example_to_tensor_native_row(
            example,
            example_index=example_index,
            tensors=tensors,
        )
        rows.append(row)
        for dtype_name in row_dtypes:
            if dtype_name not in seen_dtypes:
                dtype_summary.append(dtype_name)
                seen_dtypes.add(dtype_name)
    return {
        "format_version": 2,
        "storage_format": "tensor_native",
        "backend": backend_name,
        "model_name_or_path": model_name_or_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace_dir": str(trace_root),
        "rows": rows,
        "tensors": tensors,
        "dtype_summary": dtype_summary,
    }


def _example_to_tensor_native_row(
    example: ProbeExample,
    *,
    example_index: int,
    tensors: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    layer_tensor_keys: list[str] = []
    layer_dtypes: list[str] = []
    for layer_index, layer_hidden in enumerate(example.hidden_state_capture.hidden_states):
        tensor = _coerce_hidden_state_to_cpu_tensor(layer_hidden)
        tensor_key = f"row-{example_index}-layer-{layer_index}"
        tensors[tensor_key] = tensor
        layer_tensor_keys.append(tensor_key)
        layer_dtypes.append(_describe_tensor_dtype(tensor))
    return (
        {
            "decision_point_id": example.decision_point.decision_point_id,
            "trace_id": example.decision_point.trace_id,
            "prompt_text": example.prompt.text,
            "prompt_token_ids": list(example.prompt.token_ids),
            "selected_positions": list(example.hidden_state_capture.selected_positions),
            "layer_shapes": [list(shape) for shape in example.hidden_state_capture.layer_shapes],
            "layer_tensor_keys": layer_tensor_keys,
            "layer_dtypes": layer_dtypes,
            "metadata": example.metadata,
        },
        layer_dtypes,
    )


def _coerce_hidden_state_to_cpu_tensor(value: Any) -> Any:
    torch = _import_torch()
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "shape"):
        return value.clone() if hasattr(value, "clone") and callable(value.clone) else value
    return torch.tensor(value)


def _describe_tensor_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Tensor-native feature saving requires torch in the current environment.") from exc
    return torch


def _build_probe_example_metadata(decision_point: DecisionPointRecord) -> dict[str, Any]:
    metadata = dict(decision_point.metadata)
    metadata.update(
        {
            "decision_point_id": decision_point.decision_point_id,
            "trace_id": decision_point.trace_id,
            "decision_index": decision_point.decision_index,
            "assistant_message_index": decision_point.assistant_message_index,
        }
    )
    return metadata


def _flatten_hidden_state_capture_layer(value: Any) -> list[float]:
    if hasattr(value, "detach") and callable(value.detach):
        return _flatten_hidden_state_capture_layer(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and callable(value.tolist):
        return _flatten_hidden_state_capture_layer(value.tolist())
    flattened: list[float] = []
    _flatten_hidden_state_capture_value(value, flattened)
    return flattened


def _flatten_hidden_state_capture_value(value: Any, output: list[float]) -> None:
    if isinstance(value, (int, float)):
        output.append(float(value))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _flatten_hidden_state_capture_value(item, output)
        return
    raise TypeError(f"unsupported hidden-state value: {type(value)!r}")


def _resolve_selected_positions(selected_positions: Sequence[int], token_count: int) -> tuple[int, ...]:
    resolved: list[int] = []
    for position in selected_positions:
        absolute = position if position >= 0 else token_count + position
        if absolute < 0 or absolute >= token_count:
            raise IndexError(f"selected position {position} is out of range for {token_count} tokens")
        resolved.append(int(absolute))
    return tuple(resolved)


def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert_ids_to_tokens):
        converted = convert_ids_to_tokens([int(token_id)])
        if isinstance(converted, Sequence) and converted:
            return str(converted[0])
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return str(decode([int(token_id)], skip_special_tokens=False))
        except Exception:
            try:
                return str(decode(int(token_id), skip_special_tokens=False))
            except Exception:
                return str(token_id)
    return str(token_id)


def _normalize_prompt_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_prompt_message(message) for message in messages]


def _require_openai_chat_replay_request(decision_point: DecisionPointRecord) -> Mapping[str, Any]:
    if decision_point.replay_request_kind != "openai_chat" or decision_point.replay_request is None:
        raise ValueError(
            f"decision point {decision_point.decision_point_id} is missing an exact openai_chat replay request"
        )
    return decision_point.replay_request


def _apply_chat_template(tokenizer: Any, messages: Sequence[Mapping[str, Any]], *, tools: Sequence[Any]) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template")
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        parameters = inspect.signature(apply_chat_template).parameters
    except Exception:
        parameters = {}
    if tools and "tools" in parameters:
        kwargs["tools"] = list(tools)
    return apply_chat_template(messages, **kwargs)


def _normalize_prompt_message(message: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_chat_message(message)
    tool_calls = _normalize_tool_calls(message.get("tool_calls"))
    if tool_calls:
        normalized["tool_calls"] = tool_calls

    for key in ("tool_call_id", "name"):
        value = _coerce_optional_text(message.get(key))
        if value is not None:
            normalized[key] = value
    return normalized


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)):
        return []

    normalized_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        normalized_call = _normalize_tool_call(tool_call)
        if normalized_call is not None:
            normalized_calls.append(normalized_call)
    return normalized_calls


def _normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, Mapping):
        return None

    function_payload = tool_call.get("function")
    function_name: str | None = None
    arguments: Any = None

    if isinstance(function_payload, Mapping):
        function_name = _coerce_optional_text(
            function_payload.get("name") or function_payload.get("function")
        )
        arguments = function_payload.get("arguments")
        if arguments is None:
            arguments = function_payload.get("args")
    else:
        function_name = _coerce_optional_text(function_payload)

    if function_name is None:
        function_name = _coerce_optional_text(tool_call.get("name"))
    if arguments is None:
        arguments = tool_call.get("arguments")
    if arguments is None:
        arguments = tool_call.get("args")
    if function_name is None:
        return None

    # Newer chat templates (e.g. Qwen3.5 in transformers 5.x) expect
    # tool_call.arguments to be a dict, not a JSON string. Parse when needed.
    resolved_arguments: Any = arguments
    if isinstance(resolved_arguments, str):
        try:
            import json as _json
            resolved_arguments = _json.loads(resolved_arguments)
        except (ValueError, TypeError):
            pass
    if not isinstance(resolved_arguments, Mapping):
        resolved_arguments = resolved_arguments if resolved_arguments is not None else {}

    normalized_call: dict[str, Any] = {
        "function": {
            "name": function_name,
            "arguments": resolved_arguments,
        }
    }
    tool_call_id = _coerce_optional_text(tool_call.get("id"))
    if tool_call_id is not None:
        normalized_call["id"] = tool_call_id
    return normalized_call


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping) and "input_ids" in encoded:
        return list(encoded["input_ids"])
    raise TypeError("tokenizer must provide encode() or __call__(...)->input_ids")


def _load_tokenizer(tokenizer_name_or_path: str | None) -> Any:
    if not tokenizer_name_or_path:
        raise ValueError("tokenizer_name_or_path is required when tokenizer is not provided")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
