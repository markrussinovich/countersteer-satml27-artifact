from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ipi_aware.data_collection.agentdojo_parallel import (
        AgentDojoSampleWorkItem,
        AgentDojoSampleWorkResult,
    )

_RETRYABLE_VLLM_BAD_REQUEST_PATTERNS = (
    "already borrowed",
    "failed to acquire tokenizer in current thread",
)
_CONTEXT_LENGTH_PATTERNS = (
    "context length",
    "maximum input length",
    "reduce the length of the messages",
)
_CONTEXT_LENGTH_PARAMS = {
    "max_tokens",
    "max_completion_tokens",
    "input_tokens",
    "max_new_tokens",
}
_CONTEXT_RETRY_BUFFER_TOKENS = 64


def is_retryable_vllm_bad_request(error: Any) -> bool:
    text = _flatten_error_text(_extract_error_payload(error)).lower()
    if any(pattern in text for pattern in _RETRYABLE_VLLM_BAD_REQUEST_PATTERNS):
        return True
    return "tool parser creation" in text and "tokenizer" in text


def is_context_length_error(error: Any) -> bool:
    payload = _extract_error_payload(error)
    text = _flatten_error_text(payload).lower()
    param = _extract_error_field(payload, "param")
    code = _extract_error_field(payload, "code")
    if str(code).lower() == "context_length_exceeded":
        return True
    if param is not None and str(param).lower() in _CONTEXT_LENGTH_PARAMS:
        return True
    return any(pattern in text for pattern in _CONTEXT_LENGTH_PATTERNS)


def is_empty_assistant_output_error(error: Any) -> bool:
    if type(error).__name__ == "EmptyAssistantOutputError":
        return True
    text = _flatten_error_text(_extract_error_payload(error)).lower()
    return "empty assistant output" in text


def adjusted_max_completion_tokens(error: Any, current_max_completion_tokens: int | None) -> int | None:
    return None


def build_failed_sample_result(work_item: AgentDojoSampleWorkItem, error: Any) -> AgentDojoSampleWorkResult:
    from ipi_aware.data_collection.agentdojo_parallel import AgentDojoSampleWorkResult

    error_text = _flatten_error_text(_extract_error_payload(error)).strip() or str(error)
    return AgentDojoSampleWorkResult(
        kind=work_item.kind,
        suite_name=work_item.suite_name,
        user_task_id=work_item.user_task_id,
        injection_task_id=work_item.injection_task_id,
        utility=False,
        security=True,
        error=error_text,
        failed=True,
    )


def format_sample_progress_message(completed: int, total: int, latest_label: str) -> str:
    if total <= 0:
        percentage = 100.0
    else:
        percentage = (completed / total) * 100.0
    return f"Progress: {completed}/{total} finished ({percentage:.1f}%) latest={latest_label}"


def is_empty_assistant_output_payload(payload: Any) -> bool:
    content = _lookup_field(payload, "content")
    tool_calls = _lookup_field(payload, "tool_calls")
    reasoning_content = _lookup_field(payload, "reasoning_content")
    reasoning = _lookup_field(payload, "reasoning")

    has_visible_content = bool(_flatten_error_text(content).strip())
    has_tool_calls = bool(tool_calls)
    has_reasoning = bool(_flatten_error_text(reasoning_content).strip() or _flatten_error_text(reasoning).strip())
    return not has_visible_content and not has_tool_calls and not has_reasoning


def _extract_error_payload(error: Any) -> Any:
    for attr in ("body", "message"):
        value = getattr(error, attr, None)
        if value:
            return value
    return error


def _lookup_field(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        if key in payload:
            return payload.get(key)
        nested_error = payload.get("error")
        if isinstance(nested_error, Mapping) and key in nested_error:
            return nested_error.get(key)
        model_extra = payload.get("model_extra")
        if isinstance(model_extra, Mapping):
            return model_extra.get(key)
        return None
    value = getattr(payload, key, None)
    if value is not None:
        return value
    model_extra = getattr(payload, "model_extra", None)
    if isinstance(model_extra, Mapping):
        return model_extra.get(key)
    return None


def _flatten_error_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_flatten_error_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_flatten_error_text(item) for item in value)
    return str(value)


def _extract_error_field(payload: Any, key: str) -> Any:
    value = _lookup_field(payload, key)
    if value is not None:
        return value
    if hasattr(payload, key):
        return getattr(payload, key)
    return None


def _search_groups(text: str, pattern: str) -> tuple[str, ...] | None:
    match = re.search(pattern, text)
    if match is None:
        return None
    return match.groups()


def _first_int(groups: tuple[str, ...] | None) -> int | None:
    if not groups:
        return None
    try:
        return int(groups[0])
    except (TypeError, ValueError):
        return None
