from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def normalize_chat_message(message: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise ValueError("each message must be a JSON object")
    role = str(message.get("role") or "")
    if not role:
        raise ValueError("message role is required")

    raw_content = message.get("content")
    normalized: dict[str, Any] = {
        "role": role,
        "content": _coerce_visible_content(raw_content),
    }
    tool_calls = _normalize_response_tool_calls(message.get("tool_calls"))
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    if role == "tool":
        tool_call_id = _coerce_optional_text(message.get("tool_call_id"))
        if tool_call_id is not None:
            normalized["tool_call_id"] = tool_call_id
        tool_name = _coerce_optional_text(message.get("name"))
        if tool_name is None:
            tool_call_payload = message.get("tool_call")
            if isinstance(tool_call_payload, Mapping):
                tool_name = _coerce_optional_text(
                    tool_call_payload.get("function") or tool_call_payload.get("name")
                )
        if tool_name is not None:
            normalized["name"] = tool_name

    reasoning_content = _coerce_optional_text(message.get("reasoning_content"))
    reasoning = _coerce_optional_text(message.get("reasoning"))

    if role == "assistant":
        inferred_reasoning = _extract_reasoning_from_content(raw_content)
        if reasoning_content is None and reasoning is None and inferred_reasoning is not None:
            reasoning_content = inferred_reasoning

        if reasoning_content is None and reasoning is None:
            inline_reasoning, visible_content = _parse_think_tag_output(
                _coerce_raw_content(raw_content)
            )
            if inline_reasoning is not None:
                reasoning_content = inline_reasoning
                normalized["content"] = visible_content or ""

    if reasoning_content is not None:
        normalized["reasoning_content"] = reasoning_content
    if reasoning is not None:
        normalized["reasoning"] = reasoning
    return normalized


def assistant_message_from_model_output(
    *,
    content: Any,
    tool_calls: Any,
    reasoning_content: Any = None,
    reasoning: Any = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": [],
        "tool_calls": tool_calls,
    }
    visible_content = _coerce_visible_content(content)
    if visible_content:
        message["content"] = [{"type": "text", "content": visible_content}]
    else:
        message["content"] = None

    normalized = normalize_chat_message(
        {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
            "reasoning": reasoning,
        }
    )
    if "reasoning_content" in normalized:
        message["reasoning_content"] = normalized["reasoning_content"]
    if "reasoning" in normalized:
        message["reasoning"] = normalized["reasoning"]
    if normalized["content"]:
        message["content"] = [{"type": "text", "content": normalized["content"]}]
    else:
        message["content"] = None
    return message


def assistant_message_from_completion_response(
    response_payload: Mapping[str, Any],
    *,
    fallback_message: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    message_payload = _extract_first_assistant_response_message(response_payload)
    if not isinstance(message_payload, Mapping):
        if fallback_message is None:
            raise ValueError("completion response payload does not contain an assistant message")
        return normalize_chat_message(fallback_message)

    tool_calls = _normalize_response_tool_calls(message_payload.get("tool_calls"))
    return assistant_message_from_model_output(
        content=message_payload.get("content"),
        tool_calls=tool_calls,
        reasoning_content=_lookup_value(message_payload, "reasoning_content"),
        reasoning=_lookup_value(message_payload, "reasoning"),
    )


def extract_response_reasoning_fields(message: Any) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for key in ("reasoning_content", "reasoning"):
        value = _lookup_value(message, key)
        coerced = _coerce_optional_text(value)
        if coerced is not None:
            extracted[key] = coerced
    return extracted


def _extract_first_assistant_response_message(
    response_payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for container_key in ("choices", "output", "outputs"):
        container = response_payload.get(container_key)
        if (
            not isinstance(container, Sequence)
            or isinstance(container, (str, bytes, bytearray))
            or not container
        ):
            continue
        first_item = container[0]
        if isinstance(first_item, Mapping) and isinstance(first_item.get("message"), Mapping):
            return first_item["message"]
        if isinstance(first_item, Mapping):
            return first_item
    return None


def _normalize_response_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)):
        return None
    normalized_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
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
            continue
        normalized_call: dict[str, Any] = {
            "function": {
                "name": function_name,
                "arguments": arguments if arguments is not None else {},
            }
        }
        tool_call_id = _coerce_optional_text(tool_call.get("id"))
        if tool_call_id is not None:
            normalized_call["id"] = tool_call_id
        normalized_calls.append(normalized_call)
    return normalized_calls or None


def _extract_reasoning_from_content(content: Any) -> str | None:
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") not in {"thinking", "redacted_thinking"}:
                continue
            text = item.get("content")
            if text:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return None


def _coerce_visible_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") in {"thinking", "redacted_thinking"}:
                    continue
                text = item.get("content")
                if text is None:
                    text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _coerce_raw_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("content")
                if text is None:
                    text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, Mapping):
        for key in ("content", "text", "reasoning_content", "reasoning"):
            if key in value:
                return _coerce_optional_text(value.get(key))
        return None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        parts = [_coerce_optional_text(item) for item in value]
        filtered = [part for part in parts if part]
        return "\n".join(filtered) or None
    return str(value).strip() or None


def _lookup_value(message: Any, key: str) -> Any:
    if isinstance(message, Mapping):
        if key in message:
            return message.get(key)
        model_extra = message.get("model_extra")
        if isinstance(model_extra, Mapping):
            return model_extra.get(key)
    return None


def _parse_think_tag_output(text: str) -> tuple[str | None, str | None]:
    start_token = "<think>"
    end_token = "</think>"
    if start_token not in text:
        content = text.strip()
        return None, content or None

    start_index = text.index(start_token) + len(start_token)
    if end_token not in text[start_index:]:
        reasoning = text[start_index:].strip()
        return reasoning or None, None

    end_index = text.index(end_token, start_index)
    reasoning = text[start_index:end_index].strip()
    content = text[end_index + len(end_token) :].strip()
    return reasoning or None, content or None
