from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ipi_aware.data_collection.schema import DecisionPointRecord


class Gemma4PromptRestoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RestoredGemma4Prompt:
    messages: tuple[dict[str, Any], ...]
    prompt_text: str


def restore_gemma4_messages_from_prompt_token_ids(
    decision_point: DecisionPointRecord,
    *,
    tokenizer: Any | None = None,
) -> RestoredGemma4Prompt:
    if decision_point.prompt_token_ids is None:
        raise Gemma4PromptRestoreError(
            f"decision point {decision_point.decision_point_id} is missing prompt_token_ids"
        )
    if tokenizer is None:
        tokenizer = _load_source_tokenizer(decision_point)
    prompt_text = _decode_prompt_text(tokenizer, decision_point.prompt_token_ids)
    messages = restore_gemma4_messages_from_prompt_text(prompt_text)
    return RestoredGemma4Prompt(messages=messages, prompt_text=prompt_text)


def restore_gemma4_messages_from_prompt_text(prompt_text: str) -> tuple[dict[str, Any], ...]:
    text = prompt_text.strip()
    text = _strip_bos(text)

    messages: list[dict[str, Any]] = []
    system_text, text = _parse_system_block(text)
    if system_text is not None:
        messages.append(_text_message("system", system_text))

    while text:
        text = text.lstrip()
        if not text:
            break
        if text.startswith("<|turn>user\n"):
            content, text = _consume_turn(text, "user")
            messages.append(_text_message("user", content))
            continue
        if text.startswith("<|turn>model"):
            if _is_terminal_model_generation_boundary(text):
                break
            restored, text = _consume_model_turn(text)
            messages.extend(restored)
            continue
        raise Gemma4PromptRestoreError(f"unsupported Gemma4 prompt continuation: {text[:120]!r}")

    return tuple(messages)


def _parse_system_block(text: str) -> tuple[str | None, str]:
    if not text.startswith("<|turn>system\n"):
        return None, text
    end_idx = text.find("<turn|>\n")
    if end_idx < 0:
        raise Gemma4PromptRestoreError("system block is missing <turn|> terminator")
    body = text[len("<|turn>system\n") : end_idx]
    rest = text[end_idx + len("<turn|>\n") :]
    if body.startswith("<|think|>\n"):
        body = body[len("<|think|>\n") :]
    tool_idx = body.find("<|tool>")
    if tool_idx >= 0:
        body = body[:tool_idx]
    content = body.strip()
    return content, rest


def _consume_turn(text: str, role: str) -> tuple[str, str]:
    prefix = f"<|turn>{role}\n"
    if not text.startswith(prefix):
        raise Gemma4PromptRestoreError(f"expected {prefix!r}")
    body = text[len(prefix) :]
    end_idx = body.find("<turn|>\n")
    if end_idx < 0:
        raise Gemma4PromptRestoreError(f"{role} turn is missing <turn|> terminator")
    content = body[:end_idx].strip()
    rest = body[end_idx + len("<turn|>\n") :]
    return content, rest


def _consume_model_turn(text: str) -> tuple[list[dict[str, Any]], str]:
    if not text.startswith("<|turn>model\n"):
        raise Gemma4PromptRestoreError("expected <|turn>model prefix")
    cursor = len("<|turn>model\n")
    reasoning = None
    if text.startswith("<|channel>thought\n", cursor):
        thought_start = cursor + len("<|channel>thought\n")
        thought_end = text.find("<channel|>", thought_start)
        if thought_end < 0:
            raise Gemma4PromptRestoreError("unterminated Gemma4 thought channel")
        reasoning = text[thought_start:thought_end].rstrip("\n")
        cursor = thought_end + len("<channel|>")

    tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    tool_call_index = 0
    while text.startswith("<|tool_call>call:", cursor):
        payload_end = text.find("<tool_call|>", cursor)
        if payload_end < 0:
            raise Gemma4PromptRestoreError("unterminated Gemma4 tool_call block")
        payload = text[cursor + len("<|tool_call>call:") : payload_end]
        name, arguments = _parse_named_braced_payload(payload, "tool_call")
        tool_call_id = f"restored-tool-call-{tool_call_index}"
        tool_calls.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
        cursor = payload_end + len("<tool_call|>")
        tool_call_index += 1

    response_index = 0
    while text.startswith("<|tool_response>response:", cursor):
        payload_end = text.find("<tool_response|>", cursor)
        if payload_end < 0:
            raise Gemma4PromptRestoreError("unterminated Gemma4 tool_response block")
        payload = text[cursor + len("<|tool_response>response:") : payload_end]
        name, response_body = _parse_named_braced_payload(payload, "tool_response")
        tool_messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": tool_calls[response_index]["id"] if response_index < len(tool_calls) else None,
                "content": _decode_tool_response_body(response_body),
            }
        )
        cursor = payload_end + len("<tool_response|>")
        response_index += 1

    content = None
    if text.startswith("<turn|>\n", cursor):
        cursor += len("<turn|>\n")
    elif cursor < len(text):
        # Current strict heldout traces frequently stop on the tool_response boundary.
        # If visible text remains, treat it as assistant content and require a normal turn end.
        content_end = text.find("<turn|>\n", cursor)
        if content_end >= 0:
            remaining = text[cursor:content_end].strip()
            content = remaining or None
            cursor = content_end + len("<turn|>\n")
        else:
            remaining = text[cursor:].strip()
            if remaining:
                content = remaining
            cursor = len(text)

    assistant_message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": tool_calls or None}
    if reasoning:
        assistant_message["reasoning"] = reasoning
    if content:
        assistant_message["content"] = [{"type": "text", "text": content}]
    elif not tool_calls:
        raise Gemma4PromptRestoreError("model turn without tool_calls or visible content is unsupported")

    restored: list[dict[str, Any]] = [assistant_message]
    restored.extend(tool_messages)
    return restored, text[cursor:]


def _is_terminal_model_generation_boundary(text: str) -> bool:
    if not text.startswith("<|turn>model"):
        return False
    if text.startswith("<|turn>model\n"):
        rest = text[len("<|turn>model\n") :].strip()
    else:
        rest = text[len("<|turn>model") :].strip()
    return not rest


def _parse_named_braced_payload(payload: str, label: str) -> tuple[str, str]:
    brace_idx = payload.find("{")
    if brace_idx < 0 or not payload.endswith("}"):
        raise Gemma4PromptRestoreError(f"malformed Gemma4 {label} payload: {payload[:120]!r}")
    name = payload[:brace_idx].strip()
    if not name:
        raise Gemma4PromptRestoreError(f"missing function name in Gemma4 {label} payload")
    body = payload[brace_idx:]
    return name, body


def _decode_tool_response_body(body: str) -> str:
    if body.startswith("{") and body.endswith("}"):
        inner = body[1:-1]
    else:
        inner = body
    if inner.startswith("value:"):
        inner = inner[len("value:") :]
    inner = inner.strip()
    if inner.startswith('<|"|>') and inner.endswith('<|"|>'):
        return inner[len('<|"|>') : -len('<|"|>')]
    return inner.replace('<|"|>', '"').strip()


def _text_message(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": content}]}


def _strip_bos(text: str) -> str:
    for token in ("<bos>", "<s>"):
        if text.startswith(token):
            return text[len(token) :]
    return text


@lru_cache(maxsize=4)
def _load_tokenizer_cached(name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)


def _load_source_tokenizer(decision_point: DecisionPointRecord) -> Any:
    replay_request = decision_point.replay_request or {}
    model_name_or_path = replay_request.get("model")
    if not isinstance(model_name_or_path, str) or not model_name_or_path:
        raise Gemma4PromptRestoreError("Gemma4 source model path is missing from replay_request.model")
    return _load_tokenizer_cached(model_name_or_path)


def _decode_prompt_text(tokenizer: Any, prompt_token_ids: tuple[int, ...] | list[int]) -> str:
    try:
        return tokenizer.decode(
            list(prompt_token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(prompt_token_ids), skip_special_tokens=False)
