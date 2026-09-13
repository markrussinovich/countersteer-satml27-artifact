from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency in lightweight test envs
    yaml = None


def substitute_injection_placeholders(value: Any, injections: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        updated = value
        for key, injection in injections.items():
            updated = updated.replace(f"{{{key}}}", injection)
        return updated
    if isinstance(value, Mapping):
        return {
            key: substitute_injection_placeholders(item, injections)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [substitute_injection_placeholders(item, injections) for item in value]
    if isinstance(value, tuple):
        return tuple(substitute_injection_placeholders(item, injections) for item in value)
    return value


def extract_injection_strings(*containers: Any) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for container in containers:
        for value in _extract_strings(container):
            if not value or value in seen:
                continue
            seen.add(value)
            values.append(value)
    return tuple(values)


def extract_visible_message_text(message: Mapping[str, Any]) -> str:
    return _coerce_visible_text(message.get("content"))


def extract_visible_message_candidates(message: Mapping[str, Any]) -> tuple[str, ...]:
    visible_text = extract_visible_message_text(message)
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: Any) -> None:
        text = _coerce_visible_text(value)
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    add_candidate(visible_text)
    if visible_text:
        for parsed in _parse_serialized_text_candidates(visible_text):
            for leaf in _extract_strings(parsed):
                add_candidate(leaf)
    return tuple(candidates)


def normalize_injection_match_text(value: Any) -> str:
    text = _coerce_visible_text(value)
    if not text:
        return ""
    # YAML quoted scalars wrap long strings using backslash continuations and
    # indentation, which makes visible tool output differ from the underlying
    # injected text.
    text = re.sub(r"\\\s*\n\s*", "", text)
    text = re.sub(r"\\+\s+", " ", text)
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\r", "\n")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def _extract_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_extract_strings(item))
        return strings
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        strings: list[str] = []
        for item in value:
            strings.extend(_extract_strings(item))
        return strings
    return []


def _coerce_visible_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
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
    return str(value)


def _parse_serialized_text_candidates(text: str) -> tuple[Any, ...]:
    parsed_values: list[Any] = []
    for parser in (_parse_json_text, _parse_yaml_text):
        parsed = parser(text)
        if parsed is not None:
            parsed_values.append(parsed)
    return tuple(parsed_values)


def _parse_json_text(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _parse_yaml_text(text: str) -> Any | None:
    stripped = text.strip()
    if yaml is None or not stripped or ":" not in stripped:
        return None
    try:
        return yaml.safe_load(stripped)
    except Exception:
        return None
