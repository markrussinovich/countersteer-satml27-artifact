"""Token accounting helpers for collected model responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

INPUT_TOKEN_UNIT_COST = 1
OUTPUT_TOKEN_UNIT_COST = 5


def normalize_vllm_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Flatten vLLM usage, preserving cached prompt-token counts when present."""
    if not isinstance(usage, Mapping):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
    details = usage.get("prompt_tokens_details")
    cached_tokens = 0
    if isinstance(details, Mapping):
        cached_tokens = _int_value(details.get("cached_tokens"))
    if cached_tokens == 0:
        cached_tokens = _int_value(usage.get("cached_tokens"))
    prompt_tokens = _int_value(usage.get("prompt_tokens"))
    completion_tokens = _int_value(usage.get("completion_tokens"))
    total_tokens = _int_value(usage.get("total_tokens"))
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }


def token_unit_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Return unit-weighted token cost: input=1 unit, output=5 units."""
    normalized = normalize_vllm_usage(usage)
    input_units = normalized["prompt_tokens"] * INPUT_TOKEN_UNIT_COST
    output_units = normalized["completion_tokens"] * OUTPUT_TOKEN_UNIT_COST
    return {
        **normalized,
        "input_token_unit_cost": INPUT_TOKEN_UNIT_COST,
        "output_token_unit_cost": OUTPUT_TOKEN_UNIT_COST,
        "input_token_units": input_units,
        "output_token_units": output_units,
        "total_token_units": input_units + output_units,
    }


def unit_usage_from_counts(counts: Mapping[str, Any] | None) -> dict[str, int]:
    """Compute unit cost from already-flattened token counters."""
    normalized = normalize_vllm_usage(counts)
    return token_unit_usage(normalized)


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
