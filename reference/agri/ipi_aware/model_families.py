from __future__ import annotations

MODEL_FAMILY_POSITION_OFFSETS: dict[str, int] = {
    "qwen3": 0,
    "qwen3.5": -2,
    "gemma4": 0,
    "oai-oss": 0,
}

SUPPORTED_MODEL_FAMILIES: tuple[str, ...] = tuple(MODEL_FAMILY_POSITION_OFFSETS)

MODEL_FAMILY_NATIVE_GENERATION_TAILS: dict[str, str] = {
    "qwen3": "<|im_start|>assistant\n",
    "qwen3.5": "<|im_start|>assistant\n<think>\n",
    "gemma4": "<|turn>model\n<|channel>thought\n",
    "oai-oss": "<|start|>assistant",
}


def position_offset_for_model_family(model_family: str) -> int:
    try:
        return MODEL_FAMILY_POSITION_OFFSETS[model_family]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model family {model_family!r}; choose from {sorted(SUPPORTED_MODEL_FAMILIES)}"
        ) from exc


def native_generation_tail_for_model_family(model_family: str) -> str:
    try:
        return MODEL_FAMILY_NATIVE_GENERATION_TAILS[model_family]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model family {model_family!r}; choose from {sorted(SUPPORTED_MODEL_FAMILIES)}"
        ) from exc


def infer_model_family(*candidates: str | None) -> str:
    for candidate in candidates:
        normalized = str(candidate or "").strip().lower()
        if any(token in normalized for token in ("gpt-oss", "openai-oss", "oai-oss", "oss-20b")):
            return "oai-oss"
        if "gemma-4" in normalized or "gemma4" in normalized:
            return "gemma4"
        if "qwen3.6" in normalized or "qwen3-6" in normalized:
            return "default"
        if "qwen3.5" in normalized or "qwen3-5" in normalized:
            return "qwen3.5"
        if "qwen3" in normalized:
            return "qwen3"
    return "default"
