from __future__ import annotations

import pytest

from ipi_aware.model_families import (
    SUPPORTED_MODEL_FAMILIES,
    infer_model_family,
    position_offset_for_model_family,
)


def test_oai_oss_is_supported_with_no_position_offset():
    assert "oai-oss" in SUPPORTED_MODEL_FAMILIES
    assert position_offset_for_model_family("oai-oss") == 0


@pytest.mark.parametrize(
    ("model_family", "expected_offset"),
    [
        ("qwen3", 0),
        ("qwen3.5", -2),
        ("gemma4", 0),
    ],
)
def test_existing_model_family_position_offsets_are_preserved(model_family: str, expected_offset: int):
    assert position_offset_for_model_family(model_family) == expected_offset


def test_unknown_model_family_fails_closed():
    with pytest.raises(ValueError, match="Unsupported model family"):
        position_offset_for_model_family("unknown")


def test_qwen36_is_not_a_supported_model_family():
    assert "qwen3.6" not in SUPPORTED_MODEL_FAMILIES
    assert infer_model_family("Qwen/Qwen3.6-27B") == "default"
    with pytest.raises(ValueError, match="Unsupported model family"):
        position_offset_for_model_family("qwen3.6")


@pytest.mark.parametrize(
    ("candidate", "expected_family"),
    [
        ("Qwen/Qwen3-8B", "qwen3"),
        ("Qwen/Qwen3.5-9B", "qwen3.5"),
        ("google/gemma-4-31B-it", "gemma4"),
        ("openai/gpt-oss-20b", "oai-oss"),
    ],
)
def test_infer_model_family_detects_supported_formal_models(candidate: str, expected_family: str):
    assert infer_model_family(candidate) == expected_family
