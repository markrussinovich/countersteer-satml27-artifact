from __future__ import annotations

from ipi_aware.data_collection.token_accounting import token_unit_usage, unit_usage_from_counts


def test_token_unit_usage_weights_input_and_output_tokens_from_vllm_usage():
    usage = token_unit_usage(
        {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 9},
        }
    )

    assert usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "cached_tokens": 9,
        "input_token_unit_cost": 1,
        "output_token_unit_cost": 5,
        "input_token_units": 11,
        "output_token_units": 35,
        "total_token_units": 46,
    }


def test_unit_usage_from_counts_preserves_flat_cached_tokens():
    usage = unit_usage_from_counts(
        {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "cached_tokens": 9,
        }
    )

    assert usage["cached_tokens"] == 9
    assert usage["total_token_units"] == 46
