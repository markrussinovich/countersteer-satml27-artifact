import importlib

import pytest


def load_message_content_module():
    try:
        return importlib.import_module("ipi_aware.message_content")
    except ModuleNotFoundError as exc:
        pytest.fail(f"message content helper is missing: {exc}")


def test_normalize_chat_message_preserves_top_level_reasoning_content():
    module = load_message_content_module()

    normalized = module.normalize_chat_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "content": "I should look at the email first."}],
            "reasoning_content": "check the safety constraints",
        }
    )

    assert normalized == {
        "role": "assistant",
        "content": "I should look at the email first.",
        "reasoning_content": "check the safety constraints",
    }


def test_normalize_chat_message_preserves_top_level_reasoning_field():
    module = load_message_content_module()

    normalized = module.normalize_chat_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "content": "Here is the final answer."}],
            "reasoning": "double-check the calendar details",
        }
    )

    assert normalized == {
        "role": "assistant",
        "content": "Here is the final answer.",
        "reasoning": "double-check the calendar details",
    }


def test_normalize_chat_message_converts_inline_qwen_reasoning_to_reasoning_content():
    module = load_message_content_module()

    normalized = module.normalize_chat_message(
        {
            "role": "assistant",
            "content": "<think>\nNeed to verify the sender.\n</think>\nI will open the email.",
        }
    )

    assert normalized == {
        "role": "assistant",
        "content": "I will open the email.",
        "reasoning_content": "Need to verify the sender.",
    }


def test_normalize_chat_message_preserves_assistant_tool_calls():
    module = load_message_content_module()

    normalized = module.normalize_chat_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "read_file", "arguments": {"file_path": "a.txt"}},
                }
            ],
        }
    )

    assert normalized == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "read_file", "arguments": {"file_path": "a.txt"}},
            }
        ],
    }


def test_normalize_chat_message_preserves_tool_message_linkage():
    module = load_message_content_module()

    normalized = module.normalize_chat_message(
        {
            "role": "tool",
            "content": [{"type": "text", "content": "file body"}],
            "tool_call_id": "call-1",
            "tool_call": {"function": "read_file"},
        }
    )

    assert normalized == {
        "role": "tool",
        "content": "file body",
        "tool_call_id": "call-1",
        "name": "read_file",
    }


def test_assistant_message_from_model_output_uses_explicit_reasoning_field():
    module = load_message_content_module()

    message = module.assistant_message_from_model_output(
        content="Tool call next.",
        reasoning_content="Need to inspect the inbox first.",
        tool_calls=["call-1"],
    )

    assert message == {
        "role": "assistant",
        "content": [{"type": "text", "content": "Tool call next."}],
        "reasoning_content": "Need to inspect the inbox first.",
        "tool_calls": ["call-1"],
    }


def test_assistant_message_from_model_output_preserves_reasoning_field_name():
    module = load_message_content_module()

    message = module.assistant_message_from_model_output(
        content="I will open the email.",
        reasoning="Need to verify the sender.",
        tool_calls=None,
    )

    assert message == {
        "role": "assistant",
        "content": [{"type": "text", "content": "I will open the email."}],
        "reasoning": "Need to verify the sender.",
        "tool_calls": None,
    }


def test_assistant_message_from_completion_response_uses_raw_reasoning_payload():
    module = load_message_content_module()

    message = module.assistant_message_from_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning": "Check the date first.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "get_current_day", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
    )

    assert message == {
        "role": "assistant",
        "content": None,
        "reasoning": "Check the date first.",
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "get_current_day", "arguments": "{}"},
            }
        ],
    }
