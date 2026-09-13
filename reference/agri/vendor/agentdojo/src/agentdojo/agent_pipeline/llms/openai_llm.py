import json
from collections.abc import Mapping, Sequence
from typing import overload

import openai
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)
from ipi_aware.data_collection.agentdojo_runtime import (
    adjusted_max_completion_tokens,
    is_context_length_error,
    is_empty_assistant_output_payload,
    is_retryable_vllm_bad_request,
)
from ipi_aware.message_content import (
    assistant_message_from_model_output,
    extract_response_reasoning_fields,
)


def _extra_body_for_model(model: str) -> dict[str, object]:
    extra_body: dict[str, object] = {"return_token_ids": True}
    normalized = model.lower()
    if "qwen3" in normalized or "qwen3.5" in normalized:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    return extra_body


def _tool_call_to_openai(tool_call: FunctionCall) -> ChatCompletionMessageToolCallParam:
    if tool_call.id is None:
        raise ValueError("`tool_call.id` is required for OpenAI")
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type="function",
        function={
            "name": tool_call.function,
            "arguments": json.dumps(tool_call.args),
        },
    )


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatSystemMessage,
) -> list[ChatCompletionContentPartTextParam]: ...


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatAssistantMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None: ...


def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatAssistantMessage | ChatSystemMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None:
    if message["content"] is None:
        return None
    return [
        ChatCompletionContentPartTextParam(type="text", text=el["content"] or "")
        for el in message["content"]
        if el["type"] == "text"
    ]


def _message_to_openai(message: ChatMessage, model_name: str) -> ChatCompletionMessageParam:
    match message["role"]:
        case "system":
            return ChatCompletionSystemMessageParam(
                role="system", content=_content_blocks_to_openai_content_blocks(message)
            )
        case "user":
            return ChatCompletionUserMessageParam(
                role="user", content=_content_blocks_to_openai_content_blocks(message)
            )
        case "assistant":
            reasoning_fields = {}
            if "reasoning_content" in message and message["reasoning_content"] is not None:
                reasoning_fields["reasoning_content"] = message["reasoning_content"]
            if "reasoning" in message and message["reasoning"] is not None:
                reasoning_fields["reasoning"] = message["reasoning"]
            if message["tool_calls"] is not None and len(message["tool_calls"]) > 0:
                tool_calls = [_tool_call_to_openai(tool_call) for tool_call in message["tool_calls"]]
                return {
                    "role": "assistant",
                    "content": _content_blocks_to_openai_content_blocks(message),
                    "tool_calls": tool_calls,
                    **reasoning_fields,
                }
            return {
                "role": "assistant",
                "content": _content_blocks_to_openai_content_blocks(message),
                **reasoning_fields,
            }
        case "tool":
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` should be specified for OpenAI.")
            return ChatCompletionToolMessageParam(
                content=message["error"] or _content_blocks_to_openai_content_blocks(message),
                tool_call_id=message["tool_call_id"],
                role="tool",
                name=message["tool_call"].function,  # type: ignore -- this is actually used, and is important!
            )
        case _:
            raise ValueError(f"Invalid message type: {message}")


def _openai_to_tool_call(tool_call: ChatCompletionMessageToolCall) -> FunctionCall:
    return FunctionCall(
        function=tool_call.function.name,
        args=json.loads(tool_call.function.arguments),
        id=tool_call.id,
    )


def _openai_to_assistant_message(message: ChatCompletionMessage) -> ChatAssistantMessage:
    if message.tool_calls is not None:
        tool_calls = [_openai_to_tool_call(tool_call) for tool_call in message.tool_calls]
    else:
        tool_calls = None
    reasoning_fields = extract_response_reasoning_fields(message)
    assistant_message = assistant_message_from_model_output(
        content=message.content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_fields.get("reasoning_content"),
        reasoning=reasoning_fields.get("reasoning"),
    )
    return assistant_message  # type: ignore[return-value]


def _function_to_openai(f: Function) -> ChatCompletionToolParam:
    function_definition = FunctionDefinition(
        name=f.name,
        description=f.description,
        parameters=f.parameters.model_json_schema(),
    )
    return ChatCompletionToolParam(type="function", function=function_definition)


class EmptyAssistantOutputError(RuntimeError):
    pass


def _should_retry_completion_exception(exc: BaseException) -> bool:
    if isinstance(exc, openai.UnprocessableEntityError):
        return False
    if isinstance(exc, openai.BadRequestError):
        return is_retryable_vllm_bad_request(exc)
    return True


@retry(
    wait=wait_random_exponential(multiplier=0.1, max=1),
    stop=stop_after_attempt(4),
    reraise=True,
    retry=retry_if_exception(_should_retry_completion_exception),
)
def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam],
    reasoning_effort: ChatCompletionReasoningEffort | None,
    temperature: float | None = 0.0,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
    allow_empty_output: bool = False,
    extra_body_updates: Mapping[str, object] | None = None,
):
    extra_body = _extra_body_for_model(model)
    if extra_body_updates:
        extra_body.update(dict(extra_body_updates))
    effective_max_completion_tokens = max_completion_tokens
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools or NOT_GIVEN,
                tool_choice="auto" if tools else NOT_GIVEN,
                extra_body=extra_body,
                temperature=temperature if temperature is not None else NOT_GIVEN,
                top_p=top_p if top_p is not None else NOT_GIVEN,
                max_completion_tokens=(
                    effective_max_completion_tokens if effective_max_completion_tokens is not None else NOT_GIVEN
                ),
                reasoning_effort=reasoning_effort or NOT_GIVEN,
            )
        except openai.BadRequestError as exc:
            if not is_context_length_error(exc):
                raise
            adjusted = adjusted_max_completion_tokens(exc, effective_max_completion_tokens)
            if adjusted is None or adjusted >= (effective_max_completion_tokens or 0):
                raise
            effective_max_completion_tokens = adjusted
            continue
        break
    if is_empty_assistant_output_payload(completion.choices[0].message) and not allow_empty_output:
        raise EmptyAssistantOutputError("Local vLLM returned an empty assistant output")
    return completion


class OpenAILLM(BasePipelineElement):
    """LLM pipeline element that uses OpenAI's API.

    Args:
        client: The OpenAI client.
        model: The model name.
        temperature: The temperature to use for generation.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        reasoning_effort: ChatCompletionReasoningEffort | None = None,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        max_completion_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort: ChatCompletionReasoningEffort | None = reasoning_effort

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        completion = chat_completion_request(
            self.client,
            self.model,
            openai_messages,
            openai_tools,
            self.reasoning_effort,
            self.temperature,
            self.top_p,
            self.max_completion_tokens,
        )
        output = _openai_to_assistant_message(completion.choices[0].message)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args


class OpenAILLMToolFilter(BasePipelineElement):
    def __init__(self, prompt: str, client: openai.OpenAI, model: str, temperature: float | None = 0.0) -> None:
        self.prompt = prompt
        self.client = client
        self.model = model
        self.temperature = temperature

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        messages = [*messages, ChatUserMessage(role="user", content=[text_content_block_from_string(self.prompt)])]
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=openai_tools or NOT_GIVEN,
            tool_choice="none",
            temperature=self.temperature,
        )
        output = _openai_to_assistant_message(completion.choices[0].message)

        new_tools = {}
        for tool_name, tool in runtime.functions.items():
            if output["content"] is not None and tool_name in get_text_content_as_str(output["content"]):
                new_tools[tool_name] = tool

        runtime.update_functions(new_tools)

        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
