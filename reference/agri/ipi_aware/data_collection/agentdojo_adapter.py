from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ipi_aware.message_content import normalize_chat_message


@dataclass(frozen=True, slots=True)
class AgentDojoPromptCase:
    suite_name: str | None
    task_id: str
    messages: tuple[dict[str, Any], ...]


def build_agentdojo_prompt_case(
    user_task: Any,
    *,
    suite_name: str | None = None,
    system_prompt: str | None = None,
    prior_messages: Sequence[Mapping[str, Any]] | None = None,
) -> AgentDojoPromptCase:
    task_id = str(getattr(user_task, "ID", "") or "")
    prompt = str(getattr(user_task, "PROMPT", "") or "")
    if not task_id:
        raise ValueError("user_task.ID is required")
    if not prompt:
        raise ValueError("user_task.PROMPT is required")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in prior_messages or ():
        messages.append(_normalize_message(message))
    messages.append({"role": "user", "content": prompt})
    return AgentDojoPromptCase(
        suite_name=suite_name,
        task_id=task_id,
        messages=tuple(messages),
    )


def _normalize_message(message: Mapping[str, Any]) -> dict[str, str]:
    return normalize_chat_message(message)
