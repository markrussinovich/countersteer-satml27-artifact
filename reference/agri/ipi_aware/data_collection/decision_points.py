from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ipi_aware.message_content import assistant_message_from_completion_response

from .schema import DecisionPointRecord, RunTrace, to_jsonable


def extract_decision_points(trace: RunTrace) -> tuple[DecisionPointRecord, ...]:
    extracted: list[DecisionPointRecord] = []
    trace_id = trace.metadata.trace_id
    model_requests = list(trace.model_requests)
    assistant_request_index = 0
    for assistant_message_index, message in enumerate(trace.final_messages):
        if message.get("role") != "assistant":
            continue
        replay_request_kind = None
        replay_request = None
        assistant_response_payload = None
        prompt_token_ids = None
        response_token_ids = None
        if assistant_request_index < len(model_requests):
            replay_request_kind = model_requests[assistant_request_index].request_kind
            replay_request = model_requests[assistant_request_index].request_payload
            assistant_response_payload = model_requests[assistant_request_index].response_payload
            prompt_token_ids = model_requests[assistant_request_index].prompt_token_ids
            response_token_ids = model_requests[assistant_request_index].response_token_ids
        assistant_request_index += 1
        extracted.append(
            DecisionPointRecord(
                decision_index=len(extracted),
                decision_point_id=f"{trace_id}:assistant:{assistant_message_index}",
                trace_id=trace_id,
                assistant_message=_build_assistant_message(
                    assistant_response_payload=assistant_response_payload,
                    fallback_message=message,
                ),
                assistant_response_payload=assistant_response_payload,
                assistant_message_index=assistant_message_index,
                replay_request_kind=replay_request_kind,
                replay_request=replay_request,
                prompt_token_ids=prompt_token_ids,
                response_token_ids=response_token_ids,
                metadata=_build_metadata(trace),
            )
        )
    return tuple(extracted)


def _build_assistant_message(
    *,
    assistant_response_payload: Mapping[str, Any] | None,
    fallback_message: Mapping[str, Any],
) -> dict[str, Any]:
    if assistant_response_payload is None:
        return dict(fallback_message)
    try:
        return assistant_message_from_completion_response(
            assistant_response_payload,
            fallback_message=fallback_message,
        )
    except Exception:
        return dict(fallback_message)


def _build_metadata(trace: RunTrace) -> dict[str, object]:
    metadata = {
        "trace_id": trace.metadata.trace_id,
        "run_id": trace.metadata.trace_id,
        "benchmark_version": trace.metadata.benchmark_version,
        "suite_name": trace.metadata.suite_name,
        "task_id": trace.metadata.task_id,
        "model_name": trace.metadata.model_name,
        "injection_present": trace.metadata.injection_present,
        "attack_name": trace.metadata.attack_name,
        "attack_family": trace.metadata.attack_family,
        "attack_type": trace.metadata.attack_type,
        "injection_task_id": trace.metadata.injection_task_id,
        "defense_name": trace.metadata.defense_name,
        "injection_round_index": trace.metadata.injection_round_index,
        "utility": trace.outcome.utility,
        "security": trace.outcome.security,
        "outcome_error": trace.outcome.error,
        "trace_version": trace.trace_version,
    }
    if trace.metadata.extra:
        metadata["run_extra"] = trace.metadata.extra
    if trace.outcome.metadata:
        metadata["outcome_metadata"] = trace.outcome.metadata
    return metadata


def append_decision_points_jsonl(
    decision_points: tuple[DecisionPointRecord, ...] | list[DecisionPointRecord] | tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for point in decision_points:
            payload = point.to_dict() if isinstance(point, DecisionPointRecord) else to_jsonable(point)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path
