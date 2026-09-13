from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class RunMetadata:
    trace_id: str
    benchmark_version: str
    suite_name: str
    task_id: str
    model_name: str
    injection_present: bool
    system_prompt_key: str | None = None
    attack_name: str | None = None
    attack_family: str | None = None
    attack_type: str | None = None
    injection_task_id: str | None = None
    defense_name: str | None = None
    extra: JsonObject | None = None
    injection_round_index: list[int] | None = None

    def to_dict(self) -> JsonObject:
        payload = _json_mapping(self)
        payload["run_id"] = self.trace_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunMetadata:
        injection_round_index = payload.get("injection_round_index")
        return cls(
            trace_id=str(payload.get("trace_id") or payload["run_id"]),
            benchmark_version=str(payload["benchmark_version"]),
            suite_name=str(payload["suite_name"]),
            task_id=str(payload["task_id"]),
            model_name=str(payload["model_name"]),
            injection_present=bool(payload["injection_present"]),
            system_prompt_key=_optional_string(payload.get("system_prompt_key")),
            attack_name=_optional_string(payload.get("attack_name")),
            attack_family=_optional_string(payload.get("attack_family")),
            attack_type=_optional_string(payload.get("attack_type")),
            injection_task_id=_optional_string(payload.get("injection_task_id")),
            defense_name=_optional_string(payload.get("defense_name")),
            extra=_json_mapping(payload.get("extra") or {}),
            injection_round_index=_coerce_injection_round_indices(injection_round_index),
        )

    @property
    def run_id(self) -> str:
        return self.trace_id


@dataclass(frozen=True, slots=True)
class ModelRequestRecord:
    request_index: int
    request_kind: str
    request_payload: JsonObject
    response_payload: JsonObject | None = None
    prompt_token_ids: tuple[int, ...] | None = None
    response_token_ids: tuple[int, ...] | None = None
    metadata: JsonObject | None = None

    def to_dict(self) -> JsonObject:
        return _json_mapping(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelRequestRecord:
        return cls(
            request_index=int(payload["request_index"]),
            request_kind=str(payload["request_kind"]),
            request_payload=_json_mapping(payload.get("request_payload") or {}),
            response_payload=None
            if payload.get("response_payload") is None
            else _json_mapping(payload.get("response_payload") or {}),
            prompt_token_ids=_optional_int_sequence(payload.get("prompt_token_ids")),
            response_token_ids=_optional_int_sequence(payload.get("response_token_ids")),
            metadata=_json_mapping(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionRecord:
    tool_index: int
    tool_name: str
    arguments: Any
    result: Any = None
    error: str | None = None
    environment_before: Any = None
    environment_after: Any = None
    message_index: int | None = None
    metadata: JsonObject | None = None

    def to_dict(self) -> JsonObject:
        return _json_mapping(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ToolExecutionRecord:
        message_index = payload.get("message_index")
        return cls(
            tool_index=int(payload["tool_index"]),
            tool_name=str(payload["tool_name"]),
            arguments=to_jsonable(payload.get("arguments")),
            result=to_jsonable(payload.get("result")),
            error=_optional_string(payload.get("error")),
            environment_before=to_jsonable(payload.get("environment_before")),
            environment_after=to_jsonable(payload.get("environment_after")),
            message_index=None if message_index is None else int(message_index),
            metadata=_json_mapping(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    utility: bool | None = None
    security: bool | None = None
    error: str | None = None
    duration_seconds: float | None = None
    metadata: JsonObject | None = None

    def to_dict(self) -> JsonObject:
        return _json_mapping(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskOutcome:
        duration_seconds = payload.get("duration_seconds")
        return cls(
            utility=_optional_bool(payload.get("utility")),
            security=_optional_bool(payload.get("security")),
            error=_optional_string(payload.get("error")),
            duration_seconds=None if duration_seconds is None else float(duration_seconds),
            metadata=_json_mapping(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class RunTrace:
    metadata: RunMetadata
    tool_executions: tuple[ToolExecutionRecord, ...]
    final_messages: tuple[JsonObject, ...]
    outcome: TaskOutcome
    model_requests: tuple[ModelRequestRecord, ...] = ()
    trace_version: str = "v2"

    def to_dict(self) -> JsonObject:
        return {
            "metadata": self.metadata.to_dict(),
            "tool_executions": [item.to_dict() for item in self.tool_executions],
            "final_messages": [to_jsonable(message) for message in self.final_messages],
            "outcome": self.outcome.to_dict(),
            "model_requests": [item.to_dict() for item in self.model_requests],
            "trace_version": self.trace_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunTrace:
        if "metadata" not in payload:
            payload = {
                "metadata": _run_metadata_payload_from_flat_trace(payload),
                "model_requests": [],
                "tool_executions": payload.get("tool_executions") or (),
                "final_messages": payload.get("final_messages") or (),
                "outcome": {
                    "utility": payload.get("utility"),
                    "security": payload.get("security"),
                    "error": payload.get("outcome_error"),
                },
                "trace_version": payload.get("schema_version") or payload.get("trace_version") or "trace.v1",
            }
        return cls(
            metadata=RunMetadata.from_dict(payload["metadata"]),
            model_requests=tuple(
                ModelRequestRecord.from_dict(item)
                for item in payload.get("model_requests") or ()
            ),
            tool_executions=tuple(
                ToolExecutionRecord.from_dict(item)
                for item in payload.get("tool_executions") or ()
            ),
            final_messages=_message_sequence(payload.get("final_messages") or ()),
            outcome=TaskOutcome.from_dict(payload.get("outcome") or {}),
            trace_version=str(payload.get("trace_version") or "v2"),
        )


@dataclass(frozen=True, slots=True)
class DecisionPointRecord:
    decision_index: int
    decision_point_id: str
    trace_id: str
    assistant_message: JsonObject
    assistant_message_index: int
    replay_request_kind: str | None
    replay_request: JsonObject | None
    prompt_token_ids: tuple[int, ...] | None
    response_token_ids: tuple[int, ...] | None
    metadata: JsonObject
    assistant_response_payload: JsonObject | None = None

    def to_dict(self) -> JsonObject:
        return _json_mapping(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DecisionPointRecord:
        metadata_payload = payload.get("metadata")
        if isinstance(metadata_payload, Mapping):
            resolved_metadata = _json_mapping(metadata_payload)
        else:
            resolved_metadata = _decision_point_metadata_payload_from_flat_row(payload)
        request_kind = _optional_string(payload.get("request_kind")) or _optional_string(payload.get("replay_request_kind"))
        request_payload = payload.get("request_payload")
        if request_payload is None:
            request_payload = payload.get("replay_request")
        response_payload = payload.get("response_payload")
        if response_payload is None:
            response_payload = payload.get("assistant_response_payload")
        return cls(
            decision_index=int(payload["decision_index"]),
            decision_point_id=str(
                payload.get("decision_point_id")
                or _derive_decision_point_id(
                    trace_id=str(payload.get("trace_id") or resolved_metadata.get("trace_id") or resolved_metadata.get("run_id") or ""),
                    assistant_message_index=int(payload["assistant_message_index"]),
                )
            ),
            trace_id=str(payload.get("trace_id") or resolved_metadata.get("trace_id") or resolved_metadata.get("run_id") or ""),
            assistant_message=_json_mapping(payload.get("assistant_message") or {}),
            assistant_message_index=int(payload["assistant_message_index"]),
            replay_request_kind=request_kind,
            replay_request=None if request_payload is None else _json_mapping(request_payload or {}),
            prompt_token_ids=_optional_int_sequence(payload.get("prompt_token_ids")),
            response_token_ids=_optional_int_sequence(payload.get("response_token_ids")),
            metadata=resolved_metadata,
            assistant_response_payload=None if response_payload is None else _json_mapping(response_payload or {}),
        )


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump())
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return to_jsonable(dict_method())
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    if hasattr(value, "__dict__"):
        return {
            str(key): to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _message_sequence(messages: Sequence[Any]) -> tuple[JsonObject, ...]:
    return tuple(_json_mapping(message) for message in messages)


def _run_metadata_payload_from_flat_trace(payload: Mapping[str, Any]) -> JsonObject:
    metadata = {
        "trace_id": payload.get("trace_id") or payload.get("run_id"),
        "run_id": payload.get("trace_id") or payload.get("run_id"),
        "benchmark_version": payload.get("benchmark_version") or "unknown-version",
        "suite_name": payload.get("suite_name") or "unknown-suite",
        "task_id": payload.get("task_id") or "unknown-task",
        "model_name": payload.get("model_name") or "unknown-model",
        "injection_present": bool(payload.get("injection_present")),
        "system_prompt_key": payload.get("system_prompt_key"),
        "attack_name": payload.get("attack_name"),
        "attack_family": payload.get("attack_family"),
        "attack_type": payload.get("attack_type"),
        "injection_task_id": payload.get("injection_task_id"),
        "defense_name": payload.get("defense_name"),
        "injection_round_index": payload.get("injection_round_index"),
        "extra": {
            "case_id": payload.get("case_id"),
            "instance_id": payload.get("instance_id"),
            "repeat_index": payload.get("repeat_index"),
        },
    }
    return _json_mapping(metadata)


def _decision_point_metadata_payload_from_flat_row(payload: Mapping[str, Any]) -> JsonObject:
    metadata = {
        "trace_id": payload.get("trace_id") or payload.get("run_id"),
        "run_id": payload.get("trace_id") or payload.get("run_id"),
        "benchmark_version": payload.get("benchmark_version"),
        "suite_name": payload.get("suite_name"),
        "task_id": payload.get("task_id"),
        "model_name": payload.get("model_name"),
        "injection_present": payload.get("injection_present"),
        "system_prompt_key": payload.get("system_prompt_key"),
        "attack_name": payload.get("attack_name"),
        "attack_family": payload.get("attack_family"),
        "attack_type": payload.get("attack_type"),
        "injection_task_id": payload.get("injection_task_id"),
        "defense_name": payload.get("defense_name"),
        "injection_round_index": payload.get("injection_round_index"),
        "utility": payload.get("utility"),
        "security": payload.get("security"),
        "outcome_error": payload.get("outcome_error"),
        "trace_version": payload.get("trace_version") or payload.get("schema_version"),
        "run_extra": {
            "case_id": payload.get("case_id"),
            "instance_id": payload.get("instance_id"),
            "repeat_index": payload.get("repeat_index"),
        },
    }
    return _json_mapping(metadata)


def _json_mapping(value: Any) -> JsonObject:
    json_value = to_jsonable(value)
    if not isinstance(json_value, Mapping):
        raise TypeError(f"expected mapping-compatible value, got {type(value)!r}")
    return {str(key): to_jsonable(item) for key, item in json_value.items()}


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_int_sequence(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    return tuple(int(item) for item in value)


def _coerce_injection_round_indices(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [int(item) for item in value]
    return [int(value)]


def _derive_decision_point_id(*, trace_id: str, assistant_message_index: int) -> str:
    return f"{trace_id}:assistant:{assistant_message_index}"
