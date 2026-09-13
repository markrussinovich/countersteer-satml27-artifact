from __future__ import annotations

import concurrent.futures
import copy
import importlib
import json
import sys
import threading
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import local
from typing import Any, Callable
from urllib.parse import urlparse

import yaml

from ipi_aware.message_content import assistant_message_from_completion_response
from ipi_aware.data_collection.request_context import active_request_context
from ipi_aware.data_collection.token_accounting import token_unit_usage
from .agentdojo_runtime import (
    is_context_length_error,
    is_empty_assistant_output_error,
    is_empty_assistant_output_payload,
    is_retryable_vllm_bad_request,
)

_REQUEST_CAPTURE_STATE = local()
# Process-level cache for pipeline/suite/attacker resources.
# Shared across threads within a process so all threads reuse the same
# OpenAI client (and its httpx connection pool) rather than each creating
# their own.  Safe because OpenAILLM.query() is stateless and the
# openai.OpenAI client is thread-safe.
_PROCESS_RESOURCE_CACHE: dict[tuple, tuple[Any, Any, Any | None]] = {}
_VERBOSE_DIAG = False
_DIAG_LOG_LOCK = threading.Lock()
ROOT_DIR = Path(__file__).resolve().parents[2]
VENDORED_AGENTDOJO_SRC = ROOT_DIR / "vendor" / "agentdojo" / "src"
if VENDORED_AGENTDOJO_SRC.exists() and str(VENDORED_AGENTDOJO_SRC) not in sys.path:
    sys.path.insert(0, str(VENDORED_AGENTDOJO_SRC))


def _init_verbose_diag() -> None:
    global _VERBOSE_DIAG
    import os
    _VERBOSE_DIAG = os.environ.get("IPI_AWARE_VERBOSE_COLLECT", "").strip() in ("1", "true", "yes")


def _diag_log(msg: str) -> None:
    if not _VERBOSE_DIAG:
        return
    import os
    import time
    ts = time.strftime("%H:%M:%S", time.localtime())
    tid = threading.current_thread().name
    pid = os.getpid()
    line = f"[{ts} diag p={pid} t={tid}] {msg}"
    with _DIAG_LOG_LOCK:
        print(line, file=sys.stderr, flush=True)


@dataclass(frozen=True, slots=True)
class CollectionCase:
    suite_name: str
    user_task_id: str
    case_kind: str
    injection_present: bool
    attack_name: str | None = None
    injection_task_id: str | None = None
    system_prompt_key: str | None = None
    system_message_name: str | None = None
    system_message: str | None = None
    repeat_index: int = 0

    @property
    def case_id(self) -> str:
        if not self.injection_present:
            base = f"{self.suite_name}__{self.user_task_id}__clean"
        else:
            base = f"{self.suite_name}__{self.user_task_id}__{self.attack_name}__{self.injection_task_id}"
        if self.system_prompt_key:
            return f"{base}__prompt_{self.system_prompt_key}"
        return base

    @property
    def instance_id(self) -> str:
        if self.repeat_index <= 0:
            return self.case_id
        return f"{self.case_id}__repeat_{self.repeat_index}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["case_id"] = self.case_id
        payload["instance_id"] = self.instance_id
        return payload


@dataclass(frozen=True, slots=True)
class CollectedCaseResult:
    case: CollectionCase
    run_id: str
    utility: bool | None
    security: bool | None
    trace_row: dict[str, Any] | None
    decision_rows: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    error_type: str | None = None
    error_traceback: str | None = None
    failed: bool = False


def _context_length_collected_result(case: CollectionCase, run_id_prefix: str, error: Exception) -> CollectedCaseResult:
    return _nonfatal_error_collected_result(case, run_id_prefix, error)


def _nonfatal_error_collected_result(case: CollectionCase, run_id_prefix: str, error: Exception) -> CollectedCaseResult:
    run_id = f"{run_id_prefix}-{case.instance_id}"
    return CollectedCaseResult(
        case=case,
        run_id=run_id,
        utility=False,
        security=True,
        trace_row=None,
        decision_rows=(),
        error=str(error),
        error_type=type(error).__name__,
        error_traceback=traceback.format_exc(),
        failed=True,
    )


def build_request_injection_context(
    *,
    messages: Sequence[Mapping[str, Any]],
    injections: Any,
) -> dict[str, Any]:
    injection_round_index = _compute_injection_round_indices(
        messages=messages,
        injections=injections,
    )
    latest_message_index = len(messages) - 1 if messages else None
    latest_injection_message_index = max(injection_round_index) if injection_round_index else None
    assistant_since_latest = None
    if latest_injection_message_index is not None:
        assistant_since_latest = sum(
            1
            for index, message in enumerate(messages)
            if index > latest_injection_message_index and str(message.get("role") or "") == "assistant"
        )
    return {
        "injection_present": bool(injection_round_index),
        "injection_round_index": list(injection_round_index) if injection_round_index else None,
        "latest_message_index": latest_message_index,
        "latest_message_is_injected": latest_message_index in injection_round_index if latest_message_index is not None else False,
        "latest_injection_message_index": latest_injection_message_index,
        "decision_after_injection": latest_injection_message_index is not None,
        "messages_since_latest_injection": (
            latest_message_index - latest_injection_message_index
            if latest_message_index is not None and latest_injection_message_index is not None
            else None
        ),
        "assistant_decisions_since_latest_injection": assistant_since_latest,
    }


@dataclass(slots=True)
class RequestRecord:
    request_index: int
    request_kind: str
    request_payload: dict[str, Any]
    response_payload: dict[str, Any] | None
    prompt_token_ids: tuple[int, ...] | None
    response_token_ids: tuple[int, ...] | None


@dataclass(slots=True)
class ToolExecutionRecord:
    tool_index: int
    tool_name: str
    arguments: Any
    result: Any = None
    error: str | None = None
    environment_before: Any = None
    environment_after: Any = None
    message_index: int | None = None
    tool_call_id: str | None = None
    injection_vector_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_mapping(self)


@dataclass(slots=True)
class CaseRecorder:
    case_payload: dict[str, Any]
    requests: list[RequestRecord]
    tool_executions: list[ToolExecutionRecord]

    @classmethod
    def create(cls, *, case_payload: dict[str, Any]) -> CaseRecorder:
        return cls(case_payload=case_payload, requests=[], tool_executions=[])

    def record_request(
        self,
        *,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any] | None,
        prompt_token_ids: tuple[int, ...] | None,
        response_token_ids: tuple[int, ...] | None,
    ) -> None:
        self.requests.append(
            RequestRecord(
                request_index=len(self.requests),
                request_kind="openai_chat",
                request_payload=_json_mapping(request_payload),
                response_payload=None if response_payload is None else _json_mapping(response_payload),
                prompt_token_ids=prompt_token_ids,
                response_token_ids=response_token_ids,
            )
        )

    def snapshot_state(self) -> tuple[int, int]:
        return len(self.requests), len(self.tool_executions)

    def restore_state(self, snapshot: tuple[int, int]) -> None:
        request_count, tool_execution_count = snapshot
        del self.requests[request_count:]
        del self.tool_executions[tool_execution_count:]

    def update_last_response_payload(self, response_payload: Mapping[str, Any] | None) -> None:
        if not self.requests:
            return
        self.requests[-1].response_payload = (
            None if response_payload is None else _json_mapping(response_payload)
        )

    def record_tool_execution(
        self,
        *,
        tool_name: str,
        arguments: Any,
        result: Any = None,
        error: str | None = None,
        environment_before: Any = None,
        environment_after: Any = None,
        injection_vector_ids: Sequence[str] = (),
    ) -> None:
        self.tool_executions.append(
            ToolExecutionRecord(
                tool_index=len(self.tool_executions),
                tool_name=tool_name,
                arguments=_to_jsonable(arguments),
                result=_to_jsonable(result),
                error=error,
                environment_before=_to_jsonable(environment_before),
                environment_after=_to_jsonable(environment_after),
                injection_vector_ids=tuple(str(item) for item in injection_vector_ids),
            )
        )

    def build_rows(
        self,
        *,
        final_messages: Sequence[Mapping[str, Any]],
        utility: bool | None,
        security: bool | None,
        outcome_error: str | None = None,
        injection_round_indices: Sequence[int] | None = None,
        injection_provenance_source: str | None = None,
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        normalized_messages = tuple(_json_mapping(message) for message in final_messages)
        assistant_turns = [
            (message_index, message)
            for message_index, message in enumerate(normalized_messages)
            if str(message.get("role") or "") == "assistant"
        ]
        if len(assistant_turns) != len(self.requests):
            raise ValueError(
                "assistant/request mismatch: "
                f"assistant_turns={len(assistant_turns)} requests={len(self.requests)}"
            )

        if injection_round_indices is None:
            injection_round_indices = _compute_injection_round_indices(
                messages=normalized_messages,
                injections=(self.case_payload.get("injections") or {}),
            )
            injection_provenance_source = injection_provenance_source or "message_text_match"
        injection_round_indices = tuple(int(index) for index in injection_round_indices)
        injection_round_index = list(injection_round_indices) if injection_round_indices else None
        decision_rows: list[dict[str, Any]] = []
        for request, (assistant_message_index, fallback_message) in zip(self.requests, assistant_turns, strict=True):
            assistant_message = _build_assistant_message(
                response_payload=request.response_payload,
                fallback_message=fallback_message,
            )
            injection_context = _decision_injection_context(
                assistant_message_index=assistant_message_index,
                assistant_turns=assistant_turns,
                injection_round_index=injection_round_indices,
            )
            decision_row = {
                "schema_version": "decision_point.v1",
                **self.case_payload,
                "decision_index": request.request_index,
                "decision_point_id": f"{self.case_payload['trace_id']}:assistant:{assistant_message_index}",
                "assistant_message_index": assistant_message_index,
                "assistant_message": assistant_message,
                "request_kind": request.request_kind,
                "request_payload": request.request_payload,
                "response_payload": request.response_payload,
                "prompt_token_ids": list(request.prompt_token_ids or ()),
                "response_token_ids": list(request.response_token_ids or ()),
                "injection_context": injection_context,
            }
            response_usage = _extract_response_usage(request.response_payload)
            if response_usage is not None:
                decision_row["response_usage"] = response_usage
                decision_row["token_usage"] = token_unit_usage(response_usage)
            elif request.prompt_token_ids is not None or request.response_token_ids is not None:
                fallback_usage = {
                    "prompt_tokens": len(request.prompt_token_ids or ()),
                    "completion_tokens": len(request.response_token_ids or ()),
                }
                fallback_usage["total_tokens"] = fallback_usage["prompt_tokens"] + fallback_usage["completion_tokens"]
                decision_row["response_usage"] = {
                    **fallback_usage,
                    "usage_source": "token_id_lengths",
                }
                decision_row["token_usage"] = token_unit_usage(fallback_usage)
            decision_rows.append(decision_row)

        trace_row = {
            "schema_version": "trace.v1",
            **self.case_payload,
            "utility": utility,
            "security": security,
            "outcome_error": outcome_error,
            "injection_round_index": injection_round_index,
            "injection_provenance_source": injection_provenance_source,
            "final_messages": [dict(message) for message in normalized_messages],
            "tool_executions": [record.to_dict() for record in self._tool_executions_with_message_indices(normalized_messages)],
            "decision_point_ids": [row["decision_point_id"] for row in decision_rows],
        }
        return trace_row, tuple(decision_rows)

    def _tool_executions_with_message_indices(
        self,
        final_messages: Sequence[Mapping[str, Any]],
    ) -> tuple[ToolExecutionRecord, ...]:
        tool_messages = [
            (index, message)
            for index, message in enumerate(final_messages)
            if str(message.get("role") or "") == "tool"
        ]
        bindings: dict[int, tuple[int, str | None]] = {}
        search_start = 0
        for message_index, message in tool_messages:
            tool_call = _to_jsonable(message.get("tool_call"))
            if not isinstance(tool_call, Mapping):
                raise ValueError(f"tool message at index {message_index} has no structured tool_call")
            function = str(tool_call.get("function") or "")
            arguments = _to_jsonable(tool_call.get("args") or {})
            for record_index in range(search_start, len(self.tool_executions)):
                record = self.tool_executions[record_index]
                if record.tool_name != function or record.arguments != arguments:
                    continue
                bindings[record_index] = (message_index, _optional_str(message.get("tool_call_id")))
                search_start = record_index + 1
                break
            else:
                if message.get("error"):
                    continue
                raise ValueError(
                    "tool message has no matching runtime execution: "
                    f"message_index={message_index} function={function!r} arguments={arguments!r}"
                )

        attached: list[ToolExecutionRecord] = []
        for record_index, record in enumerate(self.tool_executions):
            message_index, tool_call_id = bindings.get(record_index, (None, None))
            attached.append(
                ToolExecutionRecord(
                    tool_index=record.tool_index,
                    tool_name=record.tool_name,
                    arguments=record.arguments,
                    result=record.result,
                    error=record.error,
                    environment_before=record.environment_before,
                    environment_after=record.environment_after,
                    message_index=message_index,
                    tool_call_id=tool_call_id,
                    injection_vector_ids=record.injection_vector_ids,
                )
            )
        return tuple(attached)


def _runtime_injection_round_indices(
    recorder: CaseRecorder,
    final_messages: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    return tuple(
        record.message_index
        for record in recorder._tool_executions_with_message_indices(final_messages)
        if record.message_index is not None and record.injection_vector_ids
    )


class active_request_capture:
    def __init__(self, recorder: CaseRecorder) -> None:
        self._recorder = recorder
        self._previous = None

    def __enter__(self) -> CaseRecorder:
        self._previous = getattr(_REQUEST_CAPTURE_STATE, "recorder", None)
        _REQUEST_CAPTURE_STATE.recorder = self._recorder
        return self._recorder

    def __exit__(self, exc_type, exc, tb) -> None:
        _REQUEST_CAPTURE_STATE.recorder = self._previous


def plan_collection_cases(
    *,
    suite_name: str,
    user_task_ids: Sequence[str],
    include_clean: bool,
    include_injected: bool,
    attack_names: Sequence[str] = (),
    injection_task_ids: Sequence[str] = (),
    system_message_variants: Sequence[Mapping[str, Any]] = (),
) -> tuple[CollectionCase, ...]:
    cases: list[CollectionCase] = []
    variants = tuple(system_message_variants) or (
        {
            "system_prompt_key": None,
            "system_message_name": None,
            "system_message": None,
        },
    )
    if include_clean:
        for user_task_id in user_task_ids:
            for variant in variants:
                cases.append(
                    CollectionCase(
                        suite_name=suite_name,
                        user_task_id=user_task_id,
                        case_kind="clean",
                        injection_present=False,
                        system_prompt_key=variant.get("system_prompt_key"),
                        system_message_name=variant.get("system_message_name"),
                        system_message=variant.get("system_message"),
                    )
                )
    if include_injected:
        for user_task_id in user_task_ids:
            for attack_name in attack_names:
                for injection_task_id in injection_task_ids:
                    for variant in variants:
                        cases.append(
                            CollectionCase(
                                suite_name=suite_name,
                                user_task_id=user_task_id,
                                case_kind="injected",
                                injection_present=True,
                                attack_name=attack_name,
                                injection_task_id=injection_task_id,
                                system_prompt_key=variant.get("system_prompt_key"),
                                system_message_name=variant.get("system_message_name"),
                                system_message=variant.get("system_message"),
                            )
                        )
    return tuple(cases)


def expand_collection_cases(cases: Sequence[CollectionCase], *, epochs: int) -> tuple[CollectionCase, ...]:
    expanded: list[CollectionCase] = []
    for repeat_index in range(max(epochs, 1)):
        for case in cases:
            expanded.append(
                CollectionCase(
                    suite_name=case.suite_name,
                    user_task_id=case.user_task_id,
                    case_kind=case.case_kind,
                    injection_present=case.injection_present,
                    attack_name=case.attack_name,
                    injection_task_id=case.injection_task_id,
                    system_prompt_key=case.system_prompt_key,
                    system_message_name=case.system_message_name,
                    system_message=case.system_message,
                    repeat_index=repeat_index,
                )
            )
    return tuple(expanded)


def resolve_suite_task_ids(
    *,
    benchmark_version: str,
    suite_name: str,
    user_task_ids: Sequence[str],
    injection_task_ids: Sequence[str],
    include_injected: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    _ensure_vendor_agentdojo_path()
    suite = importlib.import_module("agentdojo.task_suite.load_suites").get_suite(benchmark_version, suite_name)
    resolved_user = tuple(user_task_ids) or tuple(suite.user_tasks.keys())
    resolved_injection = tuple(injection_task_ids) or tuple(suite.injection_tasks.keys()) if include_injected else tuple()
    return resolved_user, resolved_injection


def collect_cases(
    cases: Sequence[CollectionCase],
    *,
    benchmark_version: str,
    model: str,
    run_id_prefix: str,
    model_id: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    tool_output_format: str | None = None,
    temperature: float | None = 0.0,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
    max_workers: int = 1,
    num_processes: int = 1,
    write_root: str | None = None,
    continue_on_error: bool = False,
    progress_callback: Callable[[int, int, int, CollectedCaseResult], None] | None = None,
    case_timeout_seconds: float | None = None,
    case_attempts: int = 3,
    upstream_ports: Sequence[int] | None = None,
) -> tuple[CollectedCaseResult, ...]:
    if not cases:
        return tuple()

    if num_processes > 1:
        return _collect_cases_multiprocess(
            cases,
            benchmark_version=benchmark_version,
            model=model,
            run_id_prefix=run_id_prefix,
            model_id=model_id,
            defense=defense,
            tool_delimiter=tool_delimiter,
            system_message_name=system_message_name,
            system_message=system_message,
            tool_output_format=tool_output_format,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            max_workers=max_workers,
            num_processes=num_processes,
            write_root=write_root,
            continue_on_error=continue_on_error,
            case_timeout_seconds=case_timeout_seconds,
            case_attempts=case_attempts,
            progress_callback=progress_callback,
            upstream_ports=upstream_ports,
        )

    total = len(cases)
    results: list[CollectedCaseResult | None] = [None] * total
    completed = 0

    def _store(index: int, result: CollectedCaseResult) -> None:
        nonlocal completed
        results[index] = result
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total, index, result)

    def _collect(index: int, case: CollectionCase) -> CollectedCaseResult:
        try:
            return collect_case(
                case,
                benchmark_version=benchmark_version,
                model=model,
                run_id_prefix=run_id_prefix,
                model_id=model_id,
                defense=defense,
                tool_delimiter=tool_delimiter,
                system_message_name=case.system_message_name if case.system_message_name is not None else system_message_name,
                system_message=case.system_message if case.system_message is not None else system_message,
                tool_output_format=tool_output_format,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_completion_tokens,
                case_timeout_seconds=case_timeout_seconds,
                case_attempts=case_attempts,
            )
        except Exception as exc:
            if is_context_length_error(exc):
                return _context_length_collected_result(case, run_id_prefix, exc)
            if is_empty_assistant_output_error(exc):
                return _nonfatal_error_collected_result(case, run_id_prefix, exc)
            if not continue_on_error:
                raise
            return CollectedCaseResult(
                case=case,
                run_id=f"{run_id_prefix}-{case.instance_id}",
                utility=None,
                security=None,
                trace_row=None,
                decision_rows=(),
                error=str(exc),
                error_type=type(exc).__name__,
                error_traceback=traceback.format_exc(),
                failed=True,
            )

    if max_workers <= 1:
        for index, case in enumerate(cases):
            _store(index, _collect(index, case))
        return tuple(result for result in results if result is not None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_collect, index, case): index
            for index, case in enumerate(cases)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            _store(index, future.result())
    return tuple(result for result in results if result is not None)


def _mp_worker(
    shard: list[tuple[int, CollectionCase]],
    workers_per_process: int,
    benchmark_version: str,
    model: str,
    run_id_prefix: str,
    model_id: str | None,
    defense: str | None,
    tool_delimiter: str,
    system_message_name: str | None,
    system_message: str | None,
    tool_output_format: str | None,
    temperature: float | None,
    top_p: float | None,
    max_completion_tokens: int | None,
    continue_on_error: bool,
    case_timeout_seconds: float | None,
    case_attempts: int,
    result_queue: Any,
    error_queue: Any = None,
    upstream_base_url: str | None = None,
) -> None:
    """Top-level worker for multiprocessing.Process. Streams results one at a time."""
    import os as _os
    import time as _time
    if upstream_base_url is not None:
        _os.environ["IPI_AWARE_UPSTREAM_BASE_URL"] = upstream_base_url
        parsed = urlparse(upstream_base_url)
        if parsed.port is not None:
            _os.environ["LOCAL_LLM_PORT"] = str(parsed.port)
    _init_verbose_diag()
    print(f"[_mp_worker pid={_os.getpid()}] starting, {len(shard)} cases, {workers_per_process} threads"
          f"{f', upstream={upstream_base_url}' if upstream_base_url else ''}",
          flush=True)
    try:
        def _collect_one(case: CollectionCase) -> CollectedCaseResult:
            _diag_log(f"START case={case.case_id} repeat={case.repeat_index}")
            _t0 = _time.monotonic()
            try:
                result = collect_case(
                    case,
                    benchmark_version=benchmark_version,
                    model=model,
                    run_id_prefix=run_id_prefix,
                    model_id=model_id,
                    defense=defense,
                    tool_delimiter=tool_delimiter,
                    system_message_name=case.system_message_name if case.system_message_name is not None else system_message_name,
                    system_message=case.system_message if case.system_message is not None else system_message,
                    tool_output_format=tool_output_format,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_completion_tokens,
                    case_timeout_seconds=case_timeout_seconds,
                    case_attempts=case_attempts,
                )
                _dt = _time.monotonic() - _t0
                n_req = len(result.trace_row.get("requests", ())) if isinstance(result.trace_row, dict) else 0
                _diag_log(f"DONE  case={case.case_id} dt={_dt:.1f}s reqs={n_req}")
                return result
            except Exception as exc:
                _dt = _time.monotonic() - _t0
                _diag_log(f"FAIL  case={case.case_id} dt={_dt:.1f}s err={type(exc).__name__}: {exc!s:.200s}")
                if is_context_length_error(exc):
                    return _context_length_collected_result(case, run_id_prefix, exc)
                if is_empty_assistant_output_error(exc):
                    return _nonfatal_error_collected_result(case, run_id_prefix, exc)
                if not continue_on_error:
                    raise
                return CollectedCaseResult(
                    case=case,
                    run_id=f"{run_id_prefix}-{case.instance_id}",
                    utility=None,
                    security=None,
                    trace_row=None,
                    decision_rows=(),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    error_traceback=traceback.format_exc(),
                    failed=True,
                )

        if workers_per_process <= 1:
            for global_i, case in shard:
                result_queue.put((global_i, _collect_one(case)))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers_per_process) as executor:
                future_to_idx = {
                    executor.submit(_collect_one, case): global_i
                    for global_i, case in shard
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    global_i = future_to_idx[future]
                    result_queue.put((global_i, future.result()))

        result_queue.put(None)  # sentinel: this shard is done
    except Exception:
        import sys
        if error_queue is not None:
            error_queue.put(sys.exc_info())
        raise


def _collect_cases_multiprocess(
    cases: Sequence[CollectionCase],
    *,
    benchmark_version: str,
    model: str,
    run_id_prefix: str,
    model_id: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    tool_output_format: str | None = None,
    temperature: float | None = 0.0,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
    max_workers: int = 1,
    num_processes: int = 1,
    write_root: str | None = None,
    continue_on_error: bool = False,
    case_timeout_seconds: float | None = None,
    case_attempts: int = 3,
    progress_callback: Callable[[int, int, int, CollectedCaseResult], None] | None = None,
    upstream_ports: Sequence[int] | None = None,
) -> tuple[CollectedCaseResult, ...]:
    """Shard cases across processes; each process runs its own thread pool."""
    import multiprocessing
    workers_per_process = max(1, max_workers // num_processes)

    # Build per-process base URLs from the port list.
    # Worker i -> upstream_ports[i % len(upstream_ports)]
    _port_list = list(upstream_ports) if upstream_ports else []
    def _base_url_for_worker(idx: int) -> str | None:
        if not _port_list:
            return None
        port = _port_list[idx % len(_port_list)]
        return f"http://127.0.0.1:{port}/v1"

    shards: list[list[tuple[int, CollectionCase]]] = [[] for _ in range(num_processes)]
    for i, case in enumerate(cases):
        shards[i % num_processes].append((i, case))

    total = len(cases)
    collected = 0

    # Use fork context so children inherit the parent's modules and env.
    # Plain Queue (not Manager) is inherited via fork, avoiding pickle issues.
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    error_queue = ctx.Queue()

    procs = []
    for p_idx, shard in enumerate(shards):
        proc = ctx.Process(
            target=_mp_worker,
            args=(
                shard,
                workers_per_process,
                benchmark_version,
                model,
                run_id_prefix,
                model_id,
                defense,
                tool_delimiter,
                system_message_name,
                system_message,
                tool_output_format,
                temperature,
                top_p,
                max_completion_tokens,
                continue_on_error,
                case_timeout_seconds,
                case_attempts,
                result_queue,
                error_queue,
                _base_url_for_worker(p_idx),
            ),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    shards_done = 0
    # Track which shards have sent their sentinel so we can detect dead workers.
    shard_finished = [False] * num_processes
    import sys
    print(f"[_collect_cases_multiprocess] {num_processes} workers started, waiting for results on queue...",
          file=sys.stderr, flush=True)
    while shards_done < num_processes:
        # Check for worker errors first
        while not error_queue.empty():
            exc_info = error_queue.get_nowait()
            for proc in procs:
                proc.terminate()
            raise exc_info[1].with_traceback(exc_info[2])

        try:
            item = result_queue.get(timeout=1.0)
        except Exception:
            # Check if any non-finished worker process died
            for p_idx, proc in enumerate(procs):
                if shard_finished[p_idx]:
                    continue
                if not proc.is_alive():
                    if proc.exitcode is not None and proc.exitcode != 0:
                        for p in procs:
                            p.terminate()
                        raise RuntimeError(
                            f"Worker {p_idx} (exitcode={proc.exitcode}) died unexpectedly. "
                            f"collected={collected}/{total}"
                        )
                    # exitcode==0 means the process exited cleanly (e.g. sentinel sent
                    # but we haven't read it yet) — skip.
            continue

        if item is None:
            shards_done += 1
            continue

        global_i, result = item
        collected += 1
        if collected <= 3 or collected % 1000 == 0:
            gp = getattr(getattr(result, 'case', None), 'suite_name', '?')
            print(f"[_collect_cases_multiprocess] got result {collected}/{total}, "
                  f"shards_done={shards_done}, suite={gp}",
                  file=sys.stderr, flush=True)
        if progress_callback is not None:
            progress_callback(collected, total, global_i, result)

    for proc in procs:
        proc.join()

    return tuple()


def collect_case(
    case: CollectionCase,
    *,
    benchmark_version: str,
    model: str,
    run_id_prefix: str,
    model_id: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    tool_output_format: str | None = None,
    temperature: float | None = 0.0,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
    case_timeout_seconds: float | None = None,
    case_attempts: int = 3,
) -> CollectedCaseResult:
    _ensure_vendor_agentdojo_path()
    install_openai_request_capture()
    task_suite_module = importlib.import_module("agentdojo.task_suite.task_suite")
    functions_runtime = importlib.import_module("agentdojo.functions_runtime")
    pipeline_errors = importlib.import_module("agentdojo.agent_pipeline.errors")
    base_tasks = importlib.import_module("agentdojo.base_tasks")

    suite, pipeline, attacker = _get_worker_resources(
        suite_name=case.suite_name,
        benchmark_version=benchmark_version,
        model=_normalize_agentdojo_model_name(model),
        model_id=model_id,
        attack_name=case.attack_name,
        defense=defense,
        tool_delimiter=tool_delimiter,
        system_message_name=system_message_name,
        system_message=system_message,
        tool_output_format=tool_output_format,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        client_timeout_seconds=case_timeout_seconds,
    )

    user_task = suite.get_user_task_by_id(case.user_task_id)
    injections: dict[str, str] = {}
    injection_task = None
    if case.injection_present:
        injection_task = suite.get_injection_task_by_id(str(case.injection_task_id))
        injections = attacker.attack(user_task, injection_task) if attacker is not None else {}

    run_id = f"{run_id_prefix}-{case.instance_id}"
    case_payload = {
        "trace_id": run_id,
        "run_id": run_id,
        "benchmark_version": benchmark_version,
        "suite_name": case.suite_name,
        "task_id": case.user_task_id,
        "model_name": model_id or _normalize_agentdojo_model_name(model),
        "injection_present": bool(case.injection_present) and bool(injections),
        "system_prompt_key": case.system_prompt_key,
        "attack_name": case.attack_name,
        "attack_family": _infer_attack_family(case.attack_name),
        "attack_type": case.attack_name,
        "injection_task_id": case.injection_task_id,
        "defense_name": defense,
        "case_id": case.case_id,
        "instance_id": case.instance_id,
        "repeat_index": case.repeat_index,
        "injections": injections,
    }
    recorder = CaseRecorder.create(case_payload=case_payload)

    environment = suite.load_and_inject_default_environment(injections)
    if isinstance(user_task, base_tasks.BaseUserTask):
        task_environment = user_task.init_environment(environment)
        prompt = user_task.PROMPT
    else:
        task_environment = environment
        prompt = user_task.GOAL
    environment_payload = yaml.safe_load(
        task_suite_module.read_suite_file(case.suite_name, "environment.yaml", suite.data_path)
    )
    provenance_paths = _injection_provenance_paths(environment_payload, injections)
    runtime_class = make_instrumented_runtime_class(
        recorder,
        functions_runtime.FunctionsRuntime,
        provenance_paths=provenance_paths,
    )
    pre_environment = _clone_environment(task_environment)
    runtime = runtime_class(suite.tools)

    messages: list[dict[str, Any]] = []
    request_context = {
        "trace_id": run_id,
        "run_id": run_id,
        "case_id": case.case_id,
        "instance_id": case.instance_id,
        "suite_name": case.suite_name,
        "task_id": case.user_task_id,
        "system_prompt_key": case.system_prompt_key,
        "attack_name": case.attack_name,
        "injection_present": bool(case.injection_present) and bool(injections),
        "_ipi_aware_injections": injections,
        "grid_point_id": f"{case.suite_name}__{case.system_prompt_key or 'unknown'}__{case.attack_name or 'clean'}",
    }
    with active_request_capture(recorder), active_request_context(request_context):
        model_output = None
        outcome_error = None
        attempts = max(1, int(case_attempts))
        for attempt_index in range(attempts):
            snapshot = recorder.snapshot_state()
            attempt_environment = _clone_environment(task_environment)
            attempt_messages = list(messages)
            try:
                _, _, task_environment, messages, _ = pipeline.query(prompt, runtime, task_environment, [], {})
            except pipeline_errors.AbortAgentError as exc:
                task_environment = exc.task_environment
                messages = exc.messages
            except Exception as exc:
                recorder.restore_state(snapshot)
                task_environment = attempt_environment
                messages = attempt_messages
                if attempt_index < attempts - 1 and is_retryable_vllm_bad_request(exc):
                    continue
                raise

            model_output = _safe_model_output(messages, task_suite_module.model_output_from_messages)
            if model_output is not None and not _latest_assistant_output_is_empty(recorder, messages):
                break

            if attempt_index < attempts - 1:
                recorder.restore_state(snapshot)
                task_environment = attempt_environment
                messages = attempt_messages
            else:
                outcome_error = f"no usable model output after {attempts} attempts: case_id={case.case_id}"
            model_output = None

    if outcome_error is not None:
        return CollectedCaseResult(
            case=case,
            run_id=run_id,
            utility=False,
            security=False,
            trace_row=None,
            decision_rows=(),
            error=outcome_error,
            error_type="NoUsableModelOutputError",
            error_traceback=None,
            failed=True,
        )

    functions_stack_trace = task_suite_module.functions_stack_trace_from_messages(messages)
    utility = suite._check_task_result(user_task, model_output or [], pre_environment, task_environment, functions_stack_trace)
    security = True if injection_task is None else suite._check_task_result(
        injection_task,
        model_output or [],
        pre_environment,
        task_environment,
        functions_stack_trace,
    )
    injection_round_indices = _runtime_injection_round_indices(recorder, messages)
    trace_row, decision_rows = recorder.build_rows(
        final_messages=messages,
        utility=utility,
        security=security,
        outcome_error=outcome_error,
        injection_round_indices=injection_round_indices,
        injection_provenance_source="agentdojo_runtime_structural_provenance",
    )
    return CollectedCaseResult(
        case=case,
        run_id=run_id,
        utility=utility,
        security=security,
        trace_row=trace_row,
        decision_rows=decision_rows,
    )


def default_run_id_prefix() -> str:
    return f"agentdojo-lite-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def make_instrumented_runtime_class(
    recorder: CaseRecorder,
    base_runtime_class: type[Any],
    *,
    provenance_paths: Mapping[tuple[str | int, ...], Sequence[str]] | None = None,
) -> type[Any]:
    configured_paths = dict(provenance_paths or {})

    class InstrumentedRuntime(base_runtime_class):
        def run_function(self, env: Any, function: str, kwargs: dict[str, Any], raise_on_error: bool = False):
            environment_before = _to_jsonable(env)
            provenance_index = _capture_injection_provenance(env, configured_paths)
            try:
                result, error = super().run_function(env, function, kwargs, raise_on_error=raise_on_error)
            except Exception as exc:
                recorder.record_tool_execution(
                    tool_name=function,
                    arguments=kwargs,
                    error=str(exc),
                    environment_before=environment_before,
                    environment_after=_to_jsonable(env),
                )
                raise
            recorder.record_tool_execution(
                tool_name=function,
                arguments=kwargs,
                result=result,
                error=error,
                environment_before=environment_before,
                environment_after=_to_jsonable(env),
                injection_vector_ids=_result_injection_vector_ids(result, provenance_index),
            )
            return result, error

    return InstrumentedRuntime


def _injection_provenance_paths(
    environment_payload: Any,
    injections: Mapping[str, str],
) -> dict[tuple[str | int, ...], tuple[str, ...]]:
    active_ids = tuple(str(vector_id) for vector_id in injections)
    paths: dict[tuple[str | int, ...], tuple[str, ...]] = {}

    def visit(value: Any, path: tuple[str | int, ...]) -> None:
        if isinstance(value, str):
            vector_ids = tuple(vector_id for vector_id in active_ids if f"{{{vector_id}}}" in value)
            if vector_ids:
                paths[path] = vector_ids
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                resolved_key: str | int = key
                if isinstance(key, str):
                    for vector_id, injection in injections.items():
                        resolved_key = str(resolved_key).replace(f"{{{vector_id}}}", injection)
                visit(item, (*path, resolved_key))
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                visit(item, (*path, index))

    visit(environment_payload, ())
    return paths


def _capture_injection_provenance(
    environment: Any,
    provenance_paths: Mapping[tuple[str | int, ...], Sequence[str]],
) -> tuple[dict[int, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    index: dict[int, set[str]] = {}
    text_sources: dict[str, set[str]] = {}
    for path, vector_ids in provenance_paths.items():
        try:
            value = _resolve_environment_path(environment, path)
        except (AttributeError, KeyError, IndexError, TypeError):
            continue
        index.setdefault(id(value), set()).update(str(item) for item in vector_ids)
        if isinstance(value, str) and value:
            text_sources.setdefault(value, set()).update(str(item) for item in vector_ids)
    return (
        {object_id: tuple(sorted(vector_ids)) for object_id, vector_ids in index.items()},
        {text: tuple(sorted(vector_ids)) for text, vector_ids in text_sources.items()},
    )


def _resolve_environment_path(value: Any, path: Sequence[str | int]) -> Any:
    current = value
    for component in path:
        if isinstance(current, Mapping):
            current = current[component]
        elif isinstance(component, int) and isinstance(current, Sequence):
            current = current[component]
        else:
            current = getattr(current, str(component))
    return current


def _result_injection_vector_ids(
    result: Any,
    provenance: tuple[Mapping[int, Sequence[str]], Mapping[str, Sequence[str]]],
) -> tuple[str, ...]:
    provenance_index, text_sources = provenance
    found: set[str] = set()
    visited: set[int] = set()

    def visit(value: Any) -> None:
        object_id = id(value)
        found.update(provenance_index.get(object_id, ()))
        if isinstance(value, str):
            for source_text, vector_ids in text_sources.items():
                if source_text in value:
                    found.update(vector_ids)
        if object_id in visited:
            return
        visited.add(object_id)
        model_fields = getattr(type(value), "model_fields", None)
        if isinstance(model_fields, Mapping):
            for field_name in model_fields:
                visit(getattr(value, field_name))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                visit(item)

    visit(result)
    return tuple(sorted(found))


def install_openai_request_capture() -> None:
    _install_openai_sdk_request_capture()
    try:
        openai_module = importlib.import_module("agentdojo.agent_pipeline.llms.openai_llm")
    except Exception:
        return
    if getattr(openai_module, "_ipi_aware_lite_request_capture_installed", False):
        return

    original_chat_completion_request = openai_module.chat_completion_request
    import inspect as _inspect
    _supported_chat_completion_kwargs = set(
        _inspect.signature(original_chat_completion_request).parameters
    )

    def wrapped_chat_completion_request(
        client: Any,
        model: str,
        messages: Any,
        tools: Any,
        reasoning_effort: Any,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        max_completion_tokens: int | None = None,
        allow_empty_output: bool = False,
        extra_body_updates: Mapping[str, Any] | None = None,
    ) -> Any:
        import time as _time
        before_count = _active_request_count()
        n_msgs = len(messages) if messages else 0
        _diag_log(f"LLM_CALL start n_msgs={n_msgs} has_tools={bool(tools)}")
        _llm_t0 = _time.monotonic()
        try:
            request_kwargs = {
                "client": client,
                "model": model,
                "messages": messages,
                "tools": tools,
                "reasoning_effort": reasoning_effort,
                "temperature": temperature,
                "top_p": top_p,
                "max_completion_tokens": max_completion_tokens,
                "allow_empty_output": allow_empty_output,
                "extra_body_updates": extra_body_updates,
            }
            response = original_chat_completion_request(
                **{
                    key: value
                    for key, value in request_kwargs.items()
                    if key in _supported_chat_completion_kwargs
                }
            )
        except Exception as _llm_exc:
            _llm_dt = _time.monotonic() - _llm_t0
            _diag_log(f"LLM_CALL FAIL dt={_llm_dt:.1f}s err={type(_llm_exc).__name__}")
            raise
        _llm_dt = _time.monotonic() - _llm_t0
        n_choices = len(response.choices) if response and response.choices else 0
        _diag_log(f"LLM_CALL done  dt={_llm_dt:.1f}s choices={n_choices}")
        if _active_request_count() == before_count:
            _record_active_request(
                request_payload={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto" if tools else None,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_completion_tokens": max_completion_tokens,
                    "reasoning_effort": reasoning_effort,
                    "extra_body_updates": dict(extra_body_updates or {}),
                },
                response_payload=_to_jsonable(response),
                prompt_token_ids=_extract_response_prompt_token_ids(response),
                response_token_ids=_extract_response_token_ids(response),
            )
        return response

    openai_module.chat_completion_request = wrapped_chat_completion_request
    openai_module._ipi_aware_lite_request_capture_installed = True


def _install_openai_sdk_request_capture() -> None:
    try:
        completions_module = importlib.import_module("openai.resources.chat.completions")
    except Exception:
        return
    if getattr(completions_module, "_ipi_aware_lite_request_capture_installed", False):
        return

    completions_class = getattr(completions_module, "Completions", None)
    async_completions_class = getattr(completions_module, "AsyncCompletions", None)

    if completions_class is not None:
        original_create = completions_class.create

        def wrapped_create(self, *args: Any, **kwargs: Any):
            response = original_create(self, *args, **kwargs)
            _record_active_request(
                request_payload=_sdk_request_payload(args, kwargs),
                response_payload=_to_jsonable(response),
                prompt_token_ids=_extract_response_prompt_token_ids(response),
                response_token_ids=_extract_response_token_ids(response),
            )
            return response

        completions_class.create = wrapped_create

    if async_completions_class is not None:
        original_async_create = async_completions_class.create

        async def wrapped_async_create(self, *args: Any, **kwargs: Any):
            response = await original_async_create(self, *args, **kwargs)
            _record_active_request(
                request_payload=_sdk_request_payload(args, kwargs),
                response_payload=_to_jsonable(response),
                prompt_token_ids=_extract_response_prompt_token_ids(response),
                response_token_ids=_extract_response_token_ids(response),
            )
            return response

        async_completions_class.create = wrapped_async_create

    completions_module._ipi_aware_lite_request_capture_installed = True

    try:
        text_completions_module = importlib.import_module("openai.resources.completions")
    except Exception:
        return
    if getattr(text_completions_module, "_ipi_aware_lite_request_capture_installed", False):
        return

    text_completions_class = getattr(text_completions_module, "Completions", None)
    async_text_completions_class = getattr(text_completions_module, "AsyncCompletions", None)

    if text_completions_class is not None:
        original_text_create = text_completions_class.create

        def wrapped_text_create(self, *args: Any, **kwargs: Any):
            response = original_text_create(self, *args, **kwargs)
            _record_active_request(
                request_payload=_sdk_request_payload(args, kwargs),
                response_payload=_to_jsonable(response),
                prompt_token_ids=_extract_response_prompt_token_ids(response),
                response_token_ids=_extract_response_token_ids(response),
            )
            return response

        text_completions_class.create = wrapped_text_create

    if async_text_completions_class is not None:
        original_async_text_create = async_text_completions_class.create

        async def wrapped_async_text_create(self, *args: Any, **kwargs: Any):
            response = await original_async_text_create(self, *args, **kwargs)
            _record_active_request(
                request_payload=_sdk_request_payload(args, kwargs),
                response_payload=_to_jsonable(response),
                prompt_token_ids=_extract_response_prompt_token_ids(response),
                response_token_ids=_extract_response_token_ids(response),
            )
            return response

        async_text_completions_class.create = wrapped_async_text_create

    text_completions_module._ipi_aware_lite_request_capture_installed = True


def _record_active_request(
    *,
    request_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any] | None,
    prompt_token_ids: tuple[int, ...] | None,
    response_token_ids: tuple[int, ...] | None,
) -> None:
    recorder = getattr(_REQUEST_CAPTURE_STATE, "recorder", None)
    if recorder is None:
        return
    recorder.record_request(
        request_payload=request_payload,
        response_payload=response_payload,
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
    )


def discard_last_captured_request() -> None:
    recorder = getattr(_REQUEST_CAPTURE_STATE, "recorder", None)
    if recorder is None or not recorder.requests:
        return
    recorder.requests.pop()


def update_last_captured_response_payload(response_payload: Mapping[str, Any] | None) -> None:
    recorder = getattr(_REQUEST_CAPTURE_STATE, "recorder", None)
    if recorder is None:
        return
    recorder.update_last_response_payload(response_payload)


def _active_request_count() -> int:
    recorder = getattr(_REQUEST_CAPTURE_STATE, "recorder", None)
    if recorder is None:
        return 0
    return len(recorder.requests)


def _get_worker_resources(
    *,
    suite_name: str,
    benchmark_version: str,
    model: str,
    model_id: str | None,
    attack_name: str | None,
    defense: str | None,
    tool_delimiter: str,
    system_message_name: str | None,
    system_message: str | None,
    tool_output_format: str | None,
    temperature: float | None,
    top_p: float | None,
    max_completion_tokens: int | None,
    client_timeout_seconds: float | None = None,
) -> tuple[Any, Any, Any | None]:
    key = (
        suite_name,
        benchmark_version,
        model,
        model_id,
        attack_name,
        defense,
        tool_delimiter,
        system_message_name,
        system_message,
        tool_output_format,
        temperature,
        top_p,
        max_completion_tokens,
        client_timeout_seconds,
    )
    if key not in _PROCESS_RESOURCE_CACHE:
        load_suites = importlib.import_module("agentdojo.task_suite.load_suites")
        pipeline_module = importlib.import_module("agentdojo.agent_pipeline.agent_pipeline")
        attack_registry = importlib.import_module("agentdojo.attacks.attack_registry")
        suite = load_suites.get_suite(benchmark_version, suite_name)
        pipeline = pipeline_module.AgentPipeline.from_config(
            pipeline_module.PipelineConfig(
                llm=model,
                model_id=model_id,
                defense=defense,
                tool_delimiter=tool_delimiter,
                system_message_name=system_message_name,
                system_message=system_message,
                tool_output_format=tool_output_format,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_completion_tokens,
                client_timeout_seconds=client_timeout_seconds,
            )
        )
        attacker = attack_registry.load_attack(attack_name, suite, pipeline) if attack_name is not None else None
        _PROCESS_RESOURCE_CACHE[key] = (suite, pipeline, attacker)
    return _PROCESS_RESOURCE_CACHE[key]


def _build_assistant_message(
    *,
    response_payload: Mapping[str, Any] | None,
    fallback_message: Mapping[str, Any],
) -> dict[str, Any]:
    if response_payload is None:
        return dict(fallback_message)
    try:
        return assistant_message_from_completion_response(response_payload, fallback_message=fallback_message)
    except Exception:
        return dict(fallback_message)


def _decision_injection_context(
    *,
    assistant_message_index: int,
    assistant_turns: Sequence[tuple[int, Mapping[str, Any]]],
    injection_round_index: Sequence[int],
) -> dict[str, Any]:
    injection_indices = tuple(int(index) for index in injection_round_index)
    prior_injections = tuple(index for index in injection_indices if index < assistant_message_index)
    latest_injection_message_index = max(prior_injections) if prior_injections else None
    latest_message_index = assistant_message_index - 1 if assistant_message_index > 0 else None
    assistant_decisions_since_latest = None
    if latest_injection_message_index is not None:
        assistant_decisions_since_latest = sum(
            1
            for turn_index, _message in assistant_turns
            if latest_injection_message_index < turn_index < assistant_message_index
        )
    return {
        "injection_present": bool(injection_indices),
        "injection_round_index": list(injection_indices) if injection_indices else None,
        "latest_message_index": latest_message_index,
        "latest_message_is_injected": latest_message_index in injection_indices if latest_message_index is not None else False,
        "latest_injection_message_index": latest_injection_message_index,
        "decision_after_injection": latest_injection_message_index is not None,
        "messages_since_latest_injection": (
            assistant_message_index - latest_injection_message_index - 1
            if latest_injection_message_index is not None
            else None
        ),
        "assistant_decisions_since_latest_injection": assistant_decisions_since_latest,
    }


def _compute_injection_round_index(
    *,
    final_messages: Sequence[Mapping[str, Any]],
    injections: Mapping[str, Any],
) -> int | None:
    indices = _compute_injection_round_indices(messages=final_messages, injections=injections)
    return indices[0] if indices else None


def _compute_injection_round_indices(
    *,
    messages: Sequence[Mapping[str, Any]],
    injections: Any,
) -> tuple[int, ...]:
    injection_strings = _extract_injection_strings(injections)
    if not injection_strings:
        return tuple()
    normalized_injections = [text for text in (_normalize_injection_match_text(item) for item in injection_strings) if text]
    indices: list[int] = []
    for message_index, message in enumerate(messages):
        for visible_text in _extract_visible_message_candidates(message):
            if any(injection in visible_text for injection in injection_strings):
                indices.append(message_index)
                break
            normalized_visible_text = _normalize_injection_match_text(visible_text)
            if normalized_visible_text and any(injection in normalized_visible_text for injection in normalized_injections):
                indices.append(message_index)
                break
    return tuple(indices)


def _extract_injection_strings(value: Any) -> tuple[str, ...]:
    strings: list[str] = []
    seen: set[str] = set()
    for item in _extract_strings(value):
        if item and item not in seen:
            seen.add(item)
            strings.append(item)
    return tuple(strings)


def _extract_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        results: list[str] = []
        for item in value.values():
            results.extend(_extract_strings(item))
        return results
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        results: list[str] = []
        for item in value:
            results.extend(_extract_strings(item))
        return results
    return []


def _extract_visible_message_candidates(message: Mapping[str, Any]) -> tuple[str, ...]:
    content = message.get("content")
    if isinstance(content, str):
        return (content,)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("content")
                if text is None:
                    text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        joined = "\n".join(parts)
        return (joined,) if joined else ()
    return (str(content),) if content else ()


def _normalize_injection_match_text(value: Any) -> str:
    text = value if isinstance(value, str) else str(value or "")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    return " ".join(text.split()).strip()


def _infer_attack_family(attack_name: str | None) -> str | None:
    if attack_name is None:
        return None
    if "instruction" in attack_name:
        return "prompt_injection"
    return attack_name


def _sdk_request_payload(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(key): _to_jsonable(value) for key, value in kwargs.items()}
    if args:
        payload["__args__"] = _to_jsonable(list(args))
    return payload


def _extract_response_usage(response_payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(response_payload, Mapping):
        return None
    usage = response_payload.get("usage")
    return _json_mapping(usage) if isinstance(usage, Mapping) else None


def _extract_response_prompt_token_ids(response: Any) -> tuple[int, ...] | None:
    for candidate in (
        _extract_int_sequence_field(response, "prompt_token_ids"),
        _extract_int_sequence_field(_extract_field(response, "usage"), "prompt_token_ids"),
        _extract_int_sequence_field(_first_choice(response), "prompt_token_ids"),
        _extract_int_sequence_field(_extract_field(_first_choice(response), "message"), "prompt_token_ids"),
    ):
        if candidate is not None:
            return candidate
    return None


def _extract_response_token_ids(response: Any) -> tuple[int, ...] | None:
    for candidate in (
        _extract_int_sequence_field(_first_choice(response), "token_ids"),
        _extract_int_sequence_field(_extract_field(_first_choice(response), "message"), "token_ids"),
        _extract_int_sequence_field(response, "completion_token_ids"),
        _extract_int_sequence_field(response, "output_token_ids"),
        _extract_int_sequence_field(response, "token_ids"),
    ):
        if candidate is not None:
            return candidate
    return None


def _first_choice(response: Any) -> Any:
    for field_name in ("choices", "output", "outputs"):
        choices = _extract_field(response, field_name)
        if not choices or isinstance(choices, (str, bytes, bytearray)):
            continue
        if isinstance(choices, Sequence):
            return choices[0]
    return None


def _extract_int_sequence_field(value: Any, field_name: str) -> tuple[int, ...] | None:
    sequence = _extract_field(value, field_name)
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes, bytearray)):
        return None
    try:
        return tuple(int(item) for item in sequence)
    except Exception:
        return None


def _extract_field(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _ensure_vendor_agentdojo_path() -> None:
    if str(VENDORED_AGENTDOJO_SRC) not in sys.path:
        sys.path.insert(0, str(VENDORED_AGENTDOJO_SRC))
    loaded = sys.modules.get("agentdojo")
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if VENDORED_AGENTDOJO_SRC.exists() and loaded_file is not None:
        loaded_path = Path(str(loaded_file)).resolve()
        vendor_root = VENDORED_AGENTDOJO_SRC.resolve()
        if loaded_path != vendor_root and vendor_root not in loaded_path.parents:
            raise RuntimeError(
                "A non-vendored agentdojo module is already loaded from "
                f"{loaded_path}. Restart the process or import this package before "
                "upstream agentdojo so vendor/agentdojo/src is used."
            )


def _normalize_agentdojo_model_name(model: str) -> str:
    _ensure_vendor_agentdojo_path()
    models_enum = importlib.import_module("agentdojo.models").ModelsEnum
    try:
        return models_enum(model).value
    except Exception:
        pass
    try:
        return models_enum[model].value
    except Exception:
        return model


def _clone_environment(environment: Any) -> Any:
    model_copy = getattr(environment, "model_copy", None)
    if callable(model_copy):
        return model_copy(deep=True)
    return copy.deepcopy(environment)


def _safe_model_output(messages: Sequence[Any], model_output_from_messages: Any) -> Any:
    if not messages:
        return None
    try:
        return model_output_from_messages(messages)
    except Exception:
        return None


def _latest_assistant_output_is_empty(recorder: CaseRecorder, messages: Sequence[Mapping[str, Any]]) -> bool:
    assistant_messages = [
        message
        for message in messages
        if isinstance(message, Mapping) and str(message.get("role") or "") == "assistant"
    ]
    if not assistant_messages:
        return True
    fallback_message = assistant_messages[-1]
    response_payload = recorder.requests[-1].response_payload if recorder.requests else None
    assistant_message = _build_assistant_message(
        response_payload=response_payload,
        fallback_message=fallback_message,
    )
    return is_empty_assistant_output_payload(assistant_message)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump())
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return _to_jsonable(dict_method())
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    if hasattr(value, "__dict__"):
        return {str(key): _to_jsonable(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _json_mapping(value: Any) -> dict[str, Any]:
    json_value = _to_jsonable(value)
    if not isinstance(json_value, Mapping):
        raise TypeError(f"expected mapping-compatible value, got {type(value)!r}")
    return dict(json_value)
