from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ipi_aware.data_collection.agentdojo import (
    collect_cases,
    default_run_id_prefix,
    expand_collection_cases,
    plan_collection_cases,
    resolve_suite_task_ids,
)

from ..io import coerce_decision_point_row, coerce_trace_row, read_json, write_json
from ..utils import GRID_POINT_SCHEMA, ProductPaths


@dataclass(frozen=True, slots=True)
class TraceCollectionSummary:
    root: Path
    run_name: str
    planned_case_count: int
    completed_trace_count: int
    completed_decision_point_count: int
    failed_case_count: int
    skipped_existing_case_count: int
    grid_point_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "run_name": self.run_name,
            "planned_case_count": self.planned_case_count,
            "completed_trace_count": self.completed_trace_count,
            "completed_decision_point_count": self.completed_decision_point_count,
            "failed_case_count": self.failed_case_count,
            "skipped_existing_case_count": self.skipped_existing_case_count,
            "grid_point_ids": list(self.grid_point_ids),
        }


def collect_agentdojo_v2(
    *,
    results_dir: str | Path = "results/probe_traces/v2",
    run_name: str | None = None,
    model: str = "vllm_parsed",
    model_id: str | None = None,
    benchmark_version: str = "v1.2.2",
    suites: Sequence[str] = ("workspace",),
    user_tasks: Sequence[str] = (),
    injection_tasks: Sequence[str] = (),
    attacks: Sequence[str] = (),
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_names: Sequence[str] = (),
    system_messages: Sequence[str] = (),
    tool_output_format: str | None = None,
    temperature: float | None = 0.0,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
    max_workers: int = 1,
    num_processes: int = 1,
    epochs: int = 1,
    continue_on_error: bool = False,
    max_cases: int | None = None,
    include_clean: bool = True,
    include_injected: bool = True,
    plan_only: bool = False,
    case_timeout_seconds: float | None = None,
    case_attempts: int = 3,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    upstream_ports: Sequence[int] | None = None,
    case_key_include_file: str | Path | None = None,
) -> TraceCollectionSummary:
    resolved_run_name = run_name or f"{default_run_id_prefix()}-v2"
    root = Path(results_dir) / resolved_run_name
    paths = ProductPaths.from_root(root)
    paths.grid_points_dir.mkdir(parents=True, exist_ok=True)

    system_message_variants = resolve_system_message_variants(
        system_message_names=system_message_names,
        system_messages=system_messages,
    )
    planned_cases = []
    for suite_name in suites:
        user_task_ids, injection_task_ids = resolve_suite_task_ids(
            benchmark_version=benchmark_version,
            suite_name=suite_name,
            user_task_ids=tuple(user_tasks),
            injection_task_ids=tuple(injection_tasks),
            include_injected=include_injected,
        )
        planned_cases.extend(
            plan_collection_cases(
                suite_name=suite_name,
                user_task_ids=user_task_ids,
                include_clean=include_clean,
                include_injected=include_injected,
                attack_names=tuple(attacks),
                injection_task_ids=injection_task_ids,
                system_message_variants=system_message_variants,
            )
        )
    planned_cases = list(expand_collection_cases(planned_cases, epochs=epochs))
    included_case_keys = _load_case_key_filter(case_key_include_file)
    if included_case_keys is not None:
        planned_cases = [
            case for case in planned_cases
            if _case_key(case.case_id, case.repeat_index) in included_case_keys
        ]
    if max_cases is not None:
        planned_cases = planned_cases[: int(max_cases)]

    completed = _completed_case_keys(root)
    original_planned_count = len(planned_cases)
    planned_cases = [
        case for case in planned_cases
        if _case_key(case.case_id, case.repeat_index) not in completed
    ]
    skipped_existing_case_count = original_planned_count - len(planned_cases)

    # Sort by prefix key so cases sharing the same system_message + attack + suite
    # are contiguous in the list.  When combined with fixed worker→port routing
    # this maximises KV-cache prefix reuse on each vLLM server.
    planned_cases.sort(key=lambda c: (c.system_message_name or "", c.attack_name or "", c.suite_name))

    import sys
    print(f"[collect_agentdojo_v2] {len(planned_cases)} cases to collect "
          f"({skipped_existing_case_count} skipped), "
          f"num_processes={num_processes}, max_workers={max_workers}, "
          f"upstream_ports={list(upstream_ports) if upstream_ports else None}",
          file=sys.stderr, flush=True)
    root_manifest = {
        "schema": "ipi_aware.probe_run.v2",
        "run_name": resolved_run_name,
        "benchmark_version": benchmark_version,
        "model": model,
        "model_id": model_id,
        "grid_points_dir": "grid_points",
        "planned_case_count": len(planned_cases),
        "skipped_existing_case_count": skipped_existing_case_count,
        "system_message_variants": list(system_message_variants),
    }
    write_json(paths.root / "manifest.json", root_manifest)
    if plan_only:
        return TraceCollectionSummary(
            root=root,
            run_name=resolved_run_name,
            planned_case_count=len(planned_cases),
            completed_trace_count=0,
            completed_decision_point_count=0,
            failed_case_count=0,
            skipped_existing_case_count=skipped_existing_case_count,
            grid_point_ids=(),
        )

    trace_count = 0
    decision_point_count = 0
    failed_case_count = 0
    grid_point_ids: set[str] = set()
    start_time = time.monotonic()
    _write_buffer: list[Any] = []
    _FLUSH_INTERVAL = 200
    _last_flush = time.monotonic()
    _FLUSH_SECONDS = 60.0

    def _flush_buffer() -> None:
        nonlocal _write_buffer, _last_flush
        if not _write_buffer:
            return
        for buf_result in _write_buffer:
            write_case_result_v2(
                root=root,
                result=buf_result,
                benchmark_version=benchmark_version,
                model_id=model_id or model,
            )
        _write_buffer = []
        _last_flush = time.monotonic()

    def _on_case_finished(completed_count: int, total: int, index: int, result: Any) -> None:
        nonlocal trace_count, decision_point_count, failed_case_count
        _write_buffer.append(result)
        elapsed = time.monotonic()
        if len(_write_buffer) >= _FLUSH_INTERVAL or (elapsed - _last_flush) >= _FLUSH_SECONDS:
            _flush_buffer()
        grid_point_id = grid_point_id_for_case(result.case)
        grid_point_ids.add(grid_point_id)
        if result.trace_row is not None:
            trace_count += 1
        decision_point_count += len(result.decision_rows)
        if getattr(result, "failed", False) or result.trace_row is None:
            failed_case_count += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "completed": completed_count,
                    "total": total,
                    "index": index,
                    "grid_point_id": grid_point_id,
                    "trace_count": trace_count,
                    "decision_point_count": decision_point_count,
                    "failed_case_count": failed_case_count,
                    "elapsed_seconds": round(time.monotonic() - start_time, 3),
                }
            )

    collect_cases(
        planned_cases,
        write_root=str(root) if num_processes > 1 else None,
        benchmark_version=benchmark_version,
        model=model,
        run_id_prefix=resolved_run_name,
        model_id=model_id,
        defense=defense,
        tool_delimiter=tool_delimiter,
        tool_output_format=tool_output_format,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        max_workers=max_workers,
        num_processes=num_processes,
        continue_on_error=continue_on_error,
        case_timeout_seconds=case_timeout_seconds,
        case_attempts=case_attempts,
        progress_callback=_on_case_finished,
        upstream_ports=tuple(upstream_ports) if upstream_ports else None,
    )
    _flush_buffer()
    root_manifest.update(
        {
            "completed_trace_count": trace_count,
            "completed_decision_point_count": decision_point_count,
            "failed_case_count": failed_case_count,
            "grid_points": sorted(grid_point_ids),
            "elapsed_seconds": round(time.monotonic() - start_time, 3),
        }
    )
    write_json(paths.root / "manifest.json", root_manifest)
    return TraceCollectionSummary(
        root=root,
        run_name=resolved_run_name,
        planned_case_count=len(planned_cases),
        completed_trace_count=trace_count,
        completed_decision_point_count=decision_point_count,
        failed_case_count=failed_case_count,
        skipped_existing_case_count=skipped_existing_case_count,
        grid_point_ids=tuple(sorted(grid_point_ids)),
    )


def write_case_result_v2(
    *,
    root: str | Path,
    result: Any,
    benchmark_version: str,
    model_id: str,
) -> Path:
    grid_point_id = grid_point_id_for_case(result.case)
    paths = ProductPaths.from_root(root)
    gp_dir = paths.grid_point_dir(grid_point_id)
    gp_dir.mkdir(parents=True, exist_ok=True)
    trace_delta = 0
    dp_delta = 0
    failure_delta = 0
    if result.trace_row is not None:
        _append_jsonl(paths.traces(grid_point_id), coerce_trace_row(result.trace_row, grid_point_id=grid_point_id))
        trace_delta = 1
    if result.decision_rows:
        for row in result.decision_rows:
            _append_jsonl(paths.decision_points(grid_point_id), coerce_decision_point_row(row, grid_point_id=grid_point_id))
        dp_delta = len(result.decision_rows)
    if getattr(result, "failed", False) or result.trace_row is None:
        _append_jsonl(
            gp_dir / "failures.jsonl",
            {
                "case_id": result.case.case_id,
                "repeat_index": result.case.repeat_index,
                "run_id": result.run_id,
                "error": getattr(result, "error", None),
                "error_type": getattr(result, "error_type", None),
                "error_traceback": getattr(result, "error_traceback", None),
            },
        )
        failure_delta = 1
    _write_grid_point_manifest(
        root=root,
        grid_point_id=grid_point_id,
        case=result.case,
        benchmark_version=benchmark_version,
        model_id=model_id,
        trace_delta=trace_delta,
        decision_point_delta=dp_delta,
        failure_delta=failure_delta,
    )
    return gp_dir


def grid_point_id_for_case(case: Any) -> str:
    attack_name = "clean" if case.attack_name in (None, "") else str(case.attack_name)
    return f"{case.suite_name}__{case.system_prompt_key or 'unknown'}__{attack_name}"


def resolve_system_message_variants(
    *,
    system_message_names: Sequence[str],
    system_messages: Sequence[str],
) -> tuple[dict[str, str | None], ...]:
    variants: list[dict[str, str | None]] = []
    for name in system_message_names:
        variants.append(
            {
                "system_prompt_key": name,
                "system_message_name": name,
                "system_message": None,
            }
        )
    for index, message in enumerate(system_messages):
        variants.append(
            {
                "system_prompt_key": f"inline_{index}",
                "system_message_name": None,
                "system_message": message,
            }
        )
    if variants:
        return tuple(variants)
    return ({"system_prompt_key": "default", "system_message_name": None, "system_message": None},)


def _write_grid_point_manifest(
    *,
    root: str | Path,
    grid_point_id: str,
    case: Any,
    benchmark_version: str,
    model_id: str,
    trace_delta: int = 0,
    decision_point_delta: int = 0,
    failure_delta: int = 0,
) -> None:
    import fcntl

    paths = ProductPaths.from_root(root)
    lock_path = paths.grid_point_dir(grid_point_id) / ".manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            attack_name = "clean" if case.attack_name in (None, "") else str(case.attack_name)
            manifest_path = paths.grid_point_manifest(grid_point_id)
            prev = read_json(manifest_path) if manifest_path.exists() else {}
            payload = {
                "schema": GRID_POINT_SCHEMA,
                "grid_point_id": grid_point_id,
                "suite_name": str(case.suite_name),
                "system_prompt_key": str(case.system_prompt_key or "default"),
                "attack_name": attack_name,
                "benchmark_version": str(benchmark_version),
                "model_id": str(model_id),
                "trace_count": prev.get("trace_count", 0) + trace_delta,
                "decision_point_count": prev.get("decision_point_count", 0) + decision_point_delta,
                "failure_count": prev.get("failure_count", 0) + failure_delta,
            }
            write_json(manifest_path, payload)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _completed_case_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for trace_path in root.glob("grid_points/*/traces.jsonl"):
        with trace_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                cid_tag = '"case_id": "'
                cid_pos = line.find(cid_tag)
                if cid_pos < 0:
                    continue
                cid_pos += len(cid_tag)
                cid_end = line.find('"', cid_pos)
                if cid_end < 0:
                    continue
                case_id = line[cid_pos:cid_end]
                ri_tag = '"repeat_index": '
                ri_pos = line.find(ri_tag)
                ri = 0
                if ri_pos >= 0:
                    ri_pos += len(ri_tag)
                    while ri_pos < len(line) and line[ri_pos].isdigit():
                        ri = ri * 10 + int(line[ri_pos])
                        ri_pos += 1
                keys.add(_case_key(case_id, ri))
    return keys


def _case_key(case_id: str, repeat_index: int) -> str:
    return f"{case_id}__repeat_{int(repeat_index)}"


def _load_case_key_filter(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    input_path = Path(path)
    keys: set[str] = set()
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            if text.startswith("{"):
                row = json.loads(text)
                case_id = row.get("case_id")
                repeat_index = row.get("repeat_index", 0)
                if not case_id:
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    case_id = metadata.get("case_id")
                    repeat_index = metadata.get("repeat_index", repeat_index)
                if case_id:
                    keys.add(_case_key(str(case_id), int(repeat_index)))
            else:
                keys.add(text)
    return keys


def _append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    import fcntl

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _count_jsonl_rows(path: str | Path) -> int:
    input_path = Path(path)
    if not input_path.exists():
        return 0
    count = 0
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count
