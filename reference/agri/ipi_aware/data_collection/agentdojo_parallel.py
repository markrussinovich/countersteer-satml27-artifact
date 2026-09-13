from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SampleWorkKind = Literal["clean_user_task", "injection_task_utility", "attacked_pair"]


@dataclass(frozen=True, slots=True)
class AgentDojoSuiteSelection:
    suite_name: str
    user_task_ids: tuple[str, ...]
    injection_task_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentDojoSampleWorkItem:
    kind: SampleWorkKind
    suite_name: str
    user_task_id: str
    injection_task_id: str | None = None


@dataclass(frozen=True, slots=True)
class AgentDojoSampleWorkResult:
    kind: SampleWorkKind
    suite_name: str
    user_task_id: str
    injection_task_id: str | None
    utility: bool
    security: bool
    error: str | None = None
    failed: bool = False


def build_sample_work_items(
    selections: tuple[AgentDojoSuiteSelection, ...],
    *,
    attack_enabled: bool,
) -> tuple[AgentDojoSampleWorkItem, ...]:
    work_items: list[AgentDojoSampleWorkItem] = []
    for selection in selections:
        if not attack_enabled:
            for user_task_id in selection.user_task_ids:
                work_items.append(
                    AgentDojoSampleWorkItem(
                        kind="clean_user_task",
                        suite_name=selection.suite_name,
                        user_task_id=user_task_id,
                    )
                )
            continue

        for injection_task_id in selection.injection_task_ids:
            work_items.append(
                AgentDojoSampleWorkItem(
                    kind="injection_task_utility",
                    suite_name=selection.suite_name,
                    user_task_id=injection_task_id,
                    injection_task_id=injection_task_id,
                )
            )
        for user_task_id in selection.user_task_ids:
            for injection_task_id in selection.injection_task_ids:
                work_items.append(
                    AgentDojoSampleWorkItem(
                        kind="attacked_pair",
                        suite_name=selection.suite_name,
                        user_task_id=user_task_id,
                        injection_task_id=injection_task_id,
                    )
                )
    return tuple(work_items)


def summarize_sample_work_items(work_items: tuple[AgentDojoSampleWorkItem, ...]) -> dict[str, int]:
    summary = {
        "clean_user_task": 0,
        "injection_task_utility": 0,
        "attacked_pair": 0,
        "total": len(work_items),
    }
    for work_item in work_items:
        summary[work_item.kind] += 1
    return summary


def aggregate_sample_results(
    suite_names: tuple[str, ...],
    work_results: tuple[AgentDojoSampleWorkResult, ...],
) -> dict[str, dict[str, dict]]:
    aggregated: dict[str, dict[str, dict]] = {
        suite_name: {
            "utility_results": {},
            "security_results": {},
            "injection_tasks_utility_results": {},
        }
        for suite_name in suite_names
    }

    for work_result in work_results:
        suite_results = aggregated[work_result.suite_name]
        if work_result.kind == "clean_user_task":
            suite_results["utility_results"][(work_result.user_task_id, "")] = work_result.utility
            suite_results["security_results"][(work_result.user_task_id, "")] = work_result.security
            continue
        if work_result.kind == "injection_task_utility":
            injection_task_id = _require_injection_task_id(work_result)
            suite_results["injection_tasks_utility_results"][injection_task_id] = work_result.utility
            continue

        injection_task_id = _require_injection_task_id(work_result)
        suite_results["utility_results"][(work_result.user_task_id, injection_task_id)] = work_result.utility
        suite_results["security_results"][(work_result.user_task_id, injection_task_id)] = work_result.security

    return aggregated


def _require_injection_task_id(work_result: AgentDojoSampleWorkResult) -> str:
    if work_result.injection_task_id is None:
        raise ValueError("injection_task_id is required for this work result")
    return work_result.injection_task_id
