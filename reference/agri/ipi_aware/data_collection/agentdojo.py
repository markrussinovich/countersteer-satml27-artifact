"""AgentDojo integration for IPI-Aware data collection.

This module owns benchmark-specific prompt adaptation, runtime heuristics, and
the lightweight collector used to emit traces and decision points.
"""

from .agentdojo_adapter import (
    AgentDojoPromptCase,
    build_agentdojo_prompt_case,
)
from .agentdojo_collector import (
    CollectedCaseResult,
    CollectionCase,
    collect_cases,
    default_run_id_prefix,
    expand_collection_cases,
    plan_collection_cases,
    resolve_suite_task_ids,
)
from .agentdojo_parallel import (
    AgentDojoSampleWorkItem,
    AgentDojoSampleWorkResult,
    AgentDojoSuiteSelection,
    aggregate_sample_results,
    build_sample_work_items,
    summarize_sample_work_items,
)
from .agentdojo_runtime import (
    build_failed_sample_result,
    format_sample_progress_message,
    is_context_length_error,
    is_empty_assistant_output_payload,
    is_retryable_vllm_bad_request,
)

__all__ = [
    "AgentDojoPromptCase",
    "AgentDojoSampleWorkItem",
    "AgentDojoSampleWorkResult",
    "AgentDojoSuiteSelection",
    "CollectedCaseResult",
    "CollectionCase",
    "aggregate_sample_results",
    "build_agentdojo_prompt_case",
    "build_failed_sample_result",
    "build_sample_work_items",
    "collect_cases",
    "default_run_id_prefix",
    "expand_collection_cases",
    "format_sample_progress_message",
    "is_context_length_error",
    "is_empty_assistant_output_payload",
    "is_retryable_vllm_bad_request",
    "plan_collection_cases",
    "resolve_suite_task_ids",
    "summarize_sample_work_items",
]
