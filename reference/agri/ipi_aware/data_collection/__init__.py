"""Data-collection surface for benchmark-driven trace and decision-point capture.

This namespace owns shared collection schema plus benchmark-specific collection
integrations. The current benchmark integration is AgentDojo, but the package
boundary is intentionally benchmark-agnostic.
"""

from .decision_points import (
    append_decision_points_jsonl,
    extract_decision_points,
)
from .injection_text import (
    extract_injection_strings,
    extract_visible_message_candidates,
    extract_visible_message_text,
    normalize_injection_match_text,
    substitute_injection_placeholders,
)
from .schema import (
    DecisionPointRecord,
    ModelRequestRecord,
    RunMetadata,
    RunTrace,
    TaskOutcome,
    ToolExecutionRecord,
)

__all__ = [
    "DecisionPointRecord",
    "ModelRequestRecord",
    "RunMetadata",
    "RunTrace",
    "TaskOutcome",
    "ToolExecutionRecord",
    "append_decision_points_jsonl",
    "extract_decision_points",
    "extract_injection_strings",
    "extract_visible_message_candidates",
    "extract_visible_message_text",
    "normalize_injection_match_text",
    "substitute_injection_placeholders",
]
