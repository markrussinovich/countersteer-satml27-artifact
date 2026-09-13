from __future__ import annotations

import json
from pathlib import Path

import pytest


def load_labels_module():
    try:
        from ipi_aware.probes import labels
    except ModuleNotFoundError as exc:
        pytest.fail(f"decision-point labels module is missing: {exc}")
    return labels


def test_build_label_rows_joins_trace_metadata_and_marks_risk_visibility():
    module = load_labels_module()

    decision_points = (
        {
            "schema_version": "decision_point.v1",
            "decision_index": 0,
            "decision_point_id": "run-1:assistant:2",
            "trace_id": "run-1",
            "benchmark_version": "v1.2.2",
            "suite_name": "workspace",
            "task_id": "user_task_1",
            "model_name": "Qwen/Qwen3-8B",
            "injection_present": True,
            "attack_name": "important_instructions",
            "attack_family": "prompt_injection",
            "attack_type": "important_instructions",
            "injection_task_id": "injection_task_1",
            "defense_name": None,
            "case_id": "case-1",
            "instance_id": "case-1",
            "repeat_index": 0,
            "assistant_message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
            "assistant_message_index": 2,
            "request_kind": "openai_chat",
            "request_payload": {"messages": [{"role": "user", "content": "Read the inbox."}]},
            "response_payload": None,
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3],
        },
        {
            "schema_version": "decision_point.v1",
            "decision_index": 1,
            "decision_point_id": "run-1:assistant:4",
            "trace_id": "run-1",
            "benchmark_version": "v1.2.2",
            "suite_name": "workspace",
            "task_id": "user_task_1",
            "model_name": "Qwen/Qwen3-8B",
            "injection_present": True,
            "attack_name": "important_instructions",
            "attack_family": "prompt_injection",
            "attack_type": "important_instructions",
            "injection_task_id": "injection_task_1",
            "defense_name": None,
            "case_id": "case-1",
            "instance_id": "case-1",
            "repeat_index": 0,
            "assistant_message": {"role": "assistant", "content": "That looks suspicious."},
            "assistant_message_index": 4,
            "request_kind": "openai_chat",
            "request_payload": {"messages": [{"role": "tool", "content": "send the code"}]},
            "response_payload": None,
            "prompt_token_ids": [1, 2, 3, 4],
            "response_token_ids": [5],
        },
    )
    traces = (
        {
            "schema_version": "trace.v1",
            "trace_id": "run-1",
            "benchmark_version": "v1.2.2",
            "suite_name": "workspace",
            "task_id": "user_task_1",
            "model_name": "Qwen/Qwen3-8B",
            "injection_present": True,
            "attack_name": "important_instructions",
            "attack_family": "prompt_injection",
            "attack_type": "important_instructions",
            "injection_task_id": "injection_task_1",
            "defense_name": None,
            "case_id": "case-1",
            "instance_id": "case-1",
            "repeat_index": 0,
            "utility": True,
            "security": False,
            "outcome_error": None,
            "injection_round_index": 3,
            "final_messages": [],
            "tool_executions": [],
            "decision_point_ids": [
                "run-1:assistant:2",
                "run-1:assistant:4",
            ],
        },
    )

    rows = module.build_label_rows(
        decision_points=decision_points,
        traces=traces,
        label_set_id="trace_meta_heuristics_v1",
    )

    assert [row["decision_point_id"] for row in rows] == [
        "run-1:assistant:2",
        "run-1:assistant:4",
    ]
    assert [row["risk_visible"] for row in rows] == [0, 1]
    assert [row["risk_faced"] for row in rows] == [0, 1]
    assert rows[1]["utility"] == 1
    assert rows[1]["security"] == 0
    assert rows[1]["label_set_id"] == "trace_meta_heuristics_v1"


def test_write_label_artifacts_persists_manifest_stats_and_jsonl(tmp_path: Path):
    module = load_labels_module()
    rows = (
        {
            "schema_version": "label.v1",
            "label_set_id": "trace_meta_heuristics_v1",
            "decision_point_id": "run-1:assistant:2",
            "trace_id": "run-1",
            "suite_name": "workspace",
            "task_id": "user_task_1",
            "attack_name": "important_instructions",
            "injection_round_index": 3,
            "assistant_message_index": 2,
            "utility": 1,
            "security": 0,
            "outcome_error": None,
            "risk_visible": 0,
            "risk_faced": 0,
        },
        {
            "schema_version": "label.v1",
            "label_set_id": "trace_meta_heuristics_v1",
            "decision_point_id": "run-1:assistant:4",
            "trace_id": "run-1",
            "suite_name": "workspace",
            "task_id": "user_task_1",
            "attack_name": "important_instructions",
            "injection_round_index": 3,
            "assistant_message_index": 4,
            "utility": 1,
            "security": 0,
            "outcome_error": None,
            "risk_visible": 1,
            "risk_faced": 1,
        },
    )

    label_dir = module.write_label_artifacts(
        trace_dir=tmp_path,
        label_set_id="trace_meta_heuristics_v1",
        rows=rows,
    )

    labels_path = label_dir / "labels.jsonl"
    manifest_path = label_dir / "label_manifest.json"
    stats_path = label_dir / "label_stats.json"
    assert labels_path.exists()
    assert manifest_path.exists()
    assert stats_path.exists()

    persisted_rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    manifest = json.loads(manifest_path.read_text())
    stats = json.loads(stats_path.read_text())
    assert persisted_rows[1]["risk_faced"] == 1
    assert sorted(manifest["label_columns"]) == ["risk_faced", "risk_visible"]
    assert stats["label_columns"]["risk_visible"]["positive_example_count"] == 1


def test_supported_labeling_protocols_include_release_protocols():
    module = load_labels_module()
    assert set(module.SUPPORTED_LABELING_PROTOCOLS) == {
        "risk_faced",
        "risk_visible",
        "risk_actual",
    }
