from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from ipi_aware.data_collection.schema import DecisionPointRecord, RunTrace

from ipi_aware.data_collection.decision_points import extract_decision_points
from ipi_aware.data_collection.injection_text import extract_injection_strings
from ipi_aware.data_collection.schema import DecisionPointRecord, RunTrace


_REPEAT_SUFFIX_RE = re.compile(r"__repeat_\d+$")
_IGNORED_DATASET_ROW_KEYS = frozenset(
    {
        "example_id",
        "messages_before",
        "tool_context",
        "assistant_message",
        "replay_request_kind",
        "replay_request",
        "prompt_token_ids",
        "response_token_ids",
        "metadata",
    }
)


@dataclass(frozen=True, slots=True)
class ProbeDatasetRow:
    decision_point_id: str
    trace_id: str
    case_id: str | None
    instance_id: str | None
    repeat_index: int
    suite_name: str
    task_id: str
    benchmark_version: str
    model_name: str
    system_prompt_key: str | None
    decision_index: int
    assistant_message_index: int
    injection_present: bool
    attack_name: str | None
    attack_family: str | None
    attack_type: str | None
    injection_task_id: str | None
    defense_name: str | None
    utility: int | None
    security: int | None
    outcome_error: str | None
    trace_version: str
    injection_strings: tuple[str, ...]
    labels: dict[str, int]

    @property
    def risk_visible(self) -> int:
        return int(self.labels.get("risk_visible", 0))

    @property
    def risk_faced(self) -> int:
        return int(self.labels.get("risk_faced", 0))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "decision_point_id": self.decision_point_id,
            "trace_id": self.trace_id,
            "run_id": self.trace_id,
            "case_id": self.case_id,
            "instance_id": self.instance_id,
            "repeat_index": self.repeat_index,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "benchmark_version": self.benchmark_version,
            "model_name": self.model_name,
            "system_prompt_key": self.system_prompt_key,
            "decision_index": self.decision_index,
            "assistant_message_index": self.assistant_message_index,
            "injection_present": self.injection_present,
            "attack_name": self.attack_name,
            "attack_family": self.attack_family,
            "attack_type": self.attack_type,
            "injection_task_id": self.injection_task_id,
            "defense_name": self.defense_name,
            "utility": self.utility,
            "security": self.security,
            "outcome_error": self.outcome_error,
            "trace_version": self.trace_version,
            "injection_strings": list(self.injection_strings),
        }
        for label_name, label_value in sorted(self.labels.items()):
            payload[label_name] = int(label_value)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProbeDatasetRow:
        payload_dict = dict(payload)
        injection_strings = tuple(str(value) for value in payload_dict.pop("injection_strings", ()) or ())
        core_field_names = {field.name for field in fields(cls) if field.name != "labels"}
        labels: dict[str, int] = {}
        core_fields: dict[str, Any] = {}
        for key, value in payload_dict.items():
            if key == "run_id" and "trace_id" not in core_fields:
                core_fields["trace_id"] = value
                continue
            if key == "example_id":
                if "decision_point_id" not in core_fields:
                    core_fields["decision_point_id"] = value
                continue
            if key in _IGNORED_DATASET_ROW_KEYS:
                continue
            if key in core_field_names:
                core_fields[key] = value
            else:
                labels[str(key)] = int(value)
        return cls(
            decision_point_id=str(
                core_fields.get("decision_point_id")
            ),
            trace_id=str(core_fields.get("trace_id") or core_fields.get("run_id")),
            case_id=_optional_string(core_fields.get("case_id")),
            instance_id=_optional_string(core_fields.get("instance_id")),
            repeat_index=int(core_fields.get("repeat_index") or 0),
            suite_name=str(core_fields["suite_name"]),
            task_id=str(core_fields["task_id"]),
            benchmark_version=str(core_fields["benchmark_version"]),
            model_name=str(core_fields["model_name"]),
            system_prompt_key=_optional_string(core_fields.get("system_prompt_key")),
            decision_index=int(core_fields["decision_index"]),
            assistant_message_index=int(core_fields["assistant_message_index"]),
            injection_present=bool(core_fields["injection_present"]),
            attack_name=_optional_string(core_fields.get("attack_name")),
            attack_family=_optional_string(core_fields.get("attack_family")),
            attack_type=_optional_string(core_fields.get("attack_type")),
            injection_task_id=_optional_string(core_fields.get("injection_task_id")),
            defense_name=_optional_string(core_fields.get("defense_name")),
            utility=_optional_int(core_fields.get("utility")),
            security=_optional_int(core_fields.get("security")),
            outcome_error=_optional_string(core_fields.get("outcome_error")),
            trace_version=str(core_fields.get("trace_version") or "v2"),
            injection_strings=injection_strings,
            labels=labels,
        )

    @property
    def run_id(self) -> str:
        return self.trace_id


def probe_example_id(decision_point: DecisionPointRecord) -> str:
    return decision_point.decision_point_id


def default_probe_dataset_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"probe-index-{timestamp}"


def build_probe_dataset_rows(
    decision_points: Sequence[DecisionPointRecord],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[ProbeDatasetRow, ...]:
    rows: list[ProbeDatasetRow] = []
    total_points = len(decision_points)
    for completed, point in enumerate(decision_points, start=1):
        metadata = point.metadata
        case_id, instance_id, repeat_index = _extract_case_identity(metadata)
        rows.append(
            ProbeDatasetRow(
                decision_point_id=point.decision_point_id,
                trace_id=str(point.trace_id or metadata.get("trace_id") or metadata.get("run_id") or "unknown-trace"),
                case_id=case_id,
                instance_id=instance_id,
                repeat_index=repeat_index,
                suite_name=str(metadata.get("suite_name") or "unknown-suite"),
                task_id=str(metadata.get("task_id") or "unknown-task"),
                benchmark_version=str(metadata.get("benchmark_version") or "unknown-version"),
                model_name=str(metadata.get("model_name") or "unknown-model"),
                system_prompt_key=_extract_system_prompt_key(metadata),
                decision_index=point.decision_index,
                assistant_message_index=point.assistant_message_index,
                injection_present=bool(metadata.get("injection_present")),
                attack_name=_optional_string(metadata.get("attack_name")),
                attack_family=_optional_string(metadata.get("attack_family")),
                attack_type=_optional_string(metadata.get("attack_type")),
                injection_task_id=_optional_string(metadata.get("injection_task_id")),
                defense_name=_optional_string(metadata.get("defense_name")),
                utility=_optional_int(metadata.get("utility")),
                security=_optional_int(metadata.get("security")),
                outcome_error=_optional_string(metadata.get("outcome_error")),
                trace_version=str(metadata.get("trace_version") or "v2"),
                injection_strings=_extract_row_injection_strings(metadata),
                labels={},
            )
        )
        if callable(progress_callback):
            progress_callback(completed, total_points)
    return tuple(rows)


def extract_decision_points_from_traces(
    traces: Sequence[RunTrace],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[DecisionPointRecord, ...]:
    decision_points: list[DecisionPointRecord] = []
    total_traces = len(traces)
    for completed, trace in enumerate(traces, start=1):
        decision_points.extend(extract_decision_points(trace))
        if callable(progress_callback):
            progress_callback(completed, total_traces)
    return tuple(decision_points)


def load_trace_rows(path: str | Path) -> tuple[RunTrace, ...]:
    input_path = Path(path)
    if input_path.is_dir():
        input_path = input_path / "traces.jsonl"
    traces: list[RunTrace] = []
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        trace = RunTrace.from_dict(json.loads(line))
        traces.append(trace)
    return tuple(traces)


def load_decision_point_rows(path: str | Path) -> tuple[DecisionPointRecord, ...]:
    input_path = Path(path)
    if input_path.is_dir():
        input_path = input_path / "decision_points.jsonl"
    points: list[DecisionPointRecord] = []
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        points.append(DecisionPointRecord.from_dict(json.loads(line)))
    return tuple(points)


def build_probe_dataset_rows_from_traces(
    traces: Sequence[RunTrace],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[ProbeDatasetRow, ...]:
    decision_points = extract_decision_points_from_traces(
        traces,
        progress_callback=progress_callback,
    )
    return build_probe_dataset_rows(decision_points)


def create_base_partition(
    rows: Sequence[ProbeDatasetRow],
    *,
    heldout_attack: str | None,
    heldout_suite: str | None,
    heldout_system_prompt: str | None = None,
    heldin_eval_ratio: float = 0.1,
    val_point_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    if not 0.0 <= heldin_eval_ratio <= 1.0:
        raise ValueError("heldin_eval_ratio must be between 0.0 and 1.0")
    if not 0.0 <= val_point_ratio <= 1.0:
        raise ValueError("val_point_ratio must be between 0.0 and 1.0")

    case_rows = _group_rows_by_case(rows)
    attack_case_ids = {
        case_id
        for case_id, grouped_rows in case_rows.items()
        if heldout_attack is not None and any(row.attack_name == heldout_attack for row in grouped_rows)
    }
    suite_case_ids = {
        case_id
        for case_id, grouped_rows in case_rows.items()
        if heldout_suite is not None and any(row.suite_name == heldout_suite for row in grouped_rows)
    }
    system_prompt_case_ids = {
        case_id
        for case_id, grouped_rows in case_rows.items()
        if heldout_system_prompt is not None and any(row.system_prompt_key == heldout_system_prompt for row in grouped_rows)
    }
    strict_case_ids = attack_case_ids & suite_case_ids & system_prompt_case_ids if heldout_system_prompt is not None else attack_case_ids & suite_case_ids
    attack_only_case_ids = attack_case_ids - strict_case_ids
    suite_only_case_ids = suite_case_ids - strict_case_ids
    system_prompt_only_case_ids = system_prompt_case_ids - strict_case_ids
    remaining_case_ids = sorted(
        set(case_rows) - attack_only_case_ids - suite_only_case_ids - system_prompt_only_case_ids - strict_case_ids
    )

    rng = random.Random(seed)
    heldin_eval_case_ids = _sample_case_ids(
        case_rows=case_rows,
        case_ids=remaining_case_ids,
        ratio=heldin_eval_ratio,
        seed=seed,
        rng=rng,
    )
    trainval_case_ids = sorted(set(remaining_case_ids) - set(heldin_eval_case_ids))

    trainval_rows = [row for row in rows if _row_case_id(row) in set(trainval_case_ids)]
    val_decision_point_ids = _sample_validation_decision_point_ids(
        trainval_rows,
        val_point_ratio=val_point_ratio,
        rng=rng,
    )
    train_decision_point_ids = sorted(
        row.decision_point_id
        for row in trainval_rows
        if row.decision_point_id not in set(val_decision_point_ids)
    )

    base_partition = {
        "partition_version": "v1",
        "seed": seed,
        "heldout_attack": heldout_attack,
        "heldout_suite": heldout_suite,
        "heldout_system_prompt": heldout_system_prompt,
        "heldin_eval_ratio": heldin_eval_ratio,
        "val_point_ratio": val_point_ratio,
        "attack_only_case_ids": sorted(attack_only_case_ids),
        "suite_only_case_ids": sorted(suite_only_case_ids),
        "system_prompt_only_case_ids": sorted(system_prompt_only_case_ids),
        "strict_case_ids": sorted(strict_case_ids),
        "heldin_eval_case_ids": sorted(heldin_eval_case_ids),
        "trainval_case_ids": trainval_case_ids,
        "attack_only_trace_ids": _trace_ids_for_cases(rows, attack_only_case_ids),
        "suite_only_trace_ids": _trace_ids_for_cases(rows, suite_only_case_ids),
        "system_prompt_only_trace_ids": _trace_ids_for_cases(rows, system_prompt_only_case_ids),
        "strict_trace_ids": _trace_ids_for_cases(rows, strict_case_ids),
        "heldin_eval_trace_ids": _trace_ids_for_cases(rows, heldin_eval_case_ids),
        "trainval_trace_ids": _trace_ids_for_cases(rows, trainval_case_ids),
        "attack_only_decision_point_ids": _decision_point_ids_for_cases(rows, attack_only_case_ids),
        "suite_only_decision_point_ids": _decision_point_ids_for_cases(rows, suite_only_case_ids),
        "system_prompt_only_decision_point_ids": _decision_point_ids_for_cases(rows, system_prompt_only_case_ids),
        "strict_decision_point_ids": _decision_point_ids_for_cases(rows, strict_case_ids),
        "heldin_eval_decision_point_ids": _decision_point_ids_for_cases(rows, heldin_eval_case_ids),
        "train_decision_point_ids": train_decision_point_ids,
        "val_decision_point_ids": sorted(val_decision_point_ids),
    }
    return base_partition


def create_grid_partition(
    rows: Sequence[ProbeDatasetRow],
    *,
    train_suite: str,
    train_system_prompt: str,
    train_attacks: Sequence[str],
    val_point_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    if not 0.0 <= val_point_ratio <= 1.0:
        raise ValueError("val_point_ratio must be between 0.0 and 1.0")
    canonical_train_attacks = tuple(sorted({_canonical_attack_name(value) for value in train_attacks}))
    if not canonical_train_attacks:
        raise ValueError("train_attacks must not be empty")

    train_rows = [
        row
        for row in rows
        if row.suite_name == train_suite
        and _optional_string(row.system_prompt_key) == train_system_prompt
        and _canonical_attack_name(row.attack_name) in canonical_train_attacks
    ]
    if not train_rows:
        raise ValueError("grid partition produced an empty training pool")

    train_row_ids = {row.decision_point_id for row in train_rows}
    heldout_rows = [row for row in rows if row.decision_point_id not in train_row_ids]

    rng = random.Random(seed)
    val_decision_point_ids = _sample_validation_decision_point_ids(
        train_rows,
        val_point_ratio=val_point_ratio,
        rng=rng,
    )
    val_id_set = set(val_decision_point_ids)
    train_decision_point_ids = sorted(
        row.decision_point_id
        for row in train_rows
        if row.decision_point_id not in val_id_set
    )

    eval_groups: dict[str, dict[str, Any]] = {}
    grouped_rows: dict[tuple[str, str, str], list[ProbeDatasetRow]] = defaultdict(list)
    for row in heldout_rows:
        grouped_rows[
            (
                row.suite_name,
                _optional_string(row.system_prompt_key) or "unknown",
                _canonical_attack_name(row.attack_name),
            )
        ].append(row)

    for (suite_name, system_prompt_key, attack_name), group_rows in sorted(grouped_rows.items()):
        split_name = f"heldout_grid__{suite_name}__{system_prompt_key}__{attack_name}"
        eval_groups[split_name] = {
            "split_name": split_name,
            "split_kind": "heldout_grid",
            "evaluation_group": split_name,
            "suite_name": suite_name,
            "system_prompt_key": system_prompt_key,
            "attack_name": attack_name,
            "eval_decision_point_ids": sorted(row.decision_point_id for row in group_rows),
            "eval_trace_ids": sorted({row.trace_id for row in group_rows}),
            "eval_case_ids": sorted({_row_case_id(row) for row in group_rows}),
        }

    return {
        "partition_version": "v2_grid_train",
        "split_mode": "grid_train",
        "seed": seed,
        "train_suite": train_suite,
        "train_system_prompt": train_system_prompt,
        "train_attacks": list(canonical_train_attacks),
        "val_point_ratio": val_point_ratio,
        "train_decision_point_ids": train_decision_point_ids,
        "val_decision_point_ids": sorted(val_decision_point_ids),
        "train_trace_ids": sorted({row.trace_id for row in train_rows if row.decision_point_id not in val_id_set}),
        "val_trace_ids": sorted({row.trace_id for row in train_rows if row.decision_point_id in val_id_set}),
        "train_case_ids": sorted({_row_case_id(row) for row in train_rows if row.decision_point_id not in val_id_set}),
        "val_case_ids": sorted({_row_case_id(row) for row in train_rows if row.decision_point_id in val_id_set}),
        "eval_groups": eval_groups,
    }


def create_grid_point_partition_metadata(
    rows: Sequence[ProbeDatasetRow],
    *,
    train_grid_points: Sequence[str],
    eval_grid_points: Sequence[str],
    grid_point_specs: Mapping[str, Mapping[str, Any]],
    val_point_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    if not 0.0 <= val_point_ratio <= 1.0:
        raise ValueError("val_point_ratio must be between 0.0 and 1.0")

    canonical_train_grid_points = list(dict.fromkeys(str(value) for value in train_grid_points))
    canonical_eval_grid_points = list(dict.fromkeys(str(value) for value in eval_grid_points))
    if not canonical_train_grid_points:
        raise ValueError("train_grid_points must not be empty")

    unknown_grid_points = sorted(
        {
            *[grid_point_id for grid_point_id in canonical_train_grid_points if grid_point_id not in grid_point_specs],
            *[grid_point_id for grid_point_id in canonical_eval_grid_points if grid_point_id not in grid_point_specs],
        }
    )
    if unknown_grid_points:
        raise ValueError(f"unknown grid points in split spec: {', '.join(unknown_grid_points)}")

    overlap = sorted(set(canonical_train_grid_points) & set(canonical_eval_grid_points))
    if overlap:
        raise ValueError(f"grid points cannot be both train and eval: {', '.join(overlap)}")

    grouped_rows: dict[str, list[ProbeDatasetRow]] = defaultdict(list)
    for row in rows:
        grouped_rows[grid_point_id_from_row(row)].append(row)

    missing_train_grid_points = sorted(
        grid_point_id for grid_point_id in canonical_train_grid_points if grid_point_id not in grouped_rows
    )
    if missing_train_grid_points:
        raise ValueError(
            "split spec selected train grid points without decision points: "
            + ", ".join(missing_train_grid_points)
        )

    train_rows = [
        row
        for grid_point_id in canonical_train_grid_points
        for row in grouped_rows.get(grid_point_id, ())
    ]
    rng = random.Random(seed)
    val_decision_point_ids = _sample_validation_decision_point_ids(
        train_rows,
        val_point_ratio=val_point_ratio,
        rng=rng,
    )
    val_id_set = set(val_decision_point_ids)
    train_decision_point_ids = sorted(
        row.decision_point_id
        for row in train_rows
        if row.decision_point_id not in val_id_set
    )

    eval_groups: dict[str, dict[str, Any]] = {}
    for grid_point_id in canonical_eval_grid_points:
        spec = dict(grid_point_specs[grid_point_id])
        split_name = f"heldout_grid__{grid_point_id}"
        eval_groups[split_name] = {
            "split_name": split_name,
            "split_kind": "heldout_grid",
            "evaluation_group": split_name,
            "grid_point_id": grid_point_id,
            "grid_point_path": spec.get("grid_point_path"),
            "suite_name": spec.get("suite_name"),
            "system_prompt_key": spec.get("system_prompt_key"),
            "attack_name": spec.get("attack_name"),
        }

    return {
        "partition_version": "v3_grid_point_refs",
        "split_mode": "grid_point_refs",
        "seed": seed,
        "val_point_ratio": val_point_ratio,
        "train_grid_points": canonical_train_grid_points,
        "eval_grid_points": canonical_eval_grid_points,
        "train_decision_point_ids": train_decision_point_ids,
        "val_decision_point_ids": sorted(val_decision_point_ids),
        "train_trace_ids": sorted({row.trace_id for row in train_rows if row.decision_point_id not in val_id_set}),
        "val_trace_ids": sorted({row.trace_id for row in train_rows if row.decision_point_id in val_id_set}),
        "eval_groups": eval_groups,
    }


def create_grid_partition_from_grid_points(
    grid_point_dirs: Sequence[tuple[Path, str]],
    trace_root: Path,
    *,
    train_suite: str,
    train_system_prompt: str,
    train_attacks: Sequence[str],
    val_point_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, Any], tuple[ProbeDatasetRow, ...]]:
    """Build a grid-train partition by iterating grid point directories.

    Unlike ``create_grid_partition`` which requires all decision point rows
    materialized upfront, this function reads only the train grid points'
    decision point files. Held-out eval groups store ``grid_point_path`` so
    that decision point IDs are resolved lazily at probe-training time via
    ``_resolve_eval_group_selection_spec``.

    Returns
    -------
    A tuple of (base_partition_dict, train_rows_tuple). train_rows includes
    all train+val decision points from matching train grid points.

    Parameters
    ----------
    grid_point_dirs
        Sequence of (grid_point_dir, grid_point_id) tuples for all grid points.
    train_suite, train_system_prompt, train_attacks
        Criteria for selecting the train cell(s).
    val_point_ratio
        Fraction of train decision points to hold out for validation.
    seed
        Random seed for reproducible train/val split.
    """
    if not grid_point_dirs:
        raise ValueError("grid_point_dirs must not be empty")
    if not 0.0 <= val_point_ratio <= 1.0:
        raise ValueError("val_point_ratio must be between 0.0 and 1.0")
    canonical_train_attacks = tuple(sorted({_canonical_attack_name(v) for v in train_attacks}))
    if not canonical_train_attacks:
        raise ValueError("train_attacks must not be empty")

    # Phase 1: read manifests to classify each grid point as train or heldout
    grid_point_specs: dict[str, dict[str, Any]] = {}
    train_grid_point_dirs: list[tuple[Path, str]] = []
    for gp_dir, gp_id in grid_point_dirs:
        manifest_path = gp_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        spec = json.loads(manifest_path.read_text())
        grid_point_specs[gp_id] = {
            "grid_point_id": gp_id,
            # Relative path from dataset-index dir (trace_root/dataset-indices/<id>/) to grid point
            "grid_point_path": str(Path("..") / ".." / gp_dir.relative_to(trace_root)),
            "suite_name": spec.get("suite_name"),
            "system_prompt_key": spec.get("system_prompt_key"),
            "attack_name": spec.get("attack_name"),
        }
        matches_suite = spec.get("suite_name") == train_suite
        matches_system = spec.get("system_prompt_key") == train_system_prompt
        matches_attack = _canonical_attack_name(spec.get("attack_name")) in canonical_train_attacks
        if matches_suite and matches_system and matches_attack:
            train_grid_point_dirs.append((gp_dir, gp_id))

    if not train_grid_point_dirs:
        raise ValueError(
            f"no grid points match train criteria: "
            f"suite={train_suite}, system_prompt={train_system_prompt}, attacks={list(canonical_train_attacks)}"
        )

    # Phase 2: load decision points from train grid points only
    train_rows: list[ProbeDatasetRow] = []
    for gp_dir, gp_id in train_grid_point_dirs:
        dp_path = gp_dir / "decision_points.jsonl"
        if not dp_path.exists():
            continue
        decision_points = load_decision_point_rows(dp_path)
        for point in decision_points:
            metadata = point.metadata
            case_id, instance_id, repeat_index = _extract_case_identity(metadata)
            train_rows.append(
                ProbeDatasetRow(
                    decision_point_id=point.decision_point_id,
                    trace_id=str(point.trace_id or metadata.get("trace_id") or metadata.get("run_id") or "unknown-trace"),
                    case_id=case_id,
                    instance_id=instance_id,
                    repeat_index=repeat_index,
                    suite_name=str(metadata.get("suite_name") or "unknown-suite"),
                    task_id=str(metadata.get("task_id") or "unknown-task"),
                    benchmark_version=str(metadata.get("benchmark_version") or "unknown-version"),
                    model_name=str(metadata.get("model_name") or "unknown-model"),
                    system_prompt_key=_extract_system_prompt_key(metadata),
                    decision_index=point.decision_index,
                    assistant_message_index=point.assistant_message_index,
                    injection_present=bool(metadata.get("injection_present")),
                    attack_name=_optional_string(metadata.get("attack_name")),
                    attack_family=_optional_string(metadata.get("attack_family")),
                    attack_type=_optional_string(metadata.get("attack_type")),
                    injection_task_id=_optional_string(metadata.get("injection_task_id")),
                    defense_name=_optional_string(metadata.get("defense_name")),
                    utility=_optional_int(metadata.get("utility")),
                    security=_optional_int(metadata.get("security")),
                    outcome_error=_optional_string(metadata.get("outcome_error")),
                    trace_version=str(metadata.get("trace_version") or "v2"),
                    injection_strings=_extract_row_injection_strings(metadata),
                    labels={},
                )
            )

    if not train_rows:
        raise ValueError("train grid points contain no decision points")

    # Phase 3: train/val split from pooled train rows
    rng = random.Random(seed)
    val_decision_point_ids = _sample_validation_decision_point_ids(
        train_rows,
        val_point_ratio=val_point_ratio,
        rng=rng,
    )
    val_id_set = set(val_decision_point_ids)
    train_decision_point_ids = sorted(
        row.decision_point_id
        for row in train_rows
        if row.decision_point_id not in val_id_set
    )
    train_trace_ids = sorted(
        row.trace_id for row in train_rows if row.decision_point_id not in val_id_set
    )
    val_trace_ids = sorted(
        row.trace_id for row in train_rows if row.decision_point_id in val_id_set
    )
    train_case_ids = sorted(
        {_row_case_id(row) for row in train_rows if row.decision_point_id not in val_id_set}
    )
    val_case_ids = sorted(
        {_row_case_id(row) for row in train_rows if row.decision_point_id in val_id_set}
    )

    # Phase 4: build eval_groups for held-out grid points (lazy — no IDs materialized)
    eval_groups: dict[str, dict[str, Any]] = {}
    train_gp_ids = {gp_id for _, gp_id in train_grid_point_dirs}
    for gp_id, spec in sorted(grid_point_specs.items()):
        if gp_id in train_gp_ids:
            continue
        split_name = f"heldout_grid__{spec['suite_name']}__{spec['system_prompt_key']}__{spec['attack_name']}"
        eval_groups[split_name] = {
            "split_name": split_name,
            "split_kind": "heldout_grid",
            "evaluation_group": split_name,
            "grid_point_id": gp_id,
            "grid_point_path": spec["grid_point_path"],
            "suite_name": spec["suite_name"],
            "system_prompt_key": spec["system_prompt_key"],
            "attack_name": spec["attack_name"],
            # eval_decision_point_ids intentionally omitted — resolved lazily at eval time
        }

    return (
        {
            "partition_version": "v4_lazy_eval",
            "split_mode": "grid_train",
            "seed": seed,
            "train_suite": train_suite,
            "train_system_prompt": train_system_prompt,
            "train_attacks": list(canonical_train_attacks),
            "val_point_ratio": val_point_ratio,
            "train_decision_point_ids": train_decision_point_ids,
            "val_decision_point_ids": sorted(val_decision_point_ids),
            "train_trace_ids": train_trace_ids,
            "val_trace_ids": val_trace_ids,
            "train_case_ids": train_case_ids,
            "val_case_ids": val_case_ids,
            "train_grid_points": [gp_id for _, gp_id in train_grid_point_dirs],
            "eval_groups": eval_groups,
        },
        tuple(train_rows),
    )


def project_split_manifests(base_partition: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if "eval_groups" in base_partition:
        shared = {
            "seed": int(base_partition["seed"]),
            "train_decision_point_ids": list(base_partition.get("train_decision_point_ids") or ()),
            "val_decision_point_ids": list(base_partition.get("val_decision_point_ids") or ()),
            "partition_metadata_path": "partition_metadata.json",
            "base_partition_path": "base_partition.json",
        }
        manifests: dict[str, dict[str, Any]] = {}
        for split_name, group in sorted((base_partition.get("eval_groups") or {}).items()):
            manifests[str(split_name)] = {
                **shared,
                "split_name": str(group.get("split_name") or split_name),
                "split_kind": str(group.get("split_kind") or "heldout_grid"),
                "evaluation_group": str(group.get("evaluation_group") or split_name),
                "grid_point_id": group.get("grid_point_id"),
                "grid_point_path": group.get("grid_point_path"),
                "suite_name": group.get("suite_name"),
                "system_prompt_key": group.get("system_prompt_key"),
                "attack_name": group.get("attack_name"),
                "eval_decision_point_ids": list(group.get("eval_decision_point_ids") or ()),
                "eval_trace_ids": list(group.get("eval_trace_ids") or ()),
                "eval_case_ids": list(group.get("eval_case_ids") or ()),
            }
        return manifests

    manifests: dict[str, dict[str, Any]] = {}
    shared = {
        "seed": int(base_partition["seed"]),
        "train_decision_point_ids": list(base_partition.get("train_decision_point_ids") or ()),
        "val_decision_point_ids": list(base_partition.get("val_decision_point_ids") or ()),
        "base_partition_path": "base_partition.json",
    }

    manifests[f"iid_seed{shared['seed']}"] = {
        **shared,
        "split_name": f"iid_seed{shared['seed']}",
        "split_kind": "heldin_eval",
        "evaluation_group": "heldin_eval",
        "eval_decision_point_ids": list(base_partition.get("heldin_eval_decision_point_ids") or ()),
        "eval_trace_ids": list(base_partition.get("heldin_eval_trace_ids") or ()),
        "eval_case_ids": list(base_partition.get("heldin_eval_case_ids") or ()),
    }

    heldout_attack = _optional_string(base_partition.get("heldout_attack"))
    heldout_suite = _optional_string(base_partition.get("heldout_suite"))
    heldout_system_prompt = _optional_string(base_partition.get("heldout_system_prompt"))

    if heldout_attack is not None:
        manifests[f"heldout_attack__{heldout_attack}"] = {
            **shared,
            "split_name": f"heldout_attack__{heldout_attack}",
            "split_kind": "heldout_attack",
            "evaluation_group": "heldout_attack",
            "heldout_attack": heldout_attack,
            "eval_decision_point_ids": list(base_partition.get("attack_only_decision_point_ids") or ()),
            "eval_trace_ids": list(base_partition.get("attack_only_trace_ids") or ()),
            "eval_case_ids": list(base_partition.get("attack_only_case_ids") or ()),
        }
    if heldout_suite is not None:
        manifests[f"heldout_suite__{heldout_suite}"] = {
            **shared,
            "split_name": f"heldout_suite__{heldout_suite}",
            "split_kind": "heldout_suite",
            "evaluation_group": "heldout_suite",
            "heldout_suite": heldout_suite,
            "eval_decision_point_ids": list(base_partition.get("suite_only_decision_point_ids") or ()),
            "eval_trace_ids": list(base_partition.get("suite_only_trace_ids") or ()),
            "eval_case_ids": list(base_partition.get("suite_only_case_ids") or ()),
        }
    if heldout_system_prompt is not None:
        manifests[f"heldout_system_prompt__{heldout_system_prompt}"] = {
            **shared,
            "split_name": f"heldout_system_prompt__{heldout_system_prompt}",
            "split_kind": "heldout_system_prompt",
            "evaluation_group": "heldout_system_prompt",
            "heldout_system_prompt": heldout_system_prompt,
            "eval_decision_point_ids": list(base_partition.get("system_prompt_only_decision_point_ids") or ()),
            "eval_trace_ids": list(base_partition.get("system_prompt_only_trace_ids") or ()),
            "eval_case_ids": list(base_partition.get("system_prompt_only_case_ids") or ()),
        }
    if heldout_attack is not None and heldout_suite is not None:
        manifests[f"heldout_strict__{heldout_attack}__{heldout_suite}"] = {
            **shared,
            "split_name": f"heldout_strict__{heldout_attack}__{heldout_suite}",
            "split_kind": "heldout_strict",
            "evaluation_group": "heldout_strict",
            "heldout_attack": heldout_attack,
            "heldout_suite": heldout_suite,
            "eval_decision_point_ids": list(base_partition.get("strict_decision_point_ids") or ()),
            "eval_trace_ids": list(base_partition.get("strict_trace_ids") or ()),
            "eval_case_ids": list(base_partition.get("strict_case_ids") or ()),
        }
    return manifests


def default_label_set_id() -> str:
    return "heuristic_risk_v1"


def write_probe_dataset_index(
    *,
    trace_dir: str | Path,
    dataset_id: str,
    rows: Sequence[ProbeDatasetRow],
    label_set_id: str | None = None,
    labels_path: str | None = None,
    labels_paths: Mapping[str, str] | None = None,
    base_partition: Mapping[str, Any],
    split_manifests: Mapping[str, Mapping[str, Any]],
) -> Path:
    trace_root = Path(trace_dir)
    dataset_dir = trace_root / "dataset-indices" / dataset_id
    splits_dir = dataset_dir / "splits"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "base_partition.json").write_text(
        json.dumps(dict(base_partition), ensure_ascii=False, indent=2) + "\n"
    )
    dataset_manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_partition_path": "base_partition.json",
        "split_paths": [
            f"splits/{split_name}.json"
            for split_name in sorted(split_manifests)
        ],
        "decision_point_count": len(rows),
        "trace_count": len({row.trace_id for row in rows}),
        "decision_points_path": "../../decision_points.jsonl",
    }
    # Root features may be absent for per-grid-point storage and are resolved
    # lazily via grid_point_path in eval_groups.
    features_root = trace_root / "features.pt"
    if features_root.exists():
        dataset_manifest["feature_path"] = "../../features.pt"
    else:
        # Root features.pt missing. For per-grid-point storage, derive the train
        # grid point's features path from base_partition.train_grid_points.
        train_grid_points = base_partition.get("train_grid_points") or []
        if len(train_grid_points) == 1:
            # Single train grid point: use its features.pt directly.
            train_gp_id = train_grid_points[0]
            train_gp_features = trace_root / "grid_points" / train_gp_id / "features.pt"
            if train_gp_features.exists():
                dataset_manifest["feature_path"] = f"../../grid_points/{train_gp_id}/features.pt"
                train_gp_feature_manifest = trace_root / "grid_points" / train_gp_id / "feature_manifest.json"
                if train_gp_feature_manifest.exists():
                    dataset_manifest["feature_manifest_path"] = f"../../grid_points/{train_gp_id}/feature_manifest.json"
    feature_manifest_root = trace_root / "feature_manifest.json"
    if feature_manifest_root.exists():
        dataset_manifest["feature_manifest_path"] = "../../feature_manifest.json"
    if label_set_id:
        dataset_manifest["label_set_id"] = label_set_id
    if labels_path:
        dataset_manifest["labels_path"] = labels_path
    if labels_paths:
        dataset_manifest["labels_paths"] = dict(sorted(labels_paths.items()))
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    for split_name, split_manifest in sorted(split_manifests.items()):
        split_path = splits_dir / f"{split_name}.json"
        split_path.write_text(json.dumps(dict(split_manifest), ensure_ascii=False, indent=2) + "\n")
    return dataset_dir


def write_probe_partition_metadata(
    *,
    trace_dir: str | Path,
    dataset_id: str,
    partition_metadata: Mapping[str, Any],
    split_manifests: Mapping[str, Mapping[str, Any]],
    label_set_id: str | None = None,
    labels_path: str | None = None,
    labels_paths: Mapping[str, str] | None = None,
) -> Path:
    trace_root = Path(trace_dir)
    dataset_dir = trace_root / "dataset-indices" / dataset_id
    splits_dir = dataset_dir / "splits"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    partition_payload = json.dumps(dict(partition_metadata), ensure_ascii=False, indent=2) + "\n"
    (dataset_dir / "partition_metadata.json").write_text(partition_payload)
    (dataset_dir / "base_partition.json").write_text(partition_payload)
    dataset_manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "partition_metadata_path": "partition_metadata.json",
        "base_partition_path": "base_partition.json",
        "split_paths": [f"splits/{split_name}.json" for split_name in sorted(split_manifests)],
    }
    if label_set_id:
        dataset_manifest["label_set_id"] = label_set_id
    if labels_path:
        dataset_manifest["labels_path"] = labels_path
    if labels_paths:
        dataset_manifest["labels_paths"] = dict(sorted(labels_paths.items()))
    feature_manifest_path = trace_root / "feature_manifest.json"
    features_path = trace_root / "features.pt"
    if feature_manifest_path.exists():
        dataset_manifest["feature_manifest_path"] = "../../feature_manifest.json"
    if features_path.exists():
        dataset_manifest["feature_path"] = "../../features.pt"
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    for split_name, split_manifest in sorted(split_manifests.items()):
        split_path = splits_dir / f"{split_name}.json"
        split_path.write_text(json.dumps(dict(split_manifest), ensure_ascii=False, indent=2) + "\n")
    return dataset_dir


def load_probe_dataset_rows(path: str | Path) -> tuple[ProbeDatasetRow, ...]:
    input_path = Path(path)
    if input_path.is_dir():
        manifest_path = input_path / "dataset_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            labels_path = manifest.get("labels_path")
            if labels_path:
                input_path = input_path / labels_path
            else:
                input_path = input_path / "index.jsonl"
        else:
            input_path = input_path / "index.jsonl"
    rows: list[ProbeDatasetRow] = []
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(ProbeDatasetRow.from_dict(json.loads(line)))
    return tuple(rows)


def build_label_manifest(rows: Sequence[ProbeDatasetRow]) -> dict[str, Any]:
    return {
        "label_columns": {
            label_name: _label_manifest_entry(label_name)
            for label_name in _label_columns(rows)
        }
    }


def build_label_stats(rows: Sequence[ProbeDatasetRow]) -> dict[str, Any]:
    return {
        "example_count": len(rows),
        "trace_count": len({row.trace_id for row in rows}),
        "label_columns": {
            label_name: {
                "positive_example_count": sum(int(row.labels.get(label_name, 0)) for row in rows),
                "negative_example_count": len(rows)
                - sum(int(row.labels.get(label_name, 0)) for row in rows),
            }
            for label_name in _label_columns(rows)
        },
    }


def _label_manifest_entry(label_name: str) -> dict[str, Any]:
    if label_name == "risk_visible":
        return {
            "kind": "heuristic",
            "source": "injection_round_index",
            "positive_meaning": "the injected content is already visible in the prompt prefix before this assistant generation",
        }
    if label_name == "risk_faced":
        return {
            "kind": "heuristic",
            "source": "injection_round_index",
            "positive_meaning": "this is the first decision point in the run whose prompt prefix already contains the injected content",
        }
    return {
        "kind": "custom",
        "source": "dataset_row",
        "positive_meaning": "label-specific positive class",
    }


def _mark_risk_faced(rows: Sequence[ProbeDatasetRow]) -> tuple[ProbeDatasetRow, ...]:
    first_visible_decision_point_ids: dict[str, str] = {}
    for row in rows:
        if row.risk_visible != 1:
            continue
        first_visible_decision_point_ids.setdefault(row.trace_id, row.decision_point_id)
    return tuple(
        replace(
            row,
            labels={
                **row.labels,
                "risk_faced": int(
                    first_visible_decision_point_ids.get(row.trace_id) == row.decision_point_id
                ),
            },
        )
        for row in rows
    )


def _extract_row_injection_strings(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    return extract_injection_strings(
        (metadata.get("run_extra") or {}).get("injections"),
        (metadata.get("outcome_metadata") or {}).get("injections"),
    )


def _extract_case_identity(metadata: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
    run_extra = metadata.get("run_extra")
    if not isinstance(run_extra, Mapping):
        return None, None, 0
    instance_id = _optional_string(run_extra.get("instance_id"))
    case_id = _optional_string(run_extra.get("case_id"))
    repeat_index_value = run_extra.get("repeat_index")
    repeat_index = int(repeat_index_value) if repeat_index_value is not None else 0
    if case_id is None and instance_id:
        case_id = _REPEAT_SUFFIX_RE.sub("", instance_id)
    return case_id, instance_id, repeat_index


def _extract_system_prompt_key(metadata: Mapping[str, Any]) -> str | None:
    direct = _optional_string(metadata.get("system_prompt_key"))
    if direct is not None:
        return direct
    run_extra = metadata.get("run_extra")
    if isinstance(run_extra, Mapping):
        return _optional_string(run_extra.get("system_prompt_key"))
    return None


def _sample_case_ids(
    *,
    case_rows: Mapping[str, Sequence[ProbeDatasetRow]],
    case_ids: Sequence[str],
    ratio: float,
    seed: int,
    rng: random.Random,
) -> list[str]:
    if ratio <= 0.0 or not case_ids:
        return []
    case_strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for case_id in case_ids:
        grouped_rows = case_rows[case_id]
        first = grouped_rows[0]
        case_strata[(first.suite_name, _case_positive(grouped_rows))].append(case_id)

    heldin_eval_case_ids: list[str] = []
    for key in sorted(case_strata):
        stratum_case_ids = list(case_strata[key])
        rng.shuffle(stratum_case_ids)
        eval_count = _allocate_ratio_count(len(stratum_case_ids), ratio)
        heldin_eval_case_ids.extend(stratum_case_ids[:eval_count])

    if ratio > 0.0 and not heldin_eval_case_ids and case_ids:
        heldin_eval_case_ids.append(sorted(case_ids)[0])
    return sorted(set(heldin_eval_case_ids))


def _sample_validation_decision_point_ids(
    rows: Sequence[ProbeDatasetRow],
    *,
    val_point_ratio: float,
    rng: random.Random,
) -> list[str]:
    if val_point_ratio <= 0.0 or not rows:
        return []
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in rows:
        strata[(row.suite_name, row.risk_visible)].append(row.decision_point_id)

    val_decision_point_ids: list[str] = []
    for key in sorted(strata):
        decision_point_ids = list(strata[key])
        rng.shuffle(decision_point_ids)
        val_count = _allocate_ratio_count(len(decision_point_ids), val_point_ratio)
        val_decision_point_ids.extend(decision_point_ids[:val_count])

    remaining_decision_point_ids = {row.decision_point_id for row in rows}
    if val_point_ratio > 0.0 and not val_decision_point_ids and len(remaining_decision_point_ids) > 1:
        val_decision_point_ids.append(sorted(remaining_decision_point_ids)[0])
    return sorted(set(val_decision_point_ids))


def _label_columns(rows: Sequence[ProbeDatasetRow]) -> list[str]:
    columns = sorted({label_name for row in rows for label_name in row.labels})
    return columns


def _trace_ids_for_cases(rows: Sequence[ProbeDatasetRow], case_ids: Sequence[str] | set[str]) -> list[str]:
    case_id_set = set(case_ids)
    return sorted({row.trace_id for row in rows if _row_case_id(row) in case_id_set})


def _decision_point_ids_for_cases(rows: Sequence[ProbeDatasetRow], case_ids: Sequence[str] | set[str]) -> list[str]:
    case_id_set = set(case_ids)
    return sorted(row.decision_point_id for row in rows if _row_case_id(row) in case_id_set)


def _group_rows_by_run(rows: Sequence[ProbeDatasetRow]) -> dict[str, list[ProbeDatasetRow]]:
    grouped: dict[str, list[ProbeDatasetRow]] = defaultdict(list)
    for row in rows:
        grouped[row.trace_id].append(row)
    return grouped


def _group_rows_by_case(rows: Sequence[ProbeDatasetRow]) -> dict[str, list[ProbeDatasetRow]]:
    grouped: dict[str, list[ProbeDatasetRow]] = defaultdict(list)
    for row in rows:
        grouped[_row_case_id(row)].append(row)
    return grouped


def _row_case_id(row: ProbeDatasetRow) -> str:
    if row.case_id:
        return row.case_id
    if row.instance_id:
        return _REPEAT_SUFFIX_RE.sub("", row.instance_id)
    return row.trace_id


def _case_positive(rows: Sequence[ProbeDatasetRow]) -> int:
    return int(any(row.risk_visible for row in rows))


def _allocate_ratio_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0.0:
        return 0
    if ratio >= 1.0:
        return total
    return min(total, int(round(total * ratio)))


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _canonical_attack_name(value: Any) -> str:
    attack_name = _optional_string(value)
    if attack_name is None or attack_name == "":
        return "clean"
    return attack_name


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def grid_point_id_from_row(row: ProbeDatasetRow | Mapping[str, Any]) -> str:
    suite_name = str(getattr(row, "suite_name", None) or row.get("suite_name") or "unknown")
    system_prompt_key = _optional_string(
        getattr(row, "system_prompt_key", None) if isinstance(row, ProbeDatasetRow) else row.get("system_prompt_key")
    ) or "unknown"
    attack_name = getattr(row, "attack_name", None) if isinstance(row, ProbeDatasetRow) else row.get("attack_name")
    return f"{suite_name}__{system_prompt_key}__{_canonical_attack_name(attack_name)}"


def parse_grid_point_id(grid_point_id: str) -> tuple[str, str, str]:
    parts = str(grid_point_id).split("__", 2)
    if len(parts) != 3:
        raise ValueError(f"invalid grid point id: {grid_point_id}")
    return parts[0], parts[1], parts[2]
