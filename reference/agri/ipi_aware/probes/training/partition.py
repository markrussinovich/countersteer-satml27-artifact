from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..io import load_decision_points, load_labels, write_json
from ..utils import DEFAULT_LABELING_PROTOCOL, PARTITION_SCHEMA, ProductPaths


# ---------------------------------------------------------------------------
# Fixed dataset definitions — train grid-point assignments
# ---------------------------------------------------------------------------
# Each dataset specifies which grid points are used for training.
# Eval groups are fixed across all datasets (see EVAL_GROUPS below).

DATASETS: dict[str, tuple[str, ...]] = {
    "narrow": (
        "workspace__default__direct",
    ),
    "expand_suite": (
        "slack__default__direct",
        "workspace__default__direct",
    ),
    "expand_suite_attack": (
        "slack__default__direct",
        "slack__default__long_horizon_important_instructions",
        "workspace__default__direct",
        "workspace__default__long_horizon_important_instructions",
    ),
    "broad": (
        "slack__default__direct",
        "slack__default__long_horizon_important_instructions",
        "slack__safety_reminder_balanced__direct",
        "slack__safety_reminder_balanced__long_horizon_important_instructions",
        "workspace__default__direct",
        "workspace__default__long_horizon_important_instructions",
        "workspace__safety_reminder_balanced__direct",
        "workspace__safety_reminder_balanced__long_horizon_important_instructions",
    ),
}

EVAL_GROUPS: dict[str, tuple[str, ...]] = {
    "strict": (
        "banking__default_tool_careful__ignore_previous",
        "banking__default_tool_careful__important_instructions",
        "banking__default_tool_careful__system_message",
        "banking__default_tool_careful__tool_knowledge",
        "banking__safety_reminder_explicit__ignore_previous",
        "banking__safety_reminder_explicit__important_instructions",
        "banking__safety_reminder_explicit__system_message",
        "banking__safety_reminder_explicit__tool_knowledge",
        "travel__default_tool_careful__ignore_previous",
        "travel__default_tool_careful__important_instructions",
        "travel__default_tool_careful__system_message",
        "travel__default_tool_careful__tool_knowledge",
        "travel__safety_reminder_explicit__ignore_previous",
        "travel__safety_reminder_explicit__important_instructions",
        "travel__safety_reminder_explicit__system_message",
        "travel__safety_reminder_explicit__tool_knowledge",
    ),
    "unseen_attacks": (
        "banking__default__ignore_previous",
        "banking__default__important_instructions",
        "banking__default__system_message",
        "banking__default__tool_knowledge",
        "banking__safety_reminder_balanced__ignore_previous",
        "banking__safety_reminder_balanced__important_instructions",
        "banking__safety_reminder_balanced__system_message",
        "banking__safety_reminder_balanced__tool_knowledge",
        "slack__default__ignore_previous",
        "slack__default__important_instructions",
        "slack__default__system_message",
        "slack__default__tool_knowledge",
        "slack__default_tool_careful__ignore_previous",
        "slack__default_tool_careful__important_instructions",
        "slack__default_tool_careful__system_message",
        "slack__default_tool_careful__tool_knowledge",
        "slack__safety_reminder_balanced__ignore_previous",
        "slack__safety_reminder_balanced__important_instructions",
        "slack__safety_reminder_balanced__system_message",
        "slack__safety_reminder_balanced__tool_knowledge",
        "slack__safety_reminder_explicit__ignore_previous",
        "slack__safety_reminder_explicit__important_instructions",
        "slack__safety_reminder_explicit__system_message",
        "slack__safety_reminder_explicit__tool_knowledge",
        "travel__default__ignore_previous",
        "travel__default__important_instructions",
        "travel__default__system_message",
        "travel__default__tool_knowledge",
        "travel__safety_reminder_balanced__ignore_previous",
        "travel__safety_reminder_balanced__important_instructions",
        "travel__safety_reminder_balanced__system_message",
        "travel__safety_reminder_balanced__tool_knowledge",
        "workspace__default__ignore_previous",
        "workspace__default__important_instructions",
        "workspace__default__system_message",
        "workspace__default_tool_careful__ignore_previous",
        "workspace__default_tool_careful__important_instructions",
        "workspace__default_tool_careful__system_message",
        "workspace__safety_reminder_balanced__ignore_previous",
        "workspace__safety_reminder_balanced__important_instructions",
        "workspace__safety_reminder_balanced__system_message",
        "workspace__safety_reminder_explicit__ignore_previous",
        "workspace__safety_reminder_explicit__important_instructions",
        "workspace__safety_reminder_explicit__system_message",
    ),
    "unseen_suites": (
        "banking__default__direct",
        "banking__default__ignore_previous",
        "banking__default__important_instructions",
        "banking__default__long_horizon_important_instructions",
        "banking__default__system_message",
        "banking__default__tool_knowledge",
        "banking__default_tool_careful__direct",
        "banking__default_tool_careful__long_horizon_important_instructions",
        "banking__safety_reminder_balanced__direct",
        "banking__safety_reminder_balanced__ignore_previous",
        "banking__safety_reminder_balanced__important_instructions",
        "banking__safety_reminder_balanced__long_horizon_important_instructions",
        "banking__safety_reminder_balanced__system_message",
        "banking__safety_reminder_balanced__tool_knowledge",
        "banking__safety_reminder_explicit__direct",
        "banking__safety_reminder_explicit__long_horizon_important_instructions",
        "travel__default__direct",
        "travel__default__ignore_previous",
        "travel__default__important_instructions",
        "travel__default__long_horizon_important_instructions",
        "travel__default__system_message",
        "travel__default__tool_knowledge",
        "travel__default_tool_careful__direct",
        "travel__default_tool_careful__long_horizon_important_instructions",
        "travel__safety_reminder_balanced__direct",
        "travel__safety_reminder_balanced__ignore_previous",
        "travel__safety_reminder_balanced__important_instructions",
        "travel__safety_reminder_balanced__long_horizon_important_instructions",
        "travel__safety_reminder_balanced__system_message",
        "travel__safety_reminder_balanced__tool_knowledge",
        "travel__safety_reminder_explicit__direct",
        "travel__safety_reminder_explicit__long_horizon_important_instructions",
    ),
    "unseen_prompts": (
        "banking__default_tool_careful__direct",
        "banking__default_tool_careful__long_horizon_important_instructions",
        "banking__safety_reminder_explicit__direct",
        "banking__safety_reminder_explicit__long_horizon_important_instructions",
        "slack__default_tool_careful__direct",
        "slack__default_tool_careful__ignore_previous",
        "slack__default_tool_careful__important_instructions",
        "slack__default_tool_careful__long_horizon_important_instructions",
        "slack__default_tool_careful__system_message",
        "slack__default_tool_careful__tool_knowledge",
        "slack__safety_reminder_explicit__direct",
        "slack__safety_reminder_explicit__ignore_previous",
        "slack__safety_reminder_explicit__important_instructions",
        "slack__safety_reminder_explicit__long_horizon_important_instructions",
        "slack__safety_reminder_explicit__system_message",
        "slack__safety_reminder_explicit__tool_knowledge",
        "travel__default_tool_careful__direct",
        "travel__default_tool_careful__long_horizon_important_instructions",
        "travel__safety_reminder_explicit__direct",
        "travel__safety_reminder_explicit__long_horizon_important_instructions",
        "workspace__default_tool_careful__direct",
        "workspace__default_tool_careful__ignore_previous",
        "workspace__default_tool_careful__important_instructions",
        "workspace__default_tool_careful__long_horizon_important_instructions",
        "workspace__default_tool_careful__system_message",
        "workspace__safety_reminder_explicit__direct",
        "workspace__safety_reminder_explicit__ignore_previous",
        "workspace__safety_reminder_explicit__important_instructions",
        "workspace__safety_reminder_explicit__long_horizon_important_instructions",
        "workspace__safety_reminder_explicit__system_message",
    ),
}


def resolve_train_grid_points(dataset: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(dataset, str):
        if dataset in DATASETS:
            return DATASETS[dataset]
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    return tuple(dict.fromkeys(str(v) for v in dataset))


def build_partition(
    *,
    root: str | Path,
    dataset_name: str,
    train_grid_points: Sequence[str],
    val_ratio: float = 0.3,
    split_seed: int = 42,
    labeling_protocol: str = DEFAULT_LABELING_PROTOCOL,
    partition_granularity: str = "decision-points",
) -> dict[str, Any]:
    if partition_granularity not in ("decision-points", "traces"):
        raise ValueError("partition_granularity must be 'decision-points' or 'traces'")

    train_grid_points = tuple(dict.fromkeys(str(v) for v in train_grid_points))
    if not train_grid_points:
        raise ValueError("train_grid_points must not be empty")

    labels = _load_labels_if_present(root, labeling_protocol)
    dp_rows_by_grid_point: dict[str, list[dict[str, Any]]] = {
        gp_id: load_decision_points(root, gp_id)
        for gp_id in train_grid_points
    }
    train_ids_by_grid_point = {
        gp_id: [str(row["decision_point_id"]) for row in rows]
        for gp_id, rows in dp_rows_by_grid_point.items()
    }

    if partition_granularity == "traces":
        trace_to_dp_ids_by_gp: dict[str, dict[str, list[str]]] = {}
        for gp_id, rows in dp_rows_by_grid_point.items():
            t2d: dict[str, list[str]] = {}
            for row in rows:
                t2d.setdefault(str(row["trace_id"]), []).append(
                    str(row["decision_point_id"])
                )
            trace_to_dp_ids_by_gp[gp_id] = t2d
        train_ids = _sample_train_ids_by_ratio_traces_per_gp(
            trace_to_dp_ids_by_gp=trace_to_dp_ids_by_gp,
            labels=labels,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )
    else:
        train_ids = _sample_train_ids_by_ratio(
            train_ids_by_grid_point=train_ids_by_grid_point,
            labels=labels,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )

    train: dict[str, list[str]] = {}
    val: dict[str, list[str]] = {}
    for gp_id, decision_point_ids in train_ids_by_grid_point.items():
        train[gp_id] = [dp_id for dp_id in decision_point_ids if dp_id in train_ids]
        val[gp_id] = [dp_id for dp_id in decision_point_ids if dp_id not in train_ids]

    split_meta: dict[str, Any] = {
        "split_seed": int(split_seed),
        "val_ratio": float(val_ratio),
        "partition_granularity": partition_granularity,
    }

    eval_groups_dict = {k: list(v) for k, v in EVAL_GROUPS.items()}

    return {
        "schema": PARTITION_SCHEMA,
        "dataset_name": str(dataset_name),
        "train": train,
        "val": val,
        "eval_groups": eval_groups_dict,
        **split_meta,
    }


def write_partition(*, root: str | Path, partition: Mapping[str, Any]) -> Path:
    dataset_name = str(partition["dataset_name"])
    return write_json(ProductPaths.from_root(root).partition(dataset_name), partition)


def _sample_train_ids_by_ratio(
    *,
    train_ids_by_grid_point: Mapping[str, Sequence[str]],
    labels: Mapping[str, int],
    val_ratio: float,
    split_seed: int,
) -> set[str]:
    all_ids = [
        dp_id
        for ids in train_ids_by_grid_point.values()
        for dp_id in ids
    ]
    if val_ratio <= 0.0:
        return set(all_ids)
    if val_ratio >= 1.0:
        return set()
    val_ids: set[str] = set()
    for gp_id in sorted(train_ids_by_grid_point):
        gp_rng = random.Random(int(hashlib.md5(f"{split_seed}:{gp_id}".encode()).hexdigest(), 16))
        gp_ids = list(train_ids_by_grid_point[gp_id])
        strata: dict[int, list[str]] = defaultdict(list)
        for dp_id in gp_ids:
            strata[int(labels.get(dp_id, -1))].append(str(dp_id))
        for key in sorted(strata):
            ids = list(strata[key])
            gp_rng.shuffle(ids)
            val_count = min(len(ids), int(round(len(ids) * val_ratio)))
            val_ids.update(ids[:val_count])
        if val_ratio > 0.0 and not (val_ids & set(gp_ids)) and len(gp_ids) > 1:
            val_ids.add(sorted(gp_ids)[0])
    return set(all_ids) - val_ids


def _sample_train_ids_by_ratio_traces(
    *,
    trace_to_dp_ids: Mapping[str, Sequence[str]],
    labels: Mapping[str, int],
    val_ratio: float,
    split_seed: int,
) -> set[str]:
    all_trace_ids = sorted(trace_to_dp_ids)
    if val_ratio <= 0.0:
        return {dp_id for dps in trace_to_dp_ids.values() for dp_id in dps}
    if val_ratio >= 1.0:
        return set()

    trace_label: dict[str, int] = {}
    for tid, dp_ids in trace_to_dp_ids.items():
        trace_label[tid] = 1 if any(labels.get(d, 0) == 1 for d in dp_ids) else 0

    rng = random.Random(split_seed)
    strata: dict[int, list[str]] = defaultdict(list)
    for tid in all_trace_ids:
        strata[trace_label[tid]].append(tid)

    val_traces: set[str] = set()
    for key in sorted(strata):
        ids = list(strata[key])
        rng.shuffle(ids)
        val_count = min(len(ids), int(round(len(ids) * val_ratio)))
        val_traces.update(ids[:val_count])

    train_traces = set(all_trace_ids) - val_traces
    return {dp_id for tid in train_traces for dp_id in trace_to_dp_ids[tid]}


def _sample_train_ids_by_ratio_traces_per_gp(
    *,
    trace_to_dp_ids_by_gp: Mapping[str, Mapping[str, Sequence[str]]],
    labels: Mapping[str, int],
    val_ratio: float,
    split_seed: int,
) -> set[str]:
    if val_ratio <= 0.0:
        return {
            dp_id
            for t2d in trace_to_dp_ids_by_gp.values()
            for dps in t2d.values()
            for dp_id in dps
        }
    if val_ratio >= 1.0:
        return set()

    all_dp_ids: set[str] = set()
    val_dp_ids: set[str] = set()

    for gp_id in sorted(trace_to_dp_ids_by_gp):
        t2d = trace_to_dp_ids_by_gp[gp_id]
        gp_rng = random.Random(
            int(hashlib.md5(f"{split_seed}:traces:{gp_id}".encode()).hexdigest(), 16)
        )
        trace_label: dict[str, int] = {}
        for tid, dp_ids in t2d.items():
            trace_label[tid] = 1 if any(labels.get(d, 0) == 1 for d in dp_ids) else 0

        strata: dict[int, list[str]] = defaultdict(list)
        for tid in t2d:
            strata[trace_label[tid]].append(tid)

        gp_val_traces: set[str] = set()
        for key in sorted(strata):
            ids = list(strata[key])
            gp_rng.shuffle(ids)
            val_count = min(len(ids), int(round(len(ids) * val_ratio)))
            gp_val_traces.update(ids[:val_count])

        gp_all = {dp_id for dps in t2d.values() for dp_id in dps}
        gp_val = {dp_id for tid in gp_val_traces for dp_id in t2d[tid]}
        all_dp_ids.update(gp_all)
        val_dp_ids.update(gp_val)

    return all_dp_ids - val_dp_ids


def _load_labels_if_present(root: str | Path, labeling_protocol: str) -> dict[str, int]:
    label_path = ProductPaths.from_root(root).labels(labeling_protocol)
    if not label_path.exists():
        return {}
    return load_labels(root, labeling_protocol)
