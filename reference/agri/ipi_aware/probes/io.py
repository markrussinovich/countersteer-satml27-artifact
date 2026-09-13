"""I/O contract for v2 probe products.

This module is the public save/load facade for the v2 probing pipeline.  The
pipeline is intentionally product-oriented: each stage writes one canonical
artifact family at a fixed path, and later stages read only the artifact
families they explicitly depend on.  In particular, v2 does not write or read
``examples.jsonl`` and does not use dataset-index manifests as an indirection
layer.

Filesystem layout
-----------------
All paths are resolved relative to a single run root, referred to as ``root``:

``grid_points/<grid_point_id>/manifest.json``
    One JSON object describing a suite/system-prompt/attack cell.  The writer
    stores ``schema="ipi_aware.grid_point.v1"`` plus the exact identity fields
    ``grid_point_id``, ``suite_name``, ``system_prompt_key``, ``attack_name``,
    ``benchmark_version``, ``model_id`` and the counts ``trace_count``,
    ``decision_point_count`` and ``failure_count``.  The manifest is a summary,
    not a source of examples.

``grid_points/<grid_point_id>/traces.jsonl``
    One JSON object per benchmark episode.  Rows are normalized to the v2 trace
    product shape: ``trace_id``, ``grid_point_id``, ``case_id``,
    ``repeat_index``, ``messages``, ``model_requests``, ``tool_executions``,
    ``outcome`` and ``injection_round_index``.  Collection-time writers may
    append to this file, but bulk conversion uses ``write_grid_point_products``
    to rewrite the complete shard deterministically.

``grid_points/<grid_point_id>/decision_points.jsonl``
    One JSON object per assistant decision.  This is the canonical example
    product for v2.  Rows are replay-ready and normalized to:
    ``decision_point_id``, ``trace_id``, ``grid_point_id``, ``case_id``,
    ``decision_index``, ``assistant_message_index``, ``replay_request``,
    ``assistant_message``, ``prompt_token_ids``, ``response_token_ids`` and
    ``injection_round_index``.  Later stages join by ``decision_point_id``.

``labels/<labeling_protocol>/labels.jsonl``
    One JSON object per labeled decision point.  Rows are binary-only and have
    exactly two fields: ``decision_point_id`` and ``label``.  The path name is
    the protocol pin; the row intentionally does not repeat protocol metadata.

``features/<feature_name>/<grid_point_id>.pt``
    One Torch payload per grid point and feature product.  The payload is a
    mapping with ``schema="ipi_aware.features.v1"``, identity fields, ordered
    ``decision_point_ids``, ``layer_indices``, ``selected_positions`` and a
    tensor named ``features`` shaped
    ``[examples, layers, positions_or_flat, hidden]``.  Tensor row order must
    match ``decision_point_ids`` exactly.

``datasets/<dataset_name>/partition.json``
    One JSON object containing split membership only.  The payload declares
    ``schema="ipi_aware.partition.v1"`` and contains ``train`` and ``val`` maps from
    grid point id to decision point ids, plus ``eval_grid_points``,
    ``val_ratio`` and ``split_seed``.  The partition does not pin labels or
    features; callers choose those products at load time.

``<probe_run_dir>/probe.pt``
    Legacy singular name; v2 training now writes per-layer best checkpoints
    under ``<probe_run_dir>/models/`` by default.

``<probe_run_dir>/models/best_layer_<n>.pt`` or
``<probe_run_dir>/models/best_layers_<n>_<m>.pt``
    One Torch checkpoint per trained layer/window.  The checkpoint declares
    ``schema="ipi_aware.probe_checkpoint.v2"`` and carries all metadata needed to
    rebuild the probe at evaluation time: dataset/product selectors, full
    available ``layer_indices``, chosen feature-composition fields,
    chosen probe architecture/config, ``input_dim``, threshold, training config
    and ``model_state_dict``.  Evaluation receives the checkpoint path directly;
    no manifest link is followed to discover it.  The run-level
    ``metrics.json`` records per-layer metrics and chooses
    ``best_checkpoint_path`` by validation AUROC first, then balanced accuracy,
    then accuracy.

Save/load protocol
------------------
``read_json``/``write_json`` and ``read_jsonl``/``write_jsonl`` are low-level
helpers.  They create parent directories on write, use UTF-8, preserve Unicode,
and terminate JSON files with a newline.  They do not perform schema-specific
validation.

Product loaders such as ``load_grid_point_manifest``, ``load_traces``,
``load_decision_points``, ``load_labels`` and ``load_partition`` validate the
minimum schema and field contract before returning plain dictionaries.  Feature
loading is exposed here as ``load_feature_payload`` but delegated to
``v2.features`` because it depends on torch.

Product writers exposed here are the stable save surface for pipeline stages.
Some implementations live in product-specific modules to avoid circular
imports, but callers should prefer importing save/load functions from
``v2.io``.  ``write_grid_point_products`` is a bulk writer/converter: it
normalizes legacy-compatible rows, rewrites the grid-point manifest, trace
shard and decision-point shard, and updates the root ``manifest.json`` with the
grid point id.  Streaming collection uses the same row coercion and manifest
schema in ``v2.trace_collection``.  Probe checkpoints are saved and loaded only
through ``save_probe_checkpoint`` and ``load_probe_checkpoint`` so checkpoint
validation is paired and evaluation cannot silently consume an underspecified
payload.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ipi_aware.data_collection.token_accounting import token_unit_usage

from .utils import GRID_POINT_SCHEMA, ProductPaths, iter_grid_point_ids
from .utils import EVAL_GROUPS_SCHEMA


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object rows at {path}")
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return output_path


def load_grid_point_manifest(root: str | Path, grid_point_id: str) -> dict[str, Any]:
    manifest = read_json(ProductPaths.from_root(root).grid_point_manifest(grid_point_id))
    _require_schema(manifest, GRID_POINT_SCHEMA, "grid point manifest")
    for key in (
        "grid_point_id",
        "suite_name",
        "system_prompt_key",
        "attack_name",
        "benchmark_version",
        "model_id",
        "trace_count",
        "decision_point_count",
        "failure_count",
    ):
        if key not in manifest:
            raise ValueError(f"grid point manifest is missing {key!r}")
    return manifest


def load_traces(root: str | Path, grid_point_id: str) -> list[dict[str, Any]]:
    rows = read_jsonl(ProductPaths.from_root(root).traces(grid_point_id))
    for row in rows:
        _validate_trace_row(row, grid_point_id=grid_point_id)
    return rows


def load_decision_points(root: str | Path, grid_point_id: str) -> list[dict[str, Any]]:
    rows = read_jsonl(ProductPaths.from_root(root).decision_points(grid_point_id))
    for row in rows:
        _validate_decision_point_row(row, grid_point_id=grid_point_id)
    return rows


def load_all_decision_points(
    root: str | Path,
    grid_point_ids: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    selected = tuple(grid_point_ids) if grid_point_ids is not None else iter_grid_point_ids(root)
    return {grid_point_id: load_decision_points(root, grid_point_id) for grid_point_id in selected}


def load_labels(root: str | Path, labeling_protocol: str) -> dict[str, int]:
    path = ProductPaths.from_root(root).labels(labeling_protocol)
    labels: dict[str, int] = {}
    for row in read_jsonl(path):
        if sorted(row) != ["decision_point_id", "label"]:
            raise ValueError(f"label row in {path} must contain only decision_point_id and label")
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"label for {row['decision_point_id']} must be binary 0/1")
        labels[str(row["decision_point_id"])] = label
    return labels


def load_partition(root: str | Path, dataset_name: str) -> dict[str, Any]:
    from .utils import PARTITION_SCHEMA

    payload = read_json(ProductPaths.from_root(root).partition(dataset_name))
    _require_schema(payload, PARTITION_SCHEMA, "partition")
    for key in ("dataset_name", "train", "val", "eval_groups", "split_seed"):
        if key not in payload:
            raise ValueError(f"partition is missing {key!r}")
    return payload


def load_feature_payload(
    root_or_path: str | Path,
    grid_point_id: str | None = None,
    feature_name: str = "default",
    layer_indices: Sequence[int] | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    from .featurization.feature_io import load_feature_payload as _load_feature_payload

    return _load_feature_payload(root_or_path, grid_point_id, feature_name, layer_indices, map_location=map_location)


def save_probe_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> Path:
    """Save a v2 probe checkpoint after validating the reload contract.

    Checkpoints are torch payloads rather than JSON because they include a
    model ``state_dict``.  The non-tensor metadata is intentionally complete:
    evaluation can rebuild the probe model and feature composition from the
    checkpoint plus the direct v2 products under ``root``.
    """
    payload = dict(checkpoint)
    validate_probe_checkpoint_payload(payload)
    torch = _import_torch_checkpoint()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def load_probe_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a v2 probe checkpoint from a fixed checkpoint path."""
    torch = _import_torch_checkpoint()
    input_path = Path(path)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"probe checkpoint at {input_path} is not a mapping")
    result = dict(payload)
    validate_probe_checkpoint_payload(result)
    return result


def validate_probe_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    from .utils import CHECKPOINT_SCHEMA

    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"probe checkpoint must declare schema={CHECKPOINT_SCHEMA!r}")
    required = (
        "dataset_name",
        "labeling_protocol",
        "feature_name",
        "layer_indices",
        "selected_positions",
        "feature_composition",
        "concat_num_layers",
        "selected_layer_indices",
        "selected_layer_offsets",
        "selected_layer_index",
        "selected_layer_offset",
        "probe_architecture",
        "probe_architecture_config",
        "input_dim",
        "threshold",
        "training_config",
        "model_state_dict",
    )
    for key in required:
        if key not in payload:
            raise ValueError(f"probe checkpoint is missing {key!r}")
    selected_layer_indices = list(payload["selected_layer_indices"])
    selected_layer_offsets = list(payload["selected_layer_offsets"])
    if not selected_layer_indices or not selected_layer_offsets:
        raise ValueError("probe checkpoint must include at least one selected layer")
    if len(selected_layer_indices) != len(selected_layer_offsets):
        raise ValueError("probe checkpoint selected_layer_indices and selected_layer_offsets are misaligned")
    if int(payload["input_dim"]) <= 0:
        raise ValueError("probe checkpoint input_dim must be positive")


def write_feature_payload(*, root: str | Path, payload: Mapping[str, Any]) -> Path:
    from .featurization.feature_io import write_feature_payload as _write_feature_payload

    return _write_feature_payload(root=root, payload=payload)


def write_labels(
    *,
    root: str | Path,
    labeling_protocol: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    from .collection.labels import write_labels as _write_labels

    return _write_labels(root=root, labeling_protocol=labeling_protocol, rows=rows)


def write_partition(*, root: str | Path, partition: Mapping[str, Any]) -> Path:
    from .training.partition import write_partition as _write_partition

    return _write_partition(root=root, partition=partition)


def load_eval_groups(path: str | Path) -> dict[str, Any]:
    """Load and validate a ``ipi_aware.eval_groups.v1`` product."""
    payload = read_json(path)
    _require_schema(payload, EVAL_GROUPS_SCHEMA, "eval-groups")
    for key in ("checkpoint_root", "best_checkpoint_path", "train_grid_point", "groups"):
        if key not in payload:
            raise ValueError(f"eval-groups product is missing {key!r}")
    return payload


def write_eval_groups(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a ``ipi_aware.eval_groups.v1`` product to disk."""
    if payload.get("schema") != EVAL_GROUPS_SCHEMA:
        raise ValueError(f"eval-groups product must declare schema={EVAL_GROUPS_SCHEMA!r}")
    return write_json(path, payload)


def write_grid_point_products(
    *,
    root: str | Path,
    grid_point_id: str,
    suite_name: str,
    system_prompt_key: str,
    attack_name: str,
    benchmark_version: str,
    model_id: str,
    traces: Sequence[Mapping[str, Any]],
    decision_points: Sequence[Mapping[str, Any]],
    failure_count: int = 0,
) -> Path:
    paths = ProductPaths.from_root(root)
    gp_dir = paths.grid_point_dir(grid_point_id)
    gp_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = [coerce_trace_row(row, grid_point_id=grid_point_id) for row in traces]
    decision_point_rows = [
        coerce_decision_point_row(row, grid_point_id=grid_point_id)
        for row in decision_points
    ]
    manifest = {
        "schema": GRID_POINT_SCHEMA,
        "grid_point_id": str(grid_point_id),
        "suite_name": str(suite_name),
        "system_prompt_key": str(system_prompt_key),
        "attack_name": str(attack_name),
        "benchmark_version": str(benchmark_version),
        "model_id": str(model_id),
        "trace_count": len(trace_rows),
        "decision_point_count": len(decision_point_rows),
        "failure_count": int(failure_count),
    }
    write_json(paths.grid_point_manifest(grid_point_id), manifest)
    write_jsonl(paths.traces(grid_point_id), trace_rows)
    write_jsonl(paths.decision_points(grid_point_id), decision_point_rows)
    root_manifest = {
        "schema": "ipi_aware.probe_run.v2",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "grid_points_dir": "grid_points",
    }
    root_manifest_path = paths.root / "manifest.json"
    if root_manifest_path.exists():
        try:
            root_manifest.update(read_json(root_manifest_path))
        except ValueError:
            pass
        root_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    root_manifest["schema"] = "ipi_aware.probe_run.v2"
    root_manifest["grid_points"] = sorted(set(root_manifest.get("grid_points") or ()) | {str(grid_point_id)})
    write_json(root_manifest_path, root_manifest)
    return gp_dir


def coerce_trace_row(payload: Mapping[str, Any], *, grid_point_id: str) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    outcome = payload.get("outcome") if isinstance(payload.get("outcome"), Mapping) else {}
    extra = metadata.get("extra") if isinstance(metadata.get("extra"), Mapping) else {}
    row = {
        "trace_id": str(payload.get("trace_id") or metadata.get("trace_id") or metadata.get("run_id") or payload.get("run_id") or ""),
        "grid_point_id": str(payload.get("grid_point_id") or grid_point_id),
        "case_id": str(payload.get("case_id") or extra.get("case_id") or ""),
        "repeat_index": int(payload.get("repeat_index") or extra.get("repeat_index") or 0),
        "messages": list(payload.get("messages") or payload.get("final_messages") or ()),
        "model_requests": list(payload.get("model_requests") or ()),
        "tool_executions": list(payload.get("tool_executions") or ()),
        "outcome": {
            "utility": _optional_bool(outcome.get("utility") if outcome else payload.get("utility")),
            "security": _optional_bool(outcome.get("security") if outcome else payload.get("security")),
            "error": outcome.get("error") if outcome else payload.get("outcome_error"),
        },
        "injection_round_index": _optional_int_list(
            payload.get("injection_round_index") if payload.get("injection_round_index") is not None else metadata.get("injection_round_index")
        ),
    }
    injection_context = payload.get("injection_context")
    if isinstance(injection_context, Mapping):
        row["injection_context"] = dict(injection_context)
    return row


def coerce_decision_point_row(payload: Mapping[str, Any], *, grid_point_id: str) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    run_extra = metadata.get("run_extra") if isinstance(metadata.get("run_extra"), Mapping) else {}
    row = {
        "decision_point_id": str(payload["decision_point_id"]),
        "trace_id": str(payload.get("trace_id") or metadata.get("trace_id") or metadata.get("run_id") or ""),
        "grid_point_id": str(payload.get("grid_point_id") or grid_point_id),
        "case_id": str(payload.get("case_id") or run_extra.get("case_id") or ""),
        "decision_index": int(payload["decision_index"]),
        "assistant_message_index": int(payload["assistant_message_index"]),
        "replay_request": dict(payload.get("replay_request") or payload.get("request_payload") or {}),
        "assistant_message": dict(payload.get("assistant_message") or {}),
        "prompt_token_ids": [int(value) for value in (payload.get("prompt_token_ids") or ())],
        "response_token_ids": [int(value) for value in (payload.get("response_token_ids") or ())],
        "injection_round_index": _optional_int_list(
            payload.get("injection_round_index") if payload.get("injection_round_index") is not None else metadata.get("injection_round_index")
        ),
    }
    response_usage = _response_usage_from_decision_payload(payload)
    if response_usage is not None:
        row["response_usage"] = response_usage
        row["token_usage"] = _flatten_response_usage(response_usage)
    injection_context = payload.get("injection_context")
    if isinstance(injection_context, Mapping):
        row["injection_context"] = dict(injection_context)
    return row


def _response_usage_from_decision_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    usage = payload.get("response_usage")
    if isinstance(usage, Mapping):
        return dict(usage)
    response_payload = payload.get("response_payload") or payload.get("assistant_response_payload")
    if isinstance(response_payload, Mapping):
        usage = response_payload.get("usage")
        if isinstance(usage, Mapping):
            return dict(usage)
    return None


def _flatten_response_usage(usage: Mapping[str, Any]) -> dict[str, int]:
    return token_unit_usage(usage)


def _validate_trace_row(row: Mapping[str, Any], *, grid_point_id: str) -> None:
    for key in (
        "trace_id",
        "grid_point_id",
        "case_id",
        "repeat_index",
        "messages",
        "model_requests",
        "tool_executions",
        "outcome",
        "injection_round_index",
    ):
        if key not in row:
            raise ValueError(f"trace row is missing {key!r}")
    if str(row["grid_point_id"]) != str(grid_point_id):
        raise ValueError(f"trace row grid_point_id does not match {grid_point_id}")


def _validate_decision_point_row(row: Mapping[str, Any], *, grid_point_id: str) -> None:
    for key in (
        "decision_point_id",
        "trace_id",
        "grid_point_id",
        "case_id",
        "decision_index",
        "assistant_message_index",
        "replay_request",
        "assistant_message",
        "prompt_token_ids",
        "response_token_ids",
        "injection_round_index",
    ):
        if key not in row:
            raise ValueError(f"decision point row is missing {key!r}")
    if str(row["grid_point_id"]) != str(grid_point_id):
        raise ValueError(f"decision point row grid_point_id does not match {grid_point_id}")


def _require_schema(payload: Mapping[str, Any], schema: str, product_name: str) -> None:
    if payload.get("schema") != schema:
        raise ValueError(f"{product_name} must declare schema={schema!r}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _import_torch_checkpoint() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("v2 probe checkpoint I/O requires torch in the current environment") from exc
    return torch
