from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
from pathlib import Path
from typing import Any

from ..utils import DEFAULT_FEATURE_NAME, FEATURE_SCHEMA, FEATURE_SPLIT_SCHEMA, FEATURE_LAYER_SCHEMA, ProductPaths


def build_feature_payload(
    *,
    grid_point_id: str,
    feature_name: str = DEFAULT_FEATURE_NAME,
    backend: str = "transformers_hook",
    model_id: str,
    decision_point_ids: Sequence[str],
    layer_indices: Sequence[int],
    selected_positions: Sequence[int],
    features: Any,
    filter_last_replay_role: str | None = None,
    pre_filter_decision_point_count: int | None = None,
    post_filter_decision_point_count: int | None = None,
) -> dict[str, Any]:
    torch = _import_torch()
    tensor = torch.as_tensor(features).detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(2)
    if tensor.ndim != 4:
        raise ValueError("features must have shape [examples, layers, positions_or_flat, hidden]")
    if tensor.shape[0] != len(decision_point_ids):
        raise ValueError("features example dimension must match decision_point_ids")
    if tensor.shape[1] != len(layer_indices):
        raise ValueError("features layer dimension must match layer_indices")
    payload = {
        "schema": FEATURE_SCHEMA,
        "grid_point_id": str(grid_point_id),
        "feature_name": str(feature_name),
        "backend": str(backend),
        "model_id": str(model_id),
        "decision_point_ids": [str(value) for value in decision_point_ids],
        "layer_indices": [int(value) for value in layer_indices],
        "selected_positions": [int(value) for value in selected_positions],
        "features": tensor,
    }
    if filter_last_replay_role is not None:
        payload["filter_last_replay_role"] = str(filter_last_replay_role)
    if pre_filter_decision_point_count is not None:
        payload["pre_filter_decision_point_count"] = int(pre_filter_decision_point_count)
    if post_filter_decision_point_count is not None:
        payload["post_filter_decision_point_count"] = int(post_filter_decision_point_count)
    return payload


def write_feature_payload(*, root: str | Path, payload: Mapping[str, Any]) -> Path:
    _validate_feature_payload(payload)
    torch = _import_torch()
    path = ProductPaths.from_root(root).feature_shard(
        str(payload["grid_point_id"]),
        str(payload.get("feature_name") or DEFAULT_FEATURE_NAME),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    return path


def load_feature_payload(
    root_or_path: str | Path,
    grid_point_id: str | None = None,
    feature_name: str = DEFAULT_FEATURE_NAME,
    layer_indices: Sequence[int] | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    torch = _import_torch()
    path = Path(root_or_path)
    if grid_point_id is not None:
        paths = ProductPaths.from_root(path)
        gp = str(grid_point_id)
        fn = str(feature_name)
        if paths.is_feature_split(gp, fn):
            return _load_from_split(torch, paths, gp, fn, layer_indices, map_location=map_location)
        path = paths.feature_shard(gp, fn)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"feature payload at {path} is not a mapping")
    result = dict(payload)
    _validate_feature_payload(result)
    if layer_indices is not None:
        result = _slice_payload_layers(result, layer_indices)
    return result


def feature_payload_to_id_map(payload: Mapping[str, Any]) -> dict[str, Any]:
    _validate_feature_payload(payload)
    return {
        str(decision_point_id): payload["features"][index]
        for index, decision_point_id in enumerate(payload["decision_point_ids"])
    }


def examples_to_feature_payload(
    *,
    grid_point_id: str,
    feature_name: str,
    backend: str,
    model_id: str,
    examples: Sequence[Any],
    requested_positions: Sequence[int] | None = None,
    filter_last_replay_role: str | None = None,
    pre_filter_decision_point_count: int | None = None,
    post_filter_decision_point_count: int | None = None,
) -> dict[str, Any]:
    if not examples:
        return None
    torch = _import_torch()
    decision_point_ids = [str(example.decision_point_id) for example in examples]
    position_count = len(examples[0].hidden_state_capture.selected_positions)
    layer_count = len(examples[0].hidden_state_capture.hidden_states)
    rows = []
    for example in examples:
        capture = example.hidden_state_capture
        if len(capture.selected_positions) != position_count:
            raise ValueError("all examples in a feature shard must have the same number of selected positions")
        if len(capture.hidden_states) != layer_count:
            raise ValueError("all examples in a feature shard must have the same layer count")
        rows.append(
            torch.stack(
                [
                    _layer_to_position_hidden_tensor(layer_hidden)
                    for layer_hidden in capture.hidden_states
                ],
                dim=0,
            )
        )
    features = torch.stack(rows, dim=0)
    first_capture = examples[0].hidden_state_capture
    layer_indices = list(first_capture.layer_ids) if first_capture.layer_ids is not None else list(range(layer_count))
    payload = build_feature_payload(
        grid_point_id=grid_point_id,
        feature_name=feature_name,
        backend=backend,
        model_id=model_id,
        decision_point_ids=decision_point_ids,
        layer_indices=layer_indices,
        selected_positions=list(requested_positions) if requested_positions else list(examples[0].hidden_state_capture.selected_positions),
        features=features,
        filter_last_replay_role=filter_last_replay_role,
        pre_filter_decision_point_count=pre_filter_decision_point_count,
        post_filter_decision_point_count=post_filter_decision_point_count,
    )
    prompt_restore_source_counts = _prompt_restore_source_counts(examples)
    if prompt_restore_source_counts:
        payload["prompt_restore_source_counts"] = prompt_restore_source_counts
    return payload


def _prompt_restore_source_counts(examples: Sequence[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        source = example.metadata.get("prompt_restore_source")
        if source:
            counts[str(source)] += 1
    return dict(counts)


def write_feature_payload_split(
    *, root: str | Path, payload: Mapping[str, Any], keep_original: bool = False,
) -> Path:
    """Split a bulk feature payload into per-layer files under a directory."""
    import json as _json

    torch = _import_torch()
    _validate_feature_payload(payload)

    paths = ProductPaths.from_root(root)
    gp_id = str(payload["grid_point_id"])
    fn = str(payload.get("feature_name") or DEFAULT_FEATURE_NAME)
    layer_dir = paths.feature_layer_dir(gp_id, fn)
    layer_dir.mkdir(parents=True, exist_ok=True)

    features = payload["features"]  # [N, L, P, H]
    layer_indices = payload["layer_indices"]

    manifest = {
        "schema": FEATURE_SPLIT_SCHEMA,
        "grid_point_id": gp_id,
        "feature_name": fn,
        "backend": str(payload["backend"]),
        "model_id": str(payload["model_id"]),
        "layer_indices": [int(v) for v in layer_indices],
        "selected_positions": [int(v) for v in payload["selected_positions"]],
        "decision_point_ids": [str(v) for v in payload["decision_point_ids"]],
        "num_examples": int(features.shape[0]),
    }
    if payload.get("filter_last_replay_role") is not None:
        manifest["filter_last_replay_role"] = str(payload["filter_last_replay_role"])
    if payload.get("pre_filter_decision_point_count") is not None:
        manifest["pre_filter_decision_point_count"] = int(payload["pre_filter_decision_point_count"])
    if payload.get("post_filter_decision_point_count") is not None:
        manifest["post_filter_decision_point_count"] = int(payload["post_filter_decision_point_count"])
    if payload.get("prompt_restore_source_counts") is not None:
        manifest["prompt_restore_source_counts"] = dict(payload["prompt_restore_source_counts"])
    (layer_dir / "manifest.json").write_text(
        _json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for i, layer_idx in enumerate(layer_indices):
        shard_payload = {
            "schema": FEATURE_LAYER_SCHEMA,
            "grid_point_id": gp_id,
            "layer_index": int(layer_idx),
            "features": features[:, i, :, :].contiguous(),
        }
        torch.save(shard_payload, layer_dir / f"L{layer_idx}.pt")

    if not keep_original:
        bulk_path = paths.feature_shard(gp_id, fn)
        if bulk_path.exists():
            bulk_path.unlink()

    return layer_dir


def _load_from_split(
    torch: Any,
    paths: ProductPaths,
    gp_id: str,
    feature_name: str,
    layer_indices: Sequence[int] | None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    import json as _json

    manifest_path = paths.feature_layer_manifest(gp_id, feature_name)
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FEATURE_SPLIT_SCHEMA:
        raise ValueError(f"manifest at {manifest_path} has wrong schema")

    all_layer_indices = manifest["layer_indices"]
    if layer_indices is not None:
        load_set = set(int(li) for li in layer_indices)
        selected = [li for li in all_layer_indices if li in load_set]
    else:
        selected = list(all_layer_indices)

    if not selected:
        raise ValueError(f"no matching layers for gp={gp_id} requested={layer_indices}")

    layer_tensors = []
    for li in selected:
        shard_path = paths.feature_layer_shard(gp_id, li, feature_name)
        shard = torch.load(shard_path, map_location=map_location, weights_only=False)
        if shard.get("schema") != FEATURE_LAYER_SCHEMA:
            raise ValueError(f"layer shard at {shard_path} has wrong schema")
        feat = shard["features"]
        if map_location != "cpu" and feat.device.type == "cpu":
            feat = feat.to(map_location)
        layer_tensors.append(feat)

    features = torch.stack(layer_tensors, dim=1)

    result = {
        "schema": FEATURE_SCHEMA,
        "grid_point_id": manifest["grid_point_id"],
        "feature_name": manifest["feature_name"],
        "backend": manifest["backend"],
        "model_id": manifest["model_id"],
        "decision_point_ids": manifest["decision_point_ids"],
        "layer_indices": selected,
        "selected_positions": manifest["selected_positions"],
        "features": features,
    }
    if manifest.get("filter_last_replay_role") is not None:
        result["filter_last_replay_role"] = manifest["filter_last_replay_role"]
    if manifest.get("pre_filter_decision_point_count") is not None:
        result["pre_filter_decision_point_count"] = int(manifest["pre_filter_decision_point_count"])
    if manifest.get("post_filter_decision_point_count") is not None:
        result["post_filter_decision_point_count"] = int(manifest["post_filter_decision_point_count"])
    if manifest.get("prompt_restore_source_counts") is not None:
        result["prompt_restore_source_counts"] = dict(manifest["prompt_restore_source_counts"])
    return result


def _slice_payload_layers(
    payload: dict[str, Any], layer_indices: Sequence[int],
) -> dict[str, Any]:
    all_li = payload["layer_indices"]
    index_map = {li: i for i, li in enumerate(all_li)}
    offsets = [index_map[li] for li in layer_indices if li in index_map]
    if not offsets:
        raise ValueError(f"no matching layers: requested={list(layer_indices)} available={all_li}")
    selected_li = [all_li[o] for o in offsets]
    return {
        **{k: v for k, v in payload.items() if k != "features"},
        "layer_indices": selected_li,
        "features": payload["features"][:, offsets, :, :],
    }


def _validate_feature_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != FEATURE_SCHEMA:
        raise ValueError(f"feature payload must declare schema={FEATURE_SCHEMA!r}")
    for key in (
        "grid_point_id",
        "feature_name",
        "backend",
        "model_id",
        "decision_point_ids",
        "layer_indices",
        "selected_positions",
        "features",
    ):
        if key not in payload:
            raise ValueError(f"feature payload is missing {key!r}")
    features = payload["features"]
    if getattr(features, "ndim", None) != 4:
        raise ValueError("feature tensor must have shape [examples, layers, positions_or_flat, hidden]")
    if len(payload["decision_point_ids"]) != int(features.shape[0]):
        raise ValueError("feature tensor and decision_point_ids are misaligned")
    if len(payload["layer_indices"]) != int(features.shape[1]):
        raise ValueError("feature tensor and layer_indices are misaligned")


def _layer_to_position_hidden_tensor(value: Any) -> Any:
    torch = _import_torch()
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    return tensor


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("v2 feature products require torch in the current environment") from exc
    return torch
