from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..io import (
    load_decision_points,
    load_feature_payload,
    load_labels,
    load_partition,
    load_probe_checkpoint,
    save_probe_checkpoint,
    write_json,
)
from ..utils import CHECKPOINT_SCHEMA, DEFAULT_FEATURE_NAME, DEFAULT_LABELING_PROTOCOL


PROBE_ARCHITECTURES = ("linear", "mlp", "residual_mlp", "bilinear", "deep_mlp")
FEATURE_COMPOSITIONS = ("single_layer", "contiguous_concat")


def _build_normalized_probe(torch: Any, probe: Any, mean: Any, std: Any, mode: str = "standard") -> Any:
    """Wrap a probe model with input normalization.

    Returns a torch.nn.Module whose ``state_dict`` includes ``input_mean``,
    ``input_std`` alongside the wrapped probe parameters under a ``probe.`` prefix.

    mode: "standard" applies per-feature z-score normalization.
          "cosine" applies per-example L2 normalization (unit sphere).
          "none" passes raw features without any normalization.
    """

    class _NormalizedProbe(torch.nn.Module):
        def __init__(self, probe: Any, mean: Any, std: Any, mode: str):
            super().__init__()
            self.probe = probe
            self.norm_mode = mode
            if mode == "standard":
                self.register_buffer("input_mean", mean)
                self.register_buffer("input_std", std)

        def forward(self, x: Any) -> Any:
            if self.norm_mode == "none":
                return self.probe(x)
            if self.norm_mode == "cosine":
                return self.probe(x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-8)))
            return self.probe((x - self.input_mean) / self.input_std)

    return _NormalizedProbe(probe, mean, std, mode)


def _wmw_auroc(probs: Any, labels: Any) -> float | None:
    return _binary_auroc([int(label) for label in labels], [float(prob) for prob in probs])


@dataclass(frozen=True, slots=True)
class V2Split:
    name: str
    decision_point_ids: list[str]
    trace_ids: list[str]
    labels: list[int]
    features: Any
    metadata_rows: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class V2LabeledDataset:
    dataset_name: str
    labeling_protocol: str
    feature_name: str
    layer_indices: list[int]
    selected_positions: list[int]
    train: V2Split
    val: V2Split
    eval_splits: dict[str, V2Split]


def load_labeled_dataset(
    *,
    root: str | Path,
    dataset_name: str,
    labeling_protocol: str = DEFAULT_LABELING_PROTOCOL,
    feature_name: str = DEFAULT_FEATURE_NAME,
    include_eval: bool = False,
    filter_last_replay_role: str | None = None,
) -> V2LabeledDataset:
    partition = load_partition(root, dataset_name)
    labels = load_labels(root, labeling_protocol)
    grid_points = set(partition["train"]) | set(partition["val"])
    if include_eval:
        grid_points |= set(partition.get("eval_grid_points") or ())
    shards = {
        grid_point_id: load_feature_payload(root, grid_point_id, feature_name)
        for grid_point_id in sorted(grid_points)
    }
    metadata = {
        grid_point_id: {
            str(row["decision_point_id"]): row
            for row in load_decision_points(root, grid_point_id)
        }
        for grid_point_id in sorted(grid_points)
    }
    sample_shard = next(iter(shards.values()))
    layer_indices = [int(value) for value in sample_shard["layer_indices"]]
    selected_positions = [int(value) for value in sample_shard["selected_positions"]]
    _assert_compatible_shards(shards.values(), layer_indices=layer_indices, selected_positions=selected_positions)

    train = _build_split(
        name="train",
        id_membership=partition["train"],
        labels=labels,
        shards=shards,
        metadata=metadata,
        strict=False,
        filter_last_replay_role=filter_last_replay_role,
    )
    val = _build_split(
        name="val",
        id_membership=partition["val"],
        labels=labels,
        shards=shards,
        metadata=metadata,
        strict=False,
        filter_last_replay_role=filter_last_replay_role,
    )
    eval_splits: dict[str, V2Split] = {}
    if include_eval:
        for grid_point_id in partition.get("eval_grid_points") or ():
            shard_ids = [str(value) for value in shards[grid_point_id]["decision_point_ids"]]
            eval_splits[grid_point_id] = _build_split(
                name=grid_point_id,
                id_membership={grid_point_id: shard_ids},
                labels=labels,
                shards=shards,
                metadata=metadata,
                strict=False,
                filter_last_replay_role=filter_last_replay_role,
            )
    return V2LabeledDataset(
        dataset_name=str(dataset_name),
        labeling_protocol=str(labeling_protocol),
        feature_name=str(feature_name),
        layer_indices=layer_indices,
        selected_positions=selected_positions,
        train=train,
        val=val,
        eval_splits=eval_splits,
    )


def train_probe(
    *,
    root: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    labeling_protocol: str = DEFAULT_LABELING_PROTOCOL,
    feature_name: str = DEFAULT_FEATURE_NAME,
    layer_index: int | Sequence[int] | None = None,
    probe_architecture: str = "linear",
    feature_composition: str = "single_layer",
    concat_num_layers: int = 3,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
    bilinear_rank: int | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    threshold: float = 0.5,
    random_seed: int = 42,
    device: str | None = None,
    skip_eval: bool = False,
    log_fn: Callable[[str], None] | None = None,
    feature_normalization: str = "standard",
    pos_weight: float | str | None = None,
    checkpoint_mode: str = "best_val",
    filter_last_replay_role: str | None = None,
) -> Path:
    _emit_log(log_fn, f"loading labeled dataset root={root} dataset={dataset_name}")
    dataset = load_labeled_dataset(
        root=root,
        dataset_name=dataset_name,
        labeling_protocol=labeling_protocol,
        feature_name=feature_name,
        include_eval=False,
        filter_last_replay_role=filter_last_replay_role,
    )
    _emit_log(
        log_fn,
        f"loaded dataset train={len(dataset.train.labels)} val={len(dataset.val.labels)} "
        f"layers={len(dataset.layer_indices)} positions={len(dataset.selected_positions)}",
    )
    return train_probe_dataset(
        dataset=dataset,
        output_dir=output_dir,
        layer_index=layer_index,
        probe_architecture=probe_architecture,
        feature_composition=feature_composition,
        concat_num_layers=concat_num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
        bilinear_rank=bilinear_rank,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        threshold=threshold,
        random_seed=random_seed,
        device=device,
        log_fn=log_fn,
        feature_normalization=feature_normalization,
        pos_weight=pos_weight,
        checkpoint_mode=checkpoint_mode,
    )


def train_probe_dataset(
    *,
    dataset: V2LabeledDataset,
    output_dir: str | Path,
    layer_index: int | Sequence[int] | None = None,
    probe_architecture: str = "linear",
    feature_composition: str = "single_layer",
    concat_num_layers: int = 3,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
    bilinear_rank: int | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    threshold: float = 0.5,
    random_seed: int = 42,
    device: str | None = None,
    log_fn: Callable[[str], None] | None = None,
    feature_normalization: str = "standard",
    pos_weight: float | str | None = None,
    monitor_split: V2Split | None = None,
    monitor_name: str | None = None,
    checkpoint_mode: str = "best_val",
    filter_last_replay_role: str | None = None,
) -> Path:
    torch = _import_torch()
    torch.manual_seed(int(random_seed))
    if len(set(dataset.train.labels)) < 2:
        raise ValueError("training split must contain both classes")
    candidates = resolve_feature_candidates(
        layer_indices=dataset.layer_indices,
        layer_indices_filter=_normalize_layer_filter(layer_index),
        feature_composition=feature_composition,
        concat_num_layers=concat_num_layers,
    )
    architecture_config = resolve_probe_architecture_config(
        probe_architecture=probe_architecture,
        hidden_dim=hidden_dim,
        dropout=dropout,
        bilinear_rank=bilinear_rank,
    )
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _emit_log(
        log_fn,
        f"training {len(candidates)} candidate(s) on device={device_name} "
        f"train={len(dataset.train.labels)} val={len(dataset.val.labels)}",
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    models_dir = output_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    layer_results: list[dict[str, Any]] = []
    best_overall: dict[str, Any] | None = None
    for candidate_index, candidate in enumerate(candidates):
        _emit_log(
            log_fn,
            f"candidate {candidate_index + 1}/{len(candidates)} "
            f"layers={candidate['selected_layer_indices']} offsets={candidate['selected_layer_offsets']}",
        )
        # Prepare monitor features for this candidate's layer offsets.
        mon_x, mon_y = None, None
        if monitor_split is not None:
            mon_x = _split_features_for_candidate(torch, monitor_split, candidate["selected_layer_offsets"], device_name)
            mon_y = torch.tensor(monitor_split.labels, dtype=torch.float32, device=device_name).unsqueeze(1)
        model, checkpoint, metrics = _train_one_candidate(
            torch=torch,
            dataset=dataset,
            dataset_name=dataset.dataset_name,
            labeling_protocol=dataset.labeling_protocol,
            feature_name=dataset.feature_name,
            candidate=candidate,
            architecture_config=architecture_config,
            probe_architecture=probe_architecture,
            feature_composition=feature_composition,
            concat_num_layers=concat_num_layers,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            threshold=threshold,
            random_seed=random_seed + candidate_index,
            device_name=device_name,
            log_fn=log_fn,
            feature_normalization=feature_normalization,
            pos_weight=pos_weight,
            monitor_x=mon_x,
            monitor_y=mon_y,
            monitor_name=monitor_name or "monitor",
            checkpoint_mode=checkpoint_mode,
            filter_last_replay_role=filter_last_replay_role,
        )
        checkpoint_path = models_dir / _checkpoint_filename(candidate["selected_layer_indices"])
        checkpoint["checkpoint_role"] = "per_layer_best"
        save_probe_checkpoint(checkpoint_path, checkpoint)
        result = {
            "checkpoint_path": str(checkpoint_path),
            "selected_layer_indices": list(candidate["selected_layer_indices"]),
            "selected_layer_offsets": list(candidate["selected_layer_offsets"]),
            **metrics,
        }
        layer_results.append(result)
        _emit_log(
            log_fn,
            f"saved candidate {candidate_index + 1}/{len(candidates)} checkpoint={checkpoint_path} "
            f"val_auroc={result['val'].get('auroc')}",
        )
        if best_overall is None or _score_metrics(result["val"]) > _score_metrics(best_overall["val"]):
            best_overall = result

    metrics = {
        "schema": "ipi_aware.probe_metrics.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_save_mode": "per_layer_best",
        "models_dir": "models",
        "probe_architecture": normalize_probe_architecture(probe_architecture),
        "probe_architecture_config": architecture_config,
        "feature_composition": normalize_feature_composition(feature_composition),
        "concat_num_layers": int(concat_num_layers),
        "layer_results": layer_results,
        "best_checkpoint_path": None if best_overall is None else best_overall["checkpoint_path"],
    }
    write_json(output_path / "metrics.json", metrics)
    _emit_log(log_fn, f"wrote metrics to {output_path / 'metrics.json'}")
    return output_path


def _train_one_candidate(
    *,
    torch: Any,
    dataset: V2LabeledDataset,
    dataset_name: str,
    labeling_protocol: str,
    feature_name: str,
    candidate: Mapping[str, Any],
    architecture_config: Mapping[str, Any],
    probe_architecture: str,
    feature_composition: str,
    concat_num_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    threshold: float,
    random_seed: int,
    device_name: str,
    log_fn: Callable[[str], None] | None,
    feature_normalization: str = "standard",
    pos_weight: float | str | None = None,
    monitor_x: Any | None = None,
    monitor_y: Any | None = None,
    monitor_name: str = "monitor",
    checkpoint_mode: str = "best_val",
    filter_last_replay_role: str | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(int(random_seed))
    train_x = _split_features_for_candidate(torch, dataset.train, candidate["selected_layer_offsets"], device_name)
    train_y = torch.tensor(dataset.train.labels, dtype=torch.float32, device=device_name).unsqueeze(1)
    val_split = dataset.val if dataset.val.decision_point_ids else dataset.train
    val_x = _split_features_for_candidate(torch, val_split, candidate["selected_layer_offsets"], device_name)
    val_y = torch.tensor(val_split.labels, dtype=torch.float32, device=device_name).unsqueeze(1)
    model = build_probe_model(
        torch=torch,
        input_dim=int(train_x.shape[1]),
        probe_architecture=probe_architecture,
        architecture_config=architecture_config,
    ).to(device_name)
    # Compute per-feature normalization from training set.
    if feature_normalization == "standard":
        feature_mean = train_x.mean(dim=0, keepdim=True).detach()
        feature_std = train_x.std(dim=0, keepdim=True).clamp(min=1e-8).detach()
    else:
        feature_mean = torch.zeros(1, train_x.shape[1], device=device_name)
        feature_std = torch.ones(1, train_x.shape[1], device=device_name)
    model = _build_normalized_probe(torch, model, feature_mean, feature_std, mode=feature_normalization).to(device_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    # Resolve pos_weight: "auto" computes neg/pos ratio from training labels
    resolved_pw: float | None = None
    if isinstance(pos_weight, str) and pos_weight.lower() in ("auto", "balanced"):
        positive_count = int(sum(dataset.train.labels))
        negative_count = len(dataset.train.labels) - positive_count
        if positive_count > 0 and negative_count > 0:
            resolved_pw = negative_count / positive_count
            _emit_log(log_fn, f"  pos_weight=auto: {resolved_pw:.2f} (neg={negative_count} pos={positive_count})")
    elif pos_weight is not None:
        resolved_pw = float(pos_weight)
    if resolved_pw is not None:
        pw = torch.tensor([resolved_pw], device=device_name)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    # Mini-batch setup: batch_size<=0 or >= train_size means full-batch
    effective_bs = int(batch_size) if int(batch_size) > 0 else len(train_x)
    use_batches = effective_bs < len(train_x)
    if use_batches:
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=effective_bs, shuffle=True,
            generator=torch.Generator().manual_seed(int(random_seed)),
        )
    best_state = None
    best_val_loss = None
    best_mon_auroc = None
    best_epoch = None
    use_monitor_for_checkpoint = monitor_x is not None
    use_final_checkpoint = checkpoint_mode == "final"
    epoch_history: list[dict[str, Any]] = []

    # Epoch 0: random-init eval (before any gradient steps).
    model.eval()
    with torch.no_grad():
        rand_val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        rand_probs = torch.sigmoid(model(val_x)).squeeze(1).cpu().numpy()
        rand_labels = val_y.squeeze(1).cpu().numpy()
        rand_auroc = _wmw_auroc(rand_probs, rand_labels)
    rand_monitor_msg = ""
    if monitor_x is not None:
        with torch.no_grad():
            rand_mon_probs = torch.sigmoid(model(monitor_x)).squeeze(1).cpu().numpy()
            rand_mon_labels = monitor_y.squeeze(1).cpu().numpy()
            rand_mon_auroc = _wmw_auroc(rand_mon_probs, rand_mon_labels)
        rand_monitor_msg = f" {monitor_name}_auroc={_format_optional_metric(rand_mon_auroc)}"
    _emit_log(
        log_fn,
        f"epoch 0/{int(epochs)} (random) val_loss={rand_val_loss:.6f} "
        f"val_auroc={_format_optional_metric(rand_auroc)}{rand_monitor_msg}",
    )

    for epoch_index in range(int(epochs)):
        model.train()
        if use_batches:
            epoch_loss = 0.0
            n_batches = 0
            for bx, by in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(bx), by)
                epoch_loss += float(loss.detach().cpu()) * len(bx)
                n_batches += len(bx)
                loss.backward()
                optimizer.step()
            train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x), train_y)
            train_loss = float(loss.detach().cpu())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
            val_probs = torch.sigmoid(model(val_x)).squeeze(1).cpu().numpy()
            val_labels_np = val_y.squeeze(1).cpu().numpy()
            val_auroc = _wmw_auroc(val_probs, val_labels_np)
            mon_auroc = None
            if monitor_x is not None:
                mon_probs = torch.sigmoid(model(monitor_x)).squeeze(1).cpu().numpy()
                mon_labels = monitor_y.squeeze(1).cpu().numpy()
                mon_auroc = _wmw_auroc(mon_probs, mon_labels)
        if not use_final_checkpoint:
            if use_monitor_for_checkpoint:
                if mon_auroc is not None and (best_mon_auroc is None or mon_auroc > best_mon_auroc):
                    best_mon_auroc = mon_auroc
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    best_epoch = epoch_index + 1
            else:
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    best_epoch = epoch_index + 1
        epoch_history.append({
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            **({f"{monitor_name}_auroc": mon_auroc} if mon_auroc is not None else {}),
        })
        mon_msg = f" {monitor_name}_auroc={_format_optional_metric(mon_auroc)}" if monitor_x is not None else ""
        if use_final_checkpoint:
            best_msg = "checkpoint=final"
        elif use_monitor_for_checkpoint:
            best_msg = f"best_{monitor_name}_auroc={_format_optional_metric(best_mon_auroc)}"
        else:
            best_msg = f"best_val_loss={float(best_val_loss):.6f}"
        _emit_log(
            log_fn,
            f"epoch {epoch_index + 1}/{int(epochs)} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_auroc={_format_optional_metric(val_auroc)}{mon_msg} {best_msg}",
        )
    if not use_final_checkpoint and best_state is not None:
        model.load_state_dict(best_state)
    if use_final_checkpoint:
        best_epoch = int(epochs)
    checkpoint = {
        **build_probe_checkpoint_metadata(
            dataset_name=dataset_name,
            labeling_protocol=labeling_protocol,
            feature_name=feature_name,
            layer_indices=dataset.layer_indices,
            selected_positions=dataset.selected_positions,
            selected_layer_indices=candidate["selected_layer_indices"],
            selected_layer_offsets=candidate["selected_layer_offsets"],
            feature_composition=feature_composition,
            concat_num_layers=concat_num_layers,
            probe_architecture=probe_architecture,
            probe_architecture_config=architecture_config,
            input_dim=int(train_x.shape[1]),
            threshold=threshold,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_seed=random_seed,
            feature_normalization=feature_normalization,
            pos_weight=resolved_pw,
            filter_last_replay_role=filter_last_replay_role,
        ),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "best_epoch": best_epoch,
        "total_epochs": int(epochs),
    }
    metrics = {
        "train": _predict_split_metrics(torch, model, dataset.train, candidate["selected_layer_offsets"], device_name, threshold),
        "val": _predict_split_metrics(torch, model, dataset.val, candidate["selected_layer_offsets"], device_name, threshold),
        "best_epoch": best_epoch,
        "total_epochs": int(epochs),
        "epoch_history": epoch_history,
    }
    return model, checkpoint, metrics


def evaluate_checkpoint(
    *,
    root: str | Path,
    dataset_name: str | None = None,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    torch = _import_torch()
    checkpoint = load_probe_checkpoint(checkpoint_path)
    resolved_dataset_name = str(dataset_name or checkpoint["dataset_name"])
    dataset = load_labeled_dataset(
        root=root,
        dataset_name=resolved_dataset_name,
        labeling_protocol=str(checkpoint["labeling_protocol"]),
        feature_name=str(checkpoint["feature_name"]),
        include_eval=True,
    )
    selected_layer_offsets = [int(value) for value in checkpoint["selected_layer_offsets"]]
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_probe_model(
        torch=torch,
        input_dim=int(checkpoint["input_dim"]),
        probe_architecture=str(checkpoint["probe_architecture"]),
        architecture_config=dict(checkpoint.get("probe_architecture_config") or {}),
    ).to(device_name)
    sd = checkpoint["model_state_dict"]
    norm_mode = (checkpoint.get("training_config") or {}).get("feature_normalization", "standard")
    if norm_mode != "none":
        model = _build_normalized_probe(
            torch, model,
            torch.zeros(1, int(checkpoint["input_dim"])),
            torch.ones(1, int(checkpoint["input_dim"])),
            mode=norm_mode,
        ).to(device_name)
    model.load_state_dict(sd)
    model.eval()
    rows = []
    for split_name, split in sorted(dataset.eval_splits.items()):
        probabilities = _predict_probabilities(torch, model, split, selected_layer_offsets, device_name)
        metrics = _binary_metrics(split.labels, probabilities, float(checkpoint.get("threshold", 0.5)))
        rows.append(
            {
                "schema": "ipi_aware.probe_eval.v2",
                "checkpoint_path": str(checkpoint_path),
                "dataset_name": resolved_dataset_name,
                "grid_point_id": split_name,
                "example_count": len(split.decision_point_ids),
                "metrics": metrics,
            }
        )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
    return rows


def build_probe_checkpoint_metadata(
    *,
    dataset_name: str,
    labeling_protocol: str,
    feature_name: str,
    layer_indices: Sequence[int],
    selected_positions: Sequence[int],
    selected_layer_indices: Sequence[int],
    selected_layer_offsets: Sequence[int],
    feature_composition: str,
    concat_num_layers: int,
    probe_architecture: str,
    probe_architecture_config: Mapping[str, Any],
    input_dim: int,
    threshold: float,
    epochs: int,
    batch_size: int = 256,
    learning_rate: float,
    weight_decay: float,
    random_seed: int,
    feature_normalization: str = "standard",
    pos_weight: float | str | None = None,
    filter_last_replay_role: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": str(dataset_name),
        "labeling_protocol": str(labeling_protocol),
        "feature_name": str(feature_name),
        "layer_indices": [int(value) for value in layer_indices],
        "selected_positions": [int(value) for value in selected_positions],
        "feature_composition": normalize_feature_composition(feature_composition),
        "concat_num_layers": int(concat_num_layers),
        "selected_layer_indices": [int(value) for value in selected_layer_indices],
        "selected_layer_offsets": [int(value) for value in selected_layer_offsets],
        "selected_layer_index": int(selected_layer_indices[0]),
        "selected_layer_offset": int(selected_layer_offsets[0]),
        "probe_architecture": normalize_probe_architecture(probe_architecture),
        "probe_architecture_config": dict(probe_architecture_config),
        "input_dim": int(input_dim),
        "threshold": float(threshold),
        "training_config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "random_seed": int(random_seed),
            "feature_normalization": str(feature_normalization),
            "pos_weight": float(pos_weight) if pos_weight is not None else None,
            "filter_last_replay_role": str(filter_last_replay_role) if filter_last_replay_role is not None else None,
        },
    }


def normalize_probe_architecture(probe_architecture: str) -> str:
    value = str(probe_architecture).lower()
    if value not in PROBE_ARCHITECTURES:
        raise ValueError(f"unsupported probe_architecture: {probe_architecture}")
    return value


def normalize_feature_composition(feature_composition: str) -> str:
    value = str(feature_composition).lower()
    if value not in FEATURE_COMPOSITIONS:
        raise ValueError(f"unsupported feature_composition: {feature_composition}")
    return value


def resolve_probe_architecture_config(
    *,
    probe_architecture: str,
    hidden_dim: int | None,
    dropout: float,
    bilinear_rank: int | None,
) -> dict[str, Any]:
    architecture = normalize_probe_architecture(probe_architecture)
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")
    if architecture in {"mlp", "residual_mlp", "deep_mlp"}:
        return {
            "hidden_dim": int(hidden_dim or 256),
            "dropout": float(dropout),
        }
    if architecture == "bilinear":
        return {"bilinear_rank": int(bilinear_rank or 128)}
    return {}


def resolve_feature_candidate(
    *,
    layer_indices: Sequence[int],
    layer_index: int,
    feature_composition: str,
    concat_num_layers: int,
) -> dict[str, list[int] | str | int]:
    composition = normalize_feature_composition(feature_composition)
    if composition == "single_layer":
        offset = _resolve_layer_offset(layer_indices, layer_index)
        return {
            "feature_composition": composition,
            "concat_num_layers": 1,
            "selected_layer_offsets": [offset],
            "selected_layer_indices": [int(layer_indices[offset])],
        }
    if concat_num_layers < 2:
        raise ValueError("concat_num_layers must be at least 2 for contiguous_concat")
    if concat_num_layers > len(layer_indices):
        raise ValueError("concat_num_layers exceeds available layer count")
    start_offset = _resolve_concat_start_offset(layer_indices, layer_index, concat_num_layers)
    offsets = list(range(start_offset, start_offset + int(concat_num_layers)))
    return {
        "feature_composition": composition,
        "concat_num_layers": int(concat_num_layers),
        "selected_layer_offsets": offsets,
        "selected_layer_indices": [int(layer_indices[offset]) for offset in offsets],
    }


def resolve_feature_candidates(
    *,
    layer_indices: Sequence[int],
    layer_indices_filter: Sequence[int] | None,
    feature_composition: str,
    concat_num_layers: int,
) -> list[dict[str, Any]]:
    if layer_indices_filter:
        return [
            resolve_feature_candidate(
                layer_indices=layer_indices,
                layer_index=layer_index,
                feature_composition=feature_composition,
                concat_num_layers=concat_num_layers,
            )
            for layer_index in layer_indices_filter
        ]
    composition = normalize_feature_composition(feature_composition)
    if composition == "single_layer":
        return [
            resolve_feature_candidate(
                layer_indices=layer_indices,
                layer_index=int(layer_index),
                feature_composition=composition,
                concat_num_layers=concat_num_layers,
            )
            for layer_index in layer_indices
        ]
    max_start = len(layer_indices) - int(concat_num_layers)
    if max_start < 0:
        raise ValueError("concat_num_layers exceeds available layer count")
    return [
        resolve_feature_candidate(
            layer_indices=layer_indices,
            layer_index=int(layer_indices[start_offset]),
            feature_composition=composition,
            concat_num_layers=concat_num_layers,
        )
        for start_offset in range(max_start + 1)
    ]


def build_probe_model(
    *,
    torch: Any,
    input_dim: int,
    probe_architecture: str,
    architecture_config: Mapping[str, Any],
) -> Any:
    architecture = normalize_probe_architecture(probe_architecture)
    if architecture == "linear":
        return torch.nn.Linear(int(input_dim), 1)
    if architecture == "mlp":
        hidden_dim = int(architecture_config.get("hidden_dim") or 256)
        dropout = float(architecture_config.get("dropout") or 0.0)
        return torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
    if architecture == "residual_mlp":
        return _build_residual_mlp(torch, int(input_dim), architecture_config)
    if architecture == "deep_mlp":
        return _build_deep_mlp(torch, int(input_dim), architecture_config)
    if architecture == "bilinear":
        return _build_bilinear_probe(torch, int(input_dim), architecture_config)
    raise AssertionError(f"unreachable architecture: {architecture}")


def _normalize_layer_filter(layer_index: int | Sequence[int] | None) -> list[int] | None:
    if layer_index is None:
        return None
    if isinstance(layer_index, int):
        return [int(layer_index)]
    values = [int(value) for value in layer_index]
    return values or None


def _checkpoint_filename(selected_layer_indices: Sequence[int]) -> str:
    if len(selected_layer_indices) == 1:
        return f"best_layer_{int(selected_layer_indices[0]):02d}.pt"
    return "best_layers_" + "_".join(f"{int(layer_index):02d}" for layer_index in selected_layer_indices) + ".pt"


def _score_metrics(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    auroc = metrics.get("auroc")
    if auroc is None:
        auroc = -1.0
    balanced_accuracy = metrics.get("balanced_accuracy")
    if balanced_accuracy is None:
        balanced_accuracy = -1.0
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        accuracy = -1.0
    return float(auroc), float(balanced_accuracy), float(accuracy)


def _build_split(
    *,
    name: str,
    id_membership: Mapping[str, Sequence[str]],
    labels: Mapping[str, int],
    shards: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Mapping[str, Any]]],
    strict: bool,
    filter_last_replay_role: str | None = None,
) -> V2Split:
    torch = _import_torch()
    tensors = []
    decision_point_ids: list[str] = []
    trace_ids: list[str] = []
    split_labels: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    for grid_point_id, requested_ids in sorted(id_membership.items()):
        shard = shards[grid_point_id]
        shard_index = {str(dp_id): index for index, dp_id in enumerate(shard["decision_point_ids"])}
        for decision_point_id in requested_ids:
            decision_point_id = str(decision_point_id)
            missing = decision_point_id not in labels or decision_point_id not in shard_index
            if missing and strict:
                raise ValueError(f"{name} decision point {decision_point_id} is missing a label or feature")
            if missing:
                continue
            row = dict(metadata.get(grid_point_id, {}).get(decision_point_id, {}))
            if not _matches_last_replay_role(row, filter_last_replay_role):
                continue
            if filter_last_replay_role is not None:
                row["_filter_last_replay_role"] = str(filter_last_replay_role)
            tensors.append(shard["features"][shard_index[decision_point_id]])
            decision_point_ids.append(decision_point_id)
            trace_ids.append(str(row.get("trace_id") or ""))
            split_labels.append(int(labels[decision_point_id]))
            metadata_rows.append(row)
    if tensors:
        features = torch.stack(tensors, dim=0)
    else:
        first = next(iter(shards.values()))
        shape = first["features"].shape
        features = torch.empty((0, int(shape[1]), int(shape[2]), int(shape[3])), dtype=first["features"].dtype)
    return V2Split(
        name=name,
        decision_point_ids=decision_point_ids,
        trace_ids=trace_ids,
        labels=split_labels,
        features=features,
        metadata_rows=metadata_rows,
    )


def _matches_last_replay_role(row: Mapping[str, Any], filter_last_replay_role: str | None) -> bool:
    if filter_last_replay_role is None:
        return True
    replay_request = row.get("replay_request")
    if not isinstance(replay_request, Mapping):
        return False
    messages = replay_request.get("messages")
    if not isinstance(messages, Sequence) or not messages:
        return False
    last_message = messages[-1]
    if not isinstance(last_message, Mapping):
        return False
    return str(last_message.get("role")) == str(filter_last_replay_role)


def _assert_compatible_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    layer_indices: Sequence[int],
    selected_positions: Sequence[int],
) -> None:
    for shard in shards:
        if [int(value) for value in shard["layer_indices"]] != list(layer_indices):
            raise ValueError("all v2 feature shards in a dataset must use the same layer_indices")
        # selected_positions assertion disabled: mixed backends (transformers_hook
        # vs offline_vllm_direct) report different position encodings but extract
        # from the same actual token.


def _resolve_layer_offset(layer_indices: Sequence[int], layer_index: int) -> int:
    if layer_index < 0:
        offset = len(layer_indices) + layer_index
        if offset < 0:
            raise ValueError(f"layer_index {layer_index} is out of range")
        return offset
    try:
        return list(layer_indices).index(int(layer_index))
    except ValueError as exc:
        raise ValueError(f"layer_index {layer_index} is not present in feature layer_indices") from exc


def _resolve_concat_start_offset(
    layer_indices: Sequence[int],
    layer_index: int,
    concat_num_layers: int,
) -> int:
    max_start_offset = len(layer_indices) - int(concat_num_layers)
    if layer_index < 0:
        offset = (max_start_offset + 1) + int(layer_index)
    else:
        offset = _resolve_layer_offset(layer_indices, layer_index)
    if offset < 0 or offset > max_start_offset:
        raise ValueError(
            f"layer_index {layer_index} cannot start a contiguous_concat{concat_num_layers} "
            f"candidate over {len(layer_indices)} layers"
        )
    return offset


def _split_features_for_candidate(
    torch: Any,
    split: V2Split,
    layer_offsets: Sequence[int],
    device_name: str,
) -> Any:
    features = split.features[:, list(layer_offsets), :, :].reshape(len(split.decision_point_ids), -1)
    return features.to(device=device_name, dtype=torch.float32)


def _predict_probabilities(
    torch: Any,
    model: Any,
    split: V2Split,
    layer_offsets: Sequence[int],
    device_name: str,
) -> list[float]:
    if not split.decision_point_ids:
        return []
    features = _split_features_for_candidate(torch, split, layer_offsets, device_name)
    with torch.no_grad():
        logits = model(features).squeeze(-1)
        return [float(value) for value in torch.sigmoid(logits).detach().cpu().tolist()]


def _predict_split_metrics(
    torch: Any,
    model: Any,
    split: V2Split,
    layer_offsets: Sequence[int],
    device_name: str,
    threshold: float,
) -> dict[str, Any]:
    return _binary_metrics(
        split.labels,
        _predict_probabilities(torch, model, split, layer_offsets, device_name),
        threshold,
    )


def _build_residual_mlp(torch: Any, input_dim: int, architecture_config: Mapping[str, Any]) -> Any:
    hidden_dim = int(architecture_config.get("hidden_dim") or 256)
    dropout = float(architecture_config.get("dropout") or 0.0)

    class ResidualMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = torch.nn.Linear(input_dim, hidden_dim)
            self.block = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            self.output = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x: Any) -> Any:
            hidden = self.input_projection(x)
            return self.output(hidden + self.block(hidden))

    return ResidualMLP()


def _build_deep_mlp(torch: Any, input_dim: int, architecture_config: Mapping[str, Any]) -> Any:
    hidden_dim = int(architecture_config.get("hidden_dim") or 256)
    dropout = float(architecture_config.get("dropout") or 0.0)
    layers = [
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, hidden_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim // 2, 1),
    ]
    return torch.nn.Sequential(*layers)


def _build_bilinear_probe(torch: Any, input_dim: int, architecture_config: Mapping[str, Any]) -> Any:
    rank = int(architecture_config.get("bilinear_rank") or 128)

    class BilinearProbe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, 1)
            self.left = torch.nn.Linear(input_dim, rank, bias=False)
            self.right = torch.nn.Linear(input_dim, rank, bias=False)
            self.scale = rank ** -0.5

        def forward(self, x: Any) -> Any:
            interaction = (self.left(x) * self.right(x)).sum(dim=-1, keepdim=True) * self.scale
            return self.linear(x) + interaction

    return BilinearProbe()


def _binary_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float) -> dict[str, Any]:
    if not labels:
        return {
            "example_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "threshold": float(threshold),
            "accuracy": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": None,
            "average_precision": None,
            "best_accuracy": None,
            "best_accuracy_threshold": None,
            "best_balanced_accuracy": None,
            "best_balanced_accuracy_threshold": None,
            "best_f1": None,
            "best_f1_threshold": None,
        }
    normalized_labels = [int(label) for label in labels]
    normalized_probabilities = [float(probability) for probability in probabilities]
    predictions = [int(float(probability) >= threshold) for probability in probabilities]
    threshold_metrics = _threshold_metrics(normalized_labels, predictions)
    best_metrics = _sweep_best_threshold_metrics(normalized_labels, normalized_probabilities)
    positive_count = sum(normalized_labels)
    unique_labels = set(normalized_labels)
    auroc = _binary_auroc(normalized_labels, normalized_probabilities) if len(unique_labels) == 2 else None
    average_precision = (
        _binary_average_precision(normalized_labels, normalized_probabilities)
        if positive_count > 0
        else None
    )
    return {
        "example_count": len(labels),
        "positive_count": int(positive_count),
        "negative_count": int(len(labels) - positive_count),
        "threshold": float(threshold),
        "accuracy": threshold_metrics["accuracy"],
        "balanced_accuracy": threshold_metrics["balanced_accuracy"],
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "f1": threshold_metrics["f1"],
        "auroc": auroc,
        "average_precision": average_precision,
        "best_accuracy": best_metrics["best_accuracy"],
        "best_accuracy_threshold": best_metrics["best_accuracy_threshold"],
        "best_balanced_accuracy": best_metrics["best_balanced_accuracy"],
        "best_balanced_accuracy_threshold": best_metrics["best_balanced_accuracy_threshold"],
        "best_f1": best_metrics["best_f1"],
        "best_f1_threshold": best_metrics["best_f1_threshold"],
    }


def _threshold_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    import numpy as np
    y = np.array(labels, dtype=np.int32)
    p = np.array(predictions, dtype=np.int32)
    positive_count = int(y.sum())
    negative_count = len(y) - positive_count
    true_positive = int((y & p).sum())
    true_negative = int((~y.astype(bool) & ~p.astype(bool)).sum())
    false_positive = int(((~y.astype(bool)) & p.astype(bool)).sum())
    false_negative = int((y & ~p.astype(bool)).sum())
    accuracy = (true_positive + true_negative) / len(labels)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / positive_count if positive_count else 0.0
    negative_recall = true_negative / negative_count if negative_count else None
    balanced_parts = [recall]
    if negative_recall is not None:
        balanced_parts.append(negative_recall)
    balanced_accuracy = sum(balanced_parts) / len(balanced_parts)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _sweep_best_threshold_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float | None]:
    if not labels:
        return {
            "best_accuracy": None,
            "best_accuracy_threshold": None,
            "best_balanced_accuracy": None,
            "best_balanced_accuracy_threshold": None,
            "best_f1": None,
            "best_f1_threshold": None,
        }
    import numpy as np
    y = np.array(labels, dtype=np.int32)
    scores = np.array(probabilities, dtype=np.float64)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos

    best_accuracy = -1.0
    best_accuracy_threshold = 0.0
    best_balanced_accuracy = -1.0
    best_balanced_accuracy_threshold = 0.0
    best_f1 = -1.0
    best_f1_threshold = 0.0

    thresholds = np.arange(101) / 100.0
    preds = (scores[np.newaxis, :] >= thresholds[:, np.newaxis]).astype(np.int32)  # (101, n)

    tp = (preds * y[np.newaxis, :]).sum(axis=1)
    fp = (preds * (1 - y[np.newaxis, :])).sum(axis=1)
    fn = ((1 - preds) * y[np.newaxis, :]).sum(axis=1)
    tn = ((1 - preds) * (1 - y[np.newaxis, :])).sum(axis=1)

    accuracy = (tp + tn) / n
    tpr = tp / n_pos if n_pos > 0 else np.zeros_like(tp)
    tnr = tn / n_neg if n_neg > 0 else np.zeros_like(tn)
    balanced_accuracy = (tpr + tnr) / 2
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = tpr
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float), where=(precision + recall) > 0)

    idx = np.argmax(accuracy)
    best_accuracy = float(accuracy[idx])
    best_accuracy_threshold = float(thresholds[idx])
    idx = np.argmax(balanced_accuracy)
    best_balanced_accuracy = float(balanced_accuracy[idx])
    best_balanced_accuracy_threshold = float(thresholds[idx])
    idx = np.argmax(f1)
    best_f1 = float(f1[idx])
    best_f1_threshold = float(thresholds[idx])

    return {
        "best_accuracy": best_accuracy,
        "best_accuracy_threshold": best_accuracy_threshold,
        "best_balanced_accuracy": best_balanced_accuracy,
        "best_balanced_accuracy_threshold": best_balanced_accuracy_threshold,
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_threshold,
    }


def _binary_auroc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    from sklearn.metrics import roc_auc_score

    unique = set(int(label) for label in labels)
    if len(unique) < 2:
        return None
    return float(roc_auc_score(list(labels), list(probabilities)))


def _format_optional_metric(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def _binary_average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positive_count = sum(int(label) for label in labels)
    if positive_count == 0:
        return None
    ranked = sorted(
        zip(probabilities, labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    hit_count = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ranked, start=1):
        if int(label) != 1:
            continue
        hit_count += 1
        precision_sum += hit_count / rank
    return precision_sum / positive_count


def _emit_log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("v2 probe train/eval requires torch in the current environment") from exc
    return torch
