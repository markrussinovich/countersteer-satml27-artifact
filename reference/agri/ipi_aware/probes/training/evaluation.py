from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..io import load_decision_points, load_feature_payload, load_labels, load_partition, load_probe_checkpoint, write_json
from .training import _binary_metrics, _build_normalized_probe, _matches_last_replay_role, build_probe_model

EVAL_GROUP_NAMES = ("strict", "unseen_attacks", "unseen_suites", "unseen_prompts")


@dataclass
class ProbeEvalSpec:
    probe_id: str
    checkpoint: Mapping[str, Any]
    eval_groups_def: Mapping[str, Sequence[str]]


@dataclass
class ProbeEvalResult:
    probe_id: str
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)


class BatchedProbePredictor:
    """Groups linear probes by layer offset, stacks weights for batched matmul."""

    def __init__(
        self,
        specs: Sequence[ProbeEvalSpec],
        device: str,
        offset_map: dict[int, int] | None = None,
    ) -> None:
        torch = _import_torch()
        self._torch = torch
        self._device = device
        self._offset_map = offset_map or {}
        self._spec_index: dict[str, int] = {}
        self._probe_ids: list[str] = []
        self._offset_groups: dict[tuple[int, ...], dict[str, Any]] = {}

        by_offset: dict[tuple[int, ...], list[tuple[int, ProbeEvalSpec]]] = defaultdict(list)
        for i, spec in enumerate(specs):
            self._spec_index[spec.probe_id] = i
            self._probe_ids.append(spec.probe_id)
            offsets = tuple(int(v) for v in spec.checkpoint["selected_layer_offsets"])
            by_offset[offsets].append((i, spec))

        sub_id = 0
        for offsets, group_specs in by_offset.items():
            # Sub-group by normalization mode to avoid mixing state dict shapes
            by_norm: dict[str, list[tuple[int, ProbeEvalSpec]]] = defaultdict(list)
            for idx, spec in group_specs:
                tc = spec.checkpoint.get("training_config") or {}
                nm = tc.get("feature_normalization", "standard")
                by_norm[nm].append((idx, spec))
            for norm_mode, norm_specs in by_norm.items():
                sds = [s.checkpoint["model_state_dict"] for _, s in norm_specs]
                sub_key = offsets if len(by_norm) == 1 else (offsets, sub_id)
                sub_id += 1
                entry: dict[str, Any] = {
                    "offsets_t": torch.tensor(offsets, dtype=torch.long),
                    "indices": [idx for idx, _ in norm_specs],
                    "weights": torch.stack([sd["probe.weight"].squeeze(0) for sd in sds]).to(device),
                    "biases": torch.stack([sd["probe.bias"].squeeze(0) for sd in sds]).to(device),
                    "norm_mode": norm_mode,
                }
                if norm_mode == "standard" and sds and "input_mean" in sds[0]:
                    entry["means"] = torch.stack([sd["input_mean"].squeeze(0) for sd in sds]).to(device)
                    entry["stds"] = torch.stack([sd["input_std"].squeeze(0) for sd in sds]).to(device)
                self._offset_groups[sub_key] = entry

    @property
    def probe_ids(self) -> list[str]:
        return list(self._probe_ids)

    def predict_gp(
        self,
        feature_tensor: Any,
        labeled_indices: Sequence[int],
        eval_batch_size: int = 4096,
    ) -> dict[str, list[float]]:
        torch = self._torch
        result: dict[str, list[float]] = {pid: [] for pid in self._probe_ids}
        if not labeled_indices:
            return result

        for offsets, entry in self._offset_groups.items():
            offsets_t = entry["offsets_t"]
            if self._offset_map:
                offsets_t = torch.tensor(
                    [self._offset_map[int(o)] for o in offsets_t],
                    dtype=torch.long,
                )
            weights = entry["weights"]
            biases = entry["biases"]
            has_norm = "means" in entry
            norm_mode = entry.get("norm_mode", "standard")
            indices_in_group = entry["indices"]

            for start in range(0, len(labeled_indices), eval_batch_size):
                batch_idx = labeled_indices[start : start + eval_batch_size]
                batch_feat = (
                    feature_tensor[batch_idx][:, offsets_t, :, :]
                    .reshape(len(batch_idx), -1)
                    .to(device=self._device, dtype=torch.float32)
                )
                if norm_mode == "cosine":
                    batch_feat = batch_feat / (batch_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                    logits = batch_feat @ weights.T + biases
                elif has_norm:
                    means = entry["means"]
                    stds = entry["stds"]
                    batch_feat = (batch_feat.unsqueeze(1) - means.unsqueeze(0)) / stds.unsqueeze(0)
                    logits = (batch_feat * weights.unsqueeze(0)).sum(-1) + biases
                else:
                    logits = batch_feat @ weights.T + biases
                probs_all = torch.sigmoid(logits).detach().cpu()

                for j, spec_idx in enumerate(indices_in_group):
                    result[self._probe_ids[spec_idx]].extend(probs_all[:, j].tolist())

                del batch_feat, logits, probs_all

        return result

    def predict_gp_tensors(
        self,
        feature_tensor: Any,
        labeled_indices: Sequence[int],
        eval_batch_size: int = 4096,
    ) -> dict[str, Any]:
        """Like predict_gp but returns GPU tensors instead of Python float lists."""
        torch = self._torch
        if not labeled_indices:
            return {pid: torch.tensor([], device=self._device) for pid in self._probe_ids}

        result_chunks: dict[str, list[Any]] = {pid: [] for pid in self._probe_ids}

        for offsets, entry in self._offset_groups.items():
            offsets_t = entry["offsets_t"]
            if self._offset_map:
                offsets_t = torch.tensor(
                    [self._offset_map[int(o)] for o in offsets_t],
                    dtype=torch.long,
                )
            weights = entry["weights"]
            biases = entry["biases"]
            has_norm = "means" in entry
            norm_mode = entry.get("norm_mode", "standard")
            indices_in_group = entry["indices"]

            for start in range(0, len(labeled_indices), eval_batch_size):
                batch_idx = labeled_indices[start : start + eval_batch_size]
                batch_feat = (
                    feature_tensor[batch_idx][:, offsets_t, :, :]
                    .reshape(len(batch_idx), -1)
                    .to(device=self._device, dtype=torch.float32)
                )
                if norm_mode == "cosine":
                    batch_feat = batch_feat / (batch_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                    logits = batch_feat @ weights.T + biases
                elif has_norm:
                    means = entry["means"]
                    stds = entry["stds"]
                    batch_feat = (batch_feat.unsqueeze(1) - means.unsqueeze(0)) / stds.unsqueeze(0)
                    logits = (batch_feat * weights.unsqueeze(0)).sum(-1) + biases
                else:
                    logits = batch_feat @ weights.T + biases
                probs_all = torch.sigmoid(logits)

                for j, spec_idx in enumerate(indices_in_group):
                    result_chunks[self._probe_ids[spec_idx]].append(probs_all[:, j].detach())

                del batch_feat, logits, probs_all

        return {
            pid: torch.cat(chunks) if chunks else torch.tensor([], device=self._device)
            for pid, chunks in result_chunks.items()
        }


def evaluate_probes_gp_first(
    *,
    root: str | Path,
    feature_name: str,
    labeling_protocol: str,
    probe_specs: Sequence[ProbeEvalSpec],
    eval_gp_ids: Sequence[str],
    device: str | None = None,
    eval_batch_size: int = 4096,
    log_fn: Callable[[str], None] | None = None,
    filter_last_replay_role: str | None = None,
) -> list[ProbeEvalResult]:
    torch = _import_torch()
    device_name = _resolve_device(torch, device)
    root = Path(root)

    labels = load_labels(root, labeling_protocol)

    # Warn if any probe was trained with a different labeling protocol
    for _spec in probe_specs:
        _ckpt_proto = str(_spec.checkpoint.get("labeling_protocol", ""))
        if _ckpt_proto and _ckpt_proto != labeling_protocol:
            _emit(log_fn, f"WARNING: probe {_spec.probe_id} trained with labeling_protocol={_ckpt_proto} "
                  f"but evaluating with {labeling_protocol}")

    # Collect unique layer offsets needed across all probes
    _needed_offsets: set[int] = set()
    for _spec in probe_specs:
        _needed_offsets.update(int(v) for v in _spec.checkpoint["selected_layer_offsets"])
    needed_layer_offsets = sorted(_needed_offsets) if _needed_offsets else None

    # Split linear vs non-linear probes (BatchedProbePredictor only handles linear)
    linear_specs = [s for s in probe_specs if s.checkpoint.get("probe_architecture") == "linear"]
    other_specs = [s for s in probe_specs if s.checkpoint.get("probe_architecture") != "linear"]

    predictor = BatchedProbePredictor(linear_specs, device_name) if linear_specs else None

    # Build offset remapping for partial layer loading
    offset_map: dict[int, int] | None = None
    if needed_layer_offsets is not None:
        offset_map = {li: pos for pos, li in enumerate(needed_layer_offsets)}
    if predictor is not None and offset_map is not None:
        predictor._offset_map = offset_map

    # Accumulate as GPU tensors for fast AUROC computation
    accum_tensors: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in probe_specs:
        accum_tensors[spec.probe_id] = {
            g: {"labels_cpu": [], "probs_gpu": []} for g in EVAL_GROUP_NAMES
        }

    t0 = time.monotonic()
    for gp_idx, gp_id in enumerate(eval_gp_ids, 1):
        try:
            payload = load_feature_payload(root, gp_id, feature_name, layer_indices=needed_layer_offsets, map_location=device_name)
        except FileNotFoundError:
            continue
        feature_tensor = payload["features"]
        # Rebuild offset_map from actually loaded layers (may differ from requested)
        loaded_li = payload["layer_indices"]
        if needed_layer_offsets is not None:
            gp_offset_map = {li: pos for pos, li in enumerate(loaded_li)}
        else:
            gp_offset_map = None
        if predictor is not None and gp_offset_map is not None:
            predictor._offset_map = gp_offset_map

        dp_ids = [str(v) for v in payload["decision_point_ids"]]
        allowed_dp_ids: set[str] | None = None
        if filter_last_replay_role is not None:
            allowed_dp_ids = {
                str(row["decision_point_id"])
                for row in load_decision_points(root, gp_id)
                if _matches_last_replay_role(row, filter_last_replay_role)
            }
        labeled_indices = [
            i for i, dp_id in enumerate(dp_ids)
            if dp_id in labels and (allowed_dp_ids is None or dp_id in allowed_dp_ids)
        ]
        gp_labels = [int(labels[dp_ids[i]]) for i in labeled_indices]
        if not labeled_indices:
            del payload, feature_tensor
            _empty_cuda_cache(torch)
            continue

        # Batched prediction for linear probes — keep on GPU
        if predictor is not None:
            gp_probs = predictor.predict_gp_tensors(feature_tensor, labeled_indices, eval_batch_size)
            for spec in linear_specs:
                gp_groups = [
                    gname for gname in EVAL_GROUP_NAMES
                    if gp_id in spec.eval_groups_def.get(gname, [])
                ]
                if not gp_groups:
                    continue
                probs_t = gp_probs.get(spec.probe_id)
                if probs_t is None:
                    continue
                for gname in gp_groups:
                    accum_tensors[spec.probe_id][gname]["labels_cpu"].extend(gp_labels)
                    accum_tensors[spec.probe_id][gname]["probs_gpu"].append(probs_t)

        # Per-probe prediction for non-linear probes — keep on GPU
        for spec in other_specs:
            gp_groups = [
                gname for gname in EVAL_GROUP_NAMES
                if gp_id in spec.eval_groups_def.get(gname, [])
            ]
            if not gp_groups:
                continue
            probs_t = _predict_checkpoint_on_feature_shard_tensor(
                torch=torch,
                checkpoint=spec.checkpoint,
                feature_tensor=feature_tensor,
                labeled_indices=labeled_indices,
                device_name=device_name,
                eval_batch_size=eval_batch_size,
                offset_map=gp_offset_map,
            )
            for gname in gp_groups:
                accum_tensors[spec.probe_id][gname]["labels_cpu"].extend(gp_labels)
                accum_tensors[spec.probe_id][gname]["probs_gpu"].append(probs_t)

        del payload, feature_tensor
        _empty_cuda_cache(torch)
        if gp_idx % 20 == 0 or gp_idx == len(eval_gp_ids):
            _emit(log_fn, f"  {gp_idx}/{len(eval_gp_ids)} GPs done ({time.monotonic()-t0:.0f}s)")

    # Concatenate and compute AUROC on GPU
    results: list[ProbeEvalResult] = []
    for spec in probe_specs:
        groups: dict[str, dict[str, Any]] = {}
        for gname in EVAL_GROUP_NAMES:
            lbls = accum_tensors[spec.probe_id][gname]["labels_cpu"]
            chunks = accum_tensors[spec.probe_id][gname]["probs_gpu"]
            n = len(lbls)
            if n >= 2 and len(set(lbls)) >= 2:
                all_probs = torch.cat(chunks) if chunks else torch.tensor([], device=device_name)
                auroc = _gpu_auroc(torch, all_probs, torch.tensor(lbls, device=device_name, dtype=torch.float32))
                groups[gname] = {"n": n, "auroc": auroc}
            else:
                groups[gname] = {"n": n, "auroc": None}
            del chunks
        results.append(ProbeEvalResult(probe_id=spec.probe_id, groups=groups))

    del accum_tensors
    _empty_cuda_cache(torch)
    return results


def discover_probe_checkpoints(checkpoint_root: str | Path) -> list[Path]:
    root = Path(checkpoint_root)
    if root.is_file():
        return [root] if root.suffix == ".pt" else []
    candidates: list[Path] = []
    if (root / "models").is_dir():
        candidates.extend(sorted((root / "models").glob("best*.pt")))
    candidates.extend(sorted(root.glob("*/models/best*.pt")))
    seen: set[Path] = set()
    result: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def evaluate_probe_grid_search(
    *,
    root: str | Path,
    checkpoint_root: str | Path,
    output: str | Path,
    dataset_name: str | None = None,
    device: str | None = None,
    eval_batch_size: int = 4096,
    num_workers: int = 1,
    grid_points: Sequence[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be at least 1")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    checkpoint_paths = discover_probe_checkpoints(checkpoint_root)
    if not checkpoint_paths:
        raise FileNotFoundError(f"no v2 probe checkpoints found under {checkpoint_root}")
    checkpoint_rows = [(path, load_probe_checkpoint(path)) for path in checkpoint_paths]
    grouped = _group_checkpoints_by_products(checkpoint_rows, dataset_name=dataset_name)
    _emit(
        log_fn,
        f"Discovered {len(checkpoint_paths)} checkpoint(s) in {len(grouped)} product group(s)",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
        _emit(log_fn, f"Removed existing eval output {output_path}")

    result_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for product_key, product_checkpoints in sorted(grouped.items()):
        resolved_dataset_name, labeling_protocol, feature_name = product_key
        _emit(
            log_fn,
            f"Loading partition={resolved_dataset_name} labels={labeling_protocol} feature={feature_name}",
        )
        partition = load_partition(root, resolved_dataset_name)
        labels = load_labels(root, labeling_protocol)
        eval_grid_points = list(grid_points or partition.get("eval_grid_points") or ())
        worker_count = min(int(num_workers), len(eval_grid_points) or 1)
        _emit(
            log_fn,
            f"Evaluating {len(product_checkpoints)} checkpoints over {len(eval_grid_points)} grid points "
            f"with {worker_count} worker(s)",
        )
        if worker_count == 1:
            for grid_point_id in eval_grid_points:
                rows, skipped_rows = evaluate_grid_point_checkpoints(
                    root=root,
                    grid_point_id=grid_point_id,
                    dataset_name=resolved_dataset_name,
                    labeling_protocol=labeling_protocol,
                    feature_name=feature_name,
                    labels=labels,
                    checkpoint_rows=product_checkpoints,
                    device=device,
                    eval_batch_size=eval_batch_size,
                    log_fn=log_fn,
                )
                result_rows.extend(rows)
                skipped.extend(skipped_rows)
                with output_path.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                _emit(
                    log_fn,
                    f"Finished grid_point={grid_point_id} rows={len(rows)} skipped={len(skipped_rows)}",
                )
            continue

        rows, skipped_rows = _evaluate_product_group_parallel(
            root=root,
            product_key=product_key,
            checkpoint_paths=[path for path, _checkpoint in product_checkpoints],
            eval_grid_points=eval_grid_points,
            device=device,
            eval_batch_size=eval_batch_size,
            num_workers=worker_count,
            output_path=output_path,
            log_fn=log_fn,
        )
        result_rows.extend(rows)
        skipped.extend(skipped_rows)

    summary = _build_eval_summary(
        checkpoint_root=checkpoint_root,
        output_path=output_path,
        result_rows=result_rows,
        skipped=skipped,
        num_workers=num_workers,
    )
    summary_path = _summary_path(output_path)
    write_json(summary_path, summary)
    _emit(
        log_fn,
        f"Wrote eval output rows={len(result_rows)} skipped={len(skipped)} "
        f"output={output_path} summary={summary_path}",
    )
    return summary


def _evaluate_product_group_parallel(
    *,
    root: str | Path,
    product_key: tuple[str, str, str],
    checkpoint_paths: Sequence[Path],
    eval_grid_points: Sequence[str],
    device: str | None,
    eval_batch_size: int,
    num_workers: int,
    output_path: Path,
    log_fn: Callable[[str], None] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_name, labeling_protocol, feature_name = product_key
    chunks = _split_round_robin(list(eval_grid_points), int(num_workers))
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}_workers_",
        dir=str(output_path.parent),
    ) as tmp_dir:
        tmp_path = Path(tmp_dir)
        ctx = mp.get_context("spawn")
        processes = []
        result_shards: list[Path] = []
        skipped_shards: list[Path] = []
        for worker_id, grid_point_chunk in enumerate(chunks):
            if not grid_point_chunk:
                continue
            result_shard = tmp_path / f"worker_{worker_id:02d}.jsonl"
            skipped_shard = tmp_path / f"worker_{worker_id:02d}.skipped.jsonl"
            result_shards.append(result_shard)
            skipped_shards.append(skipped_shard)
            process = ctx.Process(
                target=_worker_evaluate_grid_points,
                kwargs={
                    "worker_id": worker_id,
                    "root": str(root),
                    "grid_points": list(grid_point_chunk),
                    "dataset_name": dataset_name,
                    "labeling_protocol": labeling_protocol,
                    "feature_name": feature_name,
                    "checkpoint_paths": [str(path) for path in checkpoint_paths],
                    "device": device,
                    "eval_batch_size": int(eval_batch_size),
                    "result_output": str(result_shard),
                    "skipped_output": str(skipped_shard),
                },
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
        failed = [process for process in processes if process.exitcode not in (0, None)]
        if failed:
            raise RuntimeError(
                "eval-grid-search worker failure(s): "
                + ", ".join(f"pid={process.pid} exitcode={process.exitcode}" for process in failed)
            )
        result_rows = _load_jsonl_shards(result_shards)
        skipped_rows = _load_jsonl_shards(skipped_shards)

    result_rows = sorted(
        result_rows,
        key=lambda row: (str(row.get("grid_point_id")), str(row.get("checkpoint_path"))),
    )
    with output_path.open("a", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _emit(log_fn, f"Merged {len(result_rows)} worker result rows for product={product_key}")
    return result_rows, skipped_rows


def _worker_evaluate_grid_points(
    *,
    worker_id: int,
    root: str,
    grid_points: Sequence[str],
    dataset_name: str,
    labeling_protocol: str,
    feature_name: str,
    checkpoint_paths: Sequence[str],
    device: str | None,
    eval_batch_size: int,
    result_output: str,
    skipped_output: str,
) -> None:
    labels = load_labels(root, labeling_protocol)
    checkpoint_rows = [(Path(path), load_probe_checkpoint(path)) for path in checkpoint_paths]
    result_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for grid_point_id in grid_points:
        rows, skipped = evaluate_grid_point_checkpoints(
            root=root,
            grid_point_id=grid_point_id,
            dataset_name=dataset_name,
            labeling_protocol=labeling_protocol,
            feature_name=feature_name,
            labels=labels,
            checkpoint_rows=checkpoint_rows,
            device=device,
            eval_batch_size=eval_batch_size,
            log_fn=None,
        )
        result_rows.extend(rows)
        skipped_rows.extend(skipped)
    _write_jsonl(Path(result_output), result_rows)
    _write_jsonl(Path(skipped_output), skipped_rows)


def evaluate_grid_point_checkpoints(
    *,
    root: str | Path,
    grid_point_id: str,
    dataset_name: str,
    labeling_protocol: str,
    feature_name: str,
    labels: Mapping[str, int],
    checkpoint_rows: Sequence[tuple[Path, Mapping[str, Any]]],
    device: str | None,
    eval_batch_size: int,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch = _import_torch()
    device_name = _resolve_device(torch, device)
    _emit(log_fn, f"Loading features once: grid_point_id={grid_point_id} feature_name={feature_name}")

    # Collect unique layer offsets needed across checkpoints
    _needed_offsets: set[int] = set()
    for _, _ckpt in checkpoint_rows:
        _needed_offsets.update(int(v) for v in _ckpt["selected_layer_offsets"])
    needed_layer_offsets = sorted(_needed_offsets) if _needed_offsets else None

    feature_payload = load_feature_payload(root, grid_point_id, feature_name, layer_indices=needed_layer_offsets)
    feature_tensor = feature_payload["features"]

    # Build offset remapping for partial layer loading
    loaded_li = feature_payload["layer_indices"]
    offset_map = {li: pos for pos, li in enumerate(loaded_li)} if needed_layer_offsets is not None else None

    decision_point_ids = [str(value) for value in feature_payload["decision_point_ids"]]
    labeled_indices: list[int] = []
    selected_labels: list[int] = []
    selected_decision_point_ids: list[str] = []
    for row_index, decision_point_id in enumerate(decision_point_ids):
        if decision_point_id not in labels:
            continue
        labeled_indices.append(row_index)
        selected_labels.append(int(labels[decision_point_id]))
        selected_decision_point_ids.append(decision_point_id)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for checkpoint_path, checkpoint in checkpoint_rows:
        try:
            _emit(log_fn, f"Evaluating checkpoint={checkpoint_path} grid_point={grid_point_id}")
            probabilities = _predict_checkpoint_on_feature_shard(
                torch=torch,
                checkpoint=checkpoint,
                feature_tensor=feature_tensor,
                labeled_indices=labeled_indices,
                device_name=device_name,
                eval_batch_size=eval_batch_size,
                offset_map=offset_map,
            )
            metrics = _binary_metrics(
                selected_labels,
                probabilities,
                float(checkpoint.get("threshold", 0.5)),
            )
        except Exception as exc:
            skipped.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "grid_point_id": grid_point_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            _emit(
                log_fn,
                f"Skipped checkpoint={checkpoint_path} grid_point={grid_point_id} "
                f"error_type={type(exc).__name__}: {exc}",
            )
            continue
        rows.append(
            {
                "schema": "ipi_aware.probe_eval.v2",
                "checkpoint_path": str(checkpoint_path),
                "dataset_name": dataset_name,
                "grid_point_id": grid_point_id,
                "feature_name": feature_name,
                "labeling_protocol": labeling_protocol,
                "selected_layer_indices": [int(value) for value in checkpoint["selected_layer_indices"]],
                "selected_layer_offsets": [int(value) for value in checkpoint["selected_layer_offsets"]],
                "probe_architecture": str(checkpoint["probe_architecture"]),
                "probe_architecture_config": dict(checkpoint.get("probe_architecture_config") or {}),
                "example_count": len(selected_labels),
                "metrics": metrics,
            }
        )
    del feature_payload, feature_tensor
    _empty_cuda_cache(torch)
    return rows, skipped


def _predict_checkpoint_on_feature_shard(
    *,
    torch: Any,
    checkpoint: Mapping[str, Any],
    feature_tensor: Any,
    labeled_indices: Sequence[int],
    device_name: str,
    eval_batch_size: int,
    offset_map: dict[int, int] | None = None,
) -> list[float]:
    if not labeled_indices:
        return []
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
    selected_layer_offsets = [int(value) for value in checkpoint["selected_layer_offsets"]]
    if offset_map:
        selected_layer_offsets = [offset_map[o] for o in selected_layer_offsets]
    probabilities: list[float] = []
    with torch.no_grad():
        for start in range(0, len(labeled_indices), eval_batch_size):
            batch_indices = labeled_indices[start : start + eval_batch_size]
            batch_features = (
                feature_tensor[batch_indices][:, selected_layer_offsets, :, :]
                .reshape(len(batch_indices), -1)
                .to(device=device_name, dtype=torch.float32)
            )
            logits = model(batch_features).squeeze(-1)
            batch_probs = torch.sigmoid(logits).detach().cpu().tolist()
            if isinstance(batch_probs, float):
                probabilities.append(float(batch_probs))
            else:
                probabilities.extend(float(value) for value in batch_probs)
            del batch_features, logits
    del model
    _empty_cuda_cache(torch)
    return probabilities


def _group_checkpoints_by_products(
    checkpoint_rows: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    dataset_name: str | None,
) -> dict[tuple[str, str, str], list[tuple[Path, Mapping[str, Any]]]]:
    grouped: dict[tuple[str, str, str], list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    for checkpoint_path, checkpoint in checkpoint_rows:
        resolved_dataset_name = str(dataset_name or checkpoint["dataset_name"])
        key = (
            resolved_dataset_name,
            str(checkpoint["labeling_protocol"]),
            str(checkpoint["feature_name"]),
        )
        grouped[key].append((checkpoint_path, checkpoint))
    return dict(grouped)


def _build_eval_summary(
    *,
    checkpoint_root: str | Path,
    output_path: Path,
    result_rows: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    num_workers: int,
) -> dict[str, Any]:
    by_checkpoint: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_grid_point: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in result_rows:
        by_checkpoint[str(row["checkpoint_path"])].append(row)
        by_grid_point[str(row["grid_point_id"])].append(row)
    checkpoint_scores = {
        checkpoint_path: _mean_auroc(rows)
        for checkpoint_path, rows in by_checkpoint.items()
    }
    best_checkpoint = None
    if checkpoint_scores:
        best_checkpoint = max(
            checkpoint_scores,
            key=lambda checkpoint_path: (
                _score_value(checkpoint_scores[checkpoint_path]),
                checkpoint_path,
            ),
        )
    best_by_grid_point: dict[str, dict[str, Any] | None] = {}
    for grid_point_id, rows in sorted(by_grid_point.items()):
        best_row = max(
            rows,
            key=lambda row: (
                _score_value((row.get("metrics") or {}).get("auroc")),
                str(row["checkpoint_path"]),
            ),
        )
        best_by_grid_point[grid_point_id] = {
            "checkpoint_path": str(best_row["checkpoint_path"]),
            "auroc": (best_row.get("metrics") or {}).get("auroc"),
        }
    return {
        "schema": "ipi_aware.probe_eval_summary.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_root": str(checkpoint_root),
        "output_path": str(output_path),
        "num_workers": int(num_workers),
        "result_count": len(result_rows),
        "skipped_count": len(skipped),
        "skipped": [dict(row) for row in skipped],
        "checkpoint_mean_auroc": checkpoint_scores,
        "best_checkpoint_path": best_checkpoint,
        "best_by_grid_point": best_by_grid_point,
    }


def _mean_auroc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = [
        float((row.get("metrics") or {})["auroc"])
        for row in rows
        if (row.get("metrics") or {}).get("auroc") is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _summary_path(output_path: Path) -> Path:
    if output_path.suffix == ".jsonl":
        return output_path.with_suffix(".summary.json")
    return output_path.with_name(output_path.name + ".summary.json")


def _score_value(value: Any) -> float:
    return -1.0 if value is None else float(value)


def _split_round_robin(values: Sequence[str], num_workers: int) -> list[list[str]]:
    chunks: list[list[str]] = [[] for _ in range(max(1, int(num_workers)))]
    for index, value in enumerate(values):
        chunks[index % len(chunks)].append(str(value))
    return chunks


def _load_jsonl_shards(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _gpu_auroc(torch: Any, probs: Any, labels: Any) -> float | None:
    """Compute AUROC with sklearn, accepting tensors from the batched GPU path."""
    del torch
    from sklearn.metrics import roc_auc_score

    if probs.numel() < 2:
        return None
    labels_cpu = [int(label) for label in labels.detach().cpu().tolist()]
    if len(set(labels_cpu)) < 2:
        return None
    return float(roc_auc_score(labels_cpu, probs.detach().cpu().tolist()))


def _predict_checkpoint_on_feature_shard_tensor(
    *,
    torch: Any,
    checkpoint: Mapping[str, Any],
    feature_tensor: Any,
    labeled_indices: Sequence[int],
    device_name: str,
    eval_batch_size: int,
    offset_map: dict[int, int] | None = None,
) -> Any:
    """Like _predict_checkpoint_on_feature_shard but returns a GPU tensor instead of a list."""
    if not labeled_indices:
        return torch.tensor([], device=device_name)
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
    selected_layer_offsets = [int(value) for value in checkpoint["selected_layer_offsets"]]
    if offset_map:
        selected_layer_offsets = [offset_map[o] for o in selected_layer_offsets]
    chunks: list[Any] = []
    with torch.no_grad():
        for start in range(0, len(labeled_indices), eval_batch_size):
            batch_indices = labeled_indices[start : start + eval_batch_size]
            batch_features = (
                feature_tensor[batch_indices][:, selected_layer_offsets, :, :]
                .reshape(len(batch_indices), -1)
                .to(device=device_name, dtype=torch.float32)
            )
            logits = model(batch_features).squeeze(-1)
            chunks.append(torch.sigmoid(logits))
            del batch_features, logits
    del model
    _empty_cuda_cache(torch)
    return torch.cat(chunks) if chunks else torch.tensor([], device=device_name)


def _resolve_device(torch: Any, requested_device: str | None) -> str:
    if requested_device is not None:
        return requested_device
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda"
    return "cpu"


def _empty_cuda_cache(torch: Any) -> None:
    cuda = getattr(torch, "cuda", None)
    empty_cache = getattr(cuda, "empty_cache", None) if cuda is not None else None
    if callable(empty_cache):
        empty_cache()


def _emit(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("v2 grid-search evaluation requires torch in the current environment") from exc
    return torch
