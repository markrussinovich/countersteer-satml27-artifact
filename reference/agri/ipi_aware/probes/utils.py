from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# ---------- schema constants ----------

GRID_POINT_SCHEMA = "ipi_aware.grid_point.v1"
FEATURE_SCHEMA = "ipi_aware.features.v1"
PARTITION_SCHEMA = "ipi_aware.partition.v1"
CHECKPOINT_SCHEMA = "ipi_aware.probe_checkpoint.v2"
EVAL_GROUPS_SCHEMA = "ipi_aware.eval_groups.v1"
FEATURE_SPLIT_SCHEMA = "ipi_aware.features.split.v1"
FEATURE_LAYER_SCHEMA = "ipi_aware.features.layer.v1"

DEFAULT_FEATURE_NAME = "default"
DEFAULT_LABELING_PROTOCOL = "risk_faced"


# ---------- filesystem layout ----------

@dataclass(frozen=True, slots=True)
class ProductPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProductPaths":
        return cls(Path(root))

    @property
    def grid_points_dir(self) -> Path:
        return self.root / "grid_points"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    def grid_point_dir(self, grid_point_id: str) -> Path:
        return self.grid_points_dir / str(grid_point_id)

    def grid_point_manifest(self, grid_point_id: str) -> Path:
        return self.grid_point_dir(grid_point_id) / "manifest.json"

    def traces(self, grid_point_id: str) -> Path:
        return self.grid_point_dir(grid_point_id) / "traces.jsonl"

    def decision_points(self, grid_point_id: str) -> Path:
        return self.grid_point_dir(grid_point_id) / "decision_points.jsonl"

    def labels(self, labeling_protocol: str = DEFAULT_LABELING_PROTOCOL) -> Path:
        return self.labels_dir / str(labeling_protocol) / "labels.jsonl"

    def feature_shard(
        self,
        grid_point_id: str,
        feature_name: str = DEFAULT_FEATURE_NAME,
    ) -> Path:
        return self.features_dir / str(feature_name) / f"{grid_point_id}.pt"

    def feature_layer_dir(self, grid_point_id: str, feature_name: str = DEFAULT_FEATURE_NAME) -> Path:
        return self.features_dir / str(feature_name) / str(grid_point_id)

    def feature_layer_manifest(self, grid_point_id: str, feature_name: str = DEFAULT_FEATURE_NAME) -> Path:
        return self.feature_layer_dir(grid_point_id, feature_name) / "manifest.json"

    def feature_layer_shard(self, grid_point_id: str, layer_index: int, feature_name: str = DEFAULT_FEATURE_NAME) -> Path:
        return self.feature_layer_dir(grid_point_id, feature_name) / f"L{layer_index}.pt"

    def is_feature_split(self, grid_point_id: str, feature_name: str = DEFAULT_FEATURE_NAME) -> bool:
        return self.feature_layer_manifest(grid_point_id, feature_name).is_file()

    def partition(self, dataset_name: str) -> Path:
        return self.datasets_dir / str(dataset_name) / "partition.json"


def iter_grid_point_ids(root: str | Path) -> tuple[str, ...]:
    paths = ProductPaths.from_root(root)
    if not paths.grid_points_dir.is_dir():
        return ()
    return tuple(
        path.name
        for path in sorted(paths.grid_points_dir.iterdir())
        if path.is_dir() and (path / "decision_points.jsonl").exists()
    )


# ---------- model config helpers ----------

def default_layer_ids(model_name_or_path: str) -> tuple[int, ...]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    hidden_layer_count = int(
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
        or 0
    )
    if hidden_layer_count <= 0 and hasattr(config, "get_text_config"):
        tc = config.get_text_config()
        hidden_layer_count = int(
            getattr(tc, "num_hidden_layers", None)
            or getattr(tc, "n_layer", None)
            or 0
        )
    if hidden_layer_count <= 0:
        raise ValueError(f"unable to infer num_hidden_layers for {model_name_or_path}")
    return tuple(range(hidden_layer_count))


# ---------- hidden-state capture helpers ----------

def normalize_positions(positions: Sequence[int], token_count: int) -> tuple[int, ...]:
    normalized: list[int] = []
    for position in positions:
        resolved = position if position >= 0 else token_count + position
        if resolved < 0 or resolved >= token_count:
            raise IndexError(f"selected position {position} is out of range for {token_count} tokens")
        normalized.append(int(resolved))
    return tuple(normalized)


def materialize_hidden_state(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if getattr(value, "shape", None) is not None:
        return value
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, list):
        return [materialize_hidden_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_hidden_state(item) for item in value)
    return value


def infer_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value),) + infer_shape(value[0])
    return ()
