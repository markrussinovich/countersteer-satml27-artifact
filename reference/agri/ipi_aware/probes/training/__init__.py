"""Training-stage modules: partition construction, probe training, and evaluation."""

from .training import (
    FEATURE_COMPOSITIONS,
    PROBE_ARCHITECTURES,
    V2LabeledDataset,
    V2Split,
    build_probe_checkpoint_metadata,
    build_probe_model,
    evaluate_checkpoint,
    load_labeled_dataset,
    normalize_feature_composition,
    normalize_probe_architecture,
    resolve_feature_candidates,
    train_probe,
    train_probe_dataset,
)
from .evaluation import (
    EVAL_GROUP_NAMES,
    ProbeEvalResult,
    ProbeEvalSpec,
    evaluate_grid_point_checkpoints,
    evaluate_probe_grid_search,
)
from .partition import build_partition, write_partition

__all__ = [
    "EVAL_GROUP_NAMES",
    "FEATURE_COMPOSITIONS",
    "PROBE_ARCHITECTURES",
    "ProbeEvalResult",
    "ProbeEvalSpec",
    "V2LabeledDataset",
    "V2Split",
    "build_partition",
    "build_probe_checkpoint_metadata",
    "build_probe_model",
    "evaluate_checkpoint",
    "evaluate_grid_point_checkpoints",
    "evaluate_probe_grid_search",
    "load_labeled_dataset",
    "normalize_feature_composition",
    "normalize_probe_architecture",
    "resolve_feature_candidates",
    "train_probe",
    "train_probe_dataset",
    "write_partition",
]
