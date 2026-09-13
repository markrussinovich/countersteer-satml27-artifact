"""Probe pipeline built around explicit filesystem products."""

from .utils import DEFAULT_FEATURE_NAME, ProductPaths

_FEATURIZATION_EXPORTS = {
    "DirectCaptureExtractor",
    "HiddenStateCapture",
    "ProbeExample",
    "ProbePrompt",
    "UnsupportedResidualStreamCaptureError",
    "build_feature_payload",
    "build_probe_examples_transformers_hook",
    "build_probe_examples_vllm_direct",
    "build_probe_prompt",
    "build_qwen3_probe_prompt",
    "calibrate_qwen35_assistant_prefill_position",
    "capture_prefill_hidden_states",
    "capture_prefill_hidden_states_batch",
    "capture_prefill_residual_stream",
    "capture_prefill_residual_stream_batch",
    "describe_probe_prompt",
    "examples_to_feature_payload",
    "feature_payload_to_id_map",
    "iter_grid_point_dirs",
    "load_decision_points_from_trace_artifact",
    "load_feature_payload",
    "write_feature_payload",
    "write_probe_dataset",
}
_COLLECTION_MODULE_EXPORTS = {"dataset", "labels"}

__all__ = [
    "DEFAULT_FEATURE_NAME",
    "DirectCaptureExtractor",
    "HiddenStateCapture",
    "ProductPaths",
    "ProbeExample",
    "ProbePrompt",
    "UnsupportedResidualStreamCaptureError",
    "build_feature_payload",
    "build_probe_examples_transformers_hook",
    "build_probe_examples_vllm_direct",
    "build_probe_prompt",
    "build_qwen3_probe_prompt",
    "calibrate_qwen35_assistant_prefill_position",
    "capture_prefill_hidden_states",
    "capture_prefill_hidden_states_batch",
    "capture_prefill_residual_stream",
    "capture_prefill_residual_stream_batch",
    "describe_probe_prompt",
    "dataset",
    "examples_to_feature_payload",
    "feature_payload_to_id_map",
    "iter_grid_point_dirs",
    "labels",
    "load_decision_points_from_trace_artifact",
    "load_feature_payload",
    "write_feature_payload",
    "write_probe_dataset",
]


def __getattr__(name: str):
    if name in _FEATURIZATION_EXPORTS:
        from . import featurization

        value = getattr(featurization, name)
        globals()[name] = value
        return value
    if name in _COLLECTION_MODULE_EXPORTS:
        from .collection import dataset, labels

        value = {"dataset": dataset, "labels": labels}[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
