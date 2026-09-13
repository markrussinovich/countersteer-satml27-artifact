from .feature_io import (
    build_feature_payload,
    examples_to_feature_payload,
    feature_payload_to_id_map,
    load_feature_payload,
    write_feature_payload,
)

_FEATURE_EXPORTS = {
    "ProbeExample",
    "ProbePrompt",
    "build_probe_examples_transformers_hook",
    "build_probe_examples_vllm_direct",
    "build_probe_prompt",
    "build_qwen3_probe_prompt",
    "calibrate_qwen35_assistant_prefill_position",
    "describe_probe_prompt",
    "iter_grid_point_dirs",
    "load_decision_points_from_trace_artifact",
    "write_probe_dataset",
}
_HIDDEN_STATE_EXPORTS = {
    "HiddenStateCapture",
    "UnsupportedResidualStreamCaptureError",
    "capture_prefill_hidden_states",
    "capture_prefill_hidden_states_batch",
    "capture_prefill_residual_stream",
    "capture_prefill_residual_stream_batch",
}
_DIRECT_CAPTURE_EXPORTS = {"DirectCaptureExtractor"}

__all__ = [
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
]


def __getattr__(name: str):
    if name in _FEATURE_EXPORTS:
        from . import features

        value = getattr(features, name)
    elif name in _HIDDEN_STATE_EXPORTS:
        from . import hidden_states

        value = getattr(hidden_states, name)
    elif name in _DIRECT_CAPTURE_EXPORTS:
        from . import direct_capture

        value = getattr(direct_capture, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
