"""IPI-Aware hidden-state capture — dumb executor layer.

The model.forward() is a pure executor: it receives capture positions
(set by GPUModelRunner) and gathers hidden states at those positions
for the configured layers. No position resolution, no state tracking,
no file I/O.

Model attributes (set by GPUModelRunner):
    _ipi_aware_capture_layers    - frozenset[int], which layer IDs to capture
    _ipi_aware_capture_positions - torch.Tensor or None, absolute batch token indices
    _ipi_aware_pass_started      - bool, set True when positions are assigned, consumed
                              on first matching layer to clear stale accumulators
    _ipi_aware_layer_outputs     - list[Tensor], accumulated [num_targets, H] per layer
    _ipi_aware_layer_ids         - list[int], global layer IDs captured this forward
    _ipi_aware_captured_layer_ids - list[int] | None, persisted after each forward
    _ipi_aware_forward_output    - Tensor or None, stacked [targets, layers, H] result

Env vars (read once at init):
    IPI_AWARE_CAPTURE_LAYERS     - comma-separated layer indices (e.g. "0,1,...,63")
"""
from __future__ import annotations

import os
import json


def _ipi_aware_debug_event(payload) -> None:
    if not os.environ.get("IPI_AWARE_DEBUG_CAPTURE"):
        return
    try:
        import torch
        if torch._dynamo.is_compiling():
            return
    except Exception:
        pass
    try:
        with open("/tmp/ipi-aware_debug_capture.jsonl", "a") as _ipi_aware_dbg_f:
            _ipi_aware_dbg_f.write(json.dumps(payload) + "\n")
    except OSError:
        return


def ipi_aware_init_capture(model) -> None:
    """Called once in Model.__init__(). Reads env vars, stores capture layers."""
    raw = os.environ.get("IPI_AWARE_CAPTURE_LAYERS", "")
    if raw:
        model._ipi_aware_capture_layers = frozenset(int(x) for x in raw.split(","))
    else:
        model._ipi_aware_capture_layers = frozenset()
    model._ipi_aware_capture_positions = None
    model._ipi_aware_pass_started = False
    model._ipi_aware_layer_outputs = []
    model._ipi_aware_layer_ids = []
    model._ipi_aware_captured_layer_ids = None
    model._ipi_aware_forward_output = None


def ipi_aware_capture_layer(model, layer_id: int, hidden_states, residual) -> None:
    """Gather target positions from one layer's hidden states.

    Called per layer inside model.forward(). Only does work when
    model._ipi_aware_capture_positions is not None (set by GPUModelRunner).
    GPUModelRunner sets _ipi_aware_pass_started = True alongside positions;
    on the first matching layer we consume that flag and clear any
    stale accumulators from an interrupted previous pass.
    """
    if model._ipi_aware_capture_positions is None:
        return
    if layer_id not in model._ipi_aware_capture_layers:
        return
    if model._ipi_aware_pass_started:
        model._ipi_aware_layer_outputs = []
        model._ipi_aware_layer_ids = []
        model._ipi_aware_pass_started = False
    pos = model._ipi_aware_capture_positions
    _ipi_aware_debug_event({
        "event": "capture_layer_enter",
        "model_class": type(model).__name__,
        "layer_id": int(layer_id),
        "positions_shape": list(pos.shape),
        "positions_preview": [int(x) for x in pos[: min(8, pos.numel())].tolist()],
        "hidden_states_shape": list(hidden_states.shape),
        "residual_present": residual is not None,
    })
    try:
        hs = hidden_states[pos]
    except Exception as exc:
        _ipi_aware_debug_event({
            "event": "capture_layer_error",
            "model_class": type(model).__name__,
            "layer_id": int(layer_id),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "positions_shape": list(pos.shape),
            "hidden_states_shape": list(hidden_states.shape),
        })
        raise
    if residual is not None:
        hs = hs + residual[pos]
    model._ipi_aware_layer_outputs.append(hs)
    model._ipi_aware_layer_ids.append(layer_id)
    _ipi_aware_debug_event({
        "event": "capture_layer_ok",
        "model_class": type(model).__name__,
        "layer_id": int(layer_id),
        "captured_shape": list(hs.shape),
    })


def ipi_aware_end_forward(model) -> None:
    """Stack gathered layer outputs into the forward result tensor.

    Called after the layer loop in model.forward().
    """
    if model._ipi_aware_pass_started:
        model._ipi_aware_layer_outputs = []
        model._ipi_aware_layer_ids = []
        model._ipi_aware_pass_started = False
    if not model._ipi_aware_layer_outputs:
        _ipi_aware_debug_event({
            "event": "end_forward_empty",
            "model_class": type(model).__name__,
            "positions_present": model._ipi_aware_capture_positions is not None,
            "layer_ids": list(model._ipi_aware_layer_ids),
        })
        model._ipi_aware_forward_output = None
        return
    stacked = __import__("torch").stack(model._ipi_aware_layer_outputs, dim=1)
    model._ipi_aware_forward_output = stacked
    model._ipi_aware_captured_layer_ids = list(model._ipi_aware_layer_ids)
    _ipi_aware_debug_event({
        "event": "end_forward_output",
        "model_class": type(model).__name__,
        "shape": list(stacked.shape),
        "layer_ids": list(model._ipi_aware_layer_ids),
    })
    model._ipi_aware_layer_outputs = []
    model._ipi_aware_layer_ids = []
