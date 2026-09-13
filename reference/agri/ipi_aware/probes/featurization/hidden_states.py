from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence


class UnsupportedResidualStreamCaptureError(ValueError):
    """Raised when hook-based residual-stream capture cannot resolve decoder blocks."""


@dataclass(frozen=True, slots=True)
class HiddenStateCapture:
    input_token_ids: tuple[int, ...]
    selected_positions: tuple[int, ...]
    layer_shapes: tuple[tuple[int, ...], ...]
    hidden_states: tuple[Any, ...]
    layer_ids: tuple[int, ...] | None = None


def capture_prefill_hidden_states(
    prompt: Any,
    *,
    model: Any | None = None,
    model_name_or_path: str | None = None,
    selected_positions: Sequence[int] | None = None,
) -> HiddenStateCapture:
    input_token_ids = tuple(int(token) for token in prompt.token_ids[: prompt.decode_start_index])
    if not input_token_ids:
        raise ValueError("prompt token_ids must not be empty")

    positions = _normalize_positions(
        selected_positions or [len(input_token_ids) - 1],
        len(input_token_ids),
    )
    model = model or _load_model(model_name_or_path)
    model_inputs = _build_model_inputs(input_token_ids, use_torch=True)
    model_inputs = _move_inputs_to_model_device({"input_ids": model_inputs}, model)["input_ids"]
    outputs = _forward_hidden_states(model, input_ids=model_inputs)
    hidden_states = tuple(getattr(outputs, "hidden_states", ()) or ())
    selected_hidden_states = tuple(
        _materialize_hidden_state(_select_positions(layer, positions))
        for layer in hidden_states
    )
    layer_shapes = tuple(_infer_shape(layer) for layer in selected_hidden_states)
    return HiddenStateCapture(
        input_token_ids=input_token_ids,
        selected_positions=positions,
        layer_shapes=layer_shapes,
        hidden_states=selected_hidden_states,
    )


def capture_prefill_hidden_states_batch(
    prompts: Sequence[Any],
    *,
    model: Any | None = None,
    model_name_or_path: str | None = None,
    selected_positions: Sequence[int] | None = None,
    tokenizer: Any | None = None,
) -> tuple[HiddenStateCapture, ...]:
    if not prompts:
        return ()

    prompt_token_ids = [
        tuple(int(token) for token in prompt.token_ids[: prompt.decode_start_index])
        for prompt in prompts
    ]
    for input_token_ids in prompt_token_ids:
        if not input_token_ids:
            raise ValueError("prompt token_ids must not be empty")

    positions_by_prompt = [
        _normalize_positions(selected_positions or [len(input_token_ids) - 1], len(input_token_ids))
        for input_token_ids in prompt_token_ids
    ]
    model = model or _load_model(model_name_or_path)
    model_inputs = _build_batched_model_inputs(
        prompt_token_ids,
        tokenizer=tokenizer,
        use_torch=True,
    )
    model_inputs = _move_inputs_to_model_device(model_inputs, model)
    outputs = _forward_hidden_states(model, **model_inputs)
    hidden_states = tuple(getattr(outputs, "hidden_states", ()) or ())

    captures: list[HiddenStateCapture] = []
    for batch_index, (input_token_ids, positions) in enumerate(zip(prompt_token_ids, positions_by_prompt, strict=True)):
        selected_hidden_states = tuple(
            _materialize_hidden_state(_select_positions_for_batch(layer, batch_index, positions))
            for layer in hidden_states
        )
        layer_shapes = tuple(_infer_shape(layer) for layer in selected_hidden_states)
        captures.append(
            HiddenStateCapture(
                input_token_ids=input_token_ids,
                selected_positions=positions,
                layer_shapes=layer_shapes,
                hidden_states=selected_hidden_states,
            )
        )
    return tuple(captures)


def capture_prefill_residual_stream(
    prompt: Any,
    *,
    model: Any | None = None,
    model_name_or_path: str | None = None,
    selected_positions: Sequence[int] | None = None,
) -> HiddenStateCapture:
    captures = capture_prefill_residual_stream_batch(
        (prompt,),
        model=model,
        model_name_or_path=model_name_or_path,
        selected_positions=selected_positions,
        tokenizer=None,
    )
    return captures[0]


def capture_prefill_residual_stream_batch(
    prompts: Sequence[Any],
    *,
    model: Any | None = None,
    model_name_or_path: str | None = None,
    selected_positions: Sequence[int] | None = None,
    tokenizer: Any | None = None,
) -> tuple[HiddenStateCapture, ...]:
    if not prompts:
        return ()

    prompt_token_ids = [
        tuple(int(token) for token in prompt.token_ids[: prompt.decode_start_index])
        for prompt in prompts
    ]
    for input_token_ids in prompt_token_ids:
        if not input_token_ids:
            raise ValueError("prompt token_ids must not be empty")

    positions_by_prompt = [
        _normalize_positions(selected_positions or [len(input_token_ids) - 1], len(input_token_ids))
        for input_token_ids in prompt_token_ids
    ]
    model = model or _load_model(model_name_or_path)
    backbone = _resolve_hidden_state_backbone(model)
    layers = _resolve_residual_stream_layers(backbone)
    model_inputs = _build_batched_model_inputs(
        prompt_token_ids,
        tokenizer=tokenizer,
        use_torch=True,
    )
    model_inputs = _move_inputs_to_model_device(model_inputs, model)
    captured_layers: list[tuple[Any, ...] | None] = [None] * len(layers)
    hook_handles = [
        layer.register_forward_hook(
            _build_residual_stream_hook(
                layer_index=layer_index,
                positions_by_prompt=positions_by_prompt,
                captured_layers=captured_layers,
            )
        )
        for layer_index, layer in enumerate(layers)
    ]
    try:
        with _inference_context():
            backbone(
                **model_inputs,
                return_dict=True,
            )
    finally:
        for handle in hook_handles:
            handle.remove()

    if any(layer_capture is None for layer_capture in captured_layers):
        raise RuntimeError("residual-stream hooks did not fire for all decoder blocks")

    captures: list[HiddenStateCapture] = []
    for batch_index, (input_token_ids, positions) in enumerate(zip(prompt_token_ids, positions_by_prompt, strict=True)):
        selected_hidden_states = tuple(
            layer_capture[batch_index]
            for layer_capture in captured_layers
            if layer_capture is not None
        )
        layer_shapes = tuple(_infer_shape(layer) for layer in selected_hidden_states)
        captures.append(
            HiddenStateCapture(
                input_token_ids=input_token_ids,
                selected_positions=positions,
                layer_shapes=layer_shapes,
                hidden_states=selected_hidden_states,
            )
        )
    return tuple(captures)


def _load_model(model_name_or_path: str | None) -> Any:
    if not model_name_or_path:
        raise ValueError("model_name_or_path is required when model is not provided")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if hasattr(model, "eval"):
        model.eval()
    return model


def _build_model_inputs(input_token_ids: tuple[int, ...], *, use_torch: bool) -> Any:
    if use_torch:
        try:
            import torch
        except ImportError:
            return [list(input_token_ids)]
        return torch.tensor([list(input_token_ids)], dtype=torch.long)
    return [list(input_token_ids)]


def _inference_context() -> Any:
    try:
        import torch
    except ImportError:
        return nullcontext()
    inference_mode = getattr(torch, "inference_mode", None)
    if callable(inference_mode):
        return inference_mode()
    no_grad = getattr(torch, "no_grad", None)
    if callable(no_grad):
        return no_grad()
    return nullcontext()


def _forward_hidden_states(model: Any, **model_inputs: Any) -> Any:
    backbone = _resolve_hidden_state_backbone(model)
    with _inference_context():
        return backbone(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )


def _resolve_hidden_state_backbone(model: Any) -> Any:
    base_model_prefix = getattr(model, "base_model_prefix", None)
    if isinstance(base_model_prefix, str) and base_model_prefix:
        backbone = getattr(model, base_model_prefix, None)
        if backbone is not None and backbone is not model:
            return backbone
    backbone = getattr(model, "model", None)
    if backbone is not None and backbone is not model:
        return backbone
    return model


def _resolve_residual_stream_layers(backbone: Any) -> tuple[Any, ...]:
    layers = getattr(backbone, "layers", None)
    if layers is None:
        # Some models nest layers deeper (e.g. Gemma4: model.model.language_model.layers)
        for child_name in ("language_model", "text_model", "decoder"):
            child = getattr(backbone, child_name, None)
            if child is not None:
                layers = getattr(child, "layers", None)
                if layers is not None:
                    break
    if layers is None:
        raise UnsupportedResidualStreamCaptureError(
            "backbone does not expose decoder blocks via .layers"
        )
    try:
        resolved_layers = tuple(layers)
    except TypeError as exc:  # pragma: no cover - defensive shape check
        raise UnsupportedResidualStreamCaptureError(
            "backbone .layers is not an iterable decoder block sequence"
        ) from exc
    if not resolved_layers:
        raise UnsupportedResidualStreamCaptureError(
            "backbone .layers is empty and cannot be used for residual-stream capture"
        )
    if not all(hasattr(layer, "register_forward_hook") for layer in resolved_layers):
        raise UnsupportedResidualStreamCaptureError(
            "backbone .layers contains objects that do not support forward hooks"
        )
    return resolved_layers


def _build_residual_stream_hook(
    *,
    layer_index: int,
    positions_by_prompt: Sequence[tuple[int, ...]],
    captured_layers: list[tuple[Any, ...] | None],
):
    def _hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden_state = _normalize_hook_output(output)
        captured_layers[layer_index] = tuple(
            _materialize_hidden_state(_select_positions_for_batch(hidden_state, batch_index, positions))
            for batch_index, positions in enumerate(positions_by_prompt)
        )

    return _hook


def _normalize_hook_output(output: Any) -> Any:
    if isinstance(output, tuple):
        if not output:
            raise RuntimeError("decoder block hook returned an empty tuple")
        return output[0]
    return output


def _build_batched_model_inputs(
    prompt_token_ids: Sequence[tuple[int, ...]],
    *,
    tokenizer: Any | None,
    use_torch: bool,
) -> dict[str, Any]:
    pad_token_id = _resolve_pad_token_id(tokenizer)
    max_length = max(len(token_ids) for token_ids in prompt_token_ids)
    padded_input_ids = [
        list(token_ids) + [pad_token_id] * (max_length - len(token_ids))
        for token_ids in prompt_token_ids
    ]
    attention_mask = [
        [1] * len(token_ids) + [0] * (max_length - len(token_ids))
        for token_ids in prompt_token_ids
    ]
    if use_torch:
        try:
            import torch
        except ImportError:
            return {"input_ids": padded_input_ids, "attention_mask": attention_mask}
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    return {"input_ids": padded_input_ids, "attention_mask": attention_mask}


def _resolve_pad_token_id(tokenizer: Any | None) -> int:
    if tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)
    return 0


def _move_inputs_to_model_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return inputs
    device = _resolve_model_device(model)
    if device is None:
        return inputs
    return {
        key: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
        for key, tensor in inputs.items()
    }


def _resolve_model_device(model: Any) -> Any:
    try:
        import torch
    except ImportError:
        return None
    # device_map="auto" places parameters on potentially multiple devices;
    # use the device of the first parameter (typically the embedding layer).
    for param in model.parameters():
        if hasattr(param, "device"):
            return param.device
    return None


def _materialize_hidden_state(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if getattr(value, "shape", None) is not None:
        return value
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, list):
        return [_materialize_hidden_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_hidden_state(item) for item in value)
    return value


def _normalize_positions(positions: Sequence[int], token_count: int) -> tuple[int, ...]:
    normalized: list[int] = []
    for position in positions:
        resolved = position if position >= 0 else token_count + position
        if resolved < 0 or resolved >= token_count:
            raise IndexError(f"selected position {position} is out of range for {token_count} tokens")
        normalized.append(int(resolved))
    return tuple(normalized)


def _select_positions(layer: Any, positions: tuple[int, ...]) -> Any:
    try:
        return layer[:, positions, :]
    except Exception:
        if isinstance(layer, list):
            return [[layer[0][position] for position in positions]]
        if isinstance(layer, tuple):
            return tuple(_select_positions(item, positions) for item in layer)
        raise


def _select_positions_for_batch(layer: Any, batch_index: int, positions: tuple[int, ...]) -> Any:
    try:
        return layer[batch_index : batch_index + 1, positions, :]
    except Exception:
        if isinstance(layer, list):
            return [[layer[batch_index][position] for position in positions]]
        if isinstance(layer, tuple):
            return tuple(_select_positions_for_batch(item, batch_index, positions) for item in layer)
        raise


def _infer_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value),) + _infer_shape(value[0])
    return ()
