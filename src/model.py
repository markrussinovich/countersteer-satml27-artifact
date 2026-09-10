"""Model introspection: loading, the decoder-block container, and the two hook sites. The PROBE captures the pre-MLP residual; STEERING writes the block output. They differ on purpose -- see CLAUDE.md."""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import *  # noqa: F401,F403


# ════════════════════════════════════════════════════════════ model introspection
def _repair_legacy_rope_config(model_id: str):
    """Config for checkpoints whose rope metadata predates transformers v5 validation.

    Phi-3-*-128k ships `original_max_position_embeddings` at the TOP level of config.json
    and a longrope/`su` `rope_scaling` WITHOUT it; transformers 5.x validates rope
    parameters at construction and raises KeyError before any kwarg override can apply.
    The repair moves the top-level value inside the rope dict -- metadata relocation only,
    no numeric change (verified: the factors and the value are the checkpoint's own).
    Returns None when the raw config does not have this exact shape.
    """
    from transformers import CONFIG_MAPPING, PretrainedConfig
    d, _ = PretrainedConfig.get_config_dict(model_id)
    rope = d.get("rope_scaling") or d.get("rope_parameters")
    if not (isinstance(rope, dict)
            and "original_max_position_embeddings" not in rope
            and "original_max_position_embeddings" in d):
        return None
    rope_key = "rope_scaling" if d.get("rope_scaling") else "rope_parameters"
    rope = dict(rope)
    rope["original_max_position_embeddings"] = d["original_max_position_embeddings"]
    d = dict(d)
    d[rope_key] = rope
    print(f"[model] legacy rope config repaired for {model_id}: "
          f"original_max_position_embeddings={rope['original_max_position_embeddings']} "
          f"moved inside rope parameters (metadata relocation, no numeric change)",
          flush=True)
    return CONFIG_MAPPING[d["model_type"]].from_dict(d)


def load_model_and_tok(model_id: str, device: str, attn_impl: str | None = None):
    """attn_impl: None/"" keeps the transformers default resolution (the certified
    serving path -- eager wherever sdpa/flash are unsupported, e.g. gpt-oss in
    transformers 5.14.1). A non-empty value (e.g. "flex_attention") is passed through as
    attn_implementation and CHANGES THE SERVING PATH: numbers produced under it carry a
    per-path label (the 23ae.10 rule -- equivalence is measured, never reasoned).
    Added 2026-09-07 because gpt-oss eager prefill materializes 3 simultaneous
    (heads x s^2) tensors -- ~96 GiB transient at s~16k (IPI Arena) -- which no 80 GB
    device-map arrangement can host; flex attention is blockwise and does not."""
    cfg = None
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except KeyError:
        # transformers v5 rejects Phi-3's pre-v5 rope metadata while resolving the
        # tokenizer class through AutoConfig; repair and retry with the fixed config.
        cfg = _repair_legacy_rope_config(model_id)
        if cfg is None:
            raise
        tok = AutoTokenizer.from_pretrained(model_id, config=cfg)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = {"config": cfg} if cfg is not None else {}
    if attn_impl:
        kw["attn_implementation"] = attn_impl
    # device == "auto" shards the model over every VISIBLE GPU (device_map="auto") --
    # required for the 80B/106B MoE bring-ups that exceed one card. Callers that build
    # tensors must use model.device (the embedding's device), never the "auto" string;
    # src/cli.py normalises args.device after load, and the Steer hook moves each
    # direction to its layer's own device.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device, **kw).eval()
    return model, tok


def layer_container(model):
    """The nn.ModuleList of decoder blocks, whatever the architecture calls it."""
    for path in ("model.layers", "model.model.layers",
                 # multimodal wrappers (Gemma4ForConditionalGeneration et al.): the text
                 # decoder sits under language_model beside the vision tower, whose own
                 # encoder.layers ModuleList must never match -- hence these two BEFORE
                 # any generic fallback, and no bare "layers" pattern anywhere
                 "model.language_model.layers", "language_model.model.layers",
                 "backbone.layers", "model.decoder.layers", "transformer.h"):
        cur = model
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            if len(cur) > 0:
                return cur
        except AttributeError:
            continue
    raise SystemExit("could not locate the decoder layer list for this architecture")


# ORDER MATTERS. Gemma-family blocks (Gemma2/3/4) use SANDWICH norms: their
# `post_attention_layernorm` normalises the ATTENTION OUTPUT before the residual add
# (`h = residual + post_attention_layernorm(attn_out)`), so its input is NOT the pre-MLP
# residual there -- the pre-MLP residual is the input to `pre_feedforward_layernorm`.
# That name must therefore be probed FIRST: it only exists on sandwich-norm blocks, where
# `post_attention_layernorm` exists too but is the wrong tensor. On every other supported
# family (gpt-oss, qwen3*, phi3, llama-likes) only `post_attention_layernorm` exists and
# its input is exactly the paper's pre-MLP residual.
PRE_MLP_SITES = ["pre_feedforward_layernorm",
                 "post_attention_layernorm", "pre_mlp_layernorm",
                 "post_attn_layernorm", "ffn_norm"]


def register_probe_capture(block, fn):
    """Capture the PRE-MLP residual stream -- the paper's `all_pre_mlp_hidden_states`.

    In a decoder layer:
        residual = h
        h = post_attention_layernorm(h)   <- its INPUT is the pre-MLP residual
        h = mlp(h); h = residual + h
    so a forward PRE-hook on that norm sees exactly the tensor the paper probes. A
    post-hook on the block would give the POST-MLP residual instead, which is a
    different representation and is not what the paper reports.

    NOTE this is the PROBE site only. Steering still writes the block output, because
    that is the only place an edit actually enters the residual stream.
    """
    for nm in PRE_MLP_SITES:
        m = getattr(block, nm, None)
        if isinstance(m, torch.nn.Module):
            return m.register_forward_pre_hook(lambda mod, inp: fn(inp[0])), nm
    return block.register_forward_hook(lambda mod, i, o: fn(tensor_of(o))), "block_out"


def pick_site(block):
    """Module to hook inside a decoder block, plus its name.

    THE RESIDUAL STREAM, i.e. the decoder layer's own output.

    Earlier versions hooked `post_attention_layernorm`, which is NOT the residual stream:

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)   # <- hooked here
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states                       # <- residual intact

    that tensor feeds ONLY the MLP, so steering it left the residual stream untouched and the
    perturbation reached the next layer only through the MLP's response. That is why every
    prior sweep needed alpha of 8-12 sigma (a step of 1.2-1.5x the mean activation norm)
    before ASR moved, why the direction stopped mattering at that magnitude, and why
    correctness collapsed alongside. CAA / ITI / Arditi all add to the residual stream.

    Hooking the block itself gives the residual stream leaving layer L, for every
    architecture, with no per-model attribute guessing.
    """
    return block, "block_out"


def tensor_of(out):
    return out[0] if isinstance(out, tuple) else out


def rewrap(out, h):
    return (h,) + out[1:] if isinstance(out, tuple) else h
