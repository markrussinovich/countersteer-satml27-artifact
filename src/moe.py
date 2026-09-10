"""MoE introspection: finding the top-k routers inside a decoder block (live model), and
reading router weight matrices straight out of a checkpoint (CPU, no model load).

Two consumers, one place:

  * `steering.RouterBlind` (ROUTER-BLIND RESIDUAL STEERING) needs the LIVE router modules
    and the pre-MLP norm that feeds them, inside a model that is already on the GPUs.
  * `tools/controls/build_routernull_direction.py` (ROUTER-NULL DIRECTION SURGERY) needs
    only the router WEIGHT MATRICES, which are read from the safetensors shards on CPU.

NOTHING HERE IS MODEL-SPECIFIC BY NAME. Routers are found structurally: a child of the
block's MLP whose `weight` is (n_experts, hidden_size) with n_experts >= 2. That matches
every MoE family installed here without a per-model table:

    gpt-oss          block.mlp.router  GptOssTopKRouter      weight (32, 2880)  + bias
    qwen3_moe        block.mlp.gate    Qwen3MoeTopKRouter    weight (128, 2048)
    qwen3_next       block.mlp.gate    Qwen3NextTopKRouter   weight (512, 2048)
    glm4_moe         block.mlp.gate    Glm4MoeTopkRouter     weight (128, 4096) + bias buffer

and it does NOT match the dense-MLP `gate_proj` (name is `gate_proj`, not `gate`) nor
Qwen3-Next's `shared_expert_gate` (weight is (1, hidden), below `min_experts`).
"""
from __future__ import annotations

import glob
import json
import os
import re

import torch

from .model import PRE_MLP_SITES

# Child names on the block's MLP that may hold a top-k router. Checked by NAME first so a
# stray (n, hidden) parameter elsewhere in the MLP cannot be mistaken for a router.
ROUTER_CHILD_NAMES = ("gate", "router")
# Qwen3-Next's per-token sigmoid gate on the SHARED expert. Not a top-k router -- it is a
# scalar mix weight -- so it is excluded unless explicitly asked for.
SHARED_GATE_CHILD_NAMES = ("shared_expert_gate",)


def _mlp_of(block):
    for nm in ("mlp", "feed_forward", "block_sparse_moe", "moe"):
        m = getattr(block, nm, None)
        if isinstance(m, torch.nn.Module):
            return m
    return None


def _looks_like_router(mod, hidden_size, min_experts=2):
    w = getattr(mod, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        return False
    return w.shape[1] == hidden_size and w.shape[0] >= min_experts


def block_routers(block, hidden_size, include_shared=False):
    """[(qualified_name, module)] for the top-k routers in one decoder block.

    Empty list for a dense block. `include_shared` additionally returns Qwen3-Next's
    `shared_expert_gate` (weight (1, hidden)), which is a mix weight rather than a router.
    """
    mlp = _mlp_of(block)
    if mlp is None:
        return []
    out = []
    for nm in ROUTER_CHILD_NAMES:
        m = getattr(mlp, nm, None)
        if isinstance(m, torch.nn.Module) and _looks_like_router(m, hidden_size):
            out.append((f"mlp.{nm}", m))
    if include_shared:
        for nm in SHARED_GATE_CHILD_NAMES:
            m = getattr(mlp, nm, None)
            if isinstance(m, torch.nn.Module) and _looks_like_router(m, hidden_size,
                                                                    min_experts=1):
                out.append((f"mlp.{nm}", m))
    return out


def pre_mlp_norm(block):
    """(module, name) of the norm whose INPUT is the pre-MLP residual, or (None, None).

    Same site list and same ordering rationale as `model.register_probe_capture` -- the
    router reads this norm's OUTPUT, so blinding the router means recomputing this norm on a
    corrected pre-norm residual. `pre_feedforward_layernorm` must be checked first because
    Gemma-family sandwich-norm blocks also carry a `post_attention_layernorm` that
    normalises the ATTENTION output instead.
    """
    for nm in PRE_MLP_SITES:
        m = getattr(block, nm, None)
        if isinstance(m, torch.nn.Module):
            return m, nm
    return None, None


# ─────────────────────────────────────────────── checkpoint-side (CPU, no model load)
def resolve_snapshot(model_id, snapshot=None, hf_home=None):
    """Local snapshot directory for a hub id, without downloading or loading anything.

    Order: an explicit `snapshot` path; then the HF cache resolved by huggingface_hub;
    then a glob under `hf_home`/hub (or $HF_HOME/hub). The fleet keeps models under
    different roots -- gpt-oss/Qwen under /datadrive/huggingface, GLM-4.5-Air under
    /datadrive2/huggingface -- so `--hf-home` is a required escape hatch, not a nicety.
    """
    if snapshot:
        if not os.path.isdir(snapshot):
            raise SystemExit(f"--snapshot {snapshot} is not a directory")
        return snapshot
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        pass
    roots = [r for r in (hf_home, os.environ.get("HF_HOME")) if r]
    for root in roots:
        pat = os.path.join(root, "hub", "models--" + model_id.replace("/", "--"),
                           "snapshots", "*")
        c = sorted(glob.glob(pat))
        if len(c) == 1:
            return c[0]
        if len(c) > 1:
            raise SystemExit(f"{pat} matched {len(c)} snapshots; pass --snapshot explicitly")
    raise SystemExit(
        f"no local snapshot for {model_id}; pass --snapshot /path/to/snapshot "
        f"(searched huggingface_hub cache and {roots or 'no HF_HOME'})")


def _weight_map(snapshot):
    idx = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    single = "model.safetensors"
    if not os.path.exists(os.path.join(snapshot, single)):
        # A metadata-only snapshot (config + tokenizer, no shards) is the normal state on a
        # fleet box that never hosted the model. Say that, rather than "no index.json" --
        # the fix is to run on the box that has the weights, not to re-download.
        have = sorted(os.listdir(snapshot))[:8]
        raise SystemExit(
            f"{snapshot} has no weight shards (neither model.safetensors.index.json nor "
            f"{single}); it holds only {have}. Router weights must be read on a host where "
            f"this checkpoint is actually downloaded, or pass --snapshot pointing at one.")
    from safetensors import safe_open
    with safe_open(os.path.join(snapshot, single), framework="pt") as f:
        return {k: single for k in f.keys()}


def load_checkpoint_tensor(snapshot, name, weight_map=None):
    from safetensors import safe_open
    wm = weight_map if weight_map is not None else _weight_map(snapshot)
    if name not in wm:
        raise SystemExit(f"{snapshot}: no tensor named {name}")
    with safe_open(os.path.join(snapshot, wm[name]), framework="pt") as f:
        return f.get_tensor(name).float()


_ROUTER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.(?:gate|router)\.weight$")


def checkpoint_router_weights(snapshot, layers=None):
    """{layer_index: (n_experts, hidden) float32 tensor} for every MoE layer.

    Discovered from the safetensors weight map by regex, so a model whose router child is
    called `router` (gpt-oss) and one whose child is called `gate` (qwen3*/glm4_moe) both
    work with no per-model configuration. Dense layers simply have no such key.
    """
    wm = _weight_map(snapshot)
    hits = {}
    for k in wm:
        m = _ROUTER_RE.search(k)
        if m:
            hits[int(m.group(1))] = k
    if not hits:
        raise SystemExit(
            f"{snapshot}: no `layers.N.mlp.(gate|router).weight` in the weight map -- "
            f"either this checkpoint is dense or its router is named something new. "
            f"Router-null surgery needs router weights; refusing to guess.")
    want = sorted(hits) if layers is None else [L for L in layers if L in hits]
    return {L: load_checkpoint_tensor(snapshot, hits[L], weight_map=wm) for L in want}
