"""Shared probe-evaluation machinery for the control experiments.

Factored out of tools/probe_report.py so the controls do not re-implement (and drift from)
probe loading, class-order handling, and per-token role-logit extraction.

Everything here works in LOGIT space. Never softmax: p_tool saturates at ~0.994 and hides
the drift, which already invalidated one conclusion in this project.
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/datadrive/huggingface/")
import xpia_defense as X  # noqa: E402


def load_probes(run_dir):
    """-> (layers, roles, Wb) with coef_ rows in `roles` order.

    Rows are reordered via the FITTER's own classes_, not by assuming `roles` order. They
    coincide only while all classes survive the MN_FIT_ROWS subsample; if one ever drops,
    naive indexing silently measures a different role.
    """
    rep = json.load(open(f"{run_dir}/probe_report.json"))
    layers = rep["layers"]
    P = {L: X.load_probe(f"{run_dir}/probe_L{L}.pkl") for L in layers}
    roles = P[layers[0]]["roles"]
    Wb = {}
    for L in layers:
        cls = list(P[L]["mn"].classes_)
        assert set(cls) == set(range(len(roles))), (
            f"L{L}: multinomial saw classes {cls}, expected all of {roles}")
        order = [cls.index(i) for i in range(len(roles))]
        Wb[L] = (torch.tensor(P[L]["mn"].coef_[order], dtype=torch.float32),
                 torch.tensor(P[L]["mn"].intercept_[order], dtype=torch.float32))
    return layers, roles, Wb


def attach_capture(model, layers):
    """Hook the PRE-MLP residual (the site these probes were fit on). -> (handles, cap)."""
    blocks = X.layer_container(model)
    cap = {}

    def mk(L):
        def store(t, L=L):
            cap[L] = t.detach()
        return store

    hs = [X.register_probe_capture(blocks[L], mk(L))[0] for L in layers]
    return hs, cap


def make_role_logits(model, cap, Wb, layers):
    """-> fn(ids, idx) -> {layer: ndarray [n_selected_tokens, n_roles]} of RAW role logits."""
    def role_logits(ids, idx):
        cap.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=model.device))
        out = {}
        for L in layers:
            W, b = Wb[L]
            out[L] = (cap[L][0].float().cpu()[idx] @ W.T + b).numpy()
        return out
    return role_logits


def dev_samples(n_dev=24):
    """The report's dev slice, so controls measure the SAME samples as Q1-Q4.

    probe_report reads sys.argv at module scope with different positional meanings than a
    control script's, so argv is neutralised across the import.
    """
    sys.path.insert(0, f"{ROOT}/tools")
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        from probe_report import dev_slice
    finally:
        sys.argv = _argv
    return dev_slice(X.build_dataset(), n_dev=n_dev)
