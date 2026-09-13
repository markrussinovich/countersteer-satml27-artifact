#!/usr/bin/env python
"""Convert an AGRI probe checkpoint (reference/agri pipeline) to the deployable spec JSON.

The checkpoint comes from THEIR training code (`ipi-signal-probe train`, schema
`ipi_aware.probe_checkpoint.v2`): a linear probe wrapped with train-split z-scoring
(state dict keys `input_mean`, `input_std`, `probe.weight`, `probe.bias`). This script
flattens it into the JSON that tools/controls/agri_gate.AGRIGate loads, carrying the
layer, threshold, position offset, active-turn window and (optionally) a prefill
override, plus provenance metadata.

Usage:
    python tools/controls/build_agri_probe.py \\
        --checkpoint <run>/probes/.../models/best_layer_20.pt \\
        --out runs/agri_probe_gptoss_L20.json
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="their per-layer checkpoint .pt (models/best_layer_<n>.pt)")
    ap.add_argument("--out", required=True, help="spec JSON path")
    ap.add_argument("--position-offset", type=int, default=0,
                    help="post-assistant token offset from the last prompt token, per "
                         "their MODEL_FAMILY_POSITION_OFFSETS (oai-oss/qwen3/gemma4: 0, "
                         "qwen3.5: -2)")
    ap.add_argument("--active-turns", type=int, default=3,
                    help="turns the intervention stays active after a fire (paper: 3)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="gate threshold on sigmoid(z); default: the checkpoint's own "
                         "stored threshold, else 0.5 (the paper's shared default)")
    ap.add_argument("--prefill", default=None,
                    help="override the anti-injection reasoning prefill text; default "
                         "None = their verbatim text (agri_gate.AGRI_PREFILL)")
    a = ap.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    w = sd["probe.weight"].float().reshape(-1)
    b = float(sd["probe.bias"].float().reshape(-1)[0])
    mean = sd["input_mean"].float().reshape(-1)
    std = sd["input_std"].float().reshape(-1)
    assert w.shape == mean.shape == std.shape, \
        f"shape mismatch: w{tuple(w.shape)} mean{tuple(mean.shape)} std{tuple(std.shape)}"
    layers = list(ckpt.get("selected_layer_indices") or ckpt.get("layer_indices") or [])
    assert len(layers) == 1, f"expected a single-layer probe, got layers={layers}"
    thr = a.threshold if a.threshold is not None else float(ckpt.get("threshold", 0.5))

    spec = {
        "layer": int(layers[0]),
        "weight": w.tolist(),
        "bias": b,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold": thr,
        "position_offset": a.position_offset,
        "active_turns": a.active_turns,
        "prefill": a.prefill,
        "meta": {
            "checkpoint": os.path.abspath(a.checkpoint),
            "schema": ckpt.get("schema"),
            "dataset": ckpt.get("dataset_name"),
            "labeling_protocol": ckpt.get("labeling_protocol"),
            "feature_name": ckpt.get("feature_name"),
            "probe_architecture": ckpt.get("probe_architecture"),
            "training_config": ckpt.get("training_config"),
            "metrics": ckpt.get("metrics"),
        },
    }
    blob = json.dumps(spec)
    with open(a.out + ".tmp", "w") as f:
        f.write(blob)
    os.replace(a.out + ".tmp", a.out)
    json.load(open(a.out))
    print(f"wrote {os.path.abspath(a.out)}: layer={spec['layer']} d={len(spec['weight'])} "
          f"threshold={thr} position_offset={a.position_offset} "
          f"active_turns={a.active_turns}")


if __name__ == "__main__":
    main()
