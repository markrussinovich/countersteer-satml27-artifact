#!/usr/bin/env python
"""What does a steering direction DO? Unembed it and read off the tokens it promotes.

Reverse-engineering entry point for the one intervention that measurably worked:

    mn_tool@2.5  ->  ASR 0.375 -> 0.042, correctness 0.208 -> 0.500 (70.6% of clean)
    magnitude-matched random@2.5 -> ASR 0.292, correctness 0.250   (control FAILS to match)

That direction came from the OLD, position-confounded probe, and is nearly ORTHOGONAL to
every direction in the validated basis (cos 0.05-0.22 with the new mn_tool; ~0.00 with the
global activation mean and with its own probe's tool role-mean). So the defense was never
doing role correction. This asks what it was doing instead.

Method: project the direction through the unembedding, `d @ W_U`, and report the most
promoted and most suppressed tokens. Standard logit-lens caveat -- a mid-stack residual
direction is read here without the intervening layers, so treat this as a strong hint about
what the edit pushes toward, not proof of the downstream computation.

Usage:
    python tools/controls/direction_logit_lens.py [RUN_DIR] [DIRECTION] [MODEL] [DEVICE] [TOPK]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT
RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/gpt-oss-20b-resid"
DIRECTION = sys.argv[2] if len(sys.argv) > 2 else "mn_tool"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "openai/gpt-oss-20b"
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cuda:0"
TOPK = int(sys.argv[5]) if len(sys.argv) > 5 else 30


def main():
    import glob
    import re
    layers = sorted(int(re.search(r"probe_L(\d+)", f).group(1))
                    for f in glob.glob(f"{RUN_DIR}/probe_L*.pkl"))
    model, tok = X.load_model_and_tok(MODEL, DEVICE)
    W_U = model.get_output_embeddings().weight        # [vocab, d_model]
    print(f"run: {RUN_DIR}   direction: {DIRECTION}   layers: {layers}")
    print(f"W_U: {tuple(W_U.shape)}")

    for L in layers:
        p = X.load_probe(f"{RUN_DIR}/probe_L{L}.pkl")
        if DIRECTION not in p["dirs"]:
            print(f"L{L}: no direction {DIRECTION} (have {list(p['dirs'])})")
            continue
        d = torch.tensor(p["dirs"][DIRECTION], dtype=W_U.dtype, device=W_U.device)
        d = d / d.norm()
        logits = (W_U @ d).float()                    # [vocab]
        top = torch.topk(logits, TOPK)
        bot = torch.topk(-logits, TOPK)
        print(f"\n{'='*78}\nlayer {L}   (|proj| mean {logits.abs().mean():.4f}, "
              f"max {logits.max():.3f}, min {logits.min():.3f})")
        print("PROMOTED :", " | ".join(repr(tok.decode([i])) for i in top.indices.tolist()))
        print("SUPPRESSED:", " | ".join(repr(tok.decode([i])) for i in bot.indices.tolist()))


if __name__ == "__main__":
    main()
