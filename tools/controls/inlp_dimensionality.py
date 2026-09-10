"""How many linear directions carry user-vs-tool information? (INLP-style iteration)

WHY. The role probe reads ONE direction per layer. The orthogonalized-steering control
proved role-relevant structure exists off that axis; this measures the CORRELATIONAL side
of the same question: fit the user-vs-tool probe, project its direction out of every
activation, refit on what remains, and repeat. The round at which held-out accuracy falls
to chance (0.5 -- the classes are exactly balanced: the same text rendered under both
headers) is the effective linear dimensionality of user-vs-tool information at that layer.
This is Ravfogel et al.'s Iterative Nullspace Projection applied to the paper's own probe
setup: same corpus (paper-exact, src/corpora.load_corpus kind=paper), same rendering
(render_single, header-only diff), same site (pre-MLP residual), same C=5e-3, same
by-sequence 25% holdout (rng(0), matching src/probes.train_probes).

TWO FIXES FROM THE ADVERSARIAL REVIEW (2026-08-25):

1. `--equalize`: the user and tool headers differ by a CONSTANT 8-token offset (content
   starts at absolute token 3 vs 11), a position confound the round-1 position-equalised
   control cleared but rounds >= 2 never were. With this flag, filler (" x"*k + "\n") is
   prepended INSIDE the content until the probed span starts at the same absolute index
   (16) in both roles (the position_confound_control.py construction); filler tokens are
   excluded from the span. Without it, the high-dimensionality reading of the tail rounds
   is UNPROVEN -- a nonlinearly-encoded position confound spawns many linear directions.
2. Each round's direction is RE-ORTHOGONALIZED against the accumulated basis before being
   projected out, and its genuinely-new fraction is recorded. Without this, float32
   rounding error (label-correlated, ~2e-7 relative -- too small to move accuracy, big
   enough to steer LBFGS) makes some rounds re-find an already-removed direction almost
   exactly (observed: cos up to 0.994 with round 1), so "k rounds" overstated the number
   of distinct directions by ~1 per layer.

Usage:
  .venv/bin/python tools/controls/inlp_dimensionality.py \
      [--probe-run runs/gpt-oss-20b-paperexact] [--layers 12,16] [--n-seqs 250] \
      [--rounds 12] [--equalize] [--device cuda:0] [--out runs/inlp_dimensionality.json]

Prints held-out accuracy per round per layer, plus each round's direction's cosine to the
round-1 direction, to the probe run's own user-tool axis, and its new (post-reorth)
fraction.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/datadrive/huggingface/")
import xpia_defense as X  # noqa: E402
from src.probes import fit_logreg_gpu  # noqa: E402

MN_FIT_ROWS = 200_000
EQ_TARGET = 16      # common absolute start index under --equalize


def probe_idx(tok, text, seq):
    """token indices of `seq` inside `text`, by char offsets (position_confound_control)."""
    lo = text.rindex(seq)
    hi = lo + len(seq)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"])
           if a >= lo and b <= hi and b > a]
    return enc["input_ids"], idx


def render_span(tok, role, seq, equalize):
    if not equalize:
        got = X.sentinel_span(
            tok, lambda c, r=role: X.render_single(tok, r, c, X.TOOL_NAME), seq)
        if not got:
            return None
        text, span = got
        return X.token_span(tok, text, span)
    for k in range(0, 24):
        pre = " x" * k + "\n"
        text = X.render_single(tok, role, pre + seq, X.TOOL_NAME)
        ids, idx = probe_idx(tok, text, seq)
        if idx and idx[0] == EQ_TARGET:
            return ids, idx
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--probe-run", dest="probe_run", default="runs/gpt-oss-20b-paperexact")
    ap.add_argument("--layers", default="12,16")
    ap.add_argument("--n-seqs", dest="n_seqs", type=int, default=250)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--equalize", action="store_true",
                    help="position-equalise the two renders (see docstring)")
    ap.add_argument("--stop-acc", dest="stop_acc", type=float, default=0.55)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/inlp_dimensionality.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    model, tok = X.load_model_and_tok(args.model, args.device)

    # the probe run's own user-tool axis, for the round-wise cosine
    ref_axis = {}
    for L in layers:
        p = X.load_probe(f"{args.probe_run}/probe_L{L}.pkl")
        cls = list(p["mn"].classes_)
        W = np.asarray(p["mn"].coef_)[[cls.index(i) for i in range(len(p["roles"]))]]
        v = W[p["roles"].index("user")] - W[p["roles"].index("tool")]
        ref_axis[L] = (v / np.linalg.norm(v)).astype(np.float32)

    seqs = X.load_corpus(tok, args.n_seqs, kind="paper")
    blocks = X.layer_container(model)
    cap: dict = {}
    hs = []
    for L in layers:
        def store(t, L=L):
            cap[L] = t.detach()
        h_, site = X.register_probe_capture(blocks[L], store)
        hs.append(h_)

    feats = {L: [] for L in layers}
    labels, groups = [], []
    for si, seq in enumerate(seqs):
        for yi, role in enumerate(("tool", "user")):        # 0 = tool, 1 = user
            got = render_span(tok, role, seq, args.equalize)
            if not got or not got[1]:
                continue
            ids, idx = got
            cap.clear()
            with torch.no_grad():
                model(torch.tensor([ids], device=model.device))
            sel = torch.tensor(idx, device=model.device)
            for L in layers:
                feats[L].append(cap[L][0, sel].float().cpu().numpy().astype(np.float16))
            labels.append(np.full(len(idx), yi, dtype=np.int64))
            groups.append(np.full(len(idx), si, dtype=np.int64))
        if (si + 1) % 50 == 0:
            print(f"[inlp] captured {si+1}/{len(seqs)}", flush=True)
    for h in hs:
        h.remove()

    y, g = np.concatenate(labels), np.concatenate(groups)
    u = np.unique(g)
    np.random.default_rng(0).shuffle(u)                     # the fit's own split rule
    test = set(u[: int(len(u) * .25)].tolist())
    tm = np.array([x in test for x in g])
    print(f"[inlp] {len(y)} tokens | {len(u)} seqs | site {site} | "
          f"test tokens {int(tm.sum())} | class balance {y.mean():.3f}", flush=True)

    results = {"meta": {"model": args.model, "probe_run": args.probe_run,
                        "layers": layers, "n_seqs": len(seqs), "site": site,
                        "C": 5e-3, "rounds": args.rounds, "n_tokens": int(len(y))},
               "per_layer": {}}
    for L in layers:
        Xf = np.concatenate(feats[L]).astype(np.float32)
        Xtr, Xte = Xf[~tm], Xf[tm]
        ytr, yte = y[~tm], y[tm]
        fit_ix = (np.random.default_rng(0).permutation(len(Xtr))[:MN_FIT_ROWS]
                  if len(Xtr) > MN_FIT_ROWS else np.arange(len(Xtr)))
        rows, u1, basis = [], None, []
        for r in range(1, args.rounds + 1):
            clf = fit_logreg_gpu(Xtr[fit_ix], ytr[fit_ix], C=5e-3, device=args.device,
                                 max_iter=2000)
            acc = float((clf.predict(Xte) == yte).mean())
            w = clf.coef_[list(clf.classes_).index(1)] - \
                clf.coef_[list(clf.classes_).index(0)]
            un = (w / np.linalg.norm(w)).astype(np.float32)
            if u1 is None:
                u1 = un
            # re-orthogonalize against everything already removed: float32 rounding is
            # label-correlated and lets LBFGS re-find a removed direction almost exactly
            # (observed cos up to 0.994); only the genuinely-new component counts
            new = un.copy()
            for b in basis:
                new -= (new @ b) * b
            new_frac = float(np.linalg.norm(new))
            new /= np.linalg.norm(new)
            rows.append({"round": r, "acc": acc,
                         "cos_round1": float(un @ u1),
                         "cos_ref_axis": float(un @ ref_axis[L]),
                         "new_fraction": new_frac})
            print(f"[inlp] L{L} round {r}: acc {acc:.4f}  cos(round1) "
                  f"{un @ u1:+.3f}  cos(ref axis) {un @ ref_axis[L]:+.3f}  "
                  f"new {new_frac:.3f}", flush=True)
            if acc < args.stop_acc:
                break
            basis.append(new)
            Xtr -= (Xtr @ new)[:, None] * new
            Xte -= (Xte @ new)[:, None] * new
        results["per_layer"][str(L)] = rows

    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(results, indent=2))
    json.load(open(tmp))
    os.replace(tmp, args.out)
    print(f"[inlp] wrote {args.out}", flush=True)

    print("\n=== held-out user-vs-tool accuracy after removing k directions "
          "(chance 0.5) ===")
    hdr = "layer " + " ".join(f"k={r-1:<4}" for r in range(1, args.rounds + 1))
    print(hdr)
    for L in layers:
        rows = results["per_layer"][str(L)]
        print(f"L{L:<4} " + " ".join(f"{r['acc']:.3f} " for r in rows))


if __name__ == "__main__":
    main()
