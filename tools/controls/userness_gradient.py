"""The CAUSAL userness direction: the gradient of downstream role reading, by backprop.

WHY. The orthogonalized-steering control (steer_probe_readout.py, FINDINGS.md 10b) proved
the causal object is not the probe's axis: a vector with zero projection onto every
downstream user-tool axis still moves the downstream reading, linearly in dose. Linearity
means the map (perturbation at layer S) -> (user-tool reading at layer L) has a well-defined
gradient, and that gradient IS the local causal userness direction at S -- computable in one
backward pass per text instead of by steering search.

WHAT IS COMPUTED. For each held-out paper-corpus text rendered under the <tool> header
(the same texts and spans as steer_probe_readout): hook block S's OUTPUT (the steer site),
cut the graph below it (detach + requires_grad), run the forward, read the probe's
user-tool logit at each readout layer L > S (pre-MLP site, graph kept), and backprop the
span-mean logit to the block-S output. The span-mean gradient g(text, L) is stored, plus:

  - cos(g, axis_L): how much of the gradient is the identity-path/pass-through term (the
    readout axis itself rides the residual stream back to S unchanged, so axis_L is the
    expected dominant component).
  - cos(g, axis_S): how far the causal direction sits from the probe's own axis at S --
    the number that quantifies the probe-vs-cause gap the perp experiment proved exists.
  - across texts: mean pairwise cosine and the SVD spectrum of the stacked normalized
    gradients (top-1 energy fraction). Near-rank-1 => causal userness at S is essentially
    ONE direction (just not the probe's); a flat spectrum => genuinely multidimensional.

Usage:
  .venv/bin/python tools/controls/userness_gradient.py \
      [--probe-run runs/gpt-oss-20b-paperexact] [--source-layer 12] \
      [--readout-layers 14,16,18,20,22] [--n 24] [--device cuda:0] \
      [--out runs/userness_gradient.json]

Writes the JSON summary and a sibling .npz with the raw mean-gradient vectors.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/datadrive/huggingface/")
import xpia_defense as X  # noqa: E402
from steer_probe_readout import held_out_texts, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--probe-run", dest="probe_run", default="runs/gpt-oss-20b-paperexact")
    ap.add_argument("--source-layer", dest="source_layer", type=int, default=12)
    ap.add_argument("--readout-layers", dest="readout_layers", default="14,16,18,20,22")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/userness_gradient.json")
    args = ap.parse_args()

    S = args.source_layer
    readout = [int(x) for x in args.readout_layers.split(",")]
    assert all(L > S for L in readout), "readout layers must be downstream of the source"

    model, tok = X.load_model_and_tok(args.model, args.device)
    model.requires_grad_(False)
    rep = json.load(open(f"{args.probe_run}/probe_report.json"))
    P = {L: X.load_probe(f"{args.probe_run}/probe_L{L}.pkl") for L in readout + [S]}
    roles = P[readout[0]]["roles"]
    ur, tr = roles.index("user"), roles.index("tool")

    def axis(L):
        cls = list(P[L]["mn"].classes_)
        W = np.asarray(P[L]["mn"].coef_)[[cls.index(i) for i in range(len(roles))]]
        v = torch.tensor(W[ur] - W[tr], dtype=torch.float32)
        return v / v.norm()

    axes = {L: axis(L) for L in readout}
    axis_S = axis(S)
    axes_gpu = {L: axes[L].to(args.device) for L in readout}

    texts = held_out_texts(tok, args.n, expect=rep.get("n_seqs"))
    print(f"[grad] {len(texts)} texts | source block-out L{S} | readout {readout}",
          flush=True)

    blocks = X.layer_container(model)
    src: dict = {}

    def src_hook(mod, inp, out):
        h = X.tensor_of(out)
        h2 = h.detach().requires_grad_(True)   # cut the graph below S: layers 0..S run
        src["h"] = h2                          # grad-free, backward stops here
        return X.rewrap(out, h2)

    cap: dict = {}
    hs = [blocks[S].register_forward_hook(src_hook)]
    for L in readout:
        def store(t, L=L):
            cap[L] = t                          # NO detach -- the graph must survive
        h_, _ = X.register_probe_capture(blocks[L], store)
        hs.append(h_)

    G = {L: [] for L in readout}                # normalized mean-gradients per text
    Gn = {L: [] for L in readout}               # their norms
    used = 0
    for _, text in texts:
        got = render(tok, "tool", text)
        if not got:
            continue
        ids, idx = got
        cap.clear(); src.clear()
        with torch.enable_grad():
            model(torch.tensor([ids], device=model.device))
            sel = torch.tensor(idx, device=model.device)
            for L in readout:
                scalar = (cap[L][0, sel].float() @ axes_gpu[L]).mean()
                g = torch.autograd.grad(scalar, src["h"], retain_graph=True)[0]
                g = g[0, sel].float().mean(0).cpu()
                Gn[L].append(float(g.norm()))
                G[L].append((g / g.norm()).numpy().astype(np.float16))
        cap.clear(); src.clear()
        used += 1
        print(f"[grad] text {used}/{len(texts)} done", flush=True)
    for h in hs:
        h.remove()

    out = {"meta": {"model": args.model, "probe_run": args.probe_run, "source_layer": S,
                    "readout_layers": readout, "n_texts": used,
                    "site_source": "block_out", "site_readout": "pre-MLP residual"},
           "per_readout": {}}
    npz = {}
    for L in readout:
        M = np.stack(G[L]).astype(np.float32)           # [n, D], unit rows
        npz[f"G_L{L}"] = M.astype(np.float16)
        mean_g = M.mean(0)
        mean_g /= np.linalg.norm(mean_g)
        C = M @ M.T
        pair = float(C[np.triu_indices(len(M), 1)].mean())
        sv = np.linalg.svd(M, compute_uv=False)
        energy = (sv ** 2) / (sv ** 2).sum()
        out["per_readout"][str(L)] = {
            "cos_meanG_axisL": float(mean_g @ axes[L].numpy()),
            "cos_meanG_axisS": float(mean_g @ axis_S.numpy()),
            "mean_pairwise_cos": pair,
            "sv_energy_top5": [float(e) for e in energy[:5]],
            "grad_norm_mean": float(np.mean(Gn[L])),
        }
    # pooled: is there ONE causal direction at S across all readouts?
    Mall = np.concatenate([np.stack(G[L]).astype(np.float32) for L in readout])
    sv = np.linalg.svd(Mall, compute_uv=False)
    energy = (sv ** 2) / (sv ** 2).sum()
    top = np.linalg.svd(Mall, full_matrices=False)[2][0]
    out["pooled"] = {"sv_energy_top5": [float(e) for e in energy[:5]],
                     "cos_top_axisS": float(top @ axis_S.numpy()),
                     "cos_top_axes": {str(L): float(top @ axes[L].numpy())
                                      for L in readout}}
    npz["pooled_top_direction"] = top.astype(np.float32)

    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(out, indent=2))
    json.load(open(tmp))
    os.replace(tmp, args.out)
    np.savez(args.out.replace(".json", ".npz"), **npz)
    print(f"[grad] wrote {args.out} (+.npz)", flush=True)

    print(f"\n=== causal gradient at block-out L{S} "
          f"(mean over {used} texts; all cosines vs unit axes) ===")
    print("readout  cos(meanG,axis_L)  cos(meanG,axis_S)  pairwise  top1-energy")
    for L in readout:
        r = out["per_readout"][str(L)]
        print(f"L{L:<7} {r['cos_meanG_axisL']:+.3f}            {r['cos_meanG_axisS']:+.3f}"
              f"            {r['mean_pairwise_cos']:.3f}     {r['sv_energy_top5'][0]:.3f}")
    p = out["pooled"]
    print(f"pooled top-1 energy {p['sv_energy_top5'][0]:.3f}  top-5 "
          f"{[round(e, 3) for e in p['sv_energy_top5']]}  cos(top, axis_S) "
          f"{p['cos_top_axisS']:+.3f}")


if __name__ == "__main__":
    main()
