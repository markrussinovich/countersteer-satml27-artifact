"""Does activation steering move the paper's role probe?

THE QUESTION. The role probe (Prompt Injection as Role Confusion, arXiv:2603.12277) reads
the pre-MLP residual and separates role headers at up to ~0.88 accuracy. Steering writes at
the block output. This measures the causal link between the two on the PAPER'S OWN DATASET:
held-out texts from the paper's probe corpus (25% C4-validation + 75% dolma3@3a8349c,
shuffle seed 123 -- src/corpora.load_corpus kind=paper), rendered under the `<tool>` header
exactly as the probe was trained, steered along a direction at the block output, with the
probe's user_logit - tool_logit read back at every probe layer.

LOGIT SPACE ONLY (project invariant): softmax saturates and has already hidden a real drift.

READOUT LAYERS <= min(steer layer) ARE A BUILT-IN NEGATIVE CONTROL: the pre-MLP site of
layer L sits before the block-L output, so a steer at L=12 can only move readouts at L>=14.
A shift at L<=12 means the measurement is broken, not that steering worked.

DIRECTIONS. Any name in the probe pickles' `dirs` (built by build_dirs, stored sigmas), plus
`probe_axis`: the multinomial probe's own w_user - w_tool at each steer layer, the most
direct "move the probe" vector. Its sigma is computed on the fly as the std of the
projection over the base pass's pre-MLP activations at the same layer (ITI convention,
same convention the stored sigmas use, different reference activations -- logged).

THE PASS-THROUGH CONFOUND, AND THE _perp CONTROL (adversarial review, 2026-08-25). A vector
added at the block output PERSISTS on the residual identity path, so a downstream pre-MLP
readout sees it directly: step * (u . axis_L) predicts 52-89% of the raw probe_axis shift
with no forward pass at all. A raw shift therefore does NOT show the network computing
anything differently. Appending `_perp` to any direction orthogonalizes it against the
readout axes (w_user - w_tool)_L, which zeroes the pass-through term EXACTLY: the capture
site is the forward PRE-hook on post_attention_layernorm (src/model.py), i.e. the RAW
pre-MLP residual BEFORE the norm, so a parked vector v reads through the probe as a plain
W.v -- no gain, no rms. (Verified: the raw-axis pass-through prediction matches the
measured raw-arm shift within 1.1-1.9x; a post-norm capture would have deflated it by
rms ~30-200x and the prediction would have overshot absurdly.) The basis also carries
layernorm-gain-corrected columns g_L * (w_user - w_tool)_L -- unnecessary under this
mechanism, retained because extra columns only remove MORE of the direction (safe side).
Any dose-responsive shift that survives in a _perp arm is network response; the review's
odd/even decomposition confirms the observed perp shift is 97-100% ODD in alpha, while
every scalar artifact (norm-preserve shrink, rms inflation) is EVEN in alpha and sits at
<= 0.62 logits. CAVEAT: a perp arm is clean ONLY in the user-tool logit -- its argmax
frac_user/frac_tool still carry pass-through via the OTHER role contrasts (user-system
etc.), so never quote perp-arm argmax fractions. Each cell also stores `passthrough_pred`:
the analytic no-forward-pass prediction per readout layer, so raw-arm excess over
pass-through is printed, not implied.

The basis is per steer layer and DOWNSTREAM ONLY (readout layers L > S). Including layer
S's own axis makes probe_axis@S orthogonal to itself: the residual is numerically zero and
the "direction" is float roundoff -- caught on 2026-08-25 when cos(perp, raw) printed
NEGATIVE, which is impossible for a genuine projection residual (cos == ||residual|| >= 0).
The retained-fraction print and its positivity assertion guard exactly that failure.

Usage:
  .venv/bin/python tools/controls/steer_probe_readout.py \
      --model openai/gpt-oss-20b --probe-run runs/gpt-oss-20b-paperexact \
      --n 24 --alphas -16,-8,-4,4,8,16 --directions probe_axis,dim_user_vs_rest,mn_tool \
      [--steer-layers 12,16,20] [--device cuda:0] [--out runs/steer_probe_readout.json]

Writes one row per (direction, alpha, readout layer): mean user-tool logit over the content
span of the tool-rendered message, the delta vs the unsteered pass, and the argmax role
fractions. The user-rendered base pass is included as the reference frame ("what the probe
reads on genuinely user-headed text").
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


def held_out_texts(tok, n, expect=None):
    """The fit's own 25% by-sequence test split (src/probes.py train_probes: rng(0) shuffle
    of unique sequence ids, first 25%), so no readout text entered the probe fit.

    ASSUMPTION (verified 0/249 skips on 2026-08-25, review): every sequence yielded a span
    for all roles during the fit, so unique(groups) == arange(len(seqs)) and the rng(0)
    shuffle here permutes the same array the fit did. A count mismatch breaks that silently,
    hence the assertion; a skip inside the fit would too and is NOT detectable here."""
    seqs = X.load_corpus(tok, 250, kind="paper")
    if expect is not None and len(seqs) != expect:
        raise SystemExit(f"corpus has {len(seqs)} texts but the fit saw {expect}; "
                         f"the re-derived holdout split would not match the fit's")
    u = np.arange(len(seqs))
    np.random.default_rng(0).shuffle(u)
    test = sorted(u[: int(len(u) * .25)].tolist())
    return [(int(i), seqs[i]) for i in test[:n]]


def render(tok, role, text):
    got = X.sentinel_span(tok, lambda c, r=role: X.render_single(tok, r, c, X.TOOL_NAME),
                          text)
    if not got:
        return None
    rendered, span = got
    ids, idx = X.token_span(tok, rendered, span)
    return ids, idx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--probe-run", dest="probe_run", default="runs/gpt-oss-20b-paperexact")
    ap.add_argument("--steer-layers", dest="steer_layers", default="12,16,20")
    ap.add_argument("--directions", default="probe_axis,dim_user_vs_rest,mn_tool")
    ap.add_argument("--alphas", default="-16,-8,-4,4,8,16")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/steer_probe_readout.json")
    args = ap.parse_args()

    steer_layers = [int(x) for x in args.steer_layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",") if float(x) != 0.0]
    directions = [d for d in args.directions.split(",") if d]

    model, tok = X.load_model_and_tok(args.model, args.device)
    rep = json.load(open(f"{args.probe_run}/probe_report.json"))
    layers = rep["layers"]
    P = {L: X.load_probe(f"{args.probe_run}/probe_L{L}.pkl") for L in layers}
    roles = P[layers[0]]["roles"]
    ur, tr = roles.index("user"), roles.index("tool")
    Wb = {}
    for L in layers:
        cls = list(P[L]["mn"].classes_)
        assert set(cls) == set(range(len(roles)))
        order = [cls.index(i) for i in range(len(roles))]
        Wb[L] = (torch.tensor(P[L]["mn"].coef_[order], dtype=torch.float32),
                 torch.tensor(P[L]["mn"].intercept_[order], dtype=torch.float32))

    texts = held_out_texts(tok, args.n, expect=rep.get("n_seqs"))
    print(f"[readout] {len(texts)} held-out paper-corpus texts | probe {args.probe_run} | "
          f"steer layers {steer_layers} | readout layers {layers}", flush=True)

    hs, cap = [], {}
    blocks = X.layer_container(model)

    def mk(L):
        def store(t, L=L):
            cap[L] = t.detach()
        return store
    for L in layers:
        h_, site = X.register_probe_capture(blocks[L], mk(L))
        hs.append(h_)

    def readout(ids, idx):
        """-> {layer: [n_tokens, n_roles]} role logits at the pre-MLP site."""
        out = {}
        for L in layers:
            W, b = Wb[L]
            out[L] = (cap[L][0].float().cpu()[idx] @ W.T + b).numpy()
        return out

    # ── base passes (no steering); also collect pre-MLP acts for probe_axis sigma
    base = {"tool": [], "user": []}          # per text: {layer: [T, R]}
    rendered = []                            # (ids, idx) of the tool-rendered message
    acts_at = {L: [] for L in steer_layers}  # pre-MLP acts on tool content tokens
    for _, text in texts:
        got_t, got_u = render(tok, "tool", text), render(tok, "user", text)
        if not got_t or not got_u:
            continue
        for key, (ids, idx) in (("tool", got_t), ("user", got_u)):
            cap.clear()
            with torch.no_grad():
                model(torch.tensor([ids], device=model.device))
            if key == "tool":
                for L in steer_layers:
                    acts_at[L].append(cap[L][0].float().cpu()[idx])
            base[key].append(readout(ids, idx))
        rendered.append(got_t)
    n_used = len(rendered)
    print(f"[readout] base passes done on {n_used} texts", flush=True)

    # ── the user-tool axis at every readout layer, raw and layernorm-gain-corrected;
    # a `_perp` direction is orthogonalized against ALL of these (see module docstring)
    axes = {L: (Wb[L][0][ur] - Wb[L][0][tr]) for L in layers}
    gains = {L: blocks[L].post_attention_layernorm.weight.detach().float().cpu()
             for L in layers}

    def ortho_basis(S):
        """Orthonormal basis of the readout axes DOWNSTREAM of steer layer S."""
        cols = [a / a.norm() for L in layers if L > S
                for a in (axes[L], axes[L] * gains[L])]
        if not cols:
            raise SystemExit(f"steer layer {S} has no downstream readout layer to "
                             f"orthogonalize against")
        return torch.linalg.qr(torch.stack(cols, dim=1)).Q  # [D, 2*n_downstream]

    def sigma_of(u, L):
        """ITI sigma over the base pass's pre-MLP acts on the eval tool spans."""
        return float((torch.cat(acts_at[L]) @ u.cpu()).std())

    # ── directions
    dirsets = {}
    for name in directions:
        base_name = name[:-5] if name.endswith("_perp") else name
        if base_name == "probe_axis":
            units = []
            for L in steer_layers:
                v = axes[L]
                units.append((v / v.norm()).to(args.device))
            abl = [{}] * len(steer_layers)
        else:
            try:
                units, _, abl = X.build_dirs(args.probe_run, steer_layers, base_name,
                                             args.device)
            except SystemExit as e:
                print(f"[readout] SKIP direction {name}: {e}", flush=True)
                continue
        if name.endswith("_perp"):
            perp, kept = [], []
            for u, S in zip(units, steer_layers):
                Q = ortho_basis(S)
                w = u.cpu() - Q @ (Q.T @ u.cpu())
                kept.append(float(w.norm()))       # == cos(perp, raw); genuine residual
                perp.append((w / w.norm()).to(args.device))
            assert all(k > 0.05 for k in kept), (
                f"projection residual nearly zero ({kept}); the perp direction would be "
                f"numerical noise, not an orthogonalized steer")
            print(f"[readout] {name}: retained fraction of the raw direction per steer "
                  f"layer {[round(k, 4) for k in kept]}", flush=True)
            units = perp
        # sigma is recomputed for EVERY arm over the same base-pass activations, so rows
        # at the same alpha are magnitude-matched across directions (stored pickle sigmas
        # come from a different reference corpus and differ by up to 11x -- review item 3)
        sigs = [sigma_of(u, L) for u, L in zip(units, steer_layers)]
        print(f"[readout] {name} sigmas (computed, base-pass pre-MLP): "
              f"{dict(zip(steer_layers, [round(s, 2) for s in sigs]))}", flush=True)
        dirsets[name] = (units, sigs, abl)

    def passthrough(units, sigs, a):
        """Analytic no-forward-pass prediction of d(user-tool) at each readout layer: the
        steered vector persisting on the residual identity path, dotted with the raw
        readout axis. Linear, ignores the layernorm gain and rms -- the raw arm's measured
        shift ran 1.1-1.9x this on 2026-08-25, and a _perp arm zeroes it by construction."""
        k = len(steer_layers) ** 0.5
        return {L: sum((a / k) * s * float(u.cpu() @ axes[L])
                       for u, s, S in zip(units, sigs, steer_layers) if S < L)
                for L in layers}

    def summarize(per_text):
        """per_text: list of {layer: [T, R]} -> {layer: {ut, frac_user, frac_tool}}.
        Everything is TEXT-weighted (mean of per-text means), one denominator throughout."""
        out = {}
        for L in layers:
            ut = [float((r[L][:, ur] - r[L][:, tr]).mean()) for r in per_text]
            fu = [float((r[L].argmax(1) == ur).mean()) for r in per_text]
            ft = [float((r[L].argmax(1) == tr).mean()) for r in per_text]
            out[L] = {"user_minus_tool": float(np.mean(ut)),
                      "ut_std": float(np.std(ut)),
                      "frac_user": float(np.mean(fu)),
                      "frac_tool": float(np.mean(ft))}
        return out

    results = {"meta": {"model": args.model, "probe_run": args.probe_run,
                        "steer_layers": steer_layers, "readout_layers": layers,
                        "n_texts": n_used, "roles": roles, "site_readout": site,
                        "site_steer": "block_out", "alphas": alphas,
                        "norm_preserve": True},
               "base_tool": summarize(base["tool"]),
               "base_user": summarize(base["user"]),
               "cells": []}

    for name, (units, sigs, abl) in dirsets.items():
        for a in alphas:
            per_text = []
            st = X.Steer(model, steer_layers, units, a, "sigma", sigs, abl, "add")
            for (ids, idx) in rendered:
                st.positions = [idx]
                cap.clear()
                with st, torch.no_grad():
                    model(torch.tensor([ids], device=model.device))
                per_text.append(readout(ids, idx))
            st.positions = None
            s = summarize(per_text)
            pred = passthrough(units, sigs, a)
            results["cells"].append({"direction": name, "alpha": a, "sigmas": sigs,
                                     "readout": s,
                                     "passthrough_pred": {str(L): pred[L] for L in layers}})
            line = " ".join(
                f"L{L}:{s[L]['user_minus_tool'] - results['base_tool'][L]['user_minus_tool']:+.2f}"
                f"/{pred[L]:+.2f}"
                for L in layers)
            print(f"[readout] {name:<22} a={a:+6.1f}  d(user-tool) measured/passthrough "
                  f"{line}", flush=True)

    for h in hs:
        h.remove()
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(results, indent=2))
    json.load(open(tmp))
    os.replace(tmp, args.out)
    print(f"[readout] wrote {args.out}", flush=True)

    # ── printed summary at the readout layers downstream of the first steer layer
    show = [L for L in layers if L > min(steer_layers)]
    bt, bu = results["base_tool"], results["base_user"]
    print("\n=== user-tool logit on tool-rendered held-out paper text "
          f"(base tool row is the floor, base user row the target frame) ===")
    hdr = "direction              alpha " + " ".join(f"L{L:<7}" for L in show)
    print(hdr)
    print(f"{'base tool-rendered':<22} {0.0:+6.1f} "
          + " ".join(f"{bt[L]['user_minus_tool']:+7.2f} " for L in show))
    print(f"{'base user-rendered':<22} {0.0:+6.1f} "
          + " ".join(f"{bu[L]['user_minus_tool']:+7.2f} " for L in show))
    for c in results["cells"]:
        print(f"{c['direction']:<22} {c['alpha']:+6.1f} "
              + " ".join(f"{c['readout'][L]['user_minus_tool']:+7.2f} " for L in show))

    print("\n=== network response: measured delta minus the analytic pass-through "
          "prediction (a _perp row IS its own measured delta -- pred is 0 there) ===")
    print(hdr)
    for c in results["cells"]:
        print(f"{c['direction']:<22} {c['alpha']:+6.1f} "
              + " ".join(
                  f"{c['readout'][L]['user_minus_tool'] - bt[L]['user_minus_tool'] - c['passthrough_pred'][str(L)]:+7.2f} "
                  for L in show))


if __name__ == "__main__":
    main()
