#!/usr/bin/env python
"""Calibrate the displacement-proportional step rule ON THE DIRECTION AND SITE ACTUALLY USED.

WHY THIS EXISTS. `--step-boundary auto` used to read runs/gate_separability.json, which
measures projections onto `dim_override` captured by a forward-PRE-hook on
post_attention_layernorm -- the PRE-MLP site. Steering applies whatever `--directions` names
at the BLOCK-OUTPUT site (`pick_site`). Both differ:

    cos(dim_override, dim_no_override_both) = -0.519 / -0.305 / -0.231 at L12/16/20

so the threshold did not gate at all: 93%/71%/100% of injected spans sat above it, mean steps
1.21/0.63/6.76 sigma -- at L20 a LARGER step than the fixed alpha=8 rule delivers. The
seed-0 random control got a 3.04x SMALLER displacement, making the arms un-matched in
violation of CLAUDE.md's mandatory control invariant, and the "direction beats random 24/0,
p<1e-6" that came out of it was refuted. That path now raises; this script replaces it.

WHAT IT EMITS, per direction, per layer:

  own95    the direction's OWN 95th percentile of the LEGITIMATE-token projection
           (legit_mean + 1.645*legit_sd). Calibrating on the tokens we do NOT want to move
           is the point -- a boundary fit to the injected distribution moves record content.
           This equalises the STEERED-TOKEN FRACTION across arms.

  budget   the boundary that makes this direction's expected per-token displacement
           E[max(0, p_ov - m)] equal to the REAL direction's at its own95. This equalises
           the TOTAL EDIT NORM across arms, which is the control CLAUDE.md actually demands
           -- two arms displace the residual stream equally and only the DIRECTION of
           displacement differs.

Solved by bisection on m over the observed injected-span projections, so it needs no
distributional assumption.

Usage:
    python tools/controls/step_boundary_calibrate.py [PROBE_RUN] [MODEL] [DEVICE] [N_DEV] \
        [--directions a,b,c] [--ref DIRNAME] [--corpus shipped|param_abuse] [--template T]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT


def _flag(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    v = np.concatenate([pos, neg])
    o = np.argsort(v)
    rk = np.empty(len(v))
    rk[o] = np.arange(1, len(v) + 1)
    n1, n0 = len(pos), len(neg)
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def solve_budget(p, target, lo=-50.0, hi=50.0, iters=200):
    """m such that mean(max(0, p - m)) == target. Monotone decreasing in m -> bisection."""
    f = lambda m: float(np.maximum(0.0, p - m).mean())
    if f(lo) < target:            # cannot reach the budget even steering everything
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    probe_run = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else f"{ROOT}/runs/gpt-oss-20b-userabl"
    model_id = _flag("--model", "openai/gpt-oss-20b")
    device = _flag("--device", "cuda:0")
    n_dev = int(_flag("--n-dev", "24"))
    layers = [int(x) for x in _flag("--steer-layers", "12,16,20").split(",")]
    ref = _flag("--ref", "dim_no_override_both")
    names = _flag("--directions", f"{ref},random,random2,random3,random4,random5,random7,"
                                  "shuffled").split(",")
    corpus = _flag("--corpus", "shipped")
    template = _flag("--template", "fit")

    # samples: the same corpora the sweep evaluates
    if corpus == "param_abuse":
        S = json.load(open(X.param_corpus_path("dev", template, ROOT)))["samples"]
        man = X.param_split_manifest(ROOT, "dev")
        if man:
            by = {s["id"]: s for s in S}
            S = [by[i] for i in man if i in by]
    else:
        allx = X.build_dataset()
        S = [allx[i] for i in X.build_splits(allx, n_eval=n_dev)["dev"]]
    S = [s for s in S if s.get("injection_text") and s.get("injection_field")][:n_dev]

    model, tok = X.load_model_and_tok(model_id, device)
    blocks = X.layer_container(model)
    # CAPTURE AT THE STEER SITE, not the probe site. `pick_site` is what Steer hooks.
    cap = {}
    hs = []
    for L in layers:
        mod, _ = X.pick_site(blocks[L])
        hs.append(mod.register_forward_hook(
            (lambda LL: lambda m, i, o: cap.__setitem__(LL, X.tensor_of(o)[0].float().cpu()))(L)))

    # directions exactly as the sweep builds them, magnitude-matched the same way
    D = {}
    for nm in names:
        d, sig, _ = X.build_dirs(probe_run, layers, nm, "cpu", match_sigma_to=ref)
        D[nm] = ([np.asarray(x, np.float32) for x in d], sig)

    inj = {nm: {L: [] for L in layers} for nm in names}
    leg = {nm: {L: [] for L in layers} for nm in names}
    used = 0
    for s in S:
        try:
            ids, pay, ij = X.injection_span(tok, s)
        except Exception:
            continue
        lg = [k for k in pay if k not in set(ij)]
        if len(ij) < 3 or len(lg) < 3:
            continue
        cap.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=model.device))
        for nm in names:
            dirs, sig = D[nm]
            for k, L in enumerate(layers):
                h = cap[L].numpy()
                u = dirs[k] / (np.linalg.norm(dirs[k]) + 1e-12)
                # p_ov: override-ness. Steering ADDS along d, so override-ness is -(h.d).
                p = -(h @ u) / max(sig[k], 1e-6)
                inj[nm][L].append(p[ij])
                leg[nm][L].append(p[lg])
        used += 1
    for h_ in hs:
        h_.remove()

    out = {"probe_run": probe_run, "site": "block_out", "layers": layers, "ref": ref,
           "corpus": corpus, "template": template, "n_samples": used, "rows": {}}
    print(f"\ncaptured {used} samples at the STEER site (block_out), corpus={corpus}"
          f"{'/' + template if corpus == 'param_abuse' else ''}\n")
    hdr = (f"{'direction':<22}{'L':>4}{'AUC':>7}{'inj mu':>9}{'leg mu':>9}{'leg sd':>8}"
           f"{'own95':>9}{'frac>':>7}{'E[over]':>9}{'budget m':>10}{'frac>':>7}")
    print(hdr); print("-" * len(hdr))
    # reference budget per layer, from the ref direction at its own 95th percentile
    budget = {}
    for k, L in enumerate(layers):
        lp = np.concatenate(leg[ref][L]); ip = np.concatenate(inj[ref][L])
        m = float(lp.mean() + 1.645 * lp.std())
        budget[L] = float(np.maximum(0.0, ip - m).mean())
    for nm in names:
        for k, L in enumerate(layers):
            lp = np.concatenate(leg[nm][L]); ip = np.concatenate(inj[nm][L])
            m95 = float(lp.mean() + 1.645 * lp.std())
            e95 = float(np.maximum(0.0, ip - m95).mean())
            mb = solve_budget(ip, budget[L])
            fb = float((ip > mb).mean()) if mb is not None else float("nan")
            out["rows"].setdefault(nm, {})[str(L)] = {
                "auc": auc(ip, lp), "inj_mean": float(ip.mean()), "legit_mean": float(lp.mean()),
                "legit_sd": float(lp.std()), "own95": m95, "frac_above_own95": float((ip > m95).mean()),
                "E_over_own95": e95, "budget_m": mb, "frac_above_budget": fb,
                "ref_budget": budget[L]}
            r = out["rows"][nm][str(L)]
            print(f"{nm:<22}{L:>4}{r['auc']:>7.3f}{r['inj_mean']:>9.3f}{r['legit_mean']:>9.3f}"
                  f"{r['legit_sd']:>8.3f}{m95:>9.3f}{r['frac_above_own95']:>7.2f}{e95:>9.3f}"
                  f"{(mb if mb is not None else float('nan')):>10.3f}{fb:>7.2f}")
    dst = f"{ROOT}/runs/step_boundary_calibration.json"
    blob = json.dumps(out, indent=1)
    with open(dst + ".tmp", "w") as f:
        f.write(blob)
    json.load(open(dst + ".tmp"))
    os.replace(dst + ".tmp", dst)
    print(f"\nwrote {dst}")
    print("\nPass these to xpia_defense.py as --step-boundary, per arm:")
    for nm in names:
        v = [out["rows"][nm][str(L)]["budget_m"] for L in layers]
        if all(x is not None for x in v):
            print(f"  {nm:<22} --step-boundary {','.join(f'{x:.4f}' for x in v)}   "
                  f"(matched EDIT NORM to {ref})")


if __name__ == "__main__":
    main()
