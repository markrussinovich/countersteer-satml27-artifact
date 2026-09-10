#!/usr/bin/env python
"""Is the steering direction fit at the site it is APPLIED at? Measure the mismatch.

THE DEFECT THIS MEASURES. `xpia_defense.register_probe_capture` hooks the PRE-MLP residual
(a forward-PRE-hook on `post_attention_layernorm`); `xpia_defense.pick_site` steers the BLOCK
OUTPUT. CLAUDE.md documents that split as intentional FOR THE PROBE. But
`dim_no_override_both` is not a probe -- `_probe_eval.attach_capture` uses
`register_probe_capture`, so `build_override_direction.py` computes

    d = mean(h | fired) - mean(h | not fired)        [within-sample centered]
    sigma = (A @ u).std()

on PRE-MLP activations, and both the axis AND the unit of the applied step `alpha*sigma` are
therefore measured at a site the edit never touches.

Nothing in the repo measures how large that mismatch is. It is a live candidate explanation
for the headline weakness -- at matched magnitude the direction shows no ASR advantage over a
random perturbation (pooled p = 0.080) -- because a partly-wrong axis degrades toward a random
one.

WHAT THIS RUNS. It replays the exact rows of `runs/override_slope.json` (the 3-factor
factorial: sample x override x voice x action type), rebuilding each variant prompt with
`override_slope_experiment.variant` and reusing that file's stored `fired` labels, so no
generation is needed -- one forward pass per row, capturing at BOTH sites at once. It then
refits the direction independently at each site with the identical estimator and reports:

    cos(d_preMLP, d_blockout)   how much of the axis survives the site change
    sigma_pre / sigma_block     how wrong the STEP UNIT is (alpha*sigma is in pre-MLP units)
    ||d|| at each site

Reading it: cos near 1.0 means the mismatch is cosmetic and the current fit is fine. Cos well
below 1.0 means the applied axis is partly wrong and a block-output refit is the cheapest
available shot at the direction-vs-random null. The sigma ratio is a separate defect and can
be large even when cos is high -- it rescales every alpha ever swept.

CAPTURE IS UNBATCHED, deliberately, matching override_slope_experiment: batching perturbs the
captured vectors by 1.1-1.2% relative L2 from batch-shape GEMM accumulation, and those vectors
ARE the output of this experiment.

Usage:
    python tools/controls/site_mismatch.py [--src runs/override_slope.json]
                                           [--out runs/site_mismatch.json]
                                           [--device cuda:0] [--limit N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402
import override_slope_experiment as OS  # noqa: E402

X = E.X
ROOT = E.ROOT


def attach_both(model, layers):
    """(handles, pre_cap, blk_cap) -- capture PRE-MLP and BLOCK-OUTPUT in one forward pass."""
    blocks = X.layer_container(model)
    pre, blk = {}, {}
    hs = []
    for L in layers:
        def mk_pre(L=L):
            def store(t):
                pre[L] = t.detach()
            return store

        def mk_blk(L=L):
            def hook(mod, inp, out):
                blk[L] = X.tensor_of(out).detach()
            return hook

        hs.append(X.register_probe_capture(blocks[L], mk_pre())[0])
        # pick_site returns (module, name); the module IS the block, and its forward hook
        # sees the residual stream leaving layer L -- the one place a steering edit lands.
        hs.append(X.pick_site(blocks[L])[0].register_forward_hook(mk_blk()))
    return hs, pre, blk


def fit_direction(A, y, sid):
    """The estimator build_override_direction.py uses, verbatim: within-sample centering,
    then difference in means. Returns (unit direction, sigma, ||d||)."""
    Ac = A.copy()
    for s in np.unique(sid):
        m = sid == s
        Ac[m] -= Ac[m].mean(0)
    d = Ac[y == 1].mean(0) - Ac[y == 0].mean(0)
    u = (d / (np.linalg.norm(d) + 1e-12)).astype(np.float32)
    # sigma on the UNCENTERED activations, matching the ITI convention used at build time
    return u, float((A @ u).std()), float(np.linalg.norm(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=f"{ROOT}/runs/override_slope.json",
                    help="the factorial whose rows and `fired` labels are replayed")
    ap.add_argument("--out", default=f"{ROOT}/runs/site_mismatch.json")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="probe")
    ap.add_argument("--n", type=int, default=24, help="base samples in the factorial")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap replayed rows")
    a = ap.parse_args()

    d = json.load(open(a.src))
    layers = d["layers"]
    # (sid, override, voice, action) -> fired. The stored rows are the ground truth for the
    # label; we only re-derive the ACTIVATIONS.
    fired = {(r["sid"], r["override"], r["voice"], r.get("action", "tool")): bool(r["fired"])
             for r in d["rows"]}
    print(f"[src] {len(d['rows'])} rows, {len(fired)} distinct cells, layers {layers}")

    pairs = OS.paired_samples(a.split, a.n)
    model, tok = X.load_model_and_tok(a.model, a.device)
    hs, pre_cap, blk_cap = attach_both(model, layers)

    rows_pre = {L: [] for L in layers}
    rows_blk = {L: [] for L in layers}
    ys, sids, misses = [], [], 0
    t0 = time.time()
    todo = [(orig, prm, o, v, act) for (o, v, act) in OS.FRAMINGS for (orig, prm) in pairs]
    if a.limit:
        todo = todo[:a.limit]
    for k, (orig, prm, o, v, act) in enumerate(todo):
        s = OS.variant(orig, prm, o, v, act)
        sid = orig["id"]
        key = (sid, o, v, act)
        if key not in fired:
            misses += 1
            continue
        try:
            ids, pay, inj = X.injection_span(tok, s)
        except Exception:
            misses += 1
            continue
        if not inj:
            misses += 1
            continue
        pre_cap.clear()
        blk_cap.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=model.device))
        for L in layers:
            rows_pre[L].append(pre_cap[L][0, inj].float().mean(0).cpu().numpy())
            rows_blk[L].append(blk_cap[L][0, inj].float().mean(0).cpu().numpy())
        ys.append(1.0 if fired[key] else 0.0)
        sids.append(sid)
        if (k + 1) % 96 == 0:
            el = time.time() - t0
            print(f"[cap] {k+1}/{len(todo)}  {el:.0f}s  "
                  f"eta {el/(k+1)*(len(todo)-k-1):.0f}s", flush=True)
    for h in hs:
        h.remove()
    print(f"[cap] {len(ys)} rows captured, {misses} skipped "
          f"(no stored label / injection not locatable)")

    y = np.array(ys)
    sid = np.array(sids)
    if y.std() == 0:
        raise SystemExit("every replayed row has the same `fired` label -- nothing to fit")

    out = {"src": a.src, "n_rows": len(ys), "n_skipped": misses,
           "layers": layers, "n_fired": int(y.sum()), "per_layer": {}}
    print(f"\n{'layer':<7}{'cos(pre,blk)':>14}{'sigma_pre':>12}{'sigma_blk':>12}"
          f"{'sig_blk/pre':>13}{'|d|_pre':>11}{'|d|_blk':>11}{'stored_sigma':>14}")
    for L in layers:
        Ap = np.stack(rows_pre[L]).astype(np.float32)
        Ab = np.stack(rows_blk[L]).astype(np.float32)
        up, sp, np_ = fit_direction(Ap, y, sid)
        ub, sb, nb = fit_direction(Ab, y, sid)
        c = float(up @ ub)
        stored = None
        pk = f"{ROOT}/runs/gpt-oss-20b-userabl/probe_L{L}.pkl"
        if os.path.exists(pk):
            P = X.load_probe(pk)
            stored = float(P.get("sigmas", {}).get("dim_no_override_both", 0.0)) or None
            # sanity: the replayed pre-MLP fit must reproduce the shipped direction
            if "dim_no_override_both" in P["dirs"]:
                sd = np.asarray(P["dirs"]["dim_no_override_both"], np.float32)
                sd = sd / (np.linalg.norm(sd) + 1e-12)
                out.setdefault("repro_cos_vs_shipped", {})[str(L)] = float(-up @ sd)
        out["per_layer"][str(L)] = {
            "cos_pre_vs_block": c, "sigma_pre": sp, "sigma_block": sb,
            "sigma_ratio_block_over_pre": sb / max(sp, 1e-9),
            "norm_pre": np_, "norm_block": nb, "stored_sigma": stored}
        print(f"L{L:<6}{c:>14.4f}{sp:>12.2f}{sb:>12.2f}"
              f"{sb/max(sp,1e-9):>13.3f}{np_:>11.2f}{nb:>11.2f}"
              f"{(stored if stored else float('nan')):>14.2f}")

    if "repro_cos_vs_shipped" in out:
        print("\nSANITY -- the replayed PRE-MLP fit vs the SHIPPED dim_no_override_both "
              "(sign-corrected; should be ~1.0, anything else means the replay diverged "
              "from the original capture and the table above is not comparable):")
        for L, c in out["repro_cos_vs_shipped"].items():
            print(f"  L{L:<4} cos = {c:+.4f}")

    # build the string, THEN write, THEN replace -- json.dump streams into the handle and a
    # serialisation error partway through leaves a truncated file with a fresh mtime.
    blob = json.dumps(out, indent=1)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        f.write(blob)
    os.replace(tmp, a.out)
    json.load(open(a.out))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
