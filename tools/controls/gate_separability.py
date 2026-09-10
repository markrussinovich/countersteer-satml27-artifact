#!/usr/bin/env python
"""Can a PER-TOKEN gate on the override axis separate injected from legitimate tokens?

This decides whether the gating branch is viable AT ALL, and it should have been run before
the gate was built. Gating at 0.5 sigma raised clean-payload correctness to ~the unattacked
baseline (the ceiling lifted, as designed) but ALSO weakened the defense (ASR 0.000 ->
0.200). Two incompatible explanations:

  A. injected tokens sit high on the axis and 0.5 sigma is simply above them
     => threshold tuning fixes it, gating works.
  B. injected and legitimate tokens OVERLAP on the axis
     => the direction acts diffusely across the whole span, not locally, and NO threshold
        can separate them. Gating is structurally dead and ~70% correctness is the real
        ceiling for uniform steering.

The test needs no generation: capture per-token activations on the poisoned prompt, project
each token onto the layer's `dim_override` unit vector in SIGMA units, and ask how well that
scalar separates injected tokens from the legitimate record tokens beside them.

  per-token AUC ~0.5  => B, gating is dead, stop tuning thresholds
  per-token AUC high  => A, and the reported threshold table says where to set the gate

Usage:
    python tools/controls/gate_separability.py [PROBE_RUN] [MODEL] [DEVICE] [N_DEV]
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
PROBE_RUN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/gpt-oss-20b-userabl"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-oss-20b"
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
N_DEV = int(sys.argv[4]) if len(sys.argv) > 4 else 24
THRESHOLDS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    v = np.concatenate([pos, neg])
    o = np.argsort(v)
    rk = np.empty(len(v)); rk[o] = np.arange(1, len(v) + 1)
    n1, n0 = len(pos), len(neg)
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    import glob
    import re
    layers = sorted(int(re.search(r"probe_L(\d+)", f).group(1))
                    for f in glob.glob(f"{PROBE_RUN}/probe_L*.pkl"))
    P = {L: X.load_probe(f"{PROBE_RUN}/probe_L{L}.pkl") for L in layers}
    if "dim_override" not in P[layers[0]]["dirs"]:
        raise SystemExit(f"no dim_override in {PROBE_RUN}; run build_override_direction.py")
    U = {L: (np.asarray(P[L]["dirs"]["dim_override"], np.float32) /
             (np.linalg.norm(P[L]["dirs"]["dim_override"]) + 1e-12)) for L in layers}
    SIG = {L: max(float(P[L]["sigmas"]["dim_override"]), 1e-6) for L in layers}

    all_s = X.build_dataset()
    bins = X.build_splits(all_s, n_eval=N_DEV)
    dev = [all_s[i] for i in bins["dev"]]
    dev = [s for s in dev if s.get("injection_text") and s.get("injection_field")]
    model, tok = X.load_model_and_tok(MODEL, DEVICE)
    hs, cap = E.attach_capture(model, layers)

    inj_p = {L: [] for L in layers}
    leg_p = {L: [] for L in layers}
    for s in dev:
        try:
            ids, pay, inj = X.injection_span(tok, s)
        except Exception:
            continue
        legit = [k for k in pay if k not in set(inj)]
        if not inj or not legit:
            continue
        cap.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=model.device))
        for L in layers:
            h = cap[L][0].float().cpu().numpy()
            u = U[L]
            inj_p[L].append((h[inj] @ u) / SIG[L])
            leg_p[L].append((h[legit] @ u) / SIG[L])
    for h_ in hs:
        h_.remove()

    out = {"probe_run": PROBE_RUN, "layers": layers, "thresholds": THRESHOLDS, "rows": {}}
    print(f"\nper-token projection onto dim_override, in SIGMA units "
          f"({len(inj_p[layers[0]])} samples)")
    print(f"{'layer':>6}{'AUC':>8}{'inj mean':>10}{'legit mean':>12}{'inj sd':>9}"
          f"{'legit sd':>10}{'n_inj':>8}{'n_leg':>8}")
    for L in layers:
        a = np.concatenate(inj_p[L]); b = np.concatenate(leg_p[L])
        A = auc(a, b)
        out["rows"][str(L)] = {"auc": A, "inj_mean": float(a.mean()),
                               "legit_mean": float(b.mean()), "inj_sd": float(a.std()),
                               "legit_sd": float(b.std()), "n_inj": int(a.size),
                               "n_legit": int(b.size),
                               "frac_inj_above": {str(t): float((a > t).mean())
                                                  for t in THRESHOLDS},
                               "frac_legit_above": {str(t): float((b > t).mean())
                                                    for t in THRESHOLDS}}
        print(f"{L:6d}{A:8.3f}{a.mean():10.2f}{b.mean():12.2f}{a.std():9.2f}"
              f"{b.std():10.2f}{a.size:8d}{b.size:8d}")

    best = max(layers, key=lambda L: out["rows"][str(L)]["auc"])
    print(f"\nbest separating layer: L{best}, per-token AUC "
          f"{out['rows'][str(best)]['auc']:.3f}")
    print(f"\n{'thresh(sigma)':>14}" + "".join(f"{f'L{L}':>16}" for L in layers if L in (12, 16, 20)))
    print(f"{'':>14}" + "".join(f"{'inj%/legit%':>16}" for L in layers if L in (12, 16, 20)))
    for t in THRESHOLDS:
        cells = []
        for L in layers:
            if L not in (12, 16, 20):
                continue
            r = out["rows"][str(L)]
            cells.append(f"{100*r['frac_inj_above'][str(t)]:5.0f}/"
                         f"{100*r['frac_legit_above'][str(t)]:<5.0f}")
        print(f"{t:14.2f}" + "".join(f"{c:>16}" for c in cells))

    # VIABILITY IS COVERAGE, NOT AUC. An earlier version called AUC 0.736 "viable" and was
    # wrong: a gate must keep MOST injected tokens above threshold while excluding most
    # legitimate ones. Here the best trade is ~45% injection coverage at 14% legitimate
    # (L12, -1.0 sigma), so any usable threshold leaves the majority of the injection
    # unsteered -- and the empirical sweep confirmed it (ASR 0.29-0.33 gated at every
    # threshold, vs 0.083 ungated). Judge on the coverage frontier.
    best_cov = 0.0
    for L in layers:
        r = out["rows"][str(L)]
        for t in THRESHOLDS:
            ci, cl = r["frac_inj_above"][str(t)], r["frac_legit_above"][str(t)]
            if cl <= 0.10:
                best_cov = max(best_cov, ci)
    out["best_injection_coverage_at_10pct_legit"] = best_cov
    print(f"\nbest injected-token coverage at <=10% legitimate-token coverage: "
          f"{100*best_cov:.0f}%")
    aucs = [out["rows"][str(L)]["auc"] for L in layers]
    print()
    if best_cov < 0.6:
        print(f"=> GATING NOT VIABLE. Per-token AUC reaches {max(aucs):.3f}, but no threshold\n"
              f"   covers more than {100*best_cov:.0f}% of injected tokens while sparing 90% of\n"
              "   legitimate ones. The direction acts through BROAD COVERAGE of the span,\n"
              "   not on identifiable tokens, so a gate necessarily removes most of the\n"
              "   defense. Uniform steering's correctness is the ceiling for this direction\n"
              "   alone; the route to the target is a LAYERED defense.")
    elif max(aucs) < 0.65:
        print("=> OVERLAP (B): per-token AUC is near chance at every layer. The override\n"
              "   direction acts DIFFUSELY across the span, not on identifiable tokens. NO\n"
              "   threshold can separate injected from legitimate tokens -- stop tuning the\n"
              "   gate. ~70% correctness is the real ceiling for uniform steering, and the\n"
              "   route to the target is a LAYERED defense, not a better gate.")
    else:
        print(f"=> SEPARABLE (A): per-token AUC {max(aucs):.3f} at L{best}. Gating is viable;\n"
              "   set --gate-proj where the inj%/legit% table keeps most injected tokens\n"
              "   above threshold while excluding most legitimate ones.")
    # UNITS. Everything above is a projection onto `dim_override`, where INJECTED tokens sit
    # HIGHER. `--gate-proj` consumes exactly these units. That was not true before 2026-08-04:
    # Steer._mk gated on the projection onto the STEERED direction (dim_no_override), which is
    # the negation, so a threshold read off this table selected the LEGITIMATE tokens and the
    # four archived --gate-proj runs measured an inverted gate. Fixed in xpia_defense.py; this
    # note exists so the two ends of the calibration stay pinned to one convention.
    print("\n[units] the table above, and therefore --gate-proj, are projections onto\n"
          "        dim_OVERRIDE in sigma units: HIGHER = more override-ness = more likely\n"
          "        injected. xpia_defense.Steer negates internally to match.")

    json.dump(out, open(f"{ROOT}/runs/gate_separability.json", "w"), indent=1)
    print(f"\nwrote {ROOT}/runs/gate_separability.json")


if __name__ == "__main__":
    main()
