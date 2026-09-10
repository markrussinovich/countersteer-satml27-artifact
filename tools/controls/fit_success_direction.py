#!/usr/bin/env python
"""Fit a SUCCEEDED-vs-BLOCKED direction on injected tokens — and its decisive control.

THE IDEA (why this differs from the existing `inj_dim`). `inj_dim`
(xpia_defense.py:`train_injection_probe`) labels tokens by SPAN MEMBERSHIP — injected vs
legitimate — and its own report shows that is substantially a BOILERPLATE detector: a
bag-of-words null over token ids reaches 0.988 by-sample, because 988/1271 injections share
the literal "[PRIORITY MESSAGE]" prefix. It answers "is this injected?"

Conditioning on SUCCESS instead answers "is this the kind of injected that WORKS?" Both
classes are attacker prose, in a JSON field, carrying the same boilerplate — so the
boilerplate cancels in the difference of means. The probe split is a single attacker-
template family, so that cancellation is exact rather than approximate.

    d_inj = mean(h | injected tokens, attack BLOCKED) - mean(h | injected, SUCCEEDED)

Adding `d_inj` should push a working injection toward looking like one that fails.

THE DECISIVE CONTROL. The same contrast computed on the LEGITIMATE record tokens of the
same samples:

    d_legit = mean(h | legit tokens, BLOCKED) - mean(h | legit, SUCCEEDED)

If cos(d_inj, d_legit) is HIGH, the signal is a SAMPLE-LEVEL property -- "this record is
the vulnerable kind" -- not an injection-specific axis, and steering along it would push the
whole payload rather than the attack. This is not hypothetical: on the userness axis,
legitimate tokens separated succeeded from blocked at 10/12 layers while the
injection-specific term was at chance (7/12). See runs/attribution_control.json.

Gates before this direction may be used (CLAUDE.md: evidence, not intuition):
  * cos(d_inj, d_legit) must be LOW, else it is a sample-level confound;
  * a bag-of-words null on the same labels must NOT match the activation probe's AUC,
    else it is measuring the corpus (this is what sank `inj_dim`).

Usage:
    python tools/controls/fit_success_direction.py [PROBE_RUN] [MODEL] [DEVICE] [OUT_RUN]
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
PROBE_RUN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/gpt-oss-20b-paper"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-oss-20b"
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
OUT_RUN = sys.argv[4] if len(sys.argv) > 4 else f"{ROOT}/runs/gpt-oss-20b-userabl"
LABELS = f"{ROOT}/runs/probe_split_asr_labels.json"


def unit32(v):
    """Unit vector in fp32. NEVER fp64: projecting an fp32 matrix onto an fp64 vector
    upcasts the whole matrix -- measured at 45x on the probe stage."""
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    lab = json.load(open(LABELS))
    fired = {k: v["fired"] for k, v in lab["samples"].items()}
    tmpl = {k: v.get("template", "") for k, v in lab["samples"].items()}
    print(f"labels: {lab['n']} samples — {lab['n_succeeded']} succeeded / "
          f"{lab['n_blocked']} blocked / {lab.get('n_unscoreable', 0)} unscoreable")

    all_samples = X.build_dataset()
    bins = X.build_splits(all_samples, verbose=False)
    probe = [all_samples[i] for i in bins["probe"]]
    # DROP unscoreable: `fired is None` means no attacker-specific value exists to detect.
    # That is not evidence of blocking and must not enter the negative class.
    use = [s for s in probe if fired.get(s["id"]) is not None]
    print(f"using {len(use)} scoreable samples")

    import glob
    import re
    layers = sorted(int(re.search(r"probe_L(\d+)", f).group(1))
                    for f in glob.glob(f"{PROBE_RUN}/probe_L*.pkl"))
    model, tok = X.load_model_and_tok(MODEL, DEVICE)
    hs, cap = E.attach_capture(model, layers)

    # per-SAMPLE means, then averaged across samples within a class. A flat per-token mean
    # would weight long injections ~2x (injected spans run 65-140 tokens).
    acc = {L: {"inj": {True: [], False: []}, "legit": {True: [], False: []}} for L in layers}
    kept = []
    for s in use:
        ids, pay, inj = X.injection_span(tok, s)
        legit = [k for k in pay if k not in set(inj)]
        if not inj or not legit:
            continue
        cap.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=model.device))
        f = fired[s["id"]]
        for L in layers:
            h = cap[L][0]
            # index ON DEVICE before the D2H copy: the whole-sequence-then-slice pattern
            # moves 42 GB across 3000 copies for this stage; slicing first moves 3.1 GB in
            # 250. Values are identical -- slicing changes nothing numerically.
            acc[L]["inj"][f].append(h[inj].float().mean(0).cpu().numpy())
            acc[L]["legit"][f].append(h[legit].float().mean(0).cpu().numpy())
        kept.append(s["id"])
    for h in hs:
        h.remove()

    n_s = sum(1 for k in kept if fired[k])
    n_b = len(kept) - n_s
    print(f"captured {len(kept)} samples: {n_s} succeeded / {n_b} blocked")
    if n_s < 5 or n_b < 5:
        raise SystemExit(f"too few per class ({n_s}/{n_b}) to fit a direction")

    out = {"probe_run": PROBE_RUN, "n": len(kept), "n_succeeded": n_s, "n_blocked": n_b,
           "layers": layers, "rows": {}}
    print(f"\n{'layer':>6}{'cos(d_inj,d_legit)':>20}{'||d_inj||':>12}{'||d_legit||':>13}"
          f"{'sep_inj':>10}{'sep_legit':>11}")
    for L in layers:
        S = np.stack(acc[L]["inj"][True]); B = np.stack(acc[L]["inj"][False])
        Sl = np.stack(acc[L]["legit"][True]); Bl = np.stack(acc[L]["legit"][False])
        d_inj = B.mean(0) - S.mean(0)         # toward "blocked-like"
        d_leg = Bl.mean(0) - Sl.mean(0)
        c = float(unit32(d_inj) @ unit32(d_leg))
        # separation in units of pooled sd along each own axis (effect size, not p-value)
        def sep(A, C, d):
            u = unit32(d)
            a, b = A @ u, C @ u
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
            return float((b.mean() - a.mean()) / sd)
        si, sl = sep(S, B, d_inj), sep(Sl, Bl, d_leg)
        out["rows"][str(L)] = {"cos_inj_legit": c, "norm_inj": float(np.linalg.norm(d_inj)),
                               "norm_legit": float(np.linalg.norm(d_leg)),
                               "sep_inj_d": si, "sep_legit_d": sl,
                               "d_inj": d_inj.tolist()}
        print(f"{L:6d}{c:20.4f}{np.linalg.norm(d_inj):12.2f}"
              f"{np.linalg.norm(d_leg):13.2f}{si:10.2f}{sl:11.2f}")

    hi = max(abs(out["rows"][str(L)]["cos_inj_legit"]) for L in layers)
    print(f"\nmax |cos(d_inj, d_legit)| = {hi:.3f}")
    if hi > 0.5:
        print("=> HIGH: the succeeded-vs-blocked signal is largely SAMPLE-LEVEL, not\n"
              "   injection-specific. Steering along it would push the whole payload.\n"
              "   The premise of this direction is NOT supported.")
    else:
        print("=> LOW: the contrast is injection-specific, not a property of the record.\n"
              "   Proceed to the bag-of-words null and template-family holdout gates.")

    json.dump({k: v for k, v in out.items()},
              open(f"{ROOT}/runs/success_direction.json", "w"), indent=1)
    print(f"\nwrote {ROOT}/runs/success_direction.json")


if __name__ == "__main__":
    main()
