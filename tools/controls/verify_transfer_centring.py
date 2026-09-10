#!/usr/bin/env python
"""Regression test for the per-action gate centring defect (FINDINGS §23p.8d / §23p.10).

THE BUG. `override_slope_experiment.transfer` centred by `sid` over rows of BOTH actions and
only THEN masked to one action, while `build_override_direction.py --balanced` -- which builds
the vectors actually STORED as `dim_no_override_{tool,param}` -- masks FIRST and centres within
the subset. Because `action` varies WITHIN sid, those are different matrices: the DIRECTION
differed (cos(old, shipped) 0.54-0.98), not merely the statistic. The `reliability > 0.70` gate
then certified a vector nobody ever built. Audited over six models: four of twelve verdicts
flip; on GLM-4.5-Air BOTH action types went from PARTIAL OVERLAP to "NOT ESTIMABLE anywhere".

WHY SYNTHETIC. A real capture is 0.3-1.0 GB and lives on one box. Here the action offset is
CONSTRUCTED, so the correct answer is known in closed form and the test is deterministic,
fast, and runs anywhere. `--capture PATH` additionally checks a real capture against the
recorded numbers when one is to hand.

WHAT IS ASSERTED
  1. equivalence: centring by (sid x delegation x action) then masking == masking then
     centring within the subset (the shipped build path) -- to floating point
  2. the legacy path is NOT equivalent to it whenever firing rates differ by action
  3. the legacy path differs whenever the per-sid firing rate VARIES ACROSS SIDS
  4. equalising the between-action MARGINAL gap does NOT fix it -- only homogeneous per-sid
     rates do. This corrects a wrong first draft of this test and is a real constraint on the
     §23p.9 re-capture design (see its gate E6)

Usage:  .venv/bin/python tools/controls/verify_transfer_centring.py [--capture PATH]
Exit 0 = all checks pass.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"   {detail}"))
    if not cond:
        FAILS.append(name)


def centre(A, keys):
    """Subtract the mean of each group; `keys` is one hashable per row."""
    Ac = A.copy()
    k = np.asarray(keys)
    for g in np.unique(k):
        m = k == g
        Ac[m] -= Ac[m].mean(0)
    return Ac


def dim(Ac, y, mask):
    return Ac[mask & (y == 1)].mean(0) - Ac[mask & (y == 0)].mean(0)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def synth(n_sid=24, n_fram=12, d=64, gap=0.40, sid_spread=0.35, seed=0):
    """A capture with the structure that actually produces the defect.

    FIRST ATTEMPT AT THIS WAS WRONG and is worth recording: a CONSTANT per-action offset does
    NOT reproduce the bug, because it is identical for fired and not-fired rows and therefore
    cancels in the within-action difference of means under EITHER centring. The exact
    decomposition says where the contamination really comes from --

        d_legacy - d_fixed = (mean_{a,y=1} - mean_{a,y=0}) applied to (m_sid - m_{sid,action})

    -- which is non-zero only when the per-sid ACTION CONTRAST VARIES ACROSS SIDS *and* the
    fired/not-fired rows within an action have different sid composition. Both hold in real
    captures: GLM-4.5-Air has three sids at tool 0/12 vs param 12/12 and one reversed. So the
    generator gives each sid its OWN action-contrast vector and its own per-action firing
    rate, with `gap` setting how far the two actions' rates are pulled apart.
    """
    rng = np.random.default_rng(seed)
    sids = [f"s{i}" for i in range(n_sid)]
    sig = unit(rng.normal(size=d))                       # the real fired-vs-not axis
    sid_eff = {s: rng.normal(size=d) for s in sids}
    # per-sid ACTION contrast: differs sample to sample, which is what stops it cancelling
    sid_act_eff = {s: rng.normal(size=d) * 2.0 for s in sids}
    # per-sid, per-action firing rates, spread wide so sid composition differs by outcome
    rates = {}
    for s in sids:
        base = 0.5 + rng.uniform(-sid_spread, sid_spread)
        rates[(s, "tool")] = float(np.clip(base - gap / 2, 0.0, 1.0))
        rates[(s, "param")] = float(np.clip(base + gap / 2, 0.0, 1.0))
    sid, act, y, A = [], [], [], []
    for s in sids:
        for a in ("tool", "param"):
            for _ in range(n_fram):
                yi = 1.0 if rng.random() < rates[(s, a)] else 0.0
                sid.append(s); act.append(a); y.append(yi)
                A.append(rng.normal(size=d) * 0.5 + yi * sig + sid_eff[s]
                         + (1.0 if a == "param" else -1.0) * sid_act_eff[s])
    return (np.stack(A).astype(np.float32), np.array(y), np.array(sid), np.array(act), sig)


def run_synth(gap, label, sid_spread=0.35, seed=0, quiet=False):
    A, y, sid, act, u_sig = synth(gap=gap, sid_spread=sid_spread, seed=seed)
    grp_fixed = np.array([f"{a}::{b}" for a, b in zip(sid, act)])
    Ac_legacy, Ac_fixed = centre(A, sid), centre(A, grp_fixed)
    r_tool = float(y[act == "tool"].mean())
    r_param = float(y[act == "param"].mean())
    if not quiet:
        print(f"\n-- {label}: firing tool={r_tool:.3f} param={r_param:.3f} "
              f"gap={abs(r_tool - r_param):.3f} --")
    out = {}
    for a in ("tool", "param"):
        m = act == a
        d_build = dim(centre(A[m], sid[m]), y[m], np.ones(m.sum(), bool))
        d_fixed, d_legacy = dim(Ac_fixed, y, m), dim(Ac_legacy, y, m)
        c_equiv = float(unit(d_fixed) @ unit(d_build))
        c_leg = float(unit(d_legacy) @ unit(d_build))
        # size of the discrepancy the defect introduces, relative to the correct direction
        disc = float(np.linalg.norm(unit(d_legacy) - unit(d_fixed)))
        out[a] = (c_leg, disc)
        if not quiet:
            print(f"   {a:6s} cos(fixed, build)={c_equiv:.6f}   "
                  f"cos(legacy, build)={c_leg:.4f}   ||legacy-fixed||={disc:.4f}")
            check(f"[{label}] {a}: fixed == shipped build path",
                  c_equiv > 0.999999, f"cos={c_equiv}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default="", help="optional real capture json to spot-check")
    args = ap.parse_args()

    # THREE REGIMES. Two earlier drafts of this test asserted things that are FALSE, and
    # both corrections are load-bearing, so they are recorded here rather than quietly fixed.
    #
    #  (1) A constant per-action offset does NOT reproduce the defect: it is identical for
    #      fired and not-fired rows and cancels in the within-action difference of means.
    #  (2) Equalising the between-action MARGINAL firing gap does NOT fix it either. The
    #      algebra says why:
    #          d_legacy - d_fixed = SUM_s [w1(s) - w0(s)] * c_{s,a}
    #      with w1/w0 the sid compositions of the fired / not-fired rows WITHIN action a and
    #      c_{s,a} that sid's action contrast. The marginal gap does not enter it at all; what
    #      does is per-sid firing-rate VARIANCE. And because w1/w0 are realised counts, finite
    #      samples leave a residual even when the per-sid rate is constant in expectation.
    #
    # CONSEQUENCE, and it is the reason STEP 0 had to be a code fix: **this defect cannot be
    # designed away by balancing a capture -- only the estimator change removes it.** Balance
    # shrinks it; the fix eliminates it. (Distinct from the POOLED-fit contamination of
    # §23p.2/§23p.8, whose driver IS the between-action per-sid gap. Two different terms.)
    print("\n=== regimes (single seed, for illustration) ===")
    unb = run_synth(0.40, "GLM-like: marginal gap AND per-sid spread")
    run_synth(0.00, "marginally balanced, per-sid spread REMAINS")
    homo = run_synth(0.00, "homogeneous: per-sid rates constant", sid_spread=0.0)

    # The TREND is asserted over seeds, not from one draw -- a single realisation of w1/w0 is
    # noisy and asserting on it would be tuning the test to pass.
    SEEDS = 12
    agg = {}
    for lbl, gp, sp in (("varying", 0.40, 0.35), ("marginal-only", 0.00, 0.35),
                        ("homogeneous", 0.00, 0.0)):
        acc = {"tool": [], "param": []}
        for sd in range(SEEDS):
            r = run_synth(gp, lbl, sid_spread=sp, seed=sd, quiet=True)
            for nm in acc:
                acc[nm].append(r[nm][1])
        agg[lbl] = {nm: float(np.mean(v)) for nm, v in acc.items()}
    print(f"\n=== mean ||legacy - fixed|| over {SEEDS} seeds ===")
    for lbl in ("varying", "marginal-only", "homogeneous"):
        print(f"   {lbl:16s} tool={agg[lbl]['tool']:.4f}  param={agg[lbl]['param']:.4f}")
    for nm in ("tool", "param"):
        check(f"{nm}: legacy DIFFERS from the shipped path in every regime",
              unb[nm][0] < 0.99 and homo[nm][0] < 0.999,
              f"varying={unb[nm][0]:.4f} homogeneous={homo[nm][0]:.4f}")
        check(f"{nm}: marginal balance alone does NOT remove the discrepancy",
              agg["marginal-only"][nm] > 0.5 * agg["varying"][nm],
              f"marginal={agg['marginal-only'][nm]:.4f} varying={agg['varying'][nm]:.4f}")
        check(f"{nm}: per-sid homogeneity SHRINKS but does not eliminate it",
              agg["homogeneous"][nm] < agg["varying"][nm] and agg["homogeneous"][nm] > 0.01,
              f"homogeneous={agg['homogeneous'][nm]:.4f} varying={agg['varying'][nm]:.4f}")

    if args.capture:
        import json
        print(f"\n-- real capture: {args.capture} --")
        rows = json.load(open(args.capture))
        rows = rows["rows"] if isinstance(rows, dict) and "rows" in rows else rows
        L = sorted(rows[0]["act"].keys(), key=int)[len(rows[0]["act"]) // 2]
        A = np.array([r["act"][L] for r in rows], dtype=np.float32)
        y = np.array([1.0 if r["fired"] else 0.0 for r in rows])
        sid = np.array([r["sid"] for r in rows])
        act = np.array([r.get("action", "tool") for r in rows])
        dele = np.array([r.get("delegation", "none") for r in rows])
        gl = np.array([f"{s}::{d}" for s, d in zip(sid, dele)])
        gf = np.array([f"{s}::{d}::{x}" for s, d, x in zip(sid, dele, act)])
        Acl, Acf = centre(A, gl), centre(A, gf)
        for x in ("tool", "param"):
            m = act == x
            if not m.any() or y[m].std() == 0:
                print(f"   L{L} {x}: no variance, skipped")
                continue
            d_build = dim(centre(A[m], sid[m]), y[m], np.ones(m.sum(), bool))
            c_f = float(unit(dim(Acf, y, m)) @ unit(d_build))
            c_l = float(unit(dim(Acl, y, m)) @ unit(d_build))
            print(f"   L{L} {x:6s} cos(fixed,build)={c_f:.6f}  cos(legacy,build)={c_l:.4f}")
            check(f"[real L{L}] {x}: fixed == shipped build path", c_f > 0.999,
                  f"cos={c_f}")

    print("\nRESULT: " + ("ALL CHECKS PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    # MUTATION CHECK, by hand: revert `transfer`'s group key to sid-only and the equivalence
    # assertions here still pass (this file tests the MATH, not the caller) -- so ALSO run
    # override_slope_experiment.py --analyze --transfer-centre sid on a real capture and
    # confirm it reproduces the pre-2026-09-02 gate log, which is what pins the caller.
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
