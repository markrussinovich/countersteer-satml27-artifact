#!/usr/bin/env python
"""Build the OVERRIDE direction as a steering direction and merge it into probe pickles.

PROVENANCE. From tools/controls/override_slope_experiment.py: a 4x3 factorial (override
intensity x voice), 12 framings x 24 samples, action/record/field/header constant within
sample. Override intensity is an 11x effect on ASR (none 0.056 -> supersede 0.625); voice
is secondary. The direction fit on within-sample-CENTERED injected-token activations is:

    split-half reliability  0.95-0.99   (gate >0.70; the succeeded-vs-blocked attempt: 0.30)
    held-out AUC            0.73 @ L16  (gate >0.65; held-out level = `firm`)

CAREFUL -- "held-out level = `firm`" is a fact about the GATE
(override_slope_experiment.py:238,404), NOT about this script. THIS SCRIPT APPLIES NO FRAMING
HOLDOUT: it fits over every override level, `firm` included. That is ordinary practice
(validate on a holdout, refit on all data), but it means the framing axis of CLAUDE.md's
three-axis bar is satisfied for the PROCEDURE and not for the VECTOR WE SHIP, and every
evaluation of a deployed cell holds out samples and attacker templates only. An earlier version
of this docstring said "`firm`, never fit on" inside the script that fits on `firm` -- true of
the gate, false here, and exactly the kind of sentence that becomes a paper claim by osmosis
(FINDINGS 23q.10).

This is the FIRST direction in this project to clear both gates. Every previous one either
was not estimable, or did not beat a magnitude-matched random control.

    d_override = mean(h | attack fired) - mean(h | not fired)     [within-sample centered]

Higher projection onto d_override => higher ASR, so a defense must REDUCE it. We therefore
emit `dim_no_override = -d_override`; ADDING that with --mode add steers toward "no override
claim". Both polarities are written so a sweep never has to reason about sign.

WITHIN-SAMPLE CENTERING is what makes this different from the succeeded-vs-blocked attempt:
subtracting each sample's own mean across framings removes sample identity exactly, which is
the confound that made that direction unusable (legitimate tokens separated as well as
injected ones).

Usage:
    python tools/controls/build_override_direction.py [OUT_RUN] [SRC_JSON]
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT
OUT_RUN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/gpt-oss-20b-userabl"
SRC = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/runs/override_slope.json"
# KEY SUFFIX so a refit cannot silently overwrite the direction behind the current best
# cell. `dim_no_override` was fit on disjoint-tool actions ONLY; the 3-factor refit adds
# parameter-abuse actions and writes `dim_no_override_both`, leaving the original intact so
# the two can be swept head-to-head. Overwriting would have destroyed BEST_DEFENSE.md's cell
# with no way to reproduce it.
SUF = ""
for _i, _a in enumerate(sys.argv):
    if _a == "--key-suffix" and _i + 1 < len(sys.argv):
        SUF = sys.argv[_i + 1]
# BALANCED ORTHOGONAL COMPOSITION. --balanced additionally writes per-ACTION-TYPE directions
# and an equal-weight orthogonal composition of the two.
BALANCED = "--balanced" in sys.argv
# --centre-action: THE CORRECTED FITTING RECIPE (owner directive 2026-09-02, FINDINGS
# §23p.8e). Centre by (sid x delegation x ACTION) instead of (sid x delegation). Because
# `action` varies WITHIN sid, the current key cannot remove the action main effect, and the
# residual is exactly a firing-rate-weighted action contrast -- verified as an identity:
#     d_pooled = d_actioncentred + w^T A
# On GLM-4.5-Air that term is 95% of the deployed direction's length.
#
# DEFAULT OFF, and it REQUIRES --key-suffix, deliberately. `dim_no_override_both` is the
# direction every published number in this project rests on; a rebuild that silently changed
# it would invalidate the coverage table without anyone noticing. With the flag on, EVERY key
# this run writes carries the suffix, so a corrected fit cannot collide with a shipped one.
CENTRE_ACTION = "--centre-action" in sys.argv
# BALANCED-KEY SUFFIX. Empty on the legacy path, so `--key-suffix _both --balanced` keeps
# writing `dim_no_override_{tool,param}` and `dim_no_override_bal` under exactly the names the
# shipped pickles already carry and `--directions` already references. Non-empty ONLY under
# --centre-action, where the keys must not collide with those.
BSUF = ""
if CENTRE_ACTION and not SUF:
    raise SystemExit(
        "--centre-action requires --key-suffix: it produces a DIFFERENT direction from the "
        "one every published number rests on, and must not be written under a shipped key. "
        "Suggested: --key-suffix _ac")
BSUF = SUF if CENTRE_ACTION else ""
# --dry-run computes and reports everything but writes NOTHING. The probe pickles are read by
# `build_dirs` at the start of every steered arm, so rewriting them while a sweep is running is
# a torn read waiting to happen -- and without --key-suffix this script OVERWRITES
# `dim_no_override`, the direction behind a shipped cell.
DRY = "--dry-run" in sys.argv
# --exclude-override LEVEL: withhold a FRAMING LEVEL in the fit itself (CLAUDE.md
# standard, owner-adopted 2026-09-04). Previously the framing axis was held out at the
# GATE only (override_slope_experiment.py fits train-only for the AUC gate) while the
# shipped vector was refit over ALL levels, `firm` included -- see the docstring caveat
# above. With this flag the rows of the named level are REMOVED BEFORE FITTING, so the
# held-out-level gate certifies the vector that ships, not a cousin. Requires
# --key-suffix for the same reason --centre-action does: it is a different vector from
# every shipped key. Sigma is computed over the FIT subset's activations (stated in the
# merge log).
EXCL_OVR = None
for _i, _a in enumerate(sys.argv):
    if _a == "--exclude-override" and _i + 1 < len(sys.argv):
        EXCL_OVR = sys.argv[_i + 1]
if EXCL_OVR and not SUF:
    raise SystemExit(
        "--exclude-override requires --key-suffix: the framing-held-out fit is a "
        "DIFFERENT direction from the shipped keys and must not overwrite them.")
# GUARD-HOLE CLOSED (adversarial review 2026-09-04): under `--exclude-override --balanced`
# WITHOUT --centre-action, BSUF stayed empty, so the per-action and balanced keys
# (dim_no_override_{tool,param}, dim_no_override_bal) were written UNSUFFIXED — a
# firm-excluded fit silently overwriting shipped keys, exactly what the suffix guard
# exists to prevent. Force the suffix onto the balanced keys whenever a framing level is
# excluded. (No shipped pickle was ever hit: every _ach fit ran with --centre-action,
# which skips the balanced keys entirely; verified byte-identical in the review.)
if EXCL_OVR:
    BSUF = SUF
# --delegated-only fits fired-vs-not on rows where the user DELEGATED (delegation != none).
# The pooled fit over all delegation levels is contaminated by construction (adversarial
# review 2026-08-28): delegated rows fire near ceiling, so the pooled difference-in-means
# absorbs "the user turn contains delegation text" -- a feature of the LEGITIMATE user
# message -- and steering its negation pushes toward "the user did not delegate".
DELEG_ONLY = "--delegated-only" in sys.argv


def main():
    d = json.load(open(SRC))
    rows, layers = d["rows"], d["layers"]
    if DELEG_ONLY:
        n0 = len(rows)
        rows = [r for r in rows if r.get("delegation", "none") != "none"]
        print(f"[--delegated-only] {n0} -> {len(rows)} rows (delegation != none)")
        if not rows:
            raise SystemExit("--delegated-only but the source has no delegation field")
    if EXCL_OVR:
        n0 = len(rows)
        levels = sorted({r.get("override") for r in rows})
        rows = [r for r in rows if r.get("override") != EXCL_OVR]
        print(f"[--exclude-override {EXCL_OVR}] {n0} -> {len(rows)} rows "
              f"(levels present: {levels})")
        if len(rows) == n0:
            raise SystemExit(
                f"--exclude-override {EXCL_OVR}: no rows carried that level; "
                f"levels present: {levels}")
        if not rows:
            raise SystemExit(f"--exclude-override {EXCL_OVR} removed every row")
    sids = sorted({r["sid"] for r in rows})
    has_deleg = any("delegation" in r for r in rows)
    print(f"{len(rows)} rows, {len(sids)} samples, layers {layers}"
          + f" | centering by (sid{' x delegation' if has_deleg else ''}"
          + (" x ACTION [--centre-action, CORRECTED recipe]" if CENTRE_ACTION else "")
          + ")" + (f" | key suffix {SUF!r}" if SUF else ""))

    report = {"source": SRC, "n_rows": len(rows), "layers": layers,
              "delegated_only": DELEG_ONLY, "exclude_override": EXCL_OVR,
              "centre_action": CENTRE_ACTION, "key_suffix": SUF, "per_layer": {}}
    merged_layers = 0
    for L in layers:
        A = np.array([r["act"][str(L)] for r in rows], dtype=np.float32)
        y = np.array([1.0 if r["fired"] else 0.0 for r in rows])
        sid = np.array([r["sid"] for r in rows])
        # within-GROUP centering: removes sample identity (the confound that sank the
        # succeeded-vs-blocked direction) AND, when the delegation factor exists, the
        # delegation-context main effect -- the user-turn text differs across delegation
        # levels within a sample, and plain sid centering leaves that feature in the fit
        # (adversarial review 2026-08-28). On pre-delegation artifacts every row maps to
        # the 'none' group, so this is byte-identical to the old sid centering.
        grp = np.array([f'{r["sid"]}::{r.get("delegation", "none")}'
                        + (f'::{r.get("action", "tool")}' if CENTRE_ACTION else "")
                        for r in rows])
        Ac = A.copy()
        for g_ in sorted(set(grp)):
            m = grp == g_
            Ac[m] -= Ac[m].mean(0)
        if y.std() == 0:
            continue
        merged_layers += 1
        d_ov = Ac[y == 1].mean(0) - Ac[y == 0].mean(0)
        # per-delegation-level agreement: if the levels' own directions disagree, the
        # pooled key is a norm-weighted average and must not be trusted blind
        if has_deleg:
            dg = np.array([r.get("delegation", "none") for r in rows])
            cos_lv = {}
            u_pool = d_ov / (np.linalg.norm(d_ov) + 1e-12)
            for g_ in sorted(set(dg)):
                m = dg == g_
                if y[m].std() == 0:
                    continue
                dl = Ac[m & (y == 1)].mean(0) - Ac[m & (y == 0)].mean(0)
                cos_lv[g_] = float(u_pool @ (dl / (np.linalg.norm(dl) + 1e-12)))
            print(f"  L{L} cos(pooled, per-level): "
                  + "  ".join(f"{k}={v:+.3f}" for k, v in cos_lv.items()))
            report["per_layer"].setdefault(str(L), {})["cos_per_delegation"] = cos_lv
        # sigma for --scale sigma: spread of the (uncentered) activations along the axis,
        # matching the ITI convention used elsewhere. fp32 to avoid the fp64 upcast trap.
        u = (d_ov / (np.linalg.norm(d_ov) + 1e-12)).astype(np.float32)
        sigma = float((A @ u).std())

        p = X.load_probe(f"{OUT_RUN}/probe_L{L}.pkl")

        if BALANCED:
            # WHY. The pooled fit is a difference in means over BOTH action types at once, so
            # it is a NORM-WEIGHTED average of two partially distinct axes -- and the tool
            # rows carry about twice the norm of the param rows. Measured:
            #     cos(tool-only fit, param-only fit) = 0.69 / 0.55 / 0.44 at L12/16/20
            #     the pooled direction sits at 0.86/0.86/0.76 to the tool half
            #                             and 0.77/0.70/0.69 to the param half
            # so parameter abuse is the under-served half by construction. MSRS
            # (arXiv:2508.10599) makes the same point generally: give each attribute its own
            # subspace instead of letting one average dominate.
            #
            # The composition is EQUAL-WEIGHT and ORTHOGONAL: take the tool axis, project the
            # param axis onto its orthogonal complement, renormalise both, and sum. Adding two
            # orthogonal unit steps of size `a` is exactly one step of size `a*sqrt(2)` along
            # the normalised sum, so this needs no pipeline change -- it is a direction key
            # like any other, and --match-sigma-to keeps it magnitude-comparable.
            # NO-OP TRAP, CLOSED (adversarial review 2026-09-02). Under --centre-action the
            # per-action fits and their balanced composition are MATHEMATICALLY UNCHANGED:
            # `action` is constant inside each per-action subset, so adding it to the centring
            # key is inert. Writing them anyway under an `_ac` suffix produced
            # `dim_no_override_bal_ac` etc. that were `np.array_equal` to the plain keys with
            # identical sigmas -- a name inviting someone to run a NO-OP arm and report it as
            # "the action-centred balanced direction". That is the §23e failure class wearing a
            # different hat. Under --centre-action we therefore write ONLY the pooled key,
            # which is the one the flag actually changes.
            # NOTE, and it is the more useful half: `dim_no_override_bal` is ALREADY
            # action-free by construction -- Gram-Schmidt over two within-action fits cannot
            # admit the between-action mean difference. Measured on Phi-3:
            # |cos(_bal, action-axis)| = 0.031-0.136 vs |cos(_ac, action-axis)| = 0.011-0.108,
            # and cos(_bal, _ac) = 0.92-0.94. If you want an action-stripped direction, `_bal`
            # already is one and may already have been evaluated.
            if CENTRE_ACTION:
                print(f"L{L:<3} [--centre-action] per-action and balanced keys SKIPPED: they "
                      f"are inert under action-centring (action is constant within each "
                      f"subset) and `dim_no_override_bal` is already action-free", flush=True)
            per = {}
            for act in (() if CENTRE_ACTION else ("tool", "param")):
                sel = np.array([r.get("action", "tool") == act for r in rows])
                if not sel.any():
                    continue
                # centre by the SAME key the pooled fit uses (minus `action`, which is
                # constant inside this subset by construction). Previously this centred by
                # `sid` alone while the pooled fit used `sid::delegation`, so on a capture
                # WITH a delegation factor the two halves were centred differently and the
                # per-action span statistic was partly a centring artifact (audit 2026-09-02).
                # Inert on every capture where delegation is `none` throughout.
                Aa, ya = A[sel], y[sel]
                sa = np.array([f'{r["sid"]}::{r.get("delegation", "none")}'
                               for r, k in zip(rows, sel) if k])
                Aac = Aa.copy()
                for s_ in np.unique(sa):
                    m_ = sa == s_
                    Aac[m_] -= Aac[m_].mean(0)
                if ya.std() == 0:
                    continue
                per[act] = Aac[ya == 1].mean(0) - Aac[ya == 0].mean(0)
                p["dirs"][f"dim_no_override_{act}" + BSUF] = -per[act]
                ua_ = (per[act] / (np.linalg.norm(per[act]) + 1e-12)).astype(np.float32)
                p["sigmas"][f"dim_no_override_{act}" + BSUF] = float((A @ ua_).std())
            if len(per) == 2:
                ut = per["tool"] / (np.linalg.norm(per["tool"]) + 1e-12)
                up = per["param"] / (np.linalg.norm(per["param"]) + 1e-12)
                up_orth = up - (up @ ut) * ut          # Gram-Schmidt
                nrm = np.linalg.norm(up_orth)
                if nrm > 1e-6:
                    up_orth = up_orth / nrm
                    bal = (ut + up_orth) / np.sqrt(2.0)
                    p["dirs"]["dim_override_bal" + BSUF] = bal
                    p["dirs"]["dim_no_override_bal" + BSUF] = -bal
                    ub_ = bal.astype(np.float32)
                    sb_ = float((A @ ub_).std())
                    p["sigmas"]["dim_override_bal" + BSUF] = sb_
                    p["sigmas"]["dim_no_override_bal" + BSUF] = sb_
                    print(f"L{L:<3} balanced: cos(tool,param)={float(ut @ up):+.3f}  "
                          f"cos(bal,pooled)={float(bal @ u):+.3f}  "
                          f"cos(bal,tool)={float(bal @ ut):+.3f}  "
                          f"cos(bal,param)={float(bal @ up):+.3f}  sigma={sb_:.2f}")

        p["dirs"][f"dim_override{SUF}"] = d_ov
        p["dirs"][f"dim_no_override{SUF}"] = -d_ov    # ADD this to steer toward safe
        p["sigmas"][f"dim_override{SUF}"] = sigma
        p["sigmas"][f"dim_no_override{SUF}"] = sigma
        p.setdefault("override_provenance", {})[f"dim_no_override{SUF}"] = {
            "source": SRC, "centered": "within-sample", "n_rows": len(rows),
            "note": "add to REDUCE override-ness; higher projection = higher ASR",
        }
        if not DRY:
            with open(f"{OUT_RUN}/probe_L{L}.pkl", "wb") as f:
                pickle.dump(p, f)

        def cos(a, b):
            a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        row = {"sigma": sigma, "norm": float(np.linalg.norm(d_ov)),
               "vs_mn_tool": cos(d_ov, p["dirs"]["mn_tool"]),
               "vs_dim_tool_vs_rest": cos(d_ov, p["dirs"]["dim_tool_vs_rest"])}
        if "dim_user_vs_rest" in p["dirs"]:
            row["vs_dim_user_vs_rest"] = cos(d_ov, p["dirs"]["dim_user_vs_rest"])
        old = f"{ROOT}/runs/gpt-oss-20b-resid/probe_L{L}.pkl"
        if os.path.exists(old):
            _o = X.load_probe(old)["dirs"]["mn_tool"]
            # cross-model diagnostic only: skip when hidden sizes differ (gpt-oss 2880
            # vs qwen 2048 crashed here, 2026-08-28)
            if np.asarray(_o).shape == np.asarray(d_ov).shape:
                row["vs_OLD_working_mn_tool"] = cos(d_ov, _o)
        # merge, don't assign: cos_per_delegation may already sit under this key
        report["per_layer"].setdefault(str(L), {}).update(row)
        print(f"L{L:<3} sigma={sigma:9.2f} |d|={row['norm']:9.2f}  " +
              "  ".join(f"{k.replace('vs_',''):<20}={v:+.3f}"
                        for k, v in row.items() if k.startswith("vs_")))

    if merged_layers == 0:
        # Previously this fell through to "merged dim_override... into probe_L*.pkl"
        # having merged NOTHING (zero variance in `fired` skips every layer) -- and the
        # first downstream consumer died on a missing direction key with a misleading
        # message. Found on Qwen3.8-27B, where the locked24 factorial fires 0/576.
        print(f"\n** NOTHING MERGED: `fired` has zero variance in {SRC} "
              f"({len(rows)} rows) -- no override direction is fittable on this "
              f"capture. No pickle touched, no report written. **")
        sys.exit(3)
    if DRY:
        print("\n[dry-run] nothing written -- no pickle touched, no report written")
        return
    # The report is keyed by the RUN it was fit for, not a global name: a Qwen rebuild
    # writing to `runs/override_direction_both.json` clobbered the gpt-oss artifact TWICE
    # (2026-08-27 and 2026-08-28, both restored via git checkout).
    run_tag = os.path.basename(os.path.normpath(OUT_RUN))
    dst = (f"{ROOT}/runs/override_direction{SUF}.json"
           if run_tag == "gpt-oss-20b-userabl"
           else f"{ROOT}/runs/override_direction_{run_tag}{SUF}.json")
    # build the string, write, then replace: json.dump streams into the handle and a
    # serialisation error partway through leaves a truncated file with a fresh mtime
    blob = json.dumps(report, indent=1)
    with open(dst + ".tmp", "w") as f:
        f.write(blob)
    os.replace(dst + ".tmp", dst)
    json.load(open(dst))
    print(f"\nmerged dim_override{SUF} / dim_no_override{SUF} into {OUT_RUN}/probe_L*.pkl")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
