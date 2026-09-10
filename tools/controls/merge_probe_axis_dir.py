"""Merge the paper-exact role probe's own user-tool axis into a sweep run's pickles.

WHY. The steering sweep (`xpia_defense.py --stage sweep`) reads directions from the OUTDIR's
probe_L*.pkl via build_dirs. The role-confusion paper's most direct axis -- the multinomial
probe's w_user - w_tool -- is not stored as a direction by train_probes, so measuring its
BEHAVIOURAL effect (ASR / correctness, not probe readout) requires merging it in. Two keys:

    probe_axis_user  = +(w_user - w_tool)/||.||   (toward user; the ATTACK direction)
    probe_axis_tool  = -(w_user - w_tool)/||.||   (toward tool; the DEFENSE sign, per the
                                                   paper's premise that injections work by
                                                   reading user-like)

The axis is taken from --axis-run (default: the paper-exact refit) and merged into
--out-run's pickles, which may be a different fit -- the sweep needs its OTHER directions
(dim_no_override_both etc.) for a same-process comparison, and FINDINGS.md 9.6 forbids
cross-process arm comparison. Provenance is recorded under `probe_axis_provenance`.

SIGMAS are copied from a steer_probe_readout artifact (--sigma-from), which computed them as
the std of the projection over base-pass pre-MLP activations on held-out paper-corpus tool
spans. They exist only for that artifact's steer layers; every other layer gets 0.0. THE
SWEEP MUST THEREFORE PASS --match-sigma-to (the locked convention passes
--match-sigma-to dim_no_override anyway, which overrides stored sigmas for every arm).

Atomic writes (pickle to tmp, read back, os.replace) -- build_dirs reloads these pickles at
the start of every steered arm, so never run this while a sweep is in flight.

Usage:
  .venv/bin/python tools/controls/merge_probe_axis_dir.py \
      [--axis-run runs/gpt-oss-20b-paperexact] [--out-run runs/gpt-oss-20b-userabl] \
      [--sigma-from runs/steer_probe_readout_perp.json] [--dry-run]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import xpia_defense as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis-run", dest="axis_run", default="runs/gpt-oss-20b-paperexact")
    ap.add_argument("--out-run", dest="out_run", default="runs/gpt-oss-20b-userabl")
    ap.add_argument("--sigma-from", dest="sigma_from",
                    default="runs/steer_probe_readout_perp.json")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    layers = json.load(open(f"{args.out_run}/probe_report.json"))["layers"]

    sig_by_layer = {}
    if os.path.exists(args.sigma_from):
        art = json.load(open(args.sigma_from))
        sl = art["meta"]["steer_layers"]
        for c in art["cells"]:
            if c["direction"] == "probe_axis":
                sig_by_layer = dict(zip(sl, c["sigmas"]))
                break
    if not sig_by_layer:
        print(f"[merge] WARNING: no probe_axis sigmas in {args.sigma_from}; all sigmas "
              f"will be 0.0 and the sweep MUST pass --match-sigma-to", flush=True)

    for L in layers:
        pa = X.load_probe(f"{args.axis_run}/probe_L{L}.pkl")
        roles = pa["roles"]
        cls = list(pa["mn"].classes_)
        assert set(cls) == set(range(len(roles)))
        W = np.asarray(pa["mn"].coef_)[[cls.index(i) for i in range(len(roles))]]
        v = W[roles.index("user")] - W[roles.index("tool")]
        u = (v / np.linalg.norm(v)).astype(np.float32)

        path = f"{args.out_run}/probe_L{L}.pkl"
        p = X.load_probe(path)
        p.setdefault("dirs", {})["probe_axis_user"] = u
        p["dirs"]["probe_axis_tool"] = -u
        s = float(sig_by_layer.get(L, 0.0))
        p.setdefault("sigmas", {})["probe_axis_user"] = s
        p["sigmas"]["probe_axis_tool"] = s
        p["probe_axis_provenance"] = {
            "axis_run": args.axis_run, "date": "2026-08-25",
            "axis": "mn w_user - w_tool, class-order corrected, unit norm",
            "sigma_from": args.sigma_from if s else None}
        # how far the paper-exact axis sits from the out-run fit's own user-tool axis
        cls_o = list(p["mn"].classes_)
        Wo = np.asarray(p["mn"].coef_)[[cls_o.index(i)
                                        for i in range(len(p["roles"]))]]
        vo = Wo[p["roles"].index("user")] - Wo[p["roles"].index("tool")]
        cos = float(u @ (vo / np.linalg.norm(vo)))
        print(f"[merge] L{L}: sigma={s:.2f} cos(paper-exact axis, out-run's own axis)="
              f"{cos:+.4f}", flush=True)
        if args.dry_run:
            continue
        blob = pickle.dumps(p)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        with open(tmp, "rb") as f:
            pickle.load(f)
        os.replace(tmp, path)
    print(f"[merge] {'DRY RUN, nothing written' if args.dry_run else 'done'}", flush=True)


if __name__ == "__main__":
    main()
