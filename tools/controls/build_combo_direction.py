"""Compose the override direction with the paper-exact role axis, as ONE stored vector.

WHY. The deployed cell's residual failures on the prose-medium param corpus are the most
user-like framings (defended goal 0.692 for user-role templates, p=0.006 — FINDINGS 10f),
and the role axis alone is independently defensive (probe_axis_tool@4: goal 0.013 at n=96,
FINDINGS 10d). The two directions carry different information (override framing vs role
reading; the probe axis contains no attack data). This composes them.

MECHANICS. The sweep applies one direction per arm with step (alpha/sqrt(k)) * sigma_L *
u_L. To make a composed arm reproduce EXACTLY the sum of the two component edits at their
chosen strengths (a1, a2) — i.e. step_L = (sigma_L/sqrt(k)) * (a1*u1 + a2*u2) — store:

    u_combo_L     = unit(a1*u1_L + a2*u2_L)
    sigma_combo_L = sigma_dno_L * ||a1*u1_L + a2*u2_L||

and run the arm at --alphas 1.0 WITHOUT --match-sigma-to (own stored sigma). sigma_dno is
`sigmas["dim_no_override"]`, the deployed cell's magnitude convention, so a1=8 alone
reproduces the deployed arm byte-for-byte modulo norm-preserve being applied once to the
summed edit instead of once per component.

Writes dirs[f"combo_ovr{a1}_pat{a2}"] (+ sigma + provenance) into --run's pickles at the
steer layers. Never run while a sweep is in flight (build_dirs reloads pickles per arm).

Usage:
  .venv/bin/python tools/controls/build_combo_direction.py \
      [--run runs/gpt-oss-20b-userabl] [--layers 12,16,20] \
      [--ratios 8:4,8:2] [--dry-run]
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import xpia_defense as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/gpt-oss-20b-userabl")
    ap.add_argument("--layers", default="12,16,20")
    ap.add_argument("--ratios", default="8:4,8:2",
                    help="comma list of a1:a2 = override:probe_axis_tool strengths")
    ap.add_argument("--override-key", default="dim_no_override_both",
                    help="which override direction to compose (e.g. dim_no_override_deleg "
                         "from the 2026-08-28 delegation refit)")
    ap.add_argument("--combo-suffix", default="",
                    help="appended to the stored combo key so a refit's composition "
                         "cannot overwrite the locked combo_ovr8_pat1")
    ap.add_argument("--sigma-ref", default="dim_no_override",
                    help="which stored sigma anchors the baked combo sigma. The gpt-oss "
                         "pkls carry the single-axis `dim_no_override`; runs whose fit "
                         "only stored the _both variant (e.g. qwen3-30b-thinking) pass "
                         "dim_no_override_both.")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    ratios = [tuple(float(x) for x in r.split(":")) for r in args.ratios.split(",")]
    for L in [int(x) for x in args.layers.split(",")]:
        path = f"{args.run}/probe_L{L}.pkl"
        p = X.load_probe(path)
        # COPY, DO NOT VIEW. `np.asarray(x, dtype=np.float32)` returns x ITSELF when x is
        # already a float32 ndarray -- which every stored direction is -- so the `/=` below
        # was normalising the direction IN PLACE inside `p["dirs"]`, and the pickle was then
        # rewritten with it. Measured 2026-09-02: every run that has ever had a combo built
        # (gpt-oss-20b-userabl, qwen3-30b-thinking) stores `dim_no_override_both` at norm
        # 1.0, while every run that has not (gemma4-31b-it 2.38/7.31/3.90, phi3-medium-128k,
        # qwen3next-80b) still holds the raw fit norm. `probe_axis_tool` was mutated the same
        # way.
        # BLAST RADIUS: NIL for behaviour -- `src/probes.build_dirs` unit-normalises on load
        # and `sigmas` are stored separately and were never touched, so no steered arm ever
        # ran differently and no published number moves. The exposure is ANALYSIS: any script
        # that reads a direction's NORM from a pickle gets 1.0 on the mutated runs and the
        # true norm elsewhere, silently incomparable across models. (FINDINGS 23m.4 is
        # unaffected -- sigma_decomp.py recomputes the norm from the capture, not the pickle.)
        u1 = np.array(p["dirs"][args.override_key], dtype=np.float32, copy=True)
        u1 /= np.linalg.norm(u1)
        u2 = np.array(p["dirs"]["probe_axis_tool"], dtype=np.float32, copy=True)
        u2 /= np.linalg.norm(u2)
        s_dno = float(p["sigmas"][args.sigma_ref])
        cos12 = float(u1 @ u2)
        for a1, a2 in ratios:
            v = a1 * u1 + a2 * u2
            nv = float(np.linalg.norm(v))
            key = f"combo_ovr{a1:g}_pat{a2:g}{args.combo_suffix}"
            p.setdefault("dirs", {})[key] = (v / nv).astype(np.float32)
            p.setdefault("sigmas", {})[key] = s_dno * nv
            p.setdefault("combo_provenance", {})[key] = {
                "components": [args.override_key, "probe_axis_tool"],
                "strengths": [a1, a2], "sigma_ref": args.sigma_ref,
                "cos_components": cos12, "date": "2026-08-26",
                "run_at": "--alphas 1.0, own sigma (no --match-sigma-to)"}
            print(f"[combo] L{L} {key}: cos(u1,u2)={cos12:+.3f} ||v||={nv:.3f} "
                  f"sigma={s_dno * nv:.1f}", flush=True)
        if args.dry_run:
            continue
        blob = pickle.dumps(p)
        with open(path + ".tmp", "wb") as f:
            f.write(blob)
        with open(path + ".tmp", "rb") as f:
            pickle.load(f)
        os.replace(path + ".tmp", path)
    print(f"[combo] {'DRY RUN, nothing written' if args.dry_run else 'done'}", flush=True)


if __name__ == "__main__":
    main()
