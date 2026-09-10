#!/usr/bin/env python
"""Merge direction vectors from an .npz into a run's probe pickles, atomically.

WHY. Analyses that FIT a candidate direction should not also WRITE it — a fitting script that
mutates the pickles a live sweep reads is a torn read waiting to happen, and
`build_combo_direction.py` has already silently mutated its source arrays once
(FINDINGS §23p.6b). So fits emit an .npz and this does the merge, separately and on purpose.

The .npz must hold `L<layer>` (the direction, in the polarity you want stored) and
`sigma_L<layer>` (its scale, computed by the fit against the SAME capture the other stored
sigmas use). A missing sigma is refused rather than defaulted to 0.0, because a zero sigma is
a SILENT NO-OP under `--scale sigma` (FINDINGS §23p.6a).

Writes via `.tmp` + read-back + `os.replace`, and refuses to run if any named key already
exists unless `--overwrite` is given. Prints, for every layer, proof that the pre-existing
deployed keys are byte-identical afterwards.

Usage:
  merge_npz_direction.py --run runs/glm45-air --npz path/to.npz --key dim_no_override_actioncentred
                         [--layers 20,24,28] [--protect dim_no_override_both] [--dry-run]
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--protect", default="dim_no_override_both",
                    help="comma list of keys asserted byte-identical after the merge")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    a = ap.parse_args()

    z = np.load(a.npz)
    layers = ([int(x) for x in a.layers.split(",")] if a.layers
              else sorted(int(k[1:]) for k in z.files if k.startswith("L") and k[1:].isdigit()))
    protect = [k for k in a.protect.split(",") if k]

    for L in layers:
        vk, sk = f"L{L}", f"sigma_L{L}"
        if vk not in z.files:
            raise SystemExit(f"{a.npz} has no `{vk}` (has {sorted(z.files)[:8]}...)")
        if sk not in z.files:
            raise SystemExit(
                f"{a.npz} has no `{sk}`. Refusing to store `{a.key}` at L{L} without a sigma: "
                f"under --scale sigma the step is alpha*sigma and a 0.0 would be a SILENT "
                f"NO-OP that still reports as a defense (FINDINGS 23p.6a).")
        v = np.asarray(z[vk], dtype=np.float32)
        s = float(z[sk])
        if not (s > 0 and np.isfinite(s)):
            raise SystemExit(f"{a.npz} `{sk}` = {s}; must be finite and > 0")
        if not np.isfinite(v).all():
            raise SystemExit(f"{a.npz} `{vk}` has non-finite entries")

        path = f"{a.run}/probe_L{L}.pkl"
        p = X.load_probe(path)
        if a.key in p.get("dirs", {}) and not a.overwrite:
            raise SystemExit(f"{path} already has `{a.key}`; pass --overwrite to replace it")
        before = {k: np.asarray(p["dirs"][k]).tobytes() for k in protect if k in p["dirs"]}

        p.setdefault("dirs", {})[a.key] = v            # copy already; np.asarray on z[] is new
        p.setdefault("sigmas", {})[a.key] = s
        p.setdefault("merge_provenance", {})[a.key] = {"npz": os.path.abspath(a.npz),
                                                       "layer": L, "sigma": s}
        after = {k: np.asarray(p["dirs"][k]).tobytes() for k in protect if k in p["dirs"]}
        bad = [k for k in before if before[k] != after[k]]
        if bad:
            raise SystemExit(f"L{L}: merging mutated protected key(s) {bad} -- aborting")

        print(f"[merge] L{L:<3d} {a.key}: |v|={np.linalg.norm(v):.4f} sigma={s:.4f}  "
              f"protected {list(before)} byte-identical", flush=True)
        if a.dry_run:
            continue
        blob = pickle.dumps(p)
        with open(path + ".tmp", "wb") as f:
            f.write(blob)
        with open(path + ".tmp", "rb") as f:
            pickle.load(f)                              # it is not written until it parses
        os.replace(path + ".tmp", path)
    print(f"[merge] {'DRY RUN, nothing written' if a.dry_run else 'done'}", flush=True)


if __name__ == "__main__":
    main()
