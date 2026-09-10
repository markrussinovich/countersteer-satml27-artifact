#!/usr/bin/env python
"""Remove direction keys that are bit-identical duplicates of another key under a MISLEADING name.

WHY (adversarial review 2026-09-02). `build_override_direction.py --centre-action --balanced`
wrote `dim_no_override_{tool,param,bal}_ac` (and the `dim_override_*` polarities) that are
`np.array_equal` to the un-suffixed keys with identical sigmas -- because `action` is constant
inside each per-action subset, so adding it to the centring key is inert. The maths is right;
the NAME is a trap. A key called `dim_no_override_bal_ac` invites someone to run it as "the
action-centred balanced direction" and report a NO-OP arm as a defense. That is the FINDINGS
§23e failure class (an arm that changes nothing while labelling itself a defense) reached by a
different route. The writer is fixed; this removes what it already wrote.

SAFETY. Refuses to remove a key that is NOT a bit-identical duplicate of its stated twin --
so it can never delete a real direction. Atomic write (`.tmp` -> read-back -> `os.replace`),
and every `--protect` key is asserted byte-identical afterwards.

Usage:
  purge_noop_keys.py --run runs/phi3-medium-128k --layers 8,12,16 \
      --pairs dim_no_override_bal_ac:dim_no_override_bal,... [--protect dim_no_override_both] [--dry-run]
  (with no --pairs, the default set below is used)
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import xpia_defense as X  # noqa: E402

DEFAULT_PAIRS = [("dim_no_override_bal_ac", "dim_no_override_bal"),
                 ("dim_override_bal_ac", "dim_override_bal"),
                 ("dim_no_override_tool_ac", "dim_no_override_tool"),
                 ("dim_no_override_param_ac", "dim_no_override_param")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--protect", default="dim_no_override_both,dim_no_override_ac")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    a = ap.parse_args()
    pairs = ([tuple(x.split(":")) for x in a.pairs.split(",")] if a.pairs else DEFAULT_PAIRS)
    protect = [k for k in a.protect.split(",") if k]

    for L in [int(x) for x in a.layers.split(",")]:
        path = f"{a.run}/probe_L{L}.pkl"
        p = X.load_probe(path)
        before = {k: np.asarray(p["dirs"][k]).tobytes() for k in protect if k in p["dirs"]}
        removed = []
        for dup, twin in pairs:
            if dup not in p.get("dirs", {}):
                continue
            if twin not in p["dirs"]:
                print(f"  L{L}: `{dup}` present but twin `{twin}` absent -- REFUSING")
                continue
            same = np.array_equal(np.asarray(p["dirs"][dup]), np.asarray(p["dirs"][twin]))
            sig_same = (p.get("sigmas", {}).get(dup) == p.get("sigmas", {}).get(twin))
            if not (same and sig_same):
                print(f"  L{L}: `{dup}` is NOT a bit-identical duplicate of `{twin}` "
                      f"(vec_same={same} sigma_same={sig_same}) -- REFUSING, this is a real "
                      f"direction")
                continue
            del p["dirs"][dup]
            p.get("sigmas", {}).pop(dup, None)
            removed.append(dup)
        after = {k: np.asarray(p["dirs"][k]).tobytes() for k in protect if k in p["dirs"]}
        bad = [k for k in before if before[k] != after[k]]
        if bad:
            raise SystemExit(f"L{L}: protected key(s) {bad} changed -- aborting")
        print(f"  L{L}: removed {removed or '(nothing)'}; protected {list(before)} intact")
        if a.dry_run or not removed:
            continue
        blob = pickle.dumps(p)
        with open(path + ".tmp", "wb") as f:
            f.write(blob)
        with open(path + ".tmp", "rb") as f:
            pickle.load(f)
        os.replace(path + ".tmp", path)
    print("[purge] " + ("DRY RUN, nothing written" if a.dry_run else "done"))


if __name__ == "__main__":
    main()
