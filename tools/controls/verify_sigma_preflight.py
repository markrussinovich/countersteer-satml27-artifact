#!/usr/bin/env python
"""Assert that a ZERO stored sigma is FATAL at preflight, not a silent no-op.

WHY THIS EXISTS. Under `--scale sigma` the per-token step is `alpha * sigma`. A direction
whose stored sigma is 0.0 therefore applies a ZERO edit while the arm still labels, logs and
scores itself as a defense -- exactly the FINDINGS §23e failure (a `mode=ablate` arm that
built no `Steer` and reported base-XPIA's numbers under a defended name). It is not a
hypothetical here: `tools/controls/merge_probe_axis_dir.py:65` writes

    sigmas["probe_axis_user"] = sigmas["probe_axis_tool"] = 0.0

with only a WARNING whenever no `steer_probe_readout` artifact is available, which is the
case for every model of the second bring-up wave -- GLM-4.5-Air, Qwen3-Next-80B and
Gemma-4-31B all carry 0.0 for both role axes today (checked below on whatever is on disk).

`src/steering.py: Steer._mk` DOES raise on `sigma <= 0` under `--scale sigma`, but it raises
from inside a forward hook -- i.e. only after the clean and base-XPIA arms have generated.
On a 106B model that is hours of an 8xH100 node spent to learn a fact readable from a pickle
in milliseconds. `src/probes.build_dirs(..., require_sigma=True)` moves the failure to
preflight; `src/cli.py` calls it for every steered direction and for the decode direction
BEFORE the clean arm runs.

WHAT THIS CHECKS (and it must be run as a MUTATION test -- see the footer):
  1. require_sigma=True RAISES on a direction whose sigma is 0.0
  2. require_sigma=True RAISES on a MISSING sigma (absent key -> 0.0 fallback)
  3. require_sigma=True PASSES when --match-sigma-to redirects to a direction that HAS one
     (this is the legitimate escape hatch the GLM composed cell will use)
  4. require_sigma=False (the default) still returns 0.0 silently -- so `--scale norm` runs
     and the config dump are untouched
  5. the error message names the direction, the layers, and the sigma source

Usage:  .venv/bin/python tools/controls/verify_sigma_preflight.py
Exit 0 = all checks pass.
"""
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.probes import build_dirs  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"   {detail}"))
    if not cond:
        FAILS.append(name)


def fake_run(tmp, layers=(0, 4)):
    """A minimal probe-pickle tree: one direction with a sigma, two without."""
    for L in layers:
        d = 8
        p = {"layer": L, "site": "test", "mean_norm": 10.0,
             "dirs": {"dim_no_override_both": np.ones(d, dtype=np.float32),
                      "probe_axis_user": np.eye(d, dtype=np.float32)[0],
                      "probe_axis_tool": -np.eye(d, dtype=np.float32)[0],
                      "no_sigma_at_all": np.eye(d, dtype=np.float32)[1]},
             # probe_axis_* stored as 0.0 -- byte-for-byte what merge_probe_axis_dir writes
             # when --sigma-from is absent; `no_sigma_at_all` has no key at all
             "sigmas": {"dim_no_override_both": 6.4161,
                        "probe_axis_user": 0.0, "probe_axis_tool": 0.0},
             "ablate_axes": {}}
        with open(f"{tmp}/probe_L{L}.pkl", "wb") as f:
            pickle.dump(p, f)
    return list(layers)


def main():
    tmp = tempfile.mkdtemp(prefix="sigma_preflight_")
    try:
        layers = fake_run(tmp)

        # 1. zero sigma is fatal
        try:
            build_dirs(tmp, layers, "probe_axis_tool", "cpu",
                       match_sigma_to="probe_axis_tool", require_sigma=True)
            check("zero sigma RAISES", False, "returned instead of raising")
            msg = ""
        except SystemExit as e:
            msg = str(e)
            check("zero sigma RAISES", True)

        # 5. the message is actionable
        check("message names the direction", "probe_axis_tool" in msg, msg[:120])
        check("message names the layers", "L0" in msg and "L4" in msg, msg[:120])
        check("message names the sigma source path", "probe_L*.pkl" in msg, msg[:120])
        check("message offers --match-sigma-to", "--match-sigma-to" in msg, msg[:120])

        # 2. a MISSING sigma key is fatal too (it falls back to 0.0)
        try:
            build_dirs(tmp, layers, "no_sigma_at_all", "cpu", require_sigma=True)
            check("missing sigma key RAISES", False, "returned instead of raising")
        except SystemExit:
            check("missing sigma key RAISES", True)

        # 3. the legitimate escape hatch still works
        try:
            _, sig, _ = build_dirs(tmp, layers, "probe_axis_tool", "cpu",
                                   match_sigma_to="dim_no_override_both",
                                   require_sigma=True)
            check("--match-sigma-to to a real sigma PASSES", all(s > 0 for s in sig),
                  f"sigmas={sig}")
        except SystemExit as e:
            check("--match-sigma-to to a real sigma PASSES", False, str(e)[:120])

        # 4. the default is unchanged -- --scale norm and the config dump must not break
        _, sig0, _ = build_dirs(tmp, layers, "probe_axis_tool", "cpu",
                                match_sigma_to="probe_axis_tool")
        check("require_sigma=False still returns 0.0 silently", sig0 == [0.0, 0.0],
              f"sigmas={sig0}")
        _, sigr, _ = build_dirs(tmp, layers, "dim_no_override_both", "cpu",
                                match_sigma_to="dim_no_override_both", require_sigma=True)
        check("a healthy direction is untouched",
              all(abs(s - 6.4161) < 1e-4 for s in sigr), f"sigmas={sigr}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # The live fleet, for the record: which real runs carry a 0.0 role-axis sigma today.
    print("\n  on-disk role-axis sigmas, ALL captured layers per run "
          "(zero = a silent no-op under --scale sigma unless --match-sigma-to redirects):")
    import glob
    import re
    sys.path.insert(0, os.path.join(ROOT, "tools", "controls"))
    import _probe_eval as E
    seen = {}
    for f in sorted(glob.glob(f"{ROOT}/runs/*/probe_L*.pkl")):
        tag = os.path.basename(os.path.dirname(f))
        try:
            p = E.X.load_probe(f)
        except Exception:
            continue
        s = p.get("sigmas", {})
        if "probe_axis_tool" not in s:
            continue
        L = int(re.search(r"L(\d+)", f).group(1))
        seen.setdefault(tag, []).append((L, float(s["probe_axis_tool"])))
    for tag, vals in sorted(seen.items()):
        zero = sorted(L for L, v in vals if not v)
        ok = sorted(L for L, v in vals if v)
        flag = "  <-- ALL ZERO" if not ok else ""
        print(f"    {tag:26s} zero at L{zero}  nonzero at L{ok}{flag}")

    print("\nRESULT: " + ("ALL CHECKS PASS" if not FAILS
                          else f"{len(FAILS)} FAILED: {FAILS}"))
    # MUTATION CHECK, run by hand when changing the guard: delete the `if require_sigma:`
    # block in src/probes.build_dirs and re-run -- this script must then fail checks 1, 2
    # and the four message checks and exit 1. A guard whose removal changes nothing is not
    # a guard.
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
