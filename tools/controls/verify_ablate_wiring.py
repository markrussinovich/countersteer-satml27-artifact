#!/usr/bin/env python
"""Regression test for the STEER-CONSTRUCTION GUARD in src/arms.run_arm.

WHY THIS EXISTS. `--mode ablate` is dose-free: it removes the direction's whole component
(h <- h - (h.d)d) irrespective of alpha, so it is invoked as `--mode ablate --alphas 0`.
run_arm's guard decides whether to build a `Steer` at all, and before 2026-09-01 it read

    if ((alpha or step_rule != "fixed" or delta_maps is not None
         or (decode_alpha and decode_dirs is not None)) and layers)

with no `mode` term. At alpha 0 every disjunct is false, so an ablation arm got
`steer = None`: no hook registered, `mode` never reaching Steer, the arm running COMPLETELY
UNDEFENDED while labelling and reporting itself as a defense. It shipped to two 8xH100 nodes
and was caught only because every metric came back BIT-IDENTICAL to the undefended arm on
both large MoE models (GLM-4.5-Air ASR 0.708 / CORRECT 0.208; Qwen3-Next ASR 0.542 /
CORRECT 0.238).

THE POINT OF THE FILE. The bug is in the WIRING, not in `Steer`. A CPU test that constructs
`Steer` directly applies the ablation correctly and passes -- one was written, and it passed,
and it did not catch this. Only a test that goes through `run_arm` can. `Steer` is stubbed
with a recorder so no model, tokenizer or GPU is needed.

Run: .venv/bin/python tools/controls/verify_ablate_wiring.py    (exit 0 = pass)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import src.arms as A  # noqa: E402
from src.steering import DOSE_FREE_MODES, MODES  # noqa: E402

FAILED = []


class _Built(Exception):
    """Raised by the stub to report that run_arm decided to construct a Steer."""

    def __init__(self, mode):
        self.mode = mode


def steer_built(**kw):
    """(was_a_Steer_constructed, mode_it_received) for one run_arm configuration.

    Stubs src.arms.Steer, then calls run_arm far enough to hit the guard. The stub raises,
    so nothing downstream (tokenizer, generate, scoring) is exercised or needed.
    """
    real = A.Steer

    class Stub:
        def __init__(self, model, layers, dirs, alpha, scale, sigmas, ablate_axes, mode,
                     *a, **k):
            raise _Built(mode)

    A.Steer = Stub
    try:
        A.run_arm(object(), object(), [], **kw)
    except _Built as b:
        return True, b.mode
    except Exception:
        # the guard chose None, so run_arm proceeded past the construction and died later
        # on the dummy model/tokenizer -- that is the "no Steer" answer
        return False, None
    finally:
        A.Steer = real
    return False, None


def check(desc, got, want):
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {desc}   got={got} want={want}")
    if not ok:
        FAILED.append(desc)


BASE = dict(layers=[0, 1, 2], dirs=[None] * 3, sigmas=[1.0] * 3)

print("=== steer-construction guard in src/arms.run_arm ===")

# THE REGRESSION. Dose-free ablation at alpha 0 must still build a Steer, and that Steer
# must receive mode='ablate' -- a Steer built with mode='add' would be an equally silent
# no-op (prefill_off short-circuits it).
built, mode = steer_built(alpha=0.0, mode="ablate", **BASE)
check("mode=ablate, alpha=0 BUILDS a Steer (the 2026-09-01 no-op bug)", built, True)
check("...and that Steer receives mode='ablate'", mode, "ablate")

built, mode = steer_built(alpha=0.0, mode="ablate_add", **BASE)
check("mode=ablate_add, alpha=0 BUILDS a Steer", built, True)
check("...and that Steer receives mode='ablate_add'", mode, "ablate_add")

# EVERY dose-free mode, driven off the shared tuple rather than a hand-written list, so a
# mode added to the operator later cannot quietly skip this regression. `ablate_mp` /
# `ablate_mp_add` (mean-preserving ablation, h <- h - ((h-mu).d)d) are the modes this covers
# today; their numeric behaviour is verified separately in verify_ablate_mp.py.
for _m in DOSE_FREE_MODES:
    built, mode = steer_built(alpha=0.0, mode=_m, **BASE)
    check(f"mode={_m}, alpha=0 BUILDS a Steer (dose-free: it sets its own magnitude)",
          built, True)
    check(f"...and that Steer receives mode={_m!r}", mode, _m)
check("`add` is the ONLY mode that is not dose-free",
      tuple(m for m in MODES if m not in DOSE_FREE_MODES), ("add",))

# THE CONTROL. Everything that was correct before must be unchanged -- this guard sits in
# front of every locked cell, so a fix that over-fires would put a hook on the clean and
# base-XPIA arms and silently move every published number.
built, _ = steer_built(alpha=0.0, mode="add", **BASE)
check("mode=add, alpha=0 builds NOTHING (clean / base-XPIA arms stay undefended)",
      built, False)

built, mode = steer_built(alpha=8.0, mode="add", **BASE)
check("mode=add, alpha=8 builds a Steer (the ordinary defended arm)", built, True)
check("...and that Steer receives mode='add'", mode, "add")

built, _ = steer_built(alpha=0.0, mode="add", layers=None, dirs=None, sigmas=None)
check("no layers builds NOTHING even under ablate-eligible settings", built, False)

built, _ = steer_built(alpha=0.0, mode="ablate", layers=None, dirs=None, sigmas=None)
check("mode=ablate with no layers still builds NOTHING", built, False)

# the other self-magnitude operators the guard already protected, as live regressions
built, _ = steer_built(alpha=0.0, mode="add", step_rule="boundary", **BASE)
check("step_rule=boundary at alpha 0 builds a Steer (pre-existing behaviour)", built, True)

print()
if FAILED:
    print(f"FAILED {len(FAILED)} check(s):")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("ALL ABLATE-WIRING CHECKS PASSED")
