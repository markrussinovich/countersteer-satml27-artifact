#!/usr/bin/env python
"""Sample a fixed, stratified set of (suite, user_task, injection_task, attack) cells to evaluate.

WHY A SEPARATE STEP. The sampling must be identical across every GPU shard and every future
re-run, or the arms are not comparable. Doing it inside the runner would mean each shard
sampling its own set, and re-running later would silently evaluate a different population.
This writes the cell list ONCE; the runner reads it.

WHY NOT THE FULL BENCHMARK. 97 user tasks x 27 injection tasks is ~629 security cells per
attack, and there are 9 attack shapes worth testing. That is days of GPU for a precision we do
not need: ~100 cells is enough to separate a defense that works from one that does not, and the
variant shapes need only a handful of examples each because their purpose is to supply framing
levels for the factorial, not to be measured precisely.

ONLY SOLVABLE TASKS. Read from the screen artifacts. A task the model cannot do unattacked
cannot show a defense preserving utility, and a task where it never calls a tool has no tool
output to inject into.

Usage:
    python tools/controls/agentdojo_pairs.py --screen 'runs/agentdojo_screen.shard*.json' \\
        --n-main 100 --n-variant 10 --out runs/agentdojo_cells.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

ROOT = E.ROOT

from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

VERSION = "v1.2.1"

# The canonical attack, then its five identity ablations, then three structurally different
# shapes. InjecAgent's port is deliberately EXCLUDED -- that benchmark does not fire on this
# model and is out of scope (CLAUDE.md). DoS attacks are excluded too: they measure refusal,
# which is a different objective from "did the attacker achieve their goal".
MAIN_ATTACK = "important_instructions"
VARIANT_ATTACKS = [
    "important_instructions_no_names",
    "important_instructions_no_user_name",
    "important_instructions_no_model_name",
    "important_instructions_wrong_user_name",
    "important_instructions_wrong_model_name",
    "ignore_previous",
    "system_message",
    "tool_knowledge",
]


def solvable(screen_glob):
    rows = []
    for f in sorted(glob.glob(screen_glob)):
        rows += json.load(open(f))["results"]
    ok = [r for r in rows if r.get("utility", 0) >= 1.0]
    return rows, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", default=f"{ROOT}/runs/agentdojo_screen.shard*.json")
    ap.add_argument("--n-main", type=int, default=100)
    ap.add_argument("--n-variant", type=int, default=10,
                    help="cells per variant attack shape")
    ap.add_argument("--only", default=None,
                    help="comma list of attacks to sample, overriding MAIN/VARIANT. Used to "
                         "POWER UP a specific shape: the first sweep gave ignore_previous, "
                         "system_message and tool_knowledge only 1-3 attacked events each, "
                         "which is a hint and not a measurement.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/runs/agentdojo_cells.json")
    a = ap.parse_args()

    allrows, ok = solvable(a.screen)
    by_suite = {}
    for r in ok:
        by_suite.setdefault(r["suite"], []).append(r["user_task"])
    print(f"[screen] {len(ok)}/{len(allrows)} tasks solvable on the clean arm")
    for s in sorted(by_suite):
        print(f"    {s:<10} {len(by_suite[s]):>3} solvable")
    if not ok:
        raise SystemExit("no solvable task -- nothing downstream can mean anything")

    rng = np.random.default_rng(a.seed)
    # Enumerate every (suite, user_task, injection_task) then sample STRATIFIED BY SUITE, so a
    # suite with many solvable tasks cannot dominate the estimate.
    pool = {}
    for s, tids in by_suite.items():
        suite = get_suite(VERSION, s)
        pool[s] = [(s, t, j) for t in tids for j in suite.injection_tasks]

    def draw(n_total, attack):
        """n_total cells, stratified by suite, REMAINDER DISTRIBUTED.

        Integer division alone silently under-delivers: with 4 suites, `--n-variant 10` gives
        10//4 = 2 per suite = 8 cells, not 10. Spread the remainder over the first suites so
        the requested count is the delivered count.
        """
        names = sorted(pool)
        base, rem = divmod(n_total, max(1, len(names)))
        cells = []
        for j, s in enumerate(names):
            p = pool[s]
            k = min(base + (1 if j < rem else 0), len(p))
            idx = rng.permutation(len(p))[:k]
            cells += [{"suite": p[i][0], "user_task": p[i][1], "injection_task": p[i][2],
                       "attack": attack} for i in idx]
        return cells

    if a.only:
        cells = []
        for att in [x.strip() for x in a.only.split(",")]:
            v = draw(a.n_main, att)
            cells += v
            print(f"\n[only] {len(v):>3} cells with {att}")
    else:
        cells = draw(a.n_main, MAIN_ATTACK)
        print(f"\n[main] {len(cells)} cells with {MAIN_ATTACK}")
        for att in VARIANT_ATTACKS:
            v = draw(a.n_variant, att)
            cells += v
            print(f"[variant] {len(v):>3} cells with {att}")

    blob = json.dumps({"config": vars(a), "n_cells": len(cells),
                       "solvable": ok, "cells": cells}, indent=1, default=str)
    with open(a.out + ".tmp", "w") as f:
        f.write(blob)
    os.replace(a.out + ".tmp", a.out)
    json.load(open(a.out))
    print(f"\nTOTAL {len(cells)} cells -> {a.out}")


if __name__ == "__main__":
    main()
