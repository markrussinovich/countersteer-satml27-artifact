#!/usr/bin/env python
"""Derive a RESTRICTED AgentDojo cell set from a completed run's failures.

WHY THIS EXISTS. The full AgentDojo cell set is ~180 cells and a few hours across 4 GPUs.
Sweeping a steering parameter over all of it to answer a question that only ~20 cells can
answer is wasted GPU. The n=93 canonical run showed both remaining gaps are CONCENTRATED:

    correctness  23 broken CLEAN+ cells are only 12 distinct user tasks, and the split is
                 deterministic -- 36 tasks never break under steering, 12 always do, 0 are
                 inconsistent. So one cell per broken task is sufficient: the CLEAN+ arm is
                 cached per (suite, user_task) by `agentdojo_run.py` and does not depend on
                 the injection task.
    compromise   7 residual defended-security events, 4 of them one user task.

This emits exactly those cells plus a CONTROL sample of tasks that currently SURVIVE steering.

THE CONTROLS ARE NOT OPTIONAL. Sweeping alpha only over tasks that already fail can only ever
show improvement -- a lower alpha that recovers 6 broken tasks while breaking 5 working ones
looks like a win on the failure set alone. The controls are what make the sweep two-sided.
They are drawn stratified across suites with a fixed seed so the set is reproducible.

Usage:
    python tools/controls/agentdojo_failure_cells.py \
        --runs 'runs/agentdojo_run.shard?.json' \
        --attack important_instructions --n-control 10 \
        --out runs/agentdojo_failure_cells.json
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_rows(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        rows += json.load(open(path))["results"]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=f"{ROOT}/runs/agentdojo_run.shard?.json")
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--n-control", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/runs/agentdojo_failure_cells.json")
    a = ap.parse_args()

    rows = [r for r in load_rows(a.runs) if r.get("complete") and r["attack"] == a.attack]
    if not rows:
        sys.exit(f"no complete cells for attack={a.attack} in {a.runs}")

    # --- the utility half: tasks whose CLEAN+ (steer, no injection) arm breaks a task the
    # clean arm solves. Classified over ALL repetitions of the task so a task that is only
    # sometimes broken is visible rather than silently counted as broken.
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[(r["suite"], r["user_task"])].append(
            (r["clean"].get("utility"), r["cleanplus"].get("utility")))

    broken, survives, inconsistent, unsolvable = [], [], [], []
    for task, obs in sorted(by_task.items()):
        clean = set(c for c, _ in obs)
        plus = set(p for _, p in obs)
        if clean == {0.0}:
            unsolvable.append(task)          # clean arm cannot do it; nothing to preserve
        elif len(plus) > 1:
            inconsistent.append(task)
        elif plus == {0.0}:
            broken.append(task)
        else:
            survives.append(task)

    rng = random.Random(a.seed)
    by_suite = collections.defaultdict(list)
    for t in survives:
        by_suite[t[0]].append(t)
    control = []
    suites = sorted(by_suite)
    while len(control) < min(a.n_control, len(survives)):
        for s in suites:                      # round-robin keeps the sample stratified
            pool = [t for t in by_suite[s] if t not in control]
            if pool and len(control) < a.n_control:
                control.append(rng.choice(pool))

    # one representative cell per task -- the injection task is irrelevant to clean/CLEAN+,
    # so take the first by sort order for determinism.
    def rep(task):
        cand = sorted(r["injection_task"] for r in rows
                      if (r["suite"], r["user_task"]) == task)
        return cand[0]

    # A CELL CAN SERVE MORE THAN ONE PURPOSE, so tags are a LIST per cell rather than one
    # label. A residual-security cell can land on a task that was also drawn as a utility
    # control; deduping to a single tag silently drops whichever purpose was added second,
    # which is how `security_caused` first came back 0 when it was 2.
    cells, tags = [], []
    index = {}

    def add(suite, user_task, injection_task, tag):
        key = (suite, user_task, injection_task)
        if key in index:
            if tag not in tags[index[key]]:
                tags[index[key]].append(tag)
            return
        index[key] = len(cells)
        cells.append({"suite": suite, "user_task": user_task,
                      "injection_task": injection_task, "attack": a.attack})
        tags.append([tag])

    for task in broken:
        add(task[0], task[1], rep(task), "utility_broken")
    for task in control:
        add(task[0], task[1], rep(task), "utility_control")

    # --- the compromise half: every cell where the DEFENDED arm was still hijacked.
    for r in rows:
        if r["defended"].get("security") != 1.0:
            continue
        # a residual event the undefended arm did NOT produce is steering-CAUSED, which is a
        # different defect from "the defense failed to block a live attack".
        add(r["suite"], r["user_task"], r["injection_task"],
            "security_residual" if r["attacked"].get("security") == 1.0
            else "security_caused")

    out = {"config": vars(a),
           "n_cells": len(cells),
           "tags": tags,
           "counts": dict(collections.Counter(t for ts in tags for t in ts)),
           "task_classification": {
               "utility_broken": [list(t) for t in broken],
               "utility_survives": [list(t) for t in survives],
               "utility_control_sampled": [list(t) for t in control],
               "utility_inconsistent": [list(t) for t in inconsistent],
               "clean_arm_unsolvable": [list(t) for t in unsolvable]},
           "cells": cells}

    # ARTIFACT IS NOT WRITTEN UNTIL IT PARSES (CLAUDE.md): build the string, write, replace.
    blob = json.dumps(out, indent=1)
    json.loads(blob)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(blob)
    os.replace(tmp, a.out)

    print(f"[tasks] broken={len(broken)} survives={len(survives)} "
          f"inconsistent={len(inconsistent)} clean-unsolvable={len(unsolvable)}")
    print(f"[cells] {len(cells)} -> {a.out}")
    for tag, n in sorted(collections.Counter(t for ts in tags for t in ts).items()):
        print(f"    {tag:20} {n}")
    dual = sum(1 for ts in tags if len(ts) > 1)
    print(f"    (cells serving >1 purpose: {dual})")


if __name__ == "__main__":
    main()
