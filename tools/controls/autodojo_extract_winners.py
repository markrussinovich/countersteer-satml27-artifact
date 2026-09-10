#!/usr/bin/env python
"""Extract WINNING candidates from AutoDojo caches for the blind fraud partition.

For every (arm, suite, task, vector) cell with best-ASR > 0 under --root, emit the
winning candidate (the trajectory entry maximizing (asr, -iteration) -- ties go to the
EARLIEST iteration, matching autodojo_score.py's best-row rule) in two forms:

  --keyed-out    full records incl. arm + asr + seed_style (winning_candidates_keyed.json
                 format; the sha10 key of [arm, suite, task, vector] joins them to
                 fraud_labels_blind.json)
  --blind-out    blind review items: {id, suite, task, vector, text} ONLY (no arm, no
                 model, no outcome), shuffled with a fixed seed -- the reviewer input
                 (fraud_rubric.md application protocol)

--labels: existing fraud_labels_blind.json; cells whose sha10 is already labeled are
SKIPPED (incremental passes only ever label NEW winners). Cells are deduplicated across
harvest snapshots with the same longest-trajectory rule as the scorer.

Usage:
  .venv/bin/python tools/controls/autodojo_extract_winners.py \
      --root runs/autodojo/harvest2/merged \
      --labels runs/autodojo/fraud_labels_blind.json \
      --keyed-out runs/autodojo/winners_new.json \
      --blind-out runs/autodojo/blind_new.json
"""
import argparse
import hashlib
import json
import os
import random


def cell_key(arm, suite, task, vector):
    return hashlib.sha256(json.dumps([arm, suite, task, vector]).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", default=None,
                    help="existing fraud_labels_blind.json; labeled cells are skipped")
    ap.add_argument("--keyed-out", required=True)
    ap.add_argument("--blind-out", required=True)
    ap.add_argument("--seed", type=int, default=7, help="shuffle seed for the blind list")
    a = ap.parse_args()

    have = set(json.load(open(a.labels))) if a.labels else set()

    cells = {}
    for dirpath, _, files in os.walk(a.root):
        if "injections.json" not in files:
            continue
        arm = os.path.basename(os.path.dirname(dirpath))
        cache = json.load(open(os.path.join(dirpath, "injections.json")))
        suite = cache["suite"]
        for task, vecs in cache.get("injection_tasks", {}).items():
            for vec, cell in vecs.items():
                traj = cell.get("trajectory", [])
                best = max(traj, key=lambda e: (e.get("asr", 0.0), -e.get("iteration", 0)),
                           default=None)
                if not best or best.get("asr", 0.0) <= 0:
                    continue
                k = (arm, suite, task, vec)
                # longest-trajectory dedupe across harvest snapshots (scorer rule)
                if k in cells and cells[k]["n_traj"] >= len(traj):
                    continue
                cells[k] = {"n_traj": len(traj), "best": best}

    keyed, blind, skipped = [], [], 0
    for (arm, suite, task, vec), d in sorted(cells.items()):
        kid = cell_key(arm, suite, task, vec)
        if kid in have:
            skipped += 1
            continue
        b = d["best"]
        keyed.append({"arm": arm, "suite": suite, "task": task, "vector": vec,
                      "asr": b.get("asr"), "iteration": b.get("iteration", 0),
                      "seed_style": b.get("seed_style"), "text": b.get("text", "")})
        blind.append({"id": kid, "suite": suite, "task": task, "vector": vec,
                      "text": b.get("text", "")})

    random.Random(a.seed).shuffle(blind)
    json.dump(keyed, open(a.keyed_out, "w"), indent=1)
    json.dump(blind, open(a.blind_out, "w"), indent=1)
    print(f"winners: {len(cells)} compromised cells; {skipped} already labeled; "
          f"{len(keyed)} NEW -> {a.keyed_out} / {a.blind_out}")


if __name__ == "__main__":
    main()
