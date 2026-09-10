#!/usr/bin/env python
"""Paired original-vs-scrubbed comparison for the injection-marker validity check.

Two completions artifacts generated from the SAME sample ids in the SAME order and batch
composition, differing only in whether the upstream Nemotron attack-surface annotations
("... is an injection vector ...", see tools/controls/scrub_injection_markers.py) are
present in the tool-schema descriptions. This script pairs them sample-for-sample and
answers: does the annotation change behaviour in any arm?

Per matched arm it reports goal / corrGoalOK / corrCompGoalOK / noact in both variants,
the per-sample flip counts, and an exact two-sided McNemar p on each.

Usage:
    python tools/controls/marker_scrub_compare.py ORIGINAL_completions.json SCRUBBED_completions.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import score_table as T  # noqa: E402

X = T.X


def load(path):
    d = json.load(open(path))
    meta = d["_meta"]
    S = T.load_corpus(meta)
    arms = {k: v for k, v in d.items() if k != "_meta"}
    assert "clean" in arms
    ref = arms["clean"]
    rows = {a: T.metrics(S, ref, c) for a, c in arms.items() if len(c) == len(S)}
    return meta, S, arms, rows


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    ma, Sa, arms_a, rows_a = load(pa)
    mb, Sb, arms_b, rows_b = load(pb)
    ids_a = ma["sample_ids"]
    ids_b = mb["sample_ids"]
    assert ids_a == ids_b, "sample ids differ -- these runs are not pairable"
    assert ma.get("corpus_sha") != mb.get("corpus_sha"), \
        "identical corpus_sha -- these are the same corpus variant twice"
    print(f"A (original): {os.path.basename(pa)}  corpus_sha={ma.get('corpus_sha')} "
          f"clean_sha={ma.get('clean_sha')}")
    print(f"B (scrubbed): {os.path.basename(pb)}  corpus_sha={mb.get('corpus_sha')} "
          f"clean_sha={mb.get('clean_sha')}")
    print(f"n={len(ids_a)} paired samples\n")

    both = sorted(set(rows_a) & set(rows_b))
    hdr = (f"{'arm':<30}{'metric':<16}{'orig':>8}{'scrub':>8}"
           f"{'A-only':>8}{'B-only':>8}{'McNemar p':>11}")
    print(hdr)
    print("-" * len(hdr))
    for arm in both:
        A, B = rows_a[arm], rows_b[arm]
        # per-sample boolean series; metrics exposes them precisely so a caller does not
        # re-derive the definitions (CLAUDE.md: reuse code)
        series = [("goalAll", A["_fired"], B["_fired"]),
                  ("corrGoalOK", A["_corr_goal_ok"], B["_corr_goal_ok"]),
                  ("corrCompGoalOK", A["_comp_goal_ok"], B["_comp_goal_ok"])]
        for name, va, vb in series:
            # restrict utility metrics to the scoreable intersection; goal over all
            if name == "goalAll":
                keep = list(range(len(ids_a)))
            else:
                keep = sorted(set(A["_idx"]) & set(B["_idx"]))
            va = [va[i] for i in keep]
            vb = [vb[i] for i in keep]
            aonly, bonly, p = T.mcnemar(va, vb)
            ra = sum(va) / len(va) if va else float("nan")
            rb = sum(vb) / len(vb) if vb else float("nan")
            flips = [ids_a[keep[j]] for j, (x, y) in enumerate(zip(va, vb)) if x != y]
            print(f"{arm:<30}{name:<16}{ra:>8.3f}{rb:>8.3f}"
                  f"{aonly:>8d}{bonly:>8d}{p:>11.4f}"
                  + (f"   flips: {flips}" if flips else ""))
        # capability guard
        na = sum(X.took_no_action(s, c) for s, c in zip(Sa, arms_a[arm]))
        nb = sum(X.took_no_action(s, c) for s, c in zip(Sb, arms_b[arm]))
        print(f"{arm:<30}{'noact(count)':<16}{na:>8d}{nb:>8d}")
        print()


if __name__ == "__main__":
    main()
