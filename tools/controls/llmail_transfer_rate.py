#!/usr/bin/env python3
"""Cross-model re-submission rate inside the LLMail-Inject challenge's own data.

The challenge (arXiv 2506.09956) ran each retrieval level against TWO victim
models under an identical defense -- phase-1 sub-level letters pair up as
{a,b}, {c,d}, {e,f}, {g,h}, {i,j} (Phi-3 vs GPT, see levels_descriptions.json).
The challenge paper discusses transfer only qualitatively and reports no rate
for it, so this script computes one from the released raw submissions.

Estimand (directed, the number the paper quotes): take every (team, retrieval
level, exact subject+body) that SUCCEEDED at one sub-level and that the SAME
team also submitted at the sibling sub-level; the rate is how often the sibling
submission also succeeded. It is therefore conditioned on what participants
CHOSE to re-submit -- not a systematic re-evaluation of all successful attacks
on the second model. `--by-defense` shows how strongly that choice varies.

A text that beat both models contributes twice (once per direction); the
per-text-pair variant, which counts each such text-pair once, is printed too.

Success requires the `objectives` dict to be NON-EMPTY and all its flags true:
a few hundred rows carry `{}`, and `all({})` is vacuously True, which inflates
the rate if the guard is dropped.

Usage:
    python3 tools/controls/llmail_transfer_rate.py \
        --submissions runs/llmail/raw_submissions_phase1.jsonl \
        --levels runs/llmail/levels_descriptions.json --phase phase1 [--by-defense]
"""
import argparse
import collections
import json
import pathlib
import re


def sibling_pairs(levels_path, phase):
    """Map each sub-level letter to its sibling (same defense, the other model)."""
    desc = json.loads(pathlib.Path(levels_path).read_text())[phase]
    by_defense = collections.defaultdict(dict)
    for letter, text in desc.items():
        model, defense = re.split(r"\s+with\s+", text, maxsplit=1)
        by_defense[defense.strip()][model.strip()] = letter
    pairs, defense_of = {}, {}
    for defense, letters in by_defense.items():
        if len(letters) != 2:
            continue  # a defense offered on only one model has no sibling
        a, b = sorted(letters.values())
        pairs[a], pairs[b] = b, a
        defense_of[a] = defense_of[b] = defense
    return pairs, defense_of


def load(submissions_path):
    """-> (present, succeeded) keyed by (team, retrieval level, sub-level, text)."""
    present, succeeded = set(), collections.defaultdict(bool)
    with open(submissions_path) as fh:
        for line in fh:
            r = json.loads(line)
            scenario = r["scenario"]
            key = (r["team_id"], scenario[:-1], scenario[-1],
                   (r.get("subject") or "") + "\x00" + (r.get("body") or ""))
            obj = json.loads(r["objectives"]) if r.get("objectives") else {}
            present.add(key)
            succeeded[key] |= bool(obj) and all(obj.values())
    return present, succeeded


def transfer(present, succeeded, pairs, defense_of):
    directed = []                       # (defense, sibling also succeeded)
    per_pair = collections.defaultdict(bool)
    for team, level, sub, text in present:
        sib = pairs.get(sub)
        if sib is None or not succeeded[(team, level, sub, text)]:
            continue
        other = (team, level, sib, text)
        if other not in present:
            continue                    # the team never re-submitted this text
        hit = succeeded[other]
        directed.append((defense_of[sub], hit))
        pair_key = (team, level, frozenset((sub, sib)), text)
        per_pair[pair_key] |= hit
    return directed, per_pair


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submissions", required=True, help="raw_submissions_<phase>.jsonl")
    ap.add_argument("--levels", required=True, help="levels_descriptions.json")
    ap.add_argument("--phase", default="phase1", help="key inside levels_descriptions.json")
    ap.add_argument("--by-defense", action="store_true", help="break the rate out per defense")
    args = ap.parse_args()

    pairs, defense_of = sibling_pairs(args.levels, args.phase)
    present, succeeded = load(args.submissions)
    directed, per_pair = transfer(present, succeeded, pairs, defense_of)

    if not directed:
        raise SystemExit("no cross-model re-submissions found -- check --phase / --submissions")
    hits = sum(hit for _, hit in directed)
    pair_hits = sum(per_pair.values())
    print(f"sibling pairs ({args.phase}): "
          + ", ".join(f"{a}/{pairs[a]} {defense_of[a]}" for a in sorted(pairs) if a < pairs[a]))
    print(f"directed re-submissions: {hits}/{len(directed)} = {hits / len(directed):.4f}")
    print(f"  spanning {len(per_pair)} distinct (team, level, defense-pair, text) instances "
          f"and {len({k[3] for k in per_pair})} distinct texts")
    print(f"per text-pair (each counted once, not per direction): "
          f"{pair_hits}/{len(per_pair)} = {pair_hits / len(per_pair):.4f}")
    if args.by_defense:
        agg = collections.Counter()
        tot = collections.Counter()
        for defense, hit in directed:
            agg[defense] += hit
            tot[defense] += 1
        for defense in sorted(tot, key=lambda d: -tot[d]):
            print(f"  {defense:<15} {agg[defense]}/{tot[defense]} = {agg[defense] / tot[defense]:.3f}")


if __name__ == "__main__":
    main()
