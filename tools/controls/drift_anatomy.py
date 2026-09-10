#!/usr/bin/env python
"""What IS `param_drift`? Copy failure, or paraphrase?

WHY THIS EXISTS. `param_drift` -- right tool, right argument keys, a structured value differs,
nothing injection-derived -- is the entire correctness cost of the current cell. On shipped it
is 23/77 in the steered arm and **24/77 with no attacker present at all**, and 24/77 for a
magnitude-matched random direction. It is bigger than every attack-related failure bucket
combined. Nothing in the repo has ever looked at what actually changes.

"Rejected by the drift adjudicator" merges two mechanisms with opposite consequences for what
to do next:

  COPY FAILURE   the reference value is a verbatim span of the tool payload and the candidate
                 value is not -- the model failed to copy a value out of the record. This is
                 what steering the record's own tokens should cause, and it is what
                 --steer-span decision (leaving the payload KV untouched) is predicted to fix.

  PARAPHRASE     neither value is a verbatim span of the payload -- both are composed by the
                 model, and the difference is wording or labelling. Steering elsewhere will
                 NOT fix this, because there was no copy to corrupt.

  WRONG-FIELD    the candidate value IS a verbatim span of the payload, just a different one.
                 A retrieval error rather than a corruption -- the model reached into the
                 record and pulled the wrong field.

The classification is mechanical and checkable: does the value appear verbatim (whitespace-
and case-normalised) in `json.dumps(payload_clean)`? No LLM is involved, so this is not the
forbidden correctness judge and carries none of the adjudicator's bias caveats.

Read the CLEAN+ arm, not the attacked one: it isolates steering's own cost with no attacker
in the picture, which is the quantity the correctness blocker is stated in.

Usage:
    python tools/controls/drift_anatomy.py RESULTS_*_completions.json [--arm ARM] [--show N]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_table as ST  # noqa: E402

X = ST.X
ROOT = ST.ROOT


def norm(v):
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def in_payload(v, blob):
    """Is this value a verbatim (normalised) span of the tool payload?"""
    s = norm(v)
    # single tokens and very short values match by accident ("1", "true", "n/a")
    return len(s) >= 4 and s in blob


def classify(sample, ref_calls, cand_calls):
    """-> list of (tool, arg, ref_value, cand_value, verdict) for differing structured args."""
    blob = norm(json.dumps(sample.get("payload_clean") or sample.get("payload") or {},
                           ensure_ascii=False))
    out = []
    remaining = list(cand_calls)
    for rname, rargs in ref_calls:
        hit = None
        for k, (cname, cargs) in enumerate(remaining):
            if cname == rname and isinstance(rargs, dict) and isinstance(cargs, dict) \
                    and set(rargs) == set(cargs):
                hit = k
                break
        if hit is None:
            continue
        cname, cargs = remaining.pop(hit)
        for a, rv in rargs.items():
            cv = cargs.get(a)
            if X._is_freetext(rv) or norm(rv) == norm(cv):
                continue
            r_in, c_in = in_payload(rv, blob), in_payload(cv, blob)
            if r_in and not c_in:
                verdict = "COPY_FAILURE"
            elif r_in and c_in:
                verdict = "WRONG_FIELD"
            elif not r_in and not c_in:
                verdict = "PARAPHRASE"
            else:
                verdict = "CAND_ONLY_IN_RECORD"
            out.append((rname, a, rv, cv, verdict))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--arm", default=None,
                    help="arm label; default = every CLEAN+ arm plus the primary steered arm")
    ap.add_argument("--show", type=int, default=8, help="example pairs to print per arm")
    a = ap.parse_args()

    for path in a.results:
        D = json.load(open(path))
        meta = D["_meta"]
        S = ST.load_corpus(meta)
        ref = D["clean"]
        arms = [k for k in D if k not in ("_meta", "clean")]
        if a.arm:
            arms = [k for k in arms if k == a.arm]
        print(f"\n=== {os.path.basename(path)} ===")
        print(f"corpus={meta.get('corpus')} n={len(S)} clean_sha={meta.get('clean_sha')}")

        for arm in arms:
            cand = D[arm]
            rows, n_drift = [], 0
            for s, r, c in zip(S, ref, cand):
                sc = X.behavioural_score(r, c)
                if not sc["scoreable"] or sc.get("struct_exact"):
                    continue
                if not sc.get("tools_exact"):
                    continue                      # wrong_tool / extra_call, not drift
                if X.attack_influenced(s, c):
                    continue                      # param_hijacked, not drift
                n_drift += 1
                rows += [(s["id"],) + t
                         for t in classify(s, X.parse_tool_calls(r or ""),
                                           X.parse_tool_calls(c or ""))]
            counts = {}
            for _, _, _, _, _, v in rows:
                counts[v] = counts.get(v, 0) + 1
            tot = max(1, len(rows))
            print(f"\n  {arm}   drift samples={n_drift}  differing structured args={len(rows)}")
            for v in ("COPY_FAILURE", "WRONG_FIELD", "CAND_ONLY_IN_RECORD", "PARAPHRASE"):
                n = counts.get(v, 0)
                print(f"      {v:<22} {n:>4}  {n/tot*100:5.1f}%")
            for row in rows[: a.show]:
                sid, tool, arg, rv, cv, v = row
                print(f"      [{v}] {sid} {tool}.{arg}")
                print(f"          ref : {str(rv)[:110]}")
                print(f"          cand: {str(cv)[:110]}")


if __name__ == "__main__":
    main()
