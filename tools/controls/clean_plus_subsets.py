#!/usr/bin/env python
"""Split CLEAN+ utility by SAMPLE COMPOSITION -- does the defense's benign cost fall where
the model has to compose prose, or where it only has to copy?

WHY THIS EXISTS. `utilBenign` is one aggregate number over the CLEAN+ arm, and an aggregate
can look healthy while the subset that matters breaks. Two facts make that a live risk here:

  * `tools/controls/drift_anatomy.py` measured that **91-100% of every drift sample's
    differing arguments are model-COMPOSED**, 0-6.5% are copy failures, and 0% are wrong-field
    retrievals. The cost of steering is concentrated in composed fields already.
  * the AlphaSteer confound controls (tools/controls/build_alphasteer.py) measured that the
    learned map fires on benign PROSE at ~2.4x below an injection and on benign IMPERATIVES
    at only ~1.8x below -- i.e. its errors are concentrated on exactly the instruction-shaped
    prose that composed fields and procedural records contain.

So an aggregate CLEAN+ win could come entirely from ordinary JSON tokens dominating the
sample mix while the composed-field subset is unchanged or worse. This script reports the
subsets separately so that cannot hide.

THE PARTITION IS ARM-INDEPENDENT. Every subset is derived from the CLEAN reference completion
and the clean payload -- never from the arm being scored. An arm-dependent partition would
let a defense sort itself into the easy bucket.

  copy        every structured argument of the reference call is a verbatim (normalised) span
              of the tool payload: the task is retrieve-and-copy
  composed    at least one structured argument is NOT a verbatim span: the model wrote it
  imperative  CROSS-CUTTING, not a third bucket: the clean payload's own text contains
              imperative or procedural prose. This is the subset the confound predicts is
              most exposed, because legitimate record text there is instruction-SHAPED.
              Heuristic (cue phrases + second person), and reported as such.

Both utility columns are printed for every subset, per CLAUDE.md: `corrGoalOK` (structured
arguments byte-exact) and `corrCompGoalOK` (composed prose fields exempted). They BRACKET
task fidelity and a subset quoted in only one of them is not a result.

Usage:
    python tools/controls/clean_plus_subsets.py RESULTS_completions.json [MORE.json ...]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402
import score_table as ST  # noqa: E402

X = E.X
ROOT = E.ROOT

# Cue phrases for procedural / instruction-shaped record text. A HEURISTIC standing in for a
# semantic distinction -- it is used to SPLIT a report, never to score one, so a
# misclassified sample moves a row between buckets and cannot change any arm's number.
IMPERATIVE_CUES = re.compile(
    r"\b(please|make sure|ensure that|ensure the|remember to|do not|don't|must be|must "
    r"submit|should be|need to|be sure to|kindly|confirm the|verify the|bring the|return "
    r"the|check the|note that|follow the|prior to|before you|you should|you must|you will "
    r"need)\b", re.I)


def payload_text(sample):
    """Every string VALUE in the clean payload, concatenated. Keys are schema, not content."""
    out = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            out.append(v)

    walk(sample.get("payload_clean") or sample.get("payload") or {})
    return " ".join(out)


def ref_provenance(sample, ref_completion):
    """(n_args, n_composed) over the REFERENCE call's structured arguments.

    Same `in_record` predicate as score_table.composed_args, and the same `_is_freetext`
    exemption -- so the partition is aligned with what `struct_exact` actually compares. A
    sample whose only composed field is exempt from comparison would otherwise be sorted into
    `composed` while contributing nothing to the composed-field failure mode.
    """
    blob = ST._norm(json.dumps(sample.get("payload_clean") or sample.get("payload") or {},
                               ensure_ascii=False))
    n_args = n_comp = 0
    for _name, args in X.parse_tool_calls(ref_completion or ""):
        if not isinstance(args, dict):
            continue
        for _a, rv in args.items():
            if X._is_freetext(rv):
                continue
            n_args += 1
            t = ST._norm(rv)
            if not (len(t) >= 4 and t in blob):
                n_comp += 1
    return n_args, n_comp


def partition(S, ref):
    """-> {subset name: [sample index]}. Derived from the clean arm ONLY."""
    subs = {"copy": [], "composed": [], "imperative": [], "no_imperative": []}
    for i, s in enumerate(S):
        n_args, n_comp = ref_provenance(s, ref[i])
        if n_args == 0:
            # nothing struct_exact would compare; it belongs to neither bucket and saying so
            # is better than silently folding it into one
            subs.setdefault("no_structured_args", []).append(i)
        elif n_comp:
            subs["composed"].append(i)
        else:
            subs["copy"].append(i)
        (subs["imperative"] if IMPERATIVE_CUES.search(payload_text(s))
         else subs["no_imperative"]).append(i)
    return subs


def report(path):
    d = json.load(open(path))
    meta = d.get("_meta")
    if meta is None:
        print(f"[skip] {os.path.basename(path)}: no _meta")
        return
    S = ST.load_corpus(meta)
    arms = {k: v for k, v in d.items() if k != "_meta" and len(v) == len(S)}
    if "clean" not in arms:
        print(f"[skip] {os.path.basename(path)}: no clean reference arm")
        return
    ref = arms["clean"]
    subs = partition(S, ref)
    rows = {a: ST.metrics(S, ref, c) for a, c in arms.items()}
    base = rows["clean"]

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"corpus={meta.get('corpus')} n={len(S)} clean_sha={meta.get('clean_sha')}")
    order = ["copy", "composed", "imperative", "no_imperative", "no_structured_args"]
    order = [k for k in order if subs.get(k)]
    print("subset sizes: " + ", ".join(f"{k}={len(subs[k])}" for k in order))
    print("  (`imperative` / `no_imperative` CROSS-CUT copy/composed -- they are a second "
          "axis, not more buckets, so their sizes sum to n on their own)")

    for k in order:
        idx = subs[k]
        print(f"\n-- {k} (n={len(idx)}) " + "-" * 46)
        print(f"{'arm':<34}{'corrGoalOK ^':>14}{'%clean ^':>10}"
              f"{'corrCompGoalOK ^':>18}{'%clean ^':>10}{'goal v':>9}")
        for a, m in rows.items():
            sel = [i for i in idx if i in set(m["_idx"])]
            if not sel:
                print(f"{a:<34}{'--':>14}{'--':>10}{'--':>18}{'--':>10}{'--':>9}")
                continue

            def rate(key, mm=m, ss=sel):
                return sum(mm[key][i] for i in ss) / len(ss)

            cg, cc = rate("_corr_goal_ok"), rate("_comp_goal_ok")
            bg = sum(base["_corr_goal_ok"][i] for i in sel) / len(sel)
            bc = sum(base["_comp_goal_ok"][i] for i in sel) / len(sel)
            gl = sum(m["_fired"][i] for i in sel) / len(sel)
            print(f"{a:<34}{cg:>14.3f}{(100 * cg / bg if bg else float('nan')):>9.1f}%"
                  f"{cc:>18.3f}{(100 * cc / bc if bc else float('nan')):>9.1f}%{gl:>9.3f}")
    print("\nDenominator: scoreable samples within each subset (the clean arm made >=1 call).")
    print("`%clean` is against the CLEAN arm restricted to the SAME subset, so the two "
          "columns are\ncomparable across subsets of different intrinsic difficulty.")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        report(p)


if __name__ == "__main__":
    main()
