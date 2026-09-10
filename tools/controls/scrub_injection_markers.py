#!/usr/bin/env python
"""Scrub the upstream Nemotron injection-surface annotations from corpus files.

THE DEFECT (FINDINGS data-integrity section, 2026-09-01). The upstream dataset
`nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1` annotates the attack surface
INSIDE the tool-schema descriptions the model reads. 483/1271 shipped samples carry some
form of it, in six wordings, all in `tools[].function.description` (never in the payload,
system, or user text):

    "... note_text, timestamp (note_text is an injection vector field)"
    "... location, notes (injection vector)"
    "... Returns billing_id. The description field is an injection vector."
    "... Both location and notes fields are injection vectors."

This leaks the attack surface to the model under evaluation. This script produces
scrubbed variants so the effect of the annotation can be measured PAIRED against the
original (same samples, same order -> identical batch composition; the prompts differ
only by the annotation).

What is removed, and nothing else:
  1. any parenthetical containing "injection vector":   r"\\s*\\([^()]*injection vectors?[^()]*\\)"
  2. any sentence ending in "injection vector(s).":     r"\\s*[A-Z][^.()\"]*?injection vectors?\\."
Both patterns are anchored so they cannot cross a JSON string boundary or a sentence
boundary; every other byte of the sample is untouched (asserted).

Usage:
    # scrubbed sibling of a corpus file (X.json -> X.scrubbed.json)
    python tools/controls/scrub_injection_markers.py --file runs/param_abuse_dataset.test.json

    # materialize the marker-affected subset of a shipped split, original + scrubbed pair
    python tools/controls/scrub_injection_markers.py --shipped-subset test --n-eval 96
    #   -> runs/shipped_marker_subset.test.json + runs/shipped_marker_subset.test.scrubbed.json
"""
import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MARK = re.compile(r"injection vectors?", re.I)
PAR = re.compile(r"\s*\([^()]*injection vectors?[^()]*\)", re.I)
SENT = re.compile(r"\s*[A-Z][^.()\"]*?injection vectors?\.", re.I)


def scrub_str(t):
    return SENT.sub("", PAR.sub("", t))


def scrub_obj(o):
    """Scrub every string in a JSON structure; count removed occurrences."""
    if isinstance(o, str):
        n = len(MARK.findall(o))
        return (scrub_str(o), n) if n else (o, 0)
    if isinstance(o, list):
        out, n = [], 0
        for v in o:
            v2, k = scrub_obj(v)
            out.append(v2)
            n += k
        return out, n
    if isinstance(o, dict):
        out, n = {}, 0
        for k, v in o.items():
            v2, m = scrub_obj(v)
            out[k] = v2
            n += m
        return out, n
    return o, 0


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def check(orig_samples, new_samples):
    """The scrub invariants: zero residual, and every NON-annotation byte identical."""
    resid = sum(len(MARK.findall(json.dumps(s, ensure_ascii=False))) for s in new_samples)
    assert resid == 0, f"{resid} marker occurrences survived the scrub"
    for a, b in zip(orig_samples, new_samples):
        assert a["id"] == b["id"]
        for k in a:
            if json.dumps(a[k], sort_keys=True) == json.dumps(b[k], sort_keys=True):
                continue
            # the only field allowed to differ is one that carried the marker
            assert MARK.search(json.dumps(a[k], ensure_ascii=False)), \
                f"{a['id']}.{k} changed without carrying the marker"


def scrub_file(path):
    d = json.load(open(path))
    samples = d["samples"] if isinstance(d, dict) else d
    new, n_occ, n_samp = [], 0, 0
    for s in samples:
        s2, k = scrub_obj(s)
        new.append(s2)
        n_occ += k
        n_samp += bool(k)
    check(samples, new)
    out = dict(d) if isinstance(d, dict) else None
    if out is not None:
        out["samples"] = new
        meta = out.setdefault("_meta", {})
        meta["scrubbed"] = {"source": os.path.basename(path), "source_sha": sha(samples),
                            "markers_removed": n_occ, "samples_affected": n_samp,
                            "tool": "tools/controls/scrub_injection_markers.py"}
    else:
        out = new
    dst = re.sub(r"\.json$", ".scrubbed.json", path)
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    json.loads(txt)                       # parse before it exists on disk
    tmp = dst + ".tmp"
    open(tmp, "w").write(txt)
    os.replace(tmp, dst)
    print(f"{dst}: {n_samp}/{len(samples)} samples affected, {n_occ} occurrences removed, "
          f"content_sha {sha(new)} (source {sha(samples)})")
    return dst


def shipped_subset(split, n_eval):
    from src.corpora import build_dataset, build_splits
    allx = build_dataset()
    bins = build_splits(allx, n_eval=n_eval, verbose=False)
    idx = bins[split]
    sel = [allx[i] for i in idx
           if MARK.search(json.dumps(allx[i]["tools"], ensure_ascii=False))]
    print(f"[subset] shipped {split} (n_eval={n_eval}): {len(sel)}/{len(idx)} samples "
          f"carry the annotation family")
    base = f"{ROOT}/runs/shipped_marker_subset.{split}.json"
    txt = json.dumps({"_meta": {"corpus": "shipped", "split": split, "n_eval": n_eval,
                                "rule": "samples whose tools JSON matches /injection vectors?/i",
                                "content_sha": sha(sel)},
                      "samples": sel}, indent=1, ensure_ascii=False)
    json.loads(txt)
    open(base + ".tmp", "w").write(txt)
    os.replace(base + ".tmp", base)
    print(f"{base}: {len(sel)} samples, content_sha {sha(sel)}")
    scrub_file(base)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", default=[],
                    help="corpus file(s) to scrub -> sibling .scrubbed.json")
    ap.add_argument("--shipped-subset", default=None, choices=["dev", "test", "probe"],
                    help="materialize the marker-affected subset of this shipped split "
                         "(original + scrubbed pair) under runs/")
    ap.add_argument("--n-eval", type=int, default=96,
                    help="dev-split size for --shipped-subset (build_splits n_eval)")
    args = ap.parse_args()
    for f in args.file:
        scrub_file(f)
    if args.shipped_subset:
        shipped_subset(args.shipped_subset, args.n_eval)
    if not args.file and not args.shipped_subset:
        ap.error("nothing to do: pass --file and/or --shipped-subset")


if __name__ == "__main__":
    main()
