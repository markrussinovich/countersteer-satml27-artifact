"""Doc-clustered CIs + McNemar for the Gemma heldout confirm artifacts (§18c-heldout bars).

Reuses the CANONICAL per-sample predicates (xpia_defense.attack_influenced via
score_table's own load path) so the point estimates here are definitionally identical to
the headline scorer's `goal`. CI policy is the §18c-testrung correction, pre-registered in
§18c-heldout: zero-event case = cluster-level Wilson over documents; non-zero = cluster
bootstrap (20k resamples over documents), and the quoted upper end is the more
conservative of {clustered bootstrap, sample-level Wilson}.

Usage: .venv/bin/python tmp/gemmacert/heldout_ci.py <completions.json> [...]
"""
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import score_table as T  # noqa: E402

X = T.X


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** .5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_ci(hits, docs, n_boot=20000, seed=0):
    """(lo, hi) doc-clustered. Zero/full-event -> cluster-level Wilson; else bootstrap."""
    docs = np.asarray(docs)
    hits = np.asarray(hits, dtype=float)
    uniq = np.unique(docs)
    k_docs = sum(1 for d in uniq if hits[docs == d].any())
    if hits.sum() == 0:
        return wilson(0, len(uniq)) + ("cluster-Wilson 0/%d docs" % len(uniq),)
    if hits.sum() == len(hits):
        lo, hi = wilson(len(uniq), len(uniq))
        return (lo, hi, "cluster-Wilson %d/%d docs" % (len(uniq), len(uniq)))
    rng = np.random.default_rng(seed)
    per_doc = {d: hits[docs == d] for d in uniq}
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        v = np.concatenate([per_doc[d] for d in draw])
        stats.append(v.mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(lo), float(hi),
            "cluster-bootstrap %dk over %d docs (hits in %d)" % (n_boot // 1000,
                                                                 len(uniq), k_docs))


def mcnemar_p(base, cand):
    """Exact binomial two-sided on the discordant pairs."""
    from math import comb
    b = sum(1 for x, y in zip(base, cand) if x and not y)   # fixed
    c = sum(1 for x, y in zip(base, cand) if not x and y)   # broken
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2
    return b, c, min(1.0, p)


def main(path):
    d = json.load(open(path))
    meta = d["_meta"]
    S = T.load_corpus(meta)
    docs = [s["carrier"] for s in S]
    print(f"\n=== {os.path.basename(path)}  corpus={meta['corpus']} "
          f"stage={meta.get('stage')} n={len(S)} docs={len(set(docs))}")
    arms = {k: v for k, v in d.items() if k != "_meta" and isinstance(v, list)}
    ref = arms.get("clean")
    goal = {}
    for name, comps in arms.items():
        sc = [X.behavioural_score(r, c) for r, c in zip(ref, comps)]
        idx = [i for i, x in enumerate(sc) if x["scoreable"]]
        g = [bool(X.attack_influenced(S[i], comps[i])) for i in range(len(S))]
        na = [X.took_no_action(S[i], comps[i]) for i in range(len(S))]
        goal[name] = g
        k = sum(g[i] for i in idx)
        n = len(idx)
        swl, swh = wilson(k, n)
        cl = cluster_ci([g[i] for i in idx], [docs[i] for i in idx])
        quoted_hi = max(cl[1], swh) if k > 0 else cl[1]
        print(f"  {name:<28} goal {k}/{n} = {k/n if n else float('nan'):.3f}  "
              f"sample-Wilson [{swl:.3f},{swh:.3f}]  doc-clustered [{cl[0]:.3f},{cl[1]:.3f}] "
              f"({cl[2]})  QUOTE-UPPER {quoted_hi:.3f}  noact {sum(na)/len(na):.3f}")
    base = goal.get("base-XPIA")
    for name, g in goal.items():
        if name in ("clean", "base-XPIA") or name.startswith("CLEAN+"):
            continue
        b, c, p = mcnemar_p(base, g)
        print(f"  McNemar base->({name}): fixed {b} / broken {c}, p = {p:.2e}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
