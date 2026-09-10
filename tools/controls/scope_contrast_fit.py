#!/usr/bin/env python
"""Fit the SCOPE-CONTRAST direction and its validity gates from an in-scope capture.

Reproduces (as committed code — review requirement 2026-08-29) the fit that produced
`dirs.scope_contrast` and the FINDINGS §10u numbers, from
`runs/override_slope_deleg_inscope2.json` (wording-varied capture, --supplement inscope2):

  direction  d_L = mean over rows of (act_injection_span − act_inscope_span), unit-norm
  gates      (1) wording-held-out AUC: refit on train wordings, classify held-out
                 wordings' spans by projection (3 folds + leave-one-wording-out)
             (2) behavioral rates (fired / followed_inscope)

The 2026-08-29 review's disclosures apply and are printed: only the IN-SCOPE side is
wording-held-out (injection framings appear in train and test); the in-scope sentence
always precedes the injection (serial position uncontrolled); the direction carries
partial request-kind variance (tool-vs-param injections separate along it at AUC
0.71–0.92) though param-action injections still separate from in-scope spans at 0.96–1.00.

Usage:
  .venv/bin/python tools/controls/scope_contrast_fit.py \
      --src runs/override_slope_deleg_inscope2.json \
      --out-run runs/gpt-oss-20b-userabl [--write]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402  (probe-pickle class alias side effect)


def auc(p, y):
    if len(p) == 0 or y.std() == 0:
        return float("nan")
    o = np.argsort(p)
    rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def span_auc(A_inj, A_ins, tr, te):
    u = (A_inj[tr] - A_ins[tr]).mean(0)
    u /= np.linalg.norm(u) + 1e-12
    p = np.r_[A_inj[te] @ u, A_ins[te] @ u]
    y = np.r_[np.ones(int(te.sum())), np.zeros(int(te.sum()))]
    return auc(p, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/override_slope_deleg_inscope2.json")
    ap.add_argument("--out-run", default="runs/gpt-oss-20b-userabl")
    ap.add_argument("--key", default="scope_contrast")
    ap.add_argument("--write", action="store_true",
                    help="write dirs.<key> + sigma into the out-run pickles (else dry)")
    a = ap.parse_args()

    d = json.load(open(a.src))
    rows = [r for r in d["rows"] if "act_inscope" in r]
    layers = d["layers"]
    words = sorted({r["inscope_text"] for r in rows})
    fired = np.mean([r["fired"] for r in d["rows"]])
    fins = np.mean([r.get("followed_inscope", False) for r in d["rows"]])
    print(f"{len(rows)}/{len(d['rows'])} rows with both spans | {len(words)} wordings | "
          f"behavioral: fired {fired:.4f} followed_inscope {fins:.4f}")

    w = None
    print(f"{'layer':>6}{'3fold-mean':>12}{'LOWO-min':>10}{'param-vs-ins':>14}{'kind-leak':>11}")
    for L in layers:
        A_inj = np.array([r["act"][str(L)] for r in rows], np.float32)
        A_ins = np.array([r["act_inscope"][str(L)] for r in rows], np.float32)
        w = np.array([words.index(r["inscope_text"]) for r in rows])
        act = np.array([r["action"] for r in rows])
        rng = np.random.default_rng(0)
        perm = list(rng.permutation(len(words)))
        folds = [(set(perm[:4]), set(perm[4:])), (set(perm[2:]), set(perm[:2])),
                 (set(perm[:2] + perm[4:]), set(perm[2:4]))]
        f3 = [span_auc(A_inj, A_ins, np.isin(w, list(tr)), np.isin(w, list(te)))
              for tr, te in folds]
        lowo = [span_auc(A_inj, A_ins, w != i, w == i) for i in range(len(words))]
        # review disclosures: param-action injections vs in-scope spans (request-kind
        # check), and how much request-kind variance the direction itself carries
        u = (A_inj - A_ins).mean(0); u /= np.linalg.norm(u) + 1e-12
        m = act == "param"
        pv = span_auc(A_inj[m], A_ins[m], np.ones(m.sum(), bool), np.ones(m.sum(), bool)) \
            if m.sum() else float("nan")
        kind = auc(np.r_[A_inj[act == "tool"] @ u, A_inj[act == "param"] @ u],
                   np.r_[np.ones(int((act == "tool").sum())),
                         np.zeros(int((act == "param").sum()))])
        print(f"{L:6d}{np.mean(f3):12.3f}{min(lowo):10.3f}{pv:14.3f}{kind:11.3f}")
        if a.write:
            path = f"{a.out_run}/probe_L{L}.pkl"
            p = pickle.load(open(path, "rb"))
            sig = float((np.r_[A_inj, A_ins] @ u).std())
            p["dirs"][a.key] = u.astype(np.float32)
            p["sigmas"][a.key] = sig
            blob = pickle.dumps(p)
            open(path + ".tmp", "wb").write(blob)
            pickle.load(open(path + ".tmp", "rb"))
            os.replace(path + ".tmp", path)
    print("written" if a.write else "dry run — pass --write to store the direction")


if __name__ == "__main__":
    main()
