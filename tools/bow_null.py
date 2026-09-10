"""Model-free null model for the injection probe, under the SAME splits.

Round-2 review reproduced the activation probe's by-sample AUC with a bag-of-words model
over token ids and no neural network. That shows by-sample AUC is not evidence. The
question it left open: does the null ALSO transfer across injection template families?

  * if BoW transfers as well as the activation probe -> the probe is a lexical detector
    and the "model linearly represents injection-ness" claim is unsupported
  * if BoW collapses cross-family where activations hold -> the activations carry
    something the surface form does not

Tokenizer only, no model weights.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/datadrive/huggingface/")
import xpia_defense as X
from transformers import AutoTokenizer

N, WINDOW, DIM = 250, 8, 2 ** 18


def family(t):
    t = t.strip()
    if "[PRIORITY MESSAGE]" in t:
        return "priority_message"
    if "URGENT" in t[:40] or "⚠" in t[:10]:
        return "urgent_must"
    return "other"


def main():
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    samples = X.build_dataset()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(samples))
    probe = [samples[i] for i in order[:N]]

    rows, cols, vals, y, fam, sid, stx = [], [], [], [], [], [], []
    r = 0
    for si, s in enumerate(probe):
        try:
            ids, pay, inj = X.injection_span(tok, s)
        except Exception:
            continue
        inj_set = set(inj)
        legit = [k for k in pay if k not in inj_set]
        if len(inj) < 3 or len(legit) < 3:
            continue
        f, t_ = family(s["injection_text"]), s["injection_text"].strip()
        for k in pay:
            # exactly the reviewer's null: current token id + bag of previous WINDOW ids
            feats = Counter()
            feats[hash(("cur", ids[k])) % DIM] += 1.0
            for d in range(1, WINDOW + 1):
                if k - d >= 0:
                    feats[hash(("prev", ids[k - d])) % DIM] += 1.0
            for c, v in feats.items():
                rows.append(r); cols.append(c); vals.append(v)
            y.append(1 if k in inj_set else 0)
            fam.append(f); sid.append(si); stx.append(t_)
            r += 1
        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{len(probe)}", flush=True)

    Xs = sparse.csr_matrix((vals, (rows, cols)), shape=(r, DIM))
    y = np.array(y); fam = np.array(fam); sid = np.array(sid); stx = np.array(stx)
    print(f"\nrows={r}  injected={int(y.sum())}  legit={int((1-y).sum())}")

    def fit(tr, te):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            return float("nan")
        c = LogisticRegression(max_iter=3000, C=X.PROBE_C, fit_intercept=True,
                               class_weight="balanced").fit(Xs[tr], y[tr])
        return float(roc_auc_score(y[te], c.decision_function(Xs[te])))

    ids_ = np.unique(sid); r2 = np.random.default_rng(0); r2.shuffle(ids_)
    hold = set(ids_[: int(len(ids_) * .30)].tolist())
    by_sample = fit(~np.isin(sid, list(hold)), np.isin(sid, list(hold)))

    st = np.unique(stx); r3 = np.random.default_rng(0); r3.shuffle(st)
    hs = set(st[: int(len(st) * .30)].tolist())
    by_string = fit(~np.isin(stx, list(hs)), np.isin(stx, list(hs)))

    pm = fam == "priority_message"
    pm2o = fit(pm, ~pm)
    o2pm = fit(~pm, pm)

    print(f"\n{'model':28s} {'by-sample':>11s} {'by-STRING':>11s} "
          f"{'PM->other':>11s} {'other->PM':>11s}")
    print("-" * 76)
    print(f"{'BoW null (no activations)':28s} {by_sample:11.3f} {by_string:11.3f} "
          f"{pm2o:11.3f} {o2pm:11.3f}")
    print(f"{'activation probe L8':28s} {1.000:11.3f} {1.000:11.3f} "
          f"{0.998:11.3f} {0.954:11.3f}")
    print("\nIf the null matches the activation probe on the FAMILY holdouts too, the")
    print("activation result is not evidence of a represented feature.")


if __name__ == "__main__":
    main()
