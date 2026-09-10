"""Role probes and steering directions: unpickling, the GPU logistic-regression fitter, probe training, the injection probe, and building unit directions with magnitude matching."""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .common import *  # noqa: F401,F403
from .corpora import load_corpus
from .model import layer_container, register_probe_capture
from .spans import injection_span
from .steering import ctrl_kind
from .templates import (apply_template, render_single, role_messages,
                        sentinel_span, supported_roles, token_span)


# ════════════════════════════════════════════════════════════ probes
def load_probe(path):
    """Unpickle a probe_L*.pkl from ANY entrypoint.

    The pickles are written while this file runs as `__main__`, so the fitted probe's class
    is recorded as `__main__.TorchLogReg`. Loading from another script (tools/probe_report.py,
    tools/controls/*) then dies with `AttributeError: Can't get attribute 'TorchLogReg' on
    <module '__main__' ...>`. Aliasing the class onto whatever __main__ currently is makes
    the existing pickles loadable without refitting.
    """
    import __main__
    if not hasattr(__main__, "TorchLogReg"):
        __main__.TorchLogReg = TorchLogReg
    with open(path, "rb") as f:
        return pickle.load(f)


class TorchLogReg:
    """sklearn-compatible surface for a probe fitted on GPU.

    Exposes coef_ / intercept_ / classes_ / predict / decision_function so everything
    downstream (build_probes, probe_report) is unchanged.
    """

    def __init__(self, coef, intercept, classes):
        self.coef_, self.intercept_, self.classes_ = coef, intercept, classes

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float32) @ self.coef_.T + self.intercept_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def fit_logreg_gpu(X, y, C, device, max_iter=1000):
    """Multinomial L2 logistic regression on GPU, sklearn's objective exactly:

        min_W  0.5*||W||^2 + C * sum_i CE(W x_i + b, y_i)

    The paper fits these with cuML on GPU; doing it on CPU with sklearn was ~5 min per
    layer and dominated the whole pipeline. LBFGS with strong-Wolfe matches sklearn's
    solver, so the fitted direction is the same object, just ~100x faster.
    """
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long, device=device)
    classes = np.unique(y)
    remap = torch.as_tensor(
        np.searchsorted(classes, y).astype(np.int64), device=device)
    k, d = len(classes), Xt.shape[1]
    W = torch.zeros(k, d, device=device, requires_grad=True)
    b = torch.zeros(k, device=device, requires_grad=True)
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=10,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-5)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = C * ce(Xt @ W.T + b, remap) + 0.5 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    del Xt, yt, remap
    torch.cuda.empty_cache()
    return TorchLogReg(W.detach().float().cpu().numpy(),
                       b.detach().float().cpu().numpy(), classes)

def train_probes(model, tok, outdir, layers, n_seqs, max_content_tokens,
                 no_think=False, C=PROBE_C, skip_ovr=False, corpus_kind="paper",
                 skip_first_n=0):
    """One-vs-rest probe per role at each layer, plus composite steering directions."""
    os.makedirs(outdir, exist_ok=True)
    roles = supported_roles(tok, no_think)
    print(f"[probe] roles supported: {roles}", flush=True)
    for need in ("tool", "user"):
        if need not in roles:
            raise SystemExit(f"template must support `{need}`")

    seqs = load_corpus(tok, n_seqs, kind=corpus_kind)
    blocks = layer_container(model)

    cap: dict[int, torch.Tensor] = {}

    def mk(L):
        def store(t, L=L):
            cap[L] = t.detach()
        return store

    handles, site_name = [], None
    for L in layers:
        h_, site_name = register_probe_capture(blocks[L], mk(L))
        handles.append(h_)
    print(f"[probe] {len(seqs)} seqs | layers {layers} | site {site_name} | C={C} | "
          f"content tokens {'uncapped' if not max_content_tokens else max_content_tokens}",
          flush=True)
    feats = {L: [] for L in layers}
    labels, groups = [], []
    pad_rng = np.random.default_rng(1234)
    t0 = time.time()
    for si, seq in enumerate(seqs):
        for ri, role in enumerate(roles):
            # PAPER CONSTRUCTION: one standalone message per role, identical except for
            # the header (see render_single). Falls back to the multi-message form only
            # for templates without a harmony-style header we can emit directly, and only
            # there is the positional pad needed.
            single = render_single(tok, role, "X") is not None
            if single:
                got = sentinel_span(
                    tok, lambda c, r=role: render_single(tok, r, c, TOOL_NAME), seq)
            else:
                npad = int(pad_rng.integers(0, 200))
                got = sentinel_span(
                    tok, lambda c, r=role, k=npad: apply_template(
                        tok, role_messages(r, c, k), TOOLS, no_think=no_think), seq)
            if not got:
                continue
            text, span = got
            ids, idx = token_span(tok, text, span)
            # PAPER: SKIP_FIRST_N = 32 if NESTED_REASONING else 0, applied as
            # token_in_seg_ix >= SKIP_FIRST_N (02-train-role-probes.ipynb). Early content
            # tokens sit inside the header's local context; nested-reasoning model
            # families are fit without them.
            if skip_first_n:
                idx = idx[skip_first_n:]
            if max_content_tokens:
                idx = idx[:max_content_tokens]
            if not idx:
                continue
            cap.clear()
            with torch.no_grad():
                model(torch.tensor([ids], device=model.device))
            # CPU index tensor, deliberately: under device_map=auto each cap[L] lives on
            # its own layer's shard, and a cuda:0 index tensor hard-crashes indexing on
            # any other device (adversarial review, 2026-08-31). CPU indices are legal
            # against every CUDA tensor.
            sel = torch.tensor(idx)
            for L in layers:
                feats[L].append(cap[L][0, sel].float().cpu().numpy().astype(np.float16))
            labels.append(np.full(len(idx), ri, dtype=np.int64))
            groups.append(np.full(len(idx), si, dtype=np.int64))
        if (si + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[probe] {si+1}/{len(seqs)}  {el:.0f}s  "
                  f"eta {el/(si+1)*(len(seqs)-si-1):.0f}s", flush=True)
    for h in handles:
        h.remove()

    y, g = np.concatenate(labels), np.concatenate(groups)
    rng = np.random.default_rng(0)
    u = np.unique(g)
    rng.shuffle(u)
    test = set(u[: int(len(u) * .25)].tolist())      # split by SEQUENCE, never by token
    tm = np.array([x in test for x in g])

    report = {}
    for L in layers:
        X = np.concatenate(feats[L]).astype(np.float32)
        Xtr, Xte, ytr, yte = X[~tm], X[tm], y[~tm], y[tm]
        ovr, accs, W = {}, {}, {}
        fit_ix = (np.random.default_rng(0).permutation(len(Xtr))[:MN_FIT_ROWS]
                  if len(Xtr) > MN_FIT_ROWS else np.arange(len(Xtr)))
        # The one-vs-rest probes cost 5 of the 6 fits per layer and are needed only for
        # the tool_ovr-family STEERING directions -- never for probe validation, which
        # uses the multinomial. Skipping them turns 72 CPU LBFGS fits into 12.
        if not skip_ovr:
            for ri, role in enumerate(roles):
                clf = LogisticRegression(max_iter=2000, C=C, fit_intercept=True)
                clf.fit(Xtr[fit_ix], (ytr[fit_ix] == ri).astype(int))
                ovr[role] = clf
                W[role] = clf.coef_[0]
                accs[role] = float(
                    accuracy_score((yte == ri).astype(int), clf.predict(Xte)))

        # MULTINOMIAL probe -- this is what the paper fits (one softmax over the role
        # space), and it is the right object for a "boost tool" direction. Under a softmax
        # the class weight vectors are coupled and sum-to-zero, so raising the tool logit
        # MECHANICALLY depresses user/system: w_tool already IS "toward tool, away from the
        # other roles". Building `tool - mean(user, system)` on top of ONE-VS-REST weights
        # (as earlier versions did) double-subtracts and distorts the axis.
        # Subsample for the softmax fit: full-data multinomial lbfgs took ~20 min PER
        # LAYER (vs seconds for the one-vs-rest fits). With C=5e-3 on 2880 dims this many
        # rows is far past what a linear probe needs, and the direction is unchanged.
        mn = fit_logreg_gpu(Xtr[fit_ix], ytr[fit_ix], C, model.device)
        mn_acc = float(accuracy_score(yte, mn.predict(Xte)))
        Wm = {r: mn.coef_[list(mn.classes_).index(i)]
              for i, r in enumerate(roles) if i in mn.classes_}

        others = [r for r in roles if r != "tool"]
        us = [r for r in ("user", "system") if r in roles]   # used by DIM dirs too
        # primary: multinomial tool-logit gradient (couples the classes, so raising tool
        # mechanically depresses user/system -- no hand-built composite needed)
        dirs = {"mn_tool": Wm["tool"] / (np.linalg.norm(Wm["tool"]) + 1e-12)}
        if not skip_ovr:
            U = {r: W[r] / (np.linalg.norm(W[r]) + 1e-12) for r in W}
            dirs.update({
                "tool_ovr": U["tool"],
                "tool_vs_user_system": U["tool"] - np.mean([U[r] for r in us], axis=0),
                "tool_vs_rest": U["tool"] - np.mean([U[r] for r in others], axis=0),
                "tool_vs_system": (U["tool"] - U["system"]) if "system" in U else U["tool"],
                "tool_vs_user": U["tool"] - U["user"],
            })
        # SOTA: DIFFERENCE-IN-MEANS. A logistic-regression weight vector is optimal for
        # CLASSIFICATION, not for intervention. Under the linear representation hypothesis
        # the difference of class means is the provably optimal steering direction and is
        # what CAA / Arditi et al. use. Provide both and let the sweep decide.
        M = {r: Xtr[ytr == roles.index(r)].mean(axis=0) for r in roles}
        Mu = {r: M[r] / (np.linalg.norm(M[r]) + 1e-12) for r in M}
        dirs.update({
            "dim_tool_ovr": M["tool"] - Xtr.mean(axis=0),
            "dim_tool_vs_user_system": M["tool"] - np.mean([M[r] for r in us], axis=0),
            "dim_tool_vs_rest": M["tool"] - np.mean([M[r] for r in others], axis=0),
        })
        # USER axes -- for SUPPRESSING userness (--mode ablate) instead of boosting tool.
        # Motivated by measurement, not symmetry:
        #   * injected text is ALREADY more tool-like than the legitimate record beside it
        #     (9/12 layers raw tool logit, 11/12 on the margin), so adding toward `tool` has
        #     little headroom -- see tools/controls/toolness_control.py.
        #   * userness DOES track compliance: succeeded attacks are more user-like than
        #     blocked ones at 12/12 layers (the attack-prediction test in probe_report.py).
        # Difference-in-means, NOT the raw class mean: Mu["user"] is cos ~0.99 with the
        # global activation mean, so ablating it removes ~45% of every token (capability
        # destruction, the failure recorded at the ablation site below).
        if "user" in roles:
            user_others = [r for r in roles if r != "user"]
            dirs.update({
                "dim_user_vs_rest": M["user"] - np.mean([M[r] for r in user_others], axis=0),
                "mn_user": Wm["user"] / (np.linalg.norm(Wm["user"]) + 1e-12),
            })
        # ITI convention: sigma = std of activations PROJECTED onto the unit direction.
        # Scaling by alpha*sigma is calibrated to the spread the model actually uses along
        # that axis; scaling by ||h|| is not (it is dominated by directions we never touch).
        sigmas, ablate = {}, {}
        for k_, v_ in dirs.items():
            # fp32, NOT fp64: Xtr is fp32, so an fp64 `u_` upcasts the whole 445k x 2880
            # matrix for the projection -- measured 4.5s vs 0.1s per direction, 45x, and
            # ~24% of the entire probe stage across 12 layers. The extra fp64 precision is
            # far below activation noise.
            u_ = np.asarray(v_, dtype=np.float32)
            u_ = u_ / (np.linalg.norm(u_) + 1e-12)
            sigmas[k_] = float((Xtr @ u_).std())
        # unit role axes kept separately: needed for directional ABLATION (project the
        # impersonated-role component out) rather than addition
        for r in ("user", "system"):
            if r in Mu:
                ablate[r] = Mu[r]
        cosmat = {a: {b: float(dirs[a] @ dirs[b] /
                               (np.linalg.norm(dirs[a]) * np.linalg.norm(dirs[b])))
                      for b in dirs} for a in dirs}
        report[L] = {"acc": accs, "mn_acc": mn_acc,
                     "mean_norm": float(np.linalg.norm(X, axis=1).mean()),
                     "dir_cos": cosmat}
        print(f"[probe] L{L:<3d} " +
              ("  ".join(f"{r}={accs[r]:.3f}" for r in roles) + "  | " if accs else "") +
              f"multinomial={mn_acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        with open(f"{outdir}/probe_L{L}.pkl", "wb") as f:
            pickle.dump({"layer": L, "site": site_name, "C": C, "roles": roles,
                         "ovr": ovr, "weights": W, "mn": mn, "mn_weights": Wm,
                         "dirs": dirs, "sigmas": sigmas,
                         "ablate_axes": ablate, "role_means": M,
                         "mean_norm": report[L]["mean_norm"]}, f)
        del X, Xtr, Xte
    json.dump({"layers": layers, "roles": roles, "site": site_name, "C": C,
               "corpus_kind": corpus_kind, "n_seqs": len(seqs),
               "report": report}, open(f"{outdir}/probe_report.json", "w"), indent=2)
    return roles, report



def train_injection_probe(model, tok, outdir, layers, samples, no_think=False,
                          C=PROBE_C, holdout=0.30):
    """Supervised INJECTION probe: injected tokens vs legitimate record tokens.

    Far stronger supervision than the C4 role probes, which only assume the role signal
    transfers. Here the positive class is literally the attacker's tokens and the negative
    class is the surrounding legitimate record, in the same prompt, same tag, same model.

    Direction convention: `inj_dim` = mean(legit) - mean(injected), so ADDING it makes an
    injected token look like ordinary record data. Difference-in-means is the provably
    optimal steering direction under the linear representation hypothesis (CAA / Arditi),
    whereas a logistic weight vector is optimal only for classification -- both are saved.

    Trained ONLY on the probe-dev split; sweep-dev and test are never touched.
    """
    blocks = layer_container(model)
    cap: dict[int, torch.Tensor] = {}

    def mk(L):
        def store(t, L=L):
            cap[L] = t.detach()
        return store
    handles, site_name = [], None
    for L in layers:
        h_, site_name = register_probe_capture(blocks[L], mk(L))
        handles.append(h_)

    P = {L: [] for L in layers}     # injected tokens
    N = {L: [] for L in layers}     # legitimate record tokens
    gp, gn = [], []                  # sample id per row, for grouped splitting
    used = 0
    for si, smp in enumerate(samples):
        try:
            ids, pay, inj = injection_span(tok, smp, no_think)
        except Exception:
            continue
        inj_set = set(inj)
        legit = [k for k in pay if k not in inj_set]
        if len(inj) < 3 or len(legit) < 3:
            continue
        cap.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=model.device))
        for L in layers:
            h = cap[L][0].float().cpu().numpy().astype(np.float16)
            P[L].append(h[inj])
            N[L].append(h[legit])
        gp += [si] * len(inj)
        gn += [si] * len(legit)
        used += 1
        if (si + 1) % 50 == 0:
            print(f"[inj-probe] {si+1}/{len(samples)} ({used} usable)", flush=True)
    for h in handles:
        h.remove()
    print(f"[inj-probe] {used}/{len(samples)} samples usable", flush=True)

    gp, gn = np.array(gp), np.array(gn)
    ids_all = np.unique(np.concatenate([gp, gn]))
    rng = np.random.default_rng(0)
    rng.shuffle(ids_all)
    hold = set(ids_all[: int(len(ids_all) * holdout)].tolist())   # split by SAMPLE
    report = {}
    for L in layers:
        Xp, Xn = np.concatenate(P[L]).astype(np.float32), np.concatenate(N[L]).astype(np.float32)
        mp, mn = np.array([g in hold for g in gp]), np.array([g in hold for g in gn])
        Xtr = np.concatenate([Xp[~mp], Xn[~mn]])
        ytr = np.concatenate([np.ones((~mp).sum()), np.zeros((~mn).sum())])
        Xte = np.concatenate([Xp[mp], Xn[mn]])
        yte = np.concatenate([np.ones(mp.sum()), np.zeros(mn.sum())])
        clf = LogisticRegression(max_iter=2000, C=C, fit_intercept=True,
                                 class_weight="balanced")
        clf.fit(Xtr, ytr)
        acc = float(accuracy_score(yte, clf.predict(Xte)))
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(yte, clf.decision_function(Xte)))
        dim = Xn[~mn].mean(axis=0) - Xp[~mp].mean(axis=0)      # legit minus injected
        u = dim / (np.linalg.norm(dim) + 1e-12)
        report[L] = {"acc": acc, "auc": auc, "n_inj": int(len(Xp)), "n_legit": int(len(Xn)),
                     "sigma": float((Xtr @ u).std())}
        print(f"[inj-probe] L{L:<3d} holdout acc={acc:.3f} AUC={auc:.3f}  "
              f"(inj tokens {len(Xp)}, legit {len(Xn)})", flush=True)
        pf = f"{outdir}/probe_L{L}.pkl"
        d = pickle.load(open(pf, "rb"))
        d["dirs"]["inj_dim"] = dim
        d["dirs"]["inj_probe"] = -clf.coef_[0]     # negate: toward legit, away from inj
        d.setdefault("sigmas", {})["inj_dim"] = report[L]["sigma"]
        d["sigmas"]["inj_probe"] = float((Xtr @ (-clf.coef_[0] /
                                          (np.linalg.norm(clf.coef_[0]) + 1e-12))).std())
        d["injection_probe"] = {"clf": clf, "acc": acc, "auc": auc}
        pickle.dump(d, open(pf, "wb"))
        del Xp, Xn, Xtr, Xte
    json.dump(report, open(f"{outdir}/injection_probe_report.json", "w"), indent=2)
    return report



def build_probes(probe_dir, layers, device):
    """(W, b, tool_row) per layer for the multinomial probe, as GPU tensors.

    Used by conditional steering to score each token's p(tool) inside the hook.
    """
    out = []
    for L in layers:
        p = pickle.load(open(f"{probe_dir}/probe_L{L}.pkl", "rb"))
        mn = p.get("mn")
        if mn is None:
            raise SystemExit(f"probe_L{L}.pkl has no multinomial probe; re-run --stage probe")
        roles = p["roles"]
        classes = list(mn.classes_)
        ti = classes.index(roles.index("tool"))
        W = torch.tensor(mn.coef_, dtype=torch.float32, device=device)
        b = torch.tensor(mn.intercept_, dtype=torch.float32, device=device)
        out.append((W, b, ti))
    return out


def build_dirs(probe_dir, layers, name, device, seed=0, match_sigma_to=None,
               require_sigma=False):
    """(unit directions, sigmas, ablation axes) per layer.

    `random`/`shuffled` are magnitude-matched controls and go through the SAME
    normalisation path as the real directions -- otherwise the control is not matched.

    `require_sigma=True` makes a MISSING OR ZERO sigma fatal HERE, at preflight, instead of
    at the first steered token. Pass it from every caller that will run `--scale sigma`.

    WHY IT IS A SEPARATE ARGUMENT AND NOT ALWAYS ON. Sigma is only the dose unit under
    `--scale sigma`; under `--scale norm` the step is alpha*||h|| and a stored 0.0 is
    harmless. And `src/cli.py`'s config dump calls this purely to RECORD effective sigmas,
    where raising would abort a run over bookkeeping.

    WHY IT EXISTS AT ALL (2026-09-02). `merge_probe_axis_dir.py:65` writes
    `sigmas["probe_axis_user"] = sigmas["probe_axis_tool"] = 0.0` with only a WARNING
    whenever no `steer_probe_readout` artifact was available -- which is the case for every
    model of the second bring-up wave (GLM-4.5-Air, Qwen3-Next-80B, Gemma-4-31B all carry
    0.0 for both axes). `Steer._mk` does raise on `sigma <= 0` under `--scale sigma`, but it
    raises from inside a forward hook AFTER the clean and base-XPIA arms have already
    generated -- on a 106B model that is hours of GPU spent to learn a fact readable from a
    pickle. This project has already lost two runs to interventions that silently did
    nothing (FINDINGS 1.1 sign-inverted gate, 23e ablate no-op); a third would be on us.
    """
    ck = ctrl_kind(name)
    rng = np.random.default_rng(ck[1] if ck else seed)
    out, sig, abl = [], [], []
    for L in layers:
        p = pickle.load(open(f"{probe_dir}/probe_L{L}.pkl", "rb"))
        if ck:
            base = p["dirs"][match_sigma_to] if match_sigma_to in p["dirs"] \
                else p["dirs"]["tool_ovr"]
            v = (rng.normal(size=np.asarray(base).shape) if ck[0] == "random"
                 else rng.permutation(np.asarray(base).copy()))
        else:
            if name not in p["dirs"]:
                raise SystemExit(f"unknown direction {name}; have {list(p['dirs'])}")
            v = p["dirs"][name]
        t = torch.tensor(np.asarray(v), dtype=torch.float32)
        u = (t / t.norm()).to(device)
        out.append(u)
        if ck:
            # A control that borrows another direction's sigma is NOT magnitude-matched:
            # under --scale sigma the step is alpha*sigma, so borrowing tool_ovr's sigma
            # gave `random` 8-11.6x LESS displacement than inj_dim at the same alpha, and
            # every "the real direction blocks, random does not" result was explained by
            # step size alone. Match the sigma of the direction being controlled for.
            sig.append(float(p.get("sigmas", {}).get(match_sigma_to, 0.0))
                       or float(p.get("sigmas", {}).get("tool_ovr", 0.0)))
        elif match_sigma_to and match_sigma_to in p.get("sigmas", {}):
            # REAL directions honour match_sigma_to too. Previously this branch was
            # control-only, so every real direction took its OWN stored sigma -- and sigma
            # is `(A @ u).std()` over whatever corpus the direction was BUILT against
            # (build_override_direction.py). Two directions fit from different SRC files
            # therefore have sigmas with different denominators, and at the same alpha they
            # apply different steps for a purely bookkeeping reason. Measured on a common
            # reference set the true ratio is 0.93-0.99, while the stored ratio reads
            # 0.79-0.91. Passing --match-sigma-to makes every arm take one direction's
            # sigma, so alpha*sigma is identical across the whole table by construction.
            sig.append(float(p["sigmas"][match_sigma_to]))
        else:
            sig.append(float(p.get("sigmas", {}).get(name, 0.0)))
        abl.append({r: torch.tensor(np.asarray(a), dtype=torch.float32).to(device)
                    for r, a in p.get("ablate_axes", {}).items()})
    if require_sigma:
        bad = [(L, s) for L, s in zip(layers, sig)
               if not (isinstance(s, float) and s > 0.0 and math.isfinite(s))]
        if bad:
            src = match_sigma_to or name
            raise SystemExit(
                f"PREFLIGHT: direction `{name}` resolves to a NON-POSITIVE sigma at "
                f"{[(f'L{L}', s) for L, s in bad]} (sigma taken from `{src}` in "
                f"{probe_dir}/probe_L*.pkl). Under --scale sigma the step is alpha*sigma, "
                f"so this arm would apply a ZERO edit and report itself as a defense.\n"
                f"  Fixes, in order of preference:\n"
                f"   1. anchor on a direction that HAS a sigma: pass --match-sigma-to "
                f"dim_no_override_both (or build a composed key with "
                f"tools/controls/build_combo_direction.py --sigma-ref dim_no_override_both, "
                f"which bakes its own sigma and must then run at --alphas 1.0 with NO "
                f"--match-sigma-to);\n"
                f"   2. populate the missing sigma: "
                f"tools/controls/merge_probe_axis_dir.py --sigma-from <steer_probe_readout "
                f"artifact> (it warns and writes 0.0 when that artifact is absent);\n"
                f"   3. run --scale norm, where alpha multiplies ||h|| and sigma is unused.")
    return out, sig, abl


MU_SOURCES = ("probe_grand", "role_mean", "span", "capture:<json>")
# The default is the EXACT probe-corpus grand mean: it is recoverable from every pickle the
# probe stage writes, it needs no extra capture staged beside the run, and it is measured
# identical to the role-balanced mean to <1e-3 sigma in projection on every model checked.
MU_DEFAULT = "probe_grand"


STEER_SITE = "block_out"          # src/model.pick_site -- where the edit actually lands


def _site_note(site, source):
    """Say, at run time, whether this mu was measured where the edit is applied.

    Silent site mismatch is how a "zero net displacement by construction" claim becomes
    false without anything failing: the mean of the PRE-MLP residual is not the mean of the
    BLOCK OUTPUT, and only the latter is what the hook edits."""
    if site == STEER_SITE:
        print(f"[mu] site={site} MATCHES the steer site -- the zero-net-displacement "
              f"property is exact up to the span/corpus mismatch", flush=True)
    else:
        print(f"[mu] SITE CAVEAT: this mu was measured at `{site}` but the hook edits "
              f"`{STEER_SITE}`. The mean shift is REDUCED, NOT ZEROED; the residual is a "
              f"small additive dose of unmeasured size on this model (measured 0.4-2.7 "
              f"sigma/layer on gpt-oss-20b). Pair this arm with --mu-source span, whose "
              f"mu is taken at the steer site by construction. (source=`{source}`)",
              flush=True)


def mu_projection(mean_acts, dirs, sigmas):
    """`mu . d_hat` per steered layer, in SIGMA units.

    This is exactly MINUS the net displacement per edited token that plain `ablate` applies,
    so it is the size of the confound `ablate_mp` removes. ONE implementation, called both
    by the sweep (src/cli.py, which has the run's real `dirs`/`sigmas` including any
    --match-sigma-to substitution) and by the launcher's preflight -- computing it twice
    would let the two disagree the moment a sigma is substituted.

    -> list of floats, one per layer; nan where the sigma is 0.
    """
    out = []
    for m, u, s in zip(mean_acts, dirs, sigmas):
        if not s:
            out.append(float("nan"))
            continue
        uu = u.float()
        uu = uu / uu.norm().clamp_min(1e-6)      # build_dirs already unit-norms; be robust
        out.append(float(m.float() @ uu) / s)
    return out


def build_means(probe_dir, layers, device, source=MU_DEFAULT, model_id=None,
                hidden_size=None):
    """Per-layer FIXED mean activation vector `mu`, for MEAN-PRESERVING ablation.

    `h <- h - ((h - mu).d)d` deletes the direction's coordinate while contributing zero net
    displacement along `d`, PROVIDED mu is the mean of the tokens being edited. Plain
    ablation is the mu=0 case, and on Qwen3-Next-80B that is not neutral: the coordinate's
    mean sits at +2.26/+0.95/-0.39 sigma at L28/32/40, so deleting it pushes -2.83 sigma
    NET along d -- additive steering at alpha ~ -1.6 in the ATTACK-favouring direction
    (FINDINGS §23k). This function is where that mean comes from.

    ══ THE PROVISO IS NOT SATISFIED EXACTLY BY ANY STORED mu, AND THE REASON IS THE SITE ══
    The probe pickles are captured at `post_attention_layernorm` (the PRE-MLP residual);
    the steering hook edits the BLOCK OUTPUT (`pick_site`). CLAUDE.md sanctions that split
    deliberately -- for the PROBE and the DIRECTION. A MEAN is a different object: the
    operator's premise is `mu ~ E[h_edited]`, and a stored mu is the mean of a different
    tensor at a different point in the block. Adversarial review (2026-09-02) measured the
    consequence on gpt-oss-20b from `runs/step_boundary_calibration.json` (which IS captured
    at `block_out`): the residual net displacement is 0.4-2.7 sigma per layer -- reduced,
    NOT zero, and the same order as the 0.83-0.87 sigma class separation the operator is
    meant to delete. On Qwen3-Next-80B and GLM-4.5-Air the steer-site mean has never been
    measured, so the size of the residual there is UNKNOWN.

    So: a fixed mu removes MOST of the mean shift, and the remainder is a small additive
    dose of unmeasured size. `span` is the only source with no such residual, because it is
    the mean of the edited activations themselves, at the site where they are edited -- at
    the cost of preserving the span-level offset along d (see Steer's hook). The two
    SOURCES BRACKET the operator and are meant to be run as a pair, not chosen between.

    SOURCES (all read the SAME artifacts the direction itself is loaded from, so a defended
    arm needs no new capture):

      probe_grand  (default) the EXACT grand mean of the probe-training activations,
                   recovered as `role_means["tool"] - dirs["dim_tool_ovr"]`: train_probes
                   writes `dim_tool_ovr = M["tool"] - Xtr.mean(0)`, so this identity is
                   exact, not an estimate. Row-weighted over the fit corpus.
      role_mean    the equal-weight mean of the five stored per-role means. Measured
                   identical to `probe_grand` to <1e-3 sigma in projection on all three
                   models checked (gpt-oss-20b, Qwen3-Next-80B, GLM-4.5-Air), because the
                   probe corpus renders the SAME content under every role header and so is
                   role-balanced by construction. Kept as an independent cross-check.
      capture:PATH the mean of the injected-span activations the DIRECTION was fit on, read
                   from an override-slope json (`rows[*].act[str(L)]`). Closest to the
                   distribution actually edited at run time, and therefore the mu for which
                   "zero net displacement" is most nearly true -- but the file is a
                   several-hundred-MB per-model artifact that is not always staged beside
                   the pickles, which is why it is not the default.

    `span` (each row's own mean over its edited positions) is NOT built here: it is not a
    stored vector but a per-input quantity, and it is passed to Steer as
    `mean_from_span=True`.

    -> list of float32 tensors on `device`, one per layer, in `layers` order.
    """
    if source == "span":
        raise SystemExit("build_means: `span` is computed inside the hook from the input, "
                         "not loaded -- pass mean_from_span=True to Steer instead.")
    cap_rows = None
    if source.startswith("capture:"):
        cap_path = source.split(":", 1)[1]
        if not os.path.exists(cap_path):
            raise SystemExit(f"--mu-source {source}: {cap_path} does not exist")
        _d = json.load(open(cap_path))
        cap_rows = _d.get("rows")
        if not cap_rows:
            raise SystemExit(f"--mu-source {source}: {cap_path} has no `rows`")
        cap_cfg = _d.get("config") or {}
        cap_model, cap_split = cap_cfg.get("model"), cap_cfg.get("split")
        cap_site = _d.get("site") or cap_cfg.get("site")
        print(f"[mu] capture {cap_path}: {len(cap_rows)} rows, layers {_d.get('layers')}, "
              f"config.model={cap_model} config.split={cap_split} site={cap_site}",
              flush=True)
        _site_note(cap_site, source)
        # HOLDOUT. mu becomes part of the operator, so it is a FIT artifact: averaging rows
        # from the dev or test split would put evaluation data inside the defense. The
        # captures these directions are fit from are `probe`-split by construction; anything
        # else is refused rather than warned about, because a warning in a 5-hour log is not
        # a control.
        if cap_split is not None and cap_split != "probe":
            raise SystemExit(
                f"--mu-source {source}: that capture is split=`{cap_split}`, not `probe`. "
                f"mu is part of the operator, so building it from dev/test rows puts "
                f"evaluation data inside the defense. Use a probe-split capture, or "
                f"--mu-source {MU_DEFAULT}.")
        if cap_split is None:
            print(f"[mu] WARNING: {cap_path} records no `config.split`; its holdout status "
                  f"cannot be checked from the artifact", flush=True)
        # A DIMENSION CHECK IS NOT ENOUGH HERE. Qwen3-30B-A3B-Thinking and Qwen3-Next-80B
        # both have hidden_size 2048, so the wrong model's capture loads silently and its
        # mean is simply a different vector -- which would make the "zero net displacement"
        # claim false while everything still ran. The capture records the model it was taken
        # on; compare it.
        if model_id and not cap_model:
            raise SystemExit(
                f"--mu-source {source}: {cap_path} records no `config.model`, so it cannot "
                f"be shown to belong to `{model_id}`. Activation means do not transfer "
                f"across models and equal hidden sizes hide the mistake -- refusing.")
        if model_id and cap_model and cap_model != model_id:
            raise SystemExit(
                f"--mu-source {source}: that capture was taken on `{cap_model}`, but this "
                f"run is `{model_id}`. Activation means are NOT transferable across models "
                f"(and equal hidden sizes make this undetectable by shape). Use the capture "
                f"for this model, or --mu-source probe_grand.")
    elif source not in [s for s in MU_SOURCES if ":" not in s and s != "span"]:
        raise SystemExit(f"unknown mu source {source!r}; have {list(MU_SOURCES)}")

    out = []
    for L in layers:
        if cap_rows is not None:
            acc, n = None, 0
            for r in cap_rows:
                v = (r.get("act") or {}).get(str(L))
                if v is None:
                    raise SystemExit(
                        f"--mu-source {source}: a row carries no activation for layer {L}. "
                        f"The capture must cover every steered layer.")
                a = np.asarray(v, dtype=np.float64)
                acc = a if acc is None else acc + a
                n += 1
            mu = acc / n
        else:
            p = pickle.load(open(f"{probe_dir}/probe_L{L}.pkl", "rb"))
            if L == layers[0]:
                _site_note(p.get("site"), source)
            M = p.get("role_means")
            if not M:
                raise SystemExit(
                    f"{probe_dir}/probe_L{L}.pkl has no `role_means`, so a mean-preserving "
                    f"ablation has no mu. Re-run --stage probe, or pass "
                    f"--mu-source capture:<override-slope json>. Refusing to fall back to "
                    f"mu=0, which would silently run plain `ablate` under this arm's name.")
            if source == "probe_grand":
                if "tool" not in M or "dim_tool_ovr" not in p.get("dirs", {}):
                    raise SystemExit(
                        f"{probe_dir}/probe_L{L}.pkl lacks role_means['tool'] or "
                        f"dirs['dim_tool_ovr'], so the probe-corpus grand mean cannot be "
                        f"recovered. Use --mu-source role_mean or capture:<json>.")
                mu = (np.asarray(M["tool"], dtype=np.float64)
                      - np.asarray(p["dirs"]["dim_tool_ovr"], dtype=np.float64))
            else:
                mu = np.mean([np.asarray(M[r], dtype=np.float64) for r in M], axis=0)
        t = torch.tensor(np.asarray(mu), dtype=torch.float32)
        if not torch.isfinite(t).all():
            raise SystemExit(f"mu for layer {L} from `{source}` is not finite")
        # A zero mu is NUMERICALLY plain `ablate`. The construction guards stop a MISSING
        # mu; this stops a degenerate one, so "never silently plain ablate" rests on the
        # value and not only on the plumbing.
        if float(t.norm()) == 0.0:
            raise SystemExit(
                f"mu for layer {L} from `{source}` is all zeros -- that is numerically "
                f"identical to plain `ablate`, so this arm would not be mean-preserving.")
        # A shape mismatch would otherwise surface as a broadcast RuntimeError INSIDE the
        # hook, i.e. after the clean and base-XPIA arms have already generated. Cheap
        # pre-flight checks go ahead of the long job.
        if hidden_size and t.numel() != hidden_size:
            raise SystemExit(
                f"mu for layer {L} from `{source}` has {t.numel()} dims but this model's "
                f"hidden_size is {hidden_size} -- wrong artifact for this model.")
        out.append(t.to(device))
    print(f"[mu] source=`{source}` layers={list(layers)} "
          f"|mu|={[round(float(t.norm()), 2) for t in out]}", flush=True)
    return out
