"""Does our role probe reproduce the paper's role-confusion result? Tables + figures.

Four questions, in order. Each must pass before the next means anything.

  Q1 SEPARATION   Does the probe separate roles at all? (per-role accuracy by layer)
  Q2 TAG CONTROL  Identical text in <user> vs <tool> must move the score. If it does not,
                  the probe is not reading role and nothing else is interpretable.
  Q3 CONFUSION    THE PAPER'S CLAIM. Inside the SAME <tool> message, injected text must
                  read as MORE USER-LIKE than the legitimate record content beside it,
                  and more user-like than the same payload with no injection at all.
  Q4 PREDICTIVE   Userness must be higher on the samples that actually tricked the model.

"userness" = user_logit - tool_logit from the multinomial probe, in LOGIT space.
Probabilities saturate (p_tool ~ 0.994) and hide the drift -- that saturation already
invalidated one conclusion in this project, so nothing here uses softmax.

Writes tables to stdout + runs/probe_report.json, and figures to runs/figs/.
"""
import json
import os
import pickle
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/datadrive/huggingface/")
import xpia_defense as X  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/gpt-oss-20b-paper"
FIGS = f"{ROOT}/runs/figs"
# Q4 needs per-sample attack-success labels. They come from UNSTEERED `base-XPIA`
# completions, so a run whose steering was invalidated is still a valid source here --
# the base model's behaviour does not depend on the probe. Override with argv[2].
BASE_RUN = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/runs/gpt-oss-20b-resid"
# model/device are parameters, not baked in -- this report must work for any model.
MODEL = sys.argv[3] if len(sys.argv) > 3 else "openai/gpt-oss-20b"
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cuda:0"
# layers come from the probe run itself (paper: every 2nd layer for a 24-layer model)
LAYERS = json.load(open(f"{OUT}/probe_report.json"))["layers"]
N = 48

# categorical hues assigned in fixed order, never cycled; CVD-safe pairing
C_INJ, C_LEG, C_CLEAN, C_USER = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"


def dev_slice(samples, n_dev=24, n_test=96, n_probe=250):
    """The sweep's dev split. Delegates to xpia_defense.build_splits -- do NOT
    reimplement.

    This used to be a second copy of the splitting logic with `quota` hardcoded to
    {test: 96, probe: 250}. It agreed with the sweep only while default CLI args were
    used: with `--n-test 48` the two returned DIFFERENT dev samples, silently, so a
    control would have been measured on a different sample set than the sweep it was
    meant to explain. Verified equivalent to the previous implementation at n_dev=24
    and n_dev=96 (identical sample ids).
    """
    return [samples[i] for i in X.build_splits(
        samples, n_test=n_test, n_probe=n_probe, n_eval=n_dev, verbose=False)["dev"]]


def main():
    os.makedirs(FIGS, exist_ok=True)
    model, tok = X.load_model_and_tok(MODEL, DEVICE)
    samples = X.build_dataset()
    dev = dev_slice(samples)

    P = {L: X.load_probe(f"{OUT}/probe_L{L}.pkl") for L in LAYERS}
    roles = P[LAYERS[0]]["roles"]
    # Index coef_ rows via the FITTER's own class order, not `roles`. They coincide only
    # while all 5 classes survive the MN_FIT_ROWS subsample; if one ever drops, the rows
    # shift and `userness` would silently measure a different pair of roles. train_probes
    # already does it this way (xpia_defense.py:599).
    cls = {L: list(P[L]["mn"].classes_) for L in LAYERS}
    for L in LAYERS:
        assert set(cls[L]) == set(range(len(roles))), (
            f"L{L}: multinomial fit saw classes {cls[L]}, expected all of {roles}")
    ui, ti = roles.index("user"), roles.index("tool")
    Wb = {L: (torch.tensor(P[L]["mn"].coef_[[cls[L].index(i) for i in range(len(roles))]],
                           dtype=torch.float32),
              torch.tensor(P[L]["mn"].intercept_[[cls[L].index(i) for i in range(len(roles))]],
                           dtype=torch.float32)) for L in LAYERS}
    rep = json.load(open(f"{OUT}/probe_report.json"))

    fired = {}
    for f in sorted(os.listdir(BASE_RUN)):
        if not f.endswith("_completions.json"):
            continue
        d = json.load(open(f"{BASE_RUN}/{f}"))
        if "base-XPIA" not in d or len(d["base-XPIA"]) != len(dev):
            continue
        for s, c in zip(dev, d["base-XPIA"]):
            v = X.attack_influenced(s, c)
            if v is not None:
                fired[s["id"]] = bool(v)
        break

    blocks = X.layer_container(model)
    cap = {}

    def mk(L):
        def store(t, L=L):
            cap[L] = t.detach()
        return store
    # PRE-MLP residual: the paper's probe site, and the site these probes were fit on
    hs = [X.register_probe_capture(blocks[L], mk(L))[0] for L in LAYERS]

    def userness(ids, idx):
        """per-token (user_logit - tool_logit) for idx, per layer."""
        cap.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=model.device))
        out = {}
        for L in LAYERS:
            W, b = Wb[L]
            lg = cap[L][0].float().cpu()[idx] @ W.T + b
            out[L] = (lg[:, ui] - lg[:, ti]).numpy()
        return out

    # ---- Q2 tag control: identical text, swapped header --------------------------
    tagctl = {L: {"as_user": [], "as_tool": []} for L in LAYERS}
    for s in dev[:N]:
        t = (s.get("injection_text") or "").strip()
        if len(t) < 40:
            continue
        for role, key in (("user", "as_user"), ("tool", "as_tool")):
            got = X.sentinel_span(
                tok, lambda c, r=role: X.render_single(tok, r, c, X.TOOL_NAME), t)
            if not got:
                continue
            ids, idx = X.token_span(tok, *got)
            if len(idx) < 3:
                continue
            u = userness(ids, idx)
            for L in LAYERS:
                tagctl[L][key].append(float(u[L].mean()))

    # ---- Q3/Q4 the real test -----------------------------------------------------
    rows, traces = [], []
    for s in dev[:N]:
        try:
            t_c, sp_c = X.prompt_and_span(tok, s, poisoned=False)
            t_p, sp_p = X.prompt_and_span(tok, s, poisoned=True)
            ids_c, idx_c = X.token_span(tok, t_c, sp_c)
            ids_p, idx_p = X.token_span(tok, t_p, sp_p)
            _, pay, inj = X.injection_span(tok, s)
        except Exception:
            continue
        inj_set = set(inj)
        legit = [k for k in pay if k not in inj_set]
        if len(idx_c) < 3 or len(inj) < 3 or len(legit) < 3:
            continue
        u_clean = userness(ids_c, idx_c)
        u_full = userness(ids_p, sorted(pay))
        pos = {k: i for i, k in enumerate(sorted(pay))}
        rows.append(dict(
            sid=s["id"],
            clean={L: float(u_clean[L].mean()) for L in LAYERS},
            inj={L: float(u_full[L][[pos[k] for k in inj]].mean()) for L in LAYERS},
            legit={L: float(u_full[L][[pos[k] for k in legit]].mean()) for L in LAYERS},
        ))
        if len(traces) < 6:
            traces.append(dict(sid=s["id"],
                               series={L: u_full[L].tolist() for L in LAYERS},
                               inj_pos=[pos[k] for k in inj]))
    for h in hs:
        h.remove()

    # ================================ TABLES ======================================
    print(f"\nprobe dir: {OUT}\nsamples: {len(rows)}   "
          f"labelled: {sum(1 for r in rows if r['sid'] in fired)} "
          f"({sum(fired.get(r['sid'], False) for r in rows)} attacks succeeded)\n")

    print("=== Q1  role separation (probe accuracy) ===")
    # per-role accuracies exist only when one-vs-rest probes were fit; under --skip-ovr
    # `acc` is {} (xpia_defense.py:573,585) and only the multinomial is available. Q1 is
    # answerable from the multinomial alone, so degrade to that instead of dying here.
    has_ovr = any(rep["report"][str(L)]["acc"] for L in LAYERS)
    hdr = "".join(f"{r:>10}" for r in roles) if has_ovr else ""
    print(f"{'layer':>6} " + hdr + f"{'multinom':>11}")
    for L in LAYERS:
        a = rep["report"][str(L)]["acc"]
        cells = "".join(f"{a[r]:10.3f}" for r in roles) if has_ovr else ""
        print(f"{L:6d} " + cells + f"{rep['report'][str(L)]['mn_acc']:11.3f}")
    if not has_ovr:
        print("      (per-role columns omitted: --skip-ovr, multinomial only)")

    print("\n=== Q2  tag control: SAME text, swapped header (userness) ===")
    print(f"{'layer':>6} {'in <user>':>12} {'in <tool>':>12} {'swing':>10}")
    for L in LAYERS:
        a, b = np.mean(tagctl[L]["as_user"]), np.mean(tagctl[L]["as_tool"])
        print(f"{L:6d} {a:12.2f} {b:12.2f} {a-b:10.2f}")

    print("\n=== Q3  ROLE CONFUSION: userness inside the SAME <tool> message ===")
    print(f"{'layer':>6} {'clean payload':>15} {'legit toks':>12} {'INJECTED':>10} "
          f"{'inj - legit':>12} {'inj - clean':>12}")
    q3 = {}
    for L in LAYERS:
        c = np.mean([r["clean"][L] for r in rows])
        g = np.mean([r["legit"][L] for r in rows])
        i_ = np.mean([r["inj"][L] for r in rows])
        q3[L] = (c, g, i_)
        print(f"{L:6d} {c:15.2f} {g:12.2f} {i_:10.2f} {i_-g:12.2f} {i_-c:12.2f}")

    print("\n=== Q4  does userness predict ATTACK SUCCESS? ===")
    print(f"{'layer':>6} {'succeeded':>12} {'blocked':>10} {'delta':>9}")
    lab = [r for r in rows if r["sid"] in fired]
    q4 = {}
    for L in LAYERS:
        a = [r["inj"][L] for r in lab if fired[r["sid"]]]
        b = [r["inj"][L] for r in lab if not fired[r["sid"]]]
        if not a or not b:
            print(f"{L:6d}   (insufficient labels)")
            continue
        q4[L] = (np.mean(a), np.mean(b))
        print(f"{L:6d} {np.mean(a):12.2f} {np.mean(b):10.2f} "
              f"{np.mean(a)-np.mean(b):+9.2f}")

    # COMPUTED verdict, not a legend. This line previously printed the literal string
    # "PASS: ..." unconditionally, describing what a pass would look like -- which reads as
    # a pass even when Q3 is inverted. Never print a verdict you did not evaluate.
    q3_pos = sum(1 for L in LAYERS if q3[L][2] - q3[L][1] > 0)
    q4_pos = sum(1 for L in q4 if q4[L][0] - q4[L][1] > 0)
    swing = [np.mean(tagctl[L]["as_user"]) - np.mean(tagctl[L]["as_tool"]) for L in LAYERS]
    verdict = {
        "Q1 role separation": ("PASS" if max(rep["report"][str(L)]["mn_acc"] for L in LAYERS)
                               > 0.5 else "FAIL"),
        "Q2 tag control": "PASS" if max(swing) > 5 else "FAIL",
        "Q3 role confusion": ("PASS" if q3_pos > len(LAYERS) / 2 else "FAIL"),
        "Q4 predicts success": ("PASS" if q4 and q4_pos > len(q4) / 2 else "FAIL"),
    }
    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        print(f"  {v:4}  {k}")
    print(f"\nQ3: 'inj - legit' positive at {q3_pos}/{len(LAYERS)} layers "
          f"(need >{len(LAYERS)//2}).  Q4: delta positive at {q4_pos}/{len(q4)} layers.")
    if verdict["Q3 role confusion"] == "FAIL":
        print("Q3 INVERTED vs the paper: injected text reads MORE TOOL-LIKE than the\n"
              "legitimate record. Do NOT report as a finding -- see todo/01-next-steps.md\n"
              "STEP 3. Note user-vs-tool cannot express a SYSTEM-styled injection; these\n"
              "payloads are authority-styled ('URGENT ... you MUST') and were previously\n"
              "measured to trade tool-ness for SYSTEM-ness (xpia_defense.py:23).")

    json.dump({"probe_dir": OUT, "rows": rows, "tagctl":
               {str(L): {k: v for k, v in tagctl[L].items()} for L in LAYERS}},
              open(f"{ROOT}/runs/probe_report.json", "w"), indent=1)

    # ================================ FIGURES =====================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.6})

    # Fig 1: Q3 -- userness by condition, per layer
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(len(LAYERS))
    for key, col, lab_ in (("clean", C_CLEAN, "clean payload (no injection)"),
                           ("legit", C_LEG, "legitimate tokens"),
                           ("inj", C_INJ, "injected tokens")):
        ys = [np.mean([r[key][L] for r in rows]) for L in LAYERS]
        se = [np.std([r[key][L] for r in rows]) / np.sqrt(len(rows)) for L in LAYERS]
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=5, label=lab_)
        ax.fill_between(xs, np.array(ys) - np.array(se), np.array(ys) + np.array(se),
                        color=col, alpha=0.15, lw=0)
    ax.set_xticks(xs); ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_ylabel("userness  (user − tool logit)")
    ax.set_title("Q3 role confusion: is injected text read as more user-like?",
                 loc="left", fontsize=10)
    ax.axhline(0, color="#333", lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/q3_confusion.png"); plt.close(fig)

    # Fig 2: Q2 tag control
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    a = [np.mean(tagctl[L]["as_user"]) for L in LAYERS]
    b = [np.mean(tagctl[L]["as_tool"]) for L in LAYERS]
    ax.plot(xs, a, "-o", color=C_USER, lw=2, ms=5, label="same text in <user>")
    ax.plot(xs, b, "-o", color=C_INJ, lw=2, ms=5, label="same text in <tool>")
    ax.set_xticks(xs); ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_ylabel("userness"); ax.axhline(0, color="#333", lw=0.8, ls=":")
    ax.set_title("Q2 tag control: identical text, header swapped", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/q2_tag_control.png"); plt.close(fig)

    # Fig 3: per-token traces with the injected span shaded
    L0 = LAYERS[min(2, len(LAYERS) - 1)]
    k = min(4, len(traces))
    if k:
        fig, axes = plt.subplots(k, 1, figsize=(6.6, 1.6 * k), sharex=False)
        axes = np.atleast_1d(axes)
        for axi, tr in zip(axes, traces[:k]):
            y = tr["series"][L0]
            axi.plot(range(len(y)), y, color=C_LEG, lw=1.4)
            ip = tr["inj_pos"]
            if ip:
                axi.axvspan(min(ip), max(ip), color=C_INJ, alpha=0.18, lw=0)
                axi.plot(ip, [y[i] for i in ip], color=C_INJ, lw=1.8)
            axi.axhline(0, color="#333", lw=0.7, ls=":")
            axi.set_ylabel("userness", fontsize=7)
            axi.set_title(tr["sid"], loc="left", fontsize=7)
        axes[-1].set_xlabel("token position within tool payload", fontsize=8)
        fig.suptitle(f"Per-token userness across the tool payload (L{L0}); "
                     f"injected span shaded", fontsize=9, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(f"{FIGS}/q3_token_traces.png"); plt.close(fig)

    # Fig 4: Q4 attack success
    if q4:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ls = sorted(q4)
        w = 0.36
        xi = np.arange(len(ls))
        ax.bar(xi - w/2, [q4[L][0] for L in ls], w, color=C_INJ, label="attack succeeded")
        ax.bar(xi + w/2, [q4[L][1] for L in ls], w, color=C_CLEAN, label="attack blocked")
        ax.set_xticks(xi); ax.set_xticklabels([f"L{L}" for L in ls])
        ax.set_ylabel("userness of injected tokens")
        ax.set_title("Q4 does userness predict attack success?", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(f"{FIGS}/q4_predictive.png"); plt.close(fig)

    print(f"\nfigures -> {FIGS}/")


if __name__ == "__main__":
    main()
