#!/usr/bin/env python
"""Per-token USERNESS-VS-REST across the whole tool payload span, injected vs not injected.

For each dev sample this plots, token by token across the tool-output span:

    userness_vs_rest = user_logit - max(other role logits)

i.e. how user-like a token reads relative to whichever role is its nearest competitor --
NOT user-minus-tool. The tool comparison alone is misleading here: injected prose already
reads more tool-like than the surrounding JSON (tools/controls/toolness_control.py), so
`user - tool` conflates "less user" with "more tool".

Two traces per sample, drawn over the SAME span:
    injected   the poisoned payload; the injected token range is shaded
    clean      the same record with the injection removed (payload_clean)

Everything is in LOGIT space. Never softmax -- p_tool saturates at ~0.994 and hides the
drift, which already invalidated one conclusion in this project.

Usage:
    python tools/controls/plot_userness_spans.py [RUN_DIR] [LAYER] [MODEL] [DEVICE] [N_DEV]
    LAYER may be an int, or "all" to emit one figure per probed layer.

Outputs: runs/figs/userness_spans_L{layer}.png  (grid, one panel per sample)
         runs/figs/userness_spans_summary.png   (mean +/- IQR by layer, inj vs legit vs clean)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X = E.X
ROOT = E.ROOT
RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/gpt-oss-20b-paper"
LAYER_ARG = sys.argv[2] if len(sys.argv) > 2 else "14"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "openai/gpt-oss-20b"
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cuda:0"
N_DEV = int(sys.argv[5]) if len(sys.argv) > 5 else 24

FIGS = f"{ROOT}/runs/figs"
# fixed hues, never cycled; CVD-safe pairing (matches tools/probe_report.py)
C_INJ, C_CLEAN, C_SHADE = "#1f77b4", "#7f7f7f", "#d62728"


def userness_vs_rest(v, ui):
    """v: [n_tok, n_roles] raw logits -> [n_tok] user minus best competing role."""
    user = v[:, ui]
    other = np.delete(v, ui, axis=1).max(axis=1)
    return user - other


def main():
    layers, roles, Wb = E.load_probes(RUN_DIR)
    ui = roles.index("user")
    want = layers if LAYER_ARG == "all" else [int(LAYER_ARG)]
    for L in want:
        if L not in layers:
            raise SystemExit(f"layer {L} not probed; available: {layers}")

    model, tok = X.load_model_and_tok(MODEL, DEVICE)
    dev = E.dev_samples(N_DEV)
    hs, cap = E.attach_capture(model, layers)
    role_logits = E.make_role_logits(model, cap, Wb, layers)

    os.makedirs(FIGS, exist_ok=True)
    per_sample = []          # one dict per sample, traces for every layer
    for s in dev:
        if not s.get("injection_text"):
            continue
        ids_p, pay_p, inj_p = X.injection_span(tok, s)
        if not pay_p or not inj_p:
            continue
        text_c, span_c = X.prompt_and_span(tok, s, poisoned=False)
        ids_c, pay_c = X.token_span(tok, text_c, span_c)
        if not pay_c:
            continue

        lg_p = role_logits(ids_p, pay_p)     # whole payload span, poisoned
        lg_c = role_logits(ids_c, pay_c)     # whole payload span, clean
        inj_set = set(inj_p)
        # positions of the injected tokens WITHIN the payload span, for shading
        inj_local = [k for k, t in enumerate(pay_p) if t in inj_set]
        per_sample.append({
            "sid": s["id"],
            "inj_local": inj_local,
            "n_pay": len(pay_p),
            "trace_inj": {L: userness_vs_rest(lg_p[L], ui) for L in layers},
            "trace_clean": {L: userness_vs_rest(lg_c[L], ui) for L in layers},
        })

    for h in hs:
        h.remove()
    n = len(per_sample)
    print(f"samples plotted: {n}   layers: {layers}   roles: {roles}")

    # ---------------- per-sample span traces ------------------------------------
    for L in want:
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.4 * nrow),
                                 squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k, ps in enumerate(per_sample):
            ax = axes[k // ncol][k % ncol]
            ax.set_visible(True)
            yi, yc = ps["trace_inj"][L], ps["trace_clean"][L]
            ax.plot(np.arange(len(yi)), yi, lw=0.9, color=C_INJ, label="injected")
            ax.plot(np.arange(len(yc)), yc, lw=0.9, color=C_CLEAN, alpha=0.85,
                    label="clean")
            if ps["inj_local"]:
                ax.axvspan(min(ps["inj_local"]), max(ps["inj_local"]),
                           color=C_SHADE, alpha=0.15, lw=0)
            ax.axhline(0, color="k", lw=0.5, ls=":")
            ax.set_title(f"{ps['sid']}  ({ps['n_pay']} tok)", fontsize=8)
            ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"userness vs rest  (user_logit - max other role)  across the tool "
                     f"payload span, layer {L}\nshaded = injected tokens; grey = same "
                     f"record with injection removed", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = f"{FIGS}/userness_spans_L{L}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)

    # ---------------- summary across layers -------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.6))
    stats = {}
    for tag, color in (("injected tokens", C_INJ), ("legit tokens (poisoned msg)", "#2ca02c"),
                       ("clean payload", C_CLEAN)):
        med, lo, hi = [], [], []
        for L in layers:
            vals = []
            for ps in per_sample:
                t_i, t_c = ps["trace_inj"][L], ps["trace_clean"][L]
                mask = np.zeros(len(t_i), dtype=bool)
                mask[ps["inj_local"]] = True
                if tag == "injected tokens":
                    v = t_i[mask]
                elif tag == "legit tokens (poisoned msg)":
                    v = t_i[~mask]
                else:
                    v = t_c
                if len(v):
                    vals.append(float(v.mean()))
            med.append(np.median(vals))
            lo.append(np.percentile(vals, 25))
            hi.append(np.percentile(vals, 75))
        stats[tag] = {"median": med, "q25": lo, "q75": hi}
        ax.plot(layers, med, marker="o", ms=3.5, color=color, label=tag)
        ax.fill_between(layers, lo, hi, color=color, alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("userness vs rest  (logit)")
    ax.set_title(f"userness vs rest by layer, median +/- IQR over {n} samples")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = f"{FIGS}/userness_spans_summary.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)

    json.dump({"run_dir": RUN_DIR, "n": n, "layers": layers, "summary": stats},
              open(f"{ROOT}/runs/userness_spans.json", "w"), indent=1)


if __name__ == "__main__":
    main()
