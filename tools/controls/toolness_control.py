#!/usr/bin/env python
"""TOOL-NESS: is injected text more or less TOOL-like than the legitimate record beside it?

Q3 in tools/probe_report.py reports `user_logit - tool_logit`. That quantity is AMBIGUOUS
for our purposes: it falls either because the user logit dropped or because the tool logit
rose, and those mean opposite things for a defense that steers untrusted content toward
`tool`. This measures tool-ness directly instead.

Two readouts per layer, both in LOGIT space (never softmax -- p_tool saturates at ~0.994):

    tool                raw tool logit
    margin              tool_logit - max(other role logits)   <- "how tool-like, vs whatever
                                                                 the nearest competitor is"

compared across three token sets drawn from the SAME tool message:

    inj      the injected attacker tokens        (exact supervision: dataset ships the string)
    legit    the surrounding record tokens
    clean    the same record with the injection removed (payload_clean)

Why it matters: the defense's premise is that untrusted text drifts toward `user` and can be
pushed back toward `tool`. If injected tokens are ALREADY more tool-like than the legitimate
record, that premise does not hold in this setup and any ASR reduction from steering is
working through some other mechanism -- which must be established before tuning alpha
against it.

Usage:
    python tools/controls/toolness_control.py [RUN_DIR] [MODEL] [DEVICE] [N_DEV]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT
RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/gpt-oss-20b-paper"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-oss-20b"
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
N_DEV = int(sys.argv[4]) if len(sys.argv) > 4 else 24


def summarise(v, ti):
    """(mean tool logit, mean margin) over tokens; v is [n_tok, n_roles]."""
    tool = v[:, ti]
    other = np.delete(v, ti, axis=1).max(axis=1)
    return float(tool.mean()), float((tool - other).mean())


def main():
    layers, roles, Wb = E.load_probes(RUN_DIR)
    ti = roles.index("tool")
    model, tok = X.load_model_and_tok(MODEL, DEVICE)
    dev = E.dev_samples(N_DEV)
    hs, cap = E.attach_capture(model, layers)
    role_logits = E.make_role_logits(model, cap, Wb, layers)

    acc = {L: {k: {"tool": [], "margin": []} for k in ("inj", "legit", "clean")}
           for L in layers}
    n_used = 0
    for s in dev:
        if not s.get("injection_text"):
            continue
        ids, pay, inj = X.injection_span(tok, s)
        legit = [k for k in pay if k not in set(inj)]
        if not inj or not legit:
            continue
        # clean payload: same record, injection removed (xpia_defense.py:986)
        text_c, span_c = X.prompt_and_span(tok, s, poisoned=False)
        ids_c, pay_c = X.token_span(tok, text_c, span_c)
        if not pay_c:
            continue

        lg_p = role_logits(ids, inj + legit)       # one forward for the poisoned prompt
        lg_c = role_logits(ids_c, pay_c)
        n_inj = len(inj)
        for L in layers:
            for key, v in (("inj", lg_p[L][:n_inj]), ("legit", lg_p[L][n_inj:]),
                           ("clean", lg_c[L])):
                t_, m_ = summarise(v, ti)
                acc[L][key]["tool"].append(t_)
                acc[L][key]["margin"].append(m_)
        n_used += 1

    for h in hs:
        h.remove()

    print(f"\nprobe dir: {RUN_DIR}   samples used: {n_used}   roles: {roles}")
    print("\n=== TOOL-NESS inside the same <tool> message (logit space) ===")
    print(f"{'layer':>6}{'tool(inj)':>11}{'tool(legit)':>13}{'tool(clean)':>13}"
          f"{'inj-legit':>11}{'t':>8}{'win%':>7}   |{'marg(inj)':>11}{'marg(legit)':>13}"
          f"{'inj-legit':>11}")
    rows = {}
    for L in layers:
        ti_, tl_, tc_ = (np.array(acc[L][k]["tool"]) for k in ("inj", "legit", "clean"))
        mi_, ml_ = (np.array(acc[L][k]["margin"]) for k in ("inj", "legit"))
        d = ti_ - tl_
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 and d.std() else float("nan")
        win = 100.0 * float((d > 0).mean())
        rows[L] = {"tool_inj": float(ti_.mean()), "tool_legit": float(tl_.mean()),
                   "tool_clean": float(tc_.mean()), "tool_inj_minus_legit": float(d.mean()),
                   "t_paired": t, "win_pct": win,
                   "margin_inj": float(mi_.mean()), "margin_legit": float(ml_.mean()),
                   "margin_inj_minus_legit": float((mi_ - ml_).mean())}
        print(f"{L:6d}{ti_.mean():11.2f}{tl_.mean():13.2f}{tc_.mean():13.2f}"
              f"{d.mean():11.2f}{t:8.2f}{win:7.0f}   |{mi_.mean():11.2f}{ml_.mean():13.2f}"
              f"{(mi_ - ml_).mean():11.2f}")

    pos = sum(1 for L in layers if rows[L]["tool_inj_minus_legit"] > 0)
    print(f"\ntool(inj) > tool(legit) at {pos}/{len(layers)} layers.")
    if pos > len(layers) / 2:
        print("=> Injected text is ALREADY MORE TOOL-LIKE than the legitimate record.\n"
              "   The 'drifts toward user, steer back to tool' premise does NOT hold here;\n"
              "   any ASR reduction from tool-steering works through another mechanism.")
    else:
        print("=> Injected text is LESS tool-like than the legitimate record: there IS tool-ness\n"
              "   headroom for steering to recover, and Q3's inversion is driven by the user\n"
              "   logit, not by injected text looking more like a tool.")

    json.dump({"run_dir": RUN_DIR, "n": n_used, "roles": roles, "layers": layers,
               "rows": rows}, open(f"{ROOT}/runs/toolness_control.json", "w"), indent=1)


if __name__ == "__main__":
    main()
