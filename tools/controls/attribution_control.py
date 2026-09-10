#!/usr/bin/env python
"""ATTRIBUTION: does the model read the injected instruction as USER-ISSUED?

The paper's premise is that a model follows an instruction when it perceives the USER as
having issued it. Earlier controls in this directory asked a weaker question -- "do these
tokens look like user prose?" -- and answered it against the surrounding JSON record, which
is out of distribution for a probe trained only on C4 prose. That comparison is confounded
by text type (see benign_prose_control.py) and is not what "user-issued" means.

Every sample already contains the right reference: the REAL user message. It is genuine
user-issued prose, in-distribution, in the same prompt. So calibrate between two anchors:

    U   real user message tokens        ceiling -- what user-issued actually looks like
    J   legitimate record tokens        floor   -- unambiguous tool data
    I   injected tokens                 the quantity of interest

    attribution = (I - J) / (U - J)

0 => the model reads the injection as tool data. 1 => as though the user said it.

This deliberately conflates header and style: U carries a <user> header, I carries <tool>.
That IS the question -- whether the injection attains user-like representation despite its
tool label. U is the ceiling, J the floor, and where I falls between them is attribution.

THE DECISIVE SPLIT is succeeded vs blocked attacks. The premise predicts successful
injections score HIGHER: they work because they are attributed to the user.

Userness is `user_logit - max(other role logits)`, in LOGIT space. Never softmax.

Usage:
    python tools/controls/attribution_control.py [RUN_DIR] [BASE_RUN] [MODEL] [DEVICE] [N_DEV]
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
BASE_RUN = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/runs/gpt-oss-20b-resid"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "openai/gpt-oss-20b"
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cuda:0"
N_DEV = int(sys.argv[5]) if len(sys.argv) > 5 else 24

# an attribution ratio is unstable when the anchors nearly coincide; report but exclude
# those samples from the ratio statistics rather than letting them dominate.
MIN_ANCHOR_GAP = 1.0


def userness(v, ui):
    return float((v[:, ui] - np.delete(v, ui, axis=1).max(axis=1)).mean())


def user_msg_span(tok, s):
    """Char span of the REAL user message inside the rendered poisoned prompt."""
    text, _ = X.prompt_and_span(tok, s, poisoned=True)
    needle = s.get("user") or ""
    if len(needle) < 20:
        return None
    lo = text.find(needle)
    if lo < 0:                     # template may reflow whitespace; try a prefix
        needle = needle.strip()[:200]
        lo = text.find(needle)
        if lo < 0:
            return None
    return text, (lo, lo + len(needle))


def main():
    layers, roles, Wb = E.load_probes(RUN_DIR)
    ui = roles.index("user")
    model, tok = X.load_model_and_tok(MODEL, DEVICE)
    dev = E.dev_samples(N_DEV)

    # attack-success labels from UNSTEERED base-XPIA completions (probe-independent)
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

    hs, cap = E.attach_capture(model, layers)
    role_logits = E.make_role_logits(model, cap, Wb, layers)

    recs = []
    for s in dev:
        if not s.get("injection_text"):
            continue
        got = user_msg_span(tok, s)
        if not got:
            continue
        text, uspan = got
        ids_u, u_idx = X.token_span(tok, text, uspan)
        ids_p, pay, inj = X.injection_span(tok, s)
        legit = [k for k in pay if k not in set(inj)]
        if not u_idx or not inj or not legit:
            continue
        # ids_u and ids_p are the same poisoned prompt, so one forward covers all three
        assert ids_u == ids_p, "user-span and injection-span prompts diverged"
        lg = role_logits(ids_p, u_idx + inj + legit)
        nu, ni = len(u_idx), len(inj)
        rec = {"sid": s["id"], "fired": fired.get(s["id"]),
               "U": {}, "I": {}, "J": {}, "attr": {}}
        for L in layers:
            U = userness(lg[L][:nu], ui)
            I = userness(lg[L][nu:nu + ni], ui)
            J = userness(lg[L][nu + ni:], ui)
            rec["U"][L], rec["I"][L], rec["J"][L] = U, I, J
            rec["attr"][L] = ((I - J) / (U - J)) if abs(U - J) >= MIN_ANCHOR_GAP else None
        recs.append(rec)

    for h in hs:
        h.remove()

    lab = [r for r in recs if r["fired"] is not None]
    succ = [r for r in lab if r["fired"]]
    blok = [r for r in lab if not r["fired"]]
    print(f"\nprobe dir: {RUN_DIR}\nsamples: {len(recs)}   labelled: {len(lab)} "
          f"({len(succ)} succeeded / {len(blok)} blocked)")

    print("\n=== ANCHORS: userness (user_logit - max other), median over samples ===")
    print(f"{'layer':>6}{'U real user':>13}{'I injected':>12}{'J legit rec':>13}"
          f"{'U-J gap':>10}{'attribution':>13}{'n_ratio':>9}")
    rows = {}
    for L in layers:
        U = np.median([r["U"][L] for r in recs])
        I = np.median([r["I"][L] for r in recs])
        J = np.median([r["J"][L] for r in recs])
        a = [r["attr"][L] for r in recs if r["attr"][L] is not None]
        med_a = float(np.median(a)) if a else float("nan")
        rows[L] = {"U": float(U), "I": float(I), "J": float(J),
                   "attribution_median": med_a, "n_ratio": len(a)}
        print(f"{L:6d}{U:13.2f}{I:12.2f}{J:13.2f}{U-J:10.2f}{med_a:13.3f}{len(a):9d}")

    print("\n=== THE TEST: attribution, succeeded vs blocked attacks ===")
    print(f"{'layer':>6}{'succeeded':>12}{'blocked':>10}{'delta':>9}{'t':>8}"
          f"{'n_s':>5}{'n_b':>5}")
    n_pos = n_tot = 0
    for L in layers:
        a = [r["attr"][L] for r in succ if r["attr"][L] is not None]
        b = [r["attr"][L] for r in blok if r["attr"][L] is not None]
        if len(a) < 2 or len(b) < 2:
            print(f"{L:6d}   (insufficient labels)")
            continue
        a, b = np.array(a), np.array(b)
        sp = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")
        d = float(a.mean() - b.mean())
        rows[L].update({"attr_succeeded": float(a.mean()),
                        "attr_blocked": float(b.mean()), "delta": d, "t": t})
        n_tot += 1
        n_pos += d > 0
        print(f"{L:6d}{a.mean():12.3f}{b.mean():10.3f}{d:+9.3f}{t:8.2f}{len(a):5d}{len(b):5d}")

    # DISAMBIGUATION. An earlier report claimed userness separates succeeded from blocked
    # at 12/12 layers, measured as raw `user - tool` on injected tokens. This control
    # changed TWO things at once -- the metric (user - max(other)) and the per-sample
    # normalisation by (U - J) -- and found no separation. Report the UNNORMALISED split
    # under this metric so the two changes can be told apart:
    #   separation returns here  -> the effect is real; normalisation removed it
    #   still absent here        -> the original result was metric-specific
    print("\n=== DISAMBIGUATION: UNNORMALISED injected userness, succeeded vs blocked ===")
    print(f"{'layer':>6}{'succeeded':>12}{'blocked':>10}{'delta':>9}{'t':>8}"
          f"   |{'U-J gap succ':>14}{'U-J gap blok':>14}")
    u_pos = u_tot = 0
    for L in layers:
        a = np.array([r["I"][L] for r in succ])
        b = np.array([r["I"][L] for r in blok])
        ga = np.array([r["U"][L] - r["J"][L] for r in succ])
        gb = np.array([r["U"][L] - r["J"][L] for r in blok])
        if len(a) < 2 or len(b) < 2:
            continue
        sp = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")
        d = float(a.mean() - b.mean())
        rows[L].update({"I_succeeded": float(a.mean()), "I_blocked": float(b.mean()),
                        "I_delta": d, "I_t": t,
                        "gap_succeeded": float(ga.mean()), "gap_blocked": float(gb.mean())})
        u_tot += 1
        u_pos += d > 0
        print(f"{L:6d}{a.mean():12.2f}{b.mean():10.2f}{d:+9.2f}{t:8.2f}"
              f"   |{ga.mean():14.2f}{gb.mean():14.2f}")
    print(f"\nUNNORMALISED injected userness higher for SUCCEEDED at {u_pos}/{u_tot} layers.")

    print(f"\nattribution higher for SUCCEEDED at {n_pos}/{n_tot} layers.")
    print("The premise -- injections work because the model reads them as USER-ISSUED --\n"
          "predicts succeeded > blocked. n is small; read the sign consistency across\n"
          "layers, not any single layer's t.")

    json.dump({"run_dir": RUN_DIR, "n": len(recs), "n_succeeded": len(succ),
               "n_blocked": len(blok), "layers": layers, "rows": rows,
               "per_sample": [{"sid": r["sid"], "fired": r["fired"],
                               "attr": {str(k): v for k, v in r["attr"].items()},
                               # save the raw anchors too -- omitting them last time meant
                               # the normalised/unnormalised question needed a whole rerun
                               "U": {str(k): v for k, v in r["U"].items()},
                               "I": {str(k): v for k, v in r["I"].items()},
                               "J": {str(k): v for k, v in r["J"].items()}}
                              for r in recs]},
              open(f"{ROOT}/runs/attribution_control.json", "w"), indent=1)


if __name__ == "__main__":
    main()
