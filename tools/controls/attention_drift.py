#!/usr/bin/env python
"""Does steering reroute attention OFF the record tokens the model must copy from?

THE HYPOTHESIS. Steering's correctness cost is not refusals and not wrong tools -- it is
`param_drift`: right tool, right argument keys, different structured values (25/77 on
shipped, and IDENTICAL with no attacker present). That is the signature of a COPY/RETRIEVAL
failure, not a decision failure. SKOP (arXiv:2605.06342) argues activation steering degrades
utility primarily by ATTENTION REROUTING -- the edit alters query-key matching and pulls
attention away from contextually important tokens.

THE TEST. No generation. Teacher-force the UNATTACKED reference completion through the model
on an INJECTION-FREE payload, twice -- once clean, once with the steering hook live -- and
measure the attention mass flowing from the completion's tokens back onto the tool payload
span. Then ask whether the per-sample DROP in that mass predicts which samples the steered
run actually got wrong.

  drop predicts struct_exact failure  => rerouting is the mechanism; SKOP's fix applies
  no relationship                     => refutes the account for this setting, and the
                                         correctness cost must be sought elsewhere

WHY FULL-ATTENTION LAYERS ONLY. gpt-oss-20b alternates sliding (window 128) and full
attention: layer_types[i] is 'sliding_attention' for even i, 'full_attention' for odd i. The
payload span sits 150-900 tokens behind the completion, far outside a 128-token window, so
completion tokens CANNOT attend to it at a sliding layer -- the mass there is 0 by
construction and averaging it in would dilute the signal to nothing. Note this also means
the three STEERED layers (12, 16, 20) are all sliding layers; whatever the edit does to
long-range copying, it reaches it indirectly.

Usage:
    python tools/controls/attention_drift.py [--n 24] [--alpha 8.0] [--device cuda:1]
        [--direction dim_no_override_both] [--steer-layers 12,16,20] [--ref RESULTS_JSON]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT


def _flag(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


def main():
    n_max = int(_flag("--n", "24"))
    alpha = float(_flag("--alpha", "8.0"))
    device = _flag("--device", "cuda:1")
    dname = _flag("--direction", "dim_no_override_both")
    steer_layers = [int(x) for x in _flag("--steer-layers", "12,16,20").split(",")]
    probe_run = _flag("--probe-run", f"{ROOT}/runs/gpt-oss-20b-userabl")
    ref_file = _flag("--ref", f"{ROOT}/runs/gpt-oss-20b-userabl/"
                              "results_add-dim-no-override-both-random-4163935_completions.json")
    model_id = _flag("--model", "openai/gpt-oss-20b")

    d = json.load(open(ref_file))
    meta = d["_meta"]
    clean_arm = d["clean"]
    steered_clean = d.get(f"CLEAN+{dname}@{alpha}")
    if steered_clean is None:
        raise SystemExit(f"{ref_file} has no CLEAN+{dname}@{alpha} arm; "
                         f"have {[k for k in d if k != '_meta']}")
    allx = X.build_dataset()
    byid = {s["id"]: s for s in allx}
    S = [byid[i] for i in meta["sample_ids"]]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # EAGER attention: sdpa/flash do not return attention weights at all.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager").eval()
    blocks = X.layer_container(model)
    types = getattr(model.config, "layer_types", None)
    full = [i for i in range(len(blocks))
            if types is None or types[i] == "full_attention"]
    print(f"[attn] {len(blocks)} layers, full-attention at {full}", flush=True)

    dirs, sig, abl = X.build_dirs(probe_run, steer_layers, dname, model.device,
                                  match_sigma_to="dim_no_override")

    # capture reduced attention mass per full-attention layer, freeing weights immediately
    want = {}
    mass = {}

    def mk(i):
        def hook(mod, inp, out):
            w = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
            if w is None or not want:
                return
            q, k = want["q"], want["k"]
            # w: [batch, heads, q_len, k_len] -> mean over heads of summed mass q->k
            sub = w[0][:, q, :][:, :, k].float()
            mass[i] = float(sub.sum(-1).mean().item())
        return hook

    hs = [blocks[i].self_attn.register_forward_hook(mk(i)) for i in full]

    rows = []
    for si, s in enumerate(S):
        if len(rows) >= n_max:
            break
        ref_c = clean_arm[si]
        if not ref_c or not X.parse_tool_calls(ref_c):
            continue
        try:
            text, span = X.prompt_and_span(tok, s, poisoned=False)
            ids, pay_idx = X.token_span(tok, text, span)
        except Exception:
            continue
        comp_ids = tok(ref_c, add_special_tokens=False)["input_ids"]
        if not comp_ids or not pay_idx:
            continue
        full_ids = ids + comp_ids
        q = list(range(len(ids), len(full_ids)))          # completion tokens as queries
        want.clear(); want.update({"q": q, "k": pay_idx})
        t = torch.tensor([full_ids], device=model.device)

        mass.clear()
        with torch.no_grad():
            model(t, output_attentions=True)
        base = dict(mass)

        st = X.Steer(model, steer_layers, dirs, alpha, "sigma", sig, abl, "add")
        st.positions = [pay_idx]
        mass.clear()
        with st, torch.no_grad():
            model(t, output_attentions=True)
        steered = dict(mass)
        st.positions = None

        if not base or not steered:
            raise SystemExit("no attention captured -- eager attention not active?")
        b = float(np.mean([base[i] for i in full]))
        a = float(np.mean([steered[i] for i in full]))
        ok = X.behavioural_score(ref_c, steered_clean[si]).get("struct_exact", False)
        rows.append({"id": s["id"], "mass_base": b, "mass_steer": a,
                     "drop": b - a, "rel_drop": (b - a) / b if b else float("nan"),
                     "clean_struct_exact": bool(ok),
                     "per_layer_base": {str(i): base[i] for i in full},
                     "per_layer_steer": {str(i): steered[i] for i in full}})
        print(f"  {s['id']:<10} mass {b:.4f} -> {a:.4f}  drop {b-a:+.4f} "
              f"({(b-a)/b*100 if b else 0:+.1f}%)  clean struct_exact={ok}", flush=True)
    for h in hs:
        h.remove()

    ok = np.array([r["clean_struct_exact"] for r in rows])
    dr = np.array([r["rel_drop"] for r in rows])
    print(f"\n=== {len(rows)} samples, steering an INJECTION-FREE payload ===")
    print(f"  relative attention drop onto the payload span: "
          f"mean {dr.mean():+.3%}  median {np.median(dr):+.3%}")
    if ok.sum() and (~ok).sum():
        print(f"  drop | steered run CORRECT   (n={int(ok.sum()):2d}): {dr[ok].mean():+.3%}")
        print(f"  drop | steered run WRONG     (n={int((~ok).sum()):2d}): {dr[~ok].mean():+.3%}")
        # point-biserial: does the drop predict failure?
        r = float(np.corrcoef(dr, (~ok).astype(float))[0, 1])
        from math import sqrt
        t = r * sqrt(max(len(rows) - 2, 1)) / sqrt(max(1 - r * r, 1e-12))
        print(f"  point-biserial r(drop, WRONG) = {r:+.3f}   t = {t:+.2f}, df = {len(rows)-2}")
        print("  => rerouting predicts drift" if r > 0.3 else
              "  => NO relationship: the rerouting account is refuted for this setting")
    else:
        print("  (all samples on one side of struct_exact -- no contrast)")
    dst = f"{ROOT}/runs/attention_drift.json"
    blob = json.dumps({"model": model_id, "direction": dname, "alpha": alpha,
                       "steer_layers": steer_layers, "full_layers": full,
                       "ref_file": os.path.basename(ref_file), "rows": rows}, indent=1)
    with open(dst + ".tmp", "w") as f:
        f.write(blob)
    json.load(open(dst + ".tmp"))
    os.replace(dst + ".tmp", dst)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
