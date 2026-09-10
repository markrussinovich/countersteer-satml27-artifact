#!/usr/bin/env python
"""Fit the CachePrune mask (arXiv:2504.21228 v3) -- offline attribution over K/V cache
coordinates. Ported from the paper; there is NO public code, so every equation reference
below is to the paper's own numbering.

METHOD (defaults are the paper's: k=1, N=8, top p=0.5%, alpha=1):
  1. For each fit sample, render the POISONED and CLEAN prompts (same renderer the
     evaluation uses: src/spans.prompt_and_span), greedy-decode both, and find the first
     token where the two continuations DIVERGE. The paper's triggering effect (their
     Fig. 3) says poisoned/clean outputs split within 1-2 response tokens; on a model with
     a reasoning preamble the split can sit a few tokens in, so we take k=1 AT THE
     DIVERGENCE POINT (j=0 reproduces the paper exactly; j is recorded per sample).
  2. One grad-enabled forward of the poisoned prompt (+ shared greedy prefix) with a
     capture cache that keeps references to the POST-RoPE cached keys and the values --
     the tensors `past_key_values.update` receives, i.e. exactly what the paper masks.
  3. The preferential attribution loss, k=1 (eq. 13):
         L = p_theta(y_p | x) - p_theta(y_c | x)
     with y_p / y_c the greedy poisoned / clean tokens at the divergence point. Two
     autograd.grad calls on the one retained graph give the loss's two components.
  4. Attribution per KV-cache coordinate (eq. 3): a_t^i = h_t^i * dL/dh_t^i, restricted to
     the CONTEXT SPAN positions (the tool-payload span -- the paper prunes only within the
     data span), then aggregated by MAX over span positions (eq. 4) and pooled by MAX over
     the N samples (the paper pools all samples' positions into one max).
  5. Normalize poisoned vs clean components (eq. 10-11), form the preference set (their
     Phi): a_p_norm > a_c_norm and |a_p_norm - a_c_norm| > 2*min(|a_p_norm|, |a_c_norm|).
  6. tau = the value such that at most p of ALL coordinates are selected within Phi
     (eq. 5), ranked by the difference-loss attribution. Mask m_i = 1 - alpha (eq. 6).

COORDINATE SPACE: n_layers x {K, V} x kv_heads x head_dim. gpt-oss-20b: 24 x 2 x 8 x 64
= 24,576 -> ~123 pruned at p=0.5%.

Fit data: the PROBE split of the shipped corpus (attacker-template-disjoint from dev and
test by build_splits), first --n-fit samples that show a divergence.

Usage:
  .venv/bin/python tools/controls/build_cacheprune_mask.py \
      --model openai/gpt-oss-20b --device cuda:0 --out runs/cacheprune_mask.json
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT

from transformers.cache_utils import DynamicCache  # noqa: E402


class CaptureCache(DynamicCache):
    """Keeps references to the GRAPH tensors entering the cache: the post-RoPE key states
    and the value states, per layer -- attribution targets, exactly what CachePrune masks."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.captured = {}

    def update(self, key_states, value_states, layer_idx, *a, **kw):
        self.captured[layer_idx] = (key_states, value_states)
        return super().update(key_states, value_states, layer_idx, *a, **kw)


def _greedy(model, tok, ids, n):
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return out[0, ids.shape[1]:].tolist()


def _divergence(gp, gc):
    """First index where the two greedy continuations differ, else None."""
    for j in range(min(len(gp), len(gc))):
        if gp[j] != gc[j]:
            return j
    return None


def fit_mask(model, tok, items, *, k=1, top_p=0.005, alpha=1.0, max_decode=64,
             n_target=None, verbose=True):
    """items: [{"id", "ids_p", "ids_c", "span_idx"}] with span_idx = token indices of the
    CONTEXT SPAN in ids_p. Consumes items in order, skipping zero-divergence ones, until
    n_target have contributed (None = use all). Returns the mask artifact dict."""
    assert k == 1, "only the paper's default k=1 is implemented"
    cfg = model.config.get_text_config()
    n_layers = cfg.num_hidden_layers
    kvh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_coords = n_layers * 2 * kvh * hd

    acc = {n: torch.full((n_layers, 2, kvh, hd), float("-inf")) for n in ("p", "c", "d")}
    per_sample, used, skipped = [], [], []

    for it in items:
        if n_target is not None and len(used) >= n_target:
            break
        ids_p = torch.tensor([it["ids_p"]], device=model.device)
        ids_c = torch.tensor([it["ids_c"]], device=model.device)
        gp = _greedy(model, tok, ids_p, max_decode)
        gc = _greedy(model, tok, ids_c, max_decode)
        j = _divergence(gp, gc)
        if j is None:
            skipped.append(it["id"])
            if verbose:
                print(f"  [fit] {it['id']}: no divergence in {max_decode} greedy tokens "
                      f"-- zero-gradient sample, skipped", flush=True)
            continue
        y_p, y_c = gp[j], gc[j]
        ids_x = torch.tensor([it["ids_p"] + gp[:j]], device=model.device)
        cache = CaptureCache(config=model.config)
        with torch.enable_grad():
            out = model(input_ids=ids_x, past_key_values=cache, use_cache=True)
            probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
            p_p, p_c = probs[y_p], probs[y_c]
            flat = [t for L in range(n_layers) for t in cache.captured[L]]
            g_p = torch.autograd.grad(p_p, flat, retain_graph=True, allow_unused=True)
            g_c = torch.autograd.grad(p_c, flat, allow_unused=True)
        sel = torch.tensor([t for t in it["span_idx"] if t < ids_x.shape[1]],
                           device=model.device)
        for L in range(n_layers):
            for kv in (0, 1):     # 0 = key, 1 = value
                h = flat[2 * L + kv][0]                      # (kvh, seq, hd)
                i = 2 * L + kv
                gp_t = g_p[i][0] if g_p[i] is not None else torch.zeros_like(h)
                gc_t = g_c[i][0] if g_c[i] is not None else torch.zeros_like(h)
                # sel lives on model.device (the EMBEDDING device); under device_map=auto
                # each layer's captured KV lives on ITS OWN device, and cuda indexing
                # requires index and tensor co-located (crashed the in-job Qwen fit on
                # cuda:1, xpia-rivals-qwen 2026-09-08). Single-GPU behaviour unchanged
                # (`.to` is a no-op there).
                sel_l = sel.to(h.device)
                ap = (h * gp_t).float()[:, sel_l, :]
                ac = (h * gc_t).float()[:, sel_l, :]
                acc["p"][L, kv] = torch.maximum(acc["p"][L, kv], ap.amax(1).cpu())
                acc["c"][L, kv] = torch.maximum(acc["c"][L, kv], ac.amax(1).cpu())
                acc["d"][L, kv] = torch.maximum(acc["d"][L, kv], (ap - ac).amax(1).cpu())
        per_sample.append({"id": it["id"], "divergence_at": j,
                           "y_p": tok.decode([y_p]), "y_c": tok.decode([y_c]),
                           "p_p": float(p_p), "p_c": float(p_c),
                           "loss": float(p_p - p_c),
                           "prompt_tokens": int(ids_x.shape[1]),
                           "span_tokens": int(sel.numel())})
        used.append(it["id"])
        if verbose:
            print(f"  [fit] {it['id']}: diverges at token {j} "
                  f"(y_p={tok.decode([y_p])!r} p={float(p_p):.4f} vs "
                  f"y_c={tok.decode([y_c])!r} p={float(p_c):.4f})", flush=True)
        del out, cache, flat, g_p, g_c
        torch.cuda.empty_cache()

    if not used:
        raise SystemExit("no fit sample produced a poisoned/clean divergence -- the "
                         "attribution loss is identically zero and there is no mask to fit")

    a_p, a_c, a_d = (acc[n].flatten() for n in ("p", "c", "d"))
    sums = {"p": float(a_p.sum()), "c": float(a_c.sum())}
    norm_note = "eq10-11: sum over coordinates"
    if sums["p"] <= 0 or sums["c"] <= 0:
        # the paper's normalizer assumes positive mass; with a negative/zero sum the sign
        # of every normalized score would FLIP. Fall back to the absolute-sum and say so.
        norm_note = ("DEVIATION: sum of attributions non-positive "
                     f"(p={sums['p']:.3e}, c={sums['c']:.3e}); normalized by sum of |a|")
        a_p_n = a_p / a_p.abs().sum().clamp_min(1e-12)
        a_c_n = a_c / a_c.abs().sum().clamp_min(1e-12)
    else:
        a_p_n = a_p / a_p.sum()
        a_c_n = a_c / a_c.sum()
    phi = (a_p_n > a_c_n) & ((a_p_n - a_c_n).abs()
                             > 2 * torch.minimum(a_p_n.abs(), a_c_n.abs()))
    n_prune = max(1, int(top_p * n_coords))
    ranked = a_d.masked_fill(~phi, float("-inf"))
    n_take = min(n_prune, int(phi.sum()))
    top = torch.topk(ranked, n_take)
    idx = top.indices[torch.isfinite(top.values)]
    pruned = []
    for i in idx.tolist():
        L, r = divmod(i, 2 * kvh * hd)
        kv, r = divmod(r, kvh * hd)
        head, dim = divmod(r, hd)
        pruned.append({"layer": L, "kv": "k" if kv == 0 else "v",
                       "head": head, "dim": dim,
                       "score": float(a_d[i]), "a_p_norm": float(a_p_n[i]),
                       "a_c_norm": float(a_c_n[i])})
    by_layer = {}
    for c in pruned:
        by_layer[c["layer"]] = by_layer.get(c["layer"], 0) + 1
    return {"n_layers": n_layers, "num_kv_heads": kvh, "head_dim": hd,
            "alpha": alpha,
            "pruned": pruned,
            "_meta": {"paper": "arXiv:2504.21228v3 (CachePrune)",
                      "model": model.config._name_or_path,
                      "k": k, "top_p": top_p, "n_coords": n_coords,
                      "n_prune_budget": n_prune, "n_pruned": len(pruned),
                      "phi_size": int(phi.sum()),
                      "n_fit": len(used), "fit_ids": used, "skipped_ids": skipped,
                      "per_sample": per_sample,
                      "agg": "max over span positions, max-pooled over samples (eq. 4 "
                             "applied to the pooled position set)",
                      "ranking": "difference-loss attribution a_p - a_c within Phi",
                      "norm": norm_note, "attr_sums": sums,
                      "pruned_per_layer": by_layer,
                      "max_decode": max_decode,
                      "time": time.strftime("%Y-%m-%d %H:%M:%S")}}


def shipped_items(tok, n_fit, max_decode, no_think=False):
    """First usable samples of the PROBE split (attacker-template-disjoint from dev/test).
    Returns more than n_fit candidates; the fitter stops after n_fit divergent ones."""
    allx = X.build_dataset()
    bins = X.build_splits(allx, verbose=False)
    items = []
    for i in bins["probe"].tolist():
        s = allx[i]
        try:
            text_p, span_p = X.prompt_and_span(tok, s, poisoned=True, no_think=no_think)
            text_c, _ = X.prompt_and_span(tok, s, poisoned=False, no_think=no_think)
        except ValueError:
            continue
        ids_p, span_idx = X.token_span(tok, text_p, span_p)
        ids_c = tok(text_c, add_special_tokens=False)["input_ids"]
        items.append({"id": s["id"], "ids_p": ids_p, "ids_c": ids_c,
                      "span_idx": span_idx})
        if len(items) >= n_fit * 3:     # headroom for zero-divergence skips
            break
    return items


def write_artifact(art, out):
    """Build the string first, verify it round-trips, then atomic-replace (CLAUDE.md)."""
    blob = json.dumps(art, indent=2)
    json.loads(blob)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        f.write(blob)
    os.replace(tmp, out)
    json.load(open(out))
    print(f"wrote {out}: {art['_meta']['n_pruned']} pruned of "
          f"{art['_meta']['n_coords']} coordinates "
          f"(budget {art['_meta']['n_prune_budget']}, Phi {art['_meta']['phi_size']}); "
          f"per-layer {art['_meta']['pruned_per_layer']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-fit", dest="n_fit", type=int, default=8,
                    help="the paper's N=8")
    ap.add_argument("--k", type=int, default=1, help="response tokens in the loss; the "
                    "paper's default (and the only implemented value) is 1")
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.005,
                    help="fraction of ALL coordinates pruned; the paper's p=0.5%%")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="masking strength: m = 1 - alpha at pruned coordinates")
    ap.add_argument("--max-decode", dest="max_decode", type=int, default=64,
                    help="greedy tokens searched for the poisoned/clean divergence")
    ap.add_argument("--no-think", dest="no_think", action="store_true", default=False)
    ap.add_argument("--out", default=f"{ROOT}/runs/cacheprune_mask.json")
    args = ap.parse_args()

    model, tok = X.load_model_and_tok(args.model, args.device)
    cand = shipped_items(tok, args.n_fit, args.max_decode, args.no_think)
    print(f"[fit] {len(cand)} candidate probe-split samples; fitting on the first "
          f"{args.n_fit} with a poisoned/clean divergence", flush=True)

    art = fit_mask(model, tok, cand, k=args.k, top_p=args.top_p,
                   alpha=args.alpha, max_decode=args.max_decode, n_target=args.n_fit)
    if art["_meta"]["n_fit"] < args.n_fit:
        print(f"[fit] WARNING only {art['_meta']['n_fit']} divergent samples "
              f"(asked for {args.n_fit})", flush=True)
    write_artifact(art, args.out)


if __name__ == "__main__":
    main()
