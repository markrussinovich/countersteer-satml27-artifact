#!/usr/bin/env python
"""CounterSteer inference-overhead measurement (paper §5.3; owner directive 2026-09-01:
report actual latency/throughput even if negligible — never claim "zero compute").

Measures, on identical prompts (n episodes from the shipped corpus, poisoned render):
  prefill latency   full forward over the prompt, steered vs unsteered
  decode throughput greedy generation of --gen tokens, steered vs unsteered
  memory            steady-state CUDA allocated, both
The steered arm is the DEPLOYED cell (direction/alpha/layers/sigma convention identical
to production). Reports median and IQR over --n episodes after --warmup discards, plus
the relative overhead percentages. Steering is prefill-only in deployment, so decode
throughput is expected identical; it is measured rather than asserted.

Usage:
  .venv/bin/python tools/controls/overhead_bench.py --n 24 --gen 256 \
      --out runs/overhead_bench.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT
from src.steering import Steer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--probe-dir", default=f"{ROOT}/runs/gpt-oss-20b-userabl")
    ap.add_argument("--direction", default="combo_ovr8_pat1")
    ap.add_argument("--alpha", type=float, default=8.06)
    ap.add_argument("--match-sigma-to", default="dim_no_override")
    ap.add_argument("--steer-layers", default="12,16,20")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    layers = [int(x) for x in a.steer_layers.split(",")]
    model, tok = X.load_model_and_tok(a.model, a.device)
    dirs, sigmas, _ = X.build_dirs(a.probe_dir, layers, a.direction, model.device,
                                   match_sigma_to=a.match_sigma_to)
    samples = X.build_dataset()
    bins = X.build_splits(samples, verbose=False)
    pool = [samples[i] for i in bins["dev"]
            if samples[i].get("injection_text") and samples[i].get("injection_field")][: a.n + a.warmup]

    def episode(s, steered):
        text, span = X.prompt_and_span(tok, s, poisoned=True)
        ids, pay = X.token_span(tok, text, span)
        ids_t = torch.tensor([ids], device=model.device)
        st = Steer(model, layers, dirs, a.alpha, scale="sigma", sigmas=sigmas) \
            if steered else None
        if st:
            st.positions = [pay]
        ctx = st if st else _null()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ctx, torch.no_grad():
            out = model(ids_t, use_cache=True)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        t0 = time.perf_counter()
        with ctx, torch.no_grad():
            for _ in range(a.gen):
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t_decode = time.perf_counter() - t0
        return len(ids), t_prefill, a.gen / t_decode, torch.cuda.max_memory_allocated() / 2**30

    class _null:
        def __enter__(self): return self
        def __exit__(self, *e): return False

    rows = {"steered": [], "unsteered": []}
    for i, s in enumerate(pool):
        for arm in ("unsteered", "steered"):
            torch.cuda.reset_peak_memory_stats()
            r = episode(s, arm == "steered")
            if i >= a.warmup:
                rows[arm].append(r)
        if i % 6 == 0:
            print(f"[{i}/{len(pool)}]", flush=True)

    def stats(k, idx):
        v = np.array([r[idx] for r in rows[k]])
        return {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
                "p75": float(np.percentile(v, 75))}
    rep = {"config": vars(a), "n": len(rows["steered"]),
           "prefill_s": {k: stats(k, 1) for k in rows},
           "decode_tok_per_s": {k: stats(k, 2) for k in rows},
           "peak_mem_gib": {k: stats(k, 3) for k in rows}}
    ps, pu = rep["prefill_s"]["steered"]["median"], rep["prefill_s"]["unsteered"]["median"]
    ds, du = rep["decode_tok_per_s"]["steered"]["median"], rep["decode_tok_per_s"]["unsteered"]["median"]
    rep["overhead"] = {"prefill_pct": 100 * (ps - pu) / pu,
                       "decode_pct": 100 * (du - ds) / du}
    blob = json.dumps(rep, indent=1)
    with open(a.out + ".tmp", "w") as f:
        f.write(blob)
    os.replace(a.out + ".tmp", a.out)
    print(f"prefill overhead {rep['overhead']['prefill_pct']:+.2f}% | "
          f"decode overhead {rep['overhead']['decode_pct']:+.2f}% | wrote {a.out}")


if __name__ == "__main__":
    main()
