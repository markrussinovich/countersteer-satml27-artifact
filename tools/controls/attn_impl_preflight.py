#!/usr/bin/env python
"""Gated preflight for running a model under a non-default attention implementation.

Two checks, one process, one GPU (this is the smoke rung for a serving-path change --
CLAUDE.md: a kernel swap is a new serving path, and equivalence is measured, never
reasoned):

  1. LONG-PROMPT FEASIBILITY under --attn-impl: prefill a synthetic prompt of
     --long-tokens tokens (default 17000, sized past the longest IPI Arena prompt).
     Exit 1 on OOM/error -- this is the gate that aborts the full run.
  2. EQUIVALENCE MEASUREMENT vs the default path: greedy-generate --gen-tokens on
     --n-prompts short chat prompts under the default resolution, then reload under
     --attn-impl and repeat. Reports exact-match count and first-divergence index per
     prompt. REPORTED, not gated: bf16 kernels are not expected to be bit-identical;
     the artifact produced under the new path carries a per-path label either way.

Usage:
  python tools/controls/attn_impl_preflight.py --model openai/gpt-oss-20b \
      --attn-impl flex_attention --device cuda:0
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from src.model import load_model_and_tok  # noqa: E402

SHORT_PROMPTS = [
    "Summarize in one sentence: the quick brown fox jumps over the lazy dog.",
    "List three prime numbers greater than 100.",
    "Translate to French: the meeting is at noon tomorrow.",
    "What tool would you call to read a file named report.txt? Answer briefly.",
    "Compute 17 * 23 and explain in one line.",
]


def gen(model, tok, prompts, n_new):
    outs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        # transformers v5: apply_chat_template defaults to return_dict=True, so the
        # return value is a BatchEncoding, never a bare tensor (a positional pass to
        # generate() crashes on .shape -- this exact bug killed xpia-ipi-gptoss3).
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs.append(out[0, ids.shape[1]:].tolist())
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn-impl", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--long-tokens", type=int, default=17000)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--skip-equivalence", action="store_true",
                    help="run only the long-prompt gate (halves the wall time)")
    a = ap.parse_args()
    prompts = SHORT_PROMPTS[: a.n_prompts]

    # ---- GATE FIRST (the only thing that may fail the preflight): load under the
    # candidate impl and prefill the long prompt. The equivalence leg runs after and
    # is crash-proofed -- a bug there must never abort a stage whose gate passed
    # (that inversion killed xpia-ipi-gptoss3 before flex was ever loaded).
    print(f"[preflight] loading {a.model} under attn_impl={a.attn_impl}", flush=True)
    model, tok = load_model_and_tok(a.model, a.device, attn_impl=a.attn_impl)

    base = tok("The quarterly report shows steady growth across regions. ").input_ids
    ids = (base * (a.long_tokens // len(base) + 1))[: a.long_tokens]
    t = torch.tensor([ids], device=model.device)
    try:
        with torch.no_grad():
            out = model.generate(t, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        assert out.shape[1] == t.shape[1] + 4
        print(f"[preflight] GATE PASS: {a.long_tokens}-token prefill + 4-token decode "
              f"under {a.attn_impl} (peak "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
    except Exception as e:  # OOM or any flex-path failure: the gate's verdict
        print(f"[preflight] GATE FAIL under {a.attn_impl} at {a.long_tokens} tokens: "
              f"{type(e).__name__}: {e}", flush=True)
        sys.exit(1)
    print(json.dumps({"gate": "PASS", "attn_impl": a.attn_impl,
                      "long_tokens": a.long_tokens}), flush=True)

    # ---- EQUIVALENCE (measured, reported, never gating) ----
    if not a.skip_equivalence:
        try:
            alt = gen(model, tok, prompts, a.gen_tokens)
            del model
            torch.cuda.empty_cache()
            print(f"[preflight] loading {a.model} under DEFAULT attention resolution",
                  flush=True)
            model, tok = load_model_and_tok(a.model, a.device)
            ref = gen(model, tok, prompts, a.gen_tokens)
            n_exact = 0
            for i, (r, c) in enumerate(zip(ref, alt)):
                div = next((j for j, (x, y) in enumerate(zip(r, c)) if x != y),
                           None if r == c else min(len(r), len(c)))
                exact = r == c
                n_exact += exact
                print(f"[preflight] equivalence prompt {i}: "
                      f"{'EXACT' if exact else f'diverges at token {div}'}", flush=True)
            print(f"[preflight] EQUIVALENCE (measured, reported, not gated): "
                  f"{n_exact}/{len(prompts)} exact greedy matches over "
                  f"{a.gen_tokens} tokens", flush=True)
        except Exception as e:
            print(f"[preflight] EQUIVALENCE LEG FAILED (reported, not gating): "
                  f"{type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
