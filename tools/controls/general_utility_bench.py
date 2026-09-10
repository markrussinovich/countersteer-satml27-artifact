"""General-utility benchmarks (MMLU / GSM8K / IFEval) under the always-on defense.

THE QUESTION. Does the locked steering cell degrade capability on ordinary, non-XPIA
work? Two delivery modes, because the defense only edits TOOL-OUTPUT token spans at
prefill:

  plain  the benchmark item as a normal user prompt. NO tool payload exists, so the
         steering hook has no positions and must be a structural no-op -- this mode
         VERIFIES that (the artifact records whether baseline and steered completions are
         byte-identical), it does not merely assume it.
  tool   the same item delivered AS A TOOL RESULT (a prefilled fetch_url turn whose
         payload carries the question/document), which is the span the defense steers on
         every call in deployment. This is where real general-utility cost would live.

Scoring is deterministic, no judge: MMLU = first standalone A-D letter in the final
answer; GSM8K = last number vs the gold after '####'; IFEval = the official
instruction-checkers (lm_eval.tasks.ifeval), prompt-level STRICT accuracy.

Arms: baseline (no steering) and steered (the locked cell) in ONE process per run, same
items, greedy decoding.

Usage:
  .venv/bin/python tools/controls/general_utility_bench.py --bench mmlu|gsm8k|ifeval \
      [--mode both|plain|tool] [--n 200] [--model openai/gpt-oss-20b] \
      [--probe-run runs/gpt-oss-20b-userabl] [--direction combo_ovr8_pat1] [--alpha 1.0] \
      [--steer-layers 12,16,20] [--batch 8] [--max-new 1024] [--device cuda:0] [--out ...]
"""
import argparse
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/datadrive/huggingface/")
import xpia_defense as X  # noqa: E402

FETCH_TOOL = [{"type": "function", "function": {
    "name": "fetch_url",
    "description": "Fetch a URL and return its full contents.",
    "parameters": {"type": "object", "properties": {
        "url": {"type": "string"}}, "required": ["url"]}}}]


def load_items(bench, n, seed=0):
    # X.hf_dataset = load_dataset at the revision pinned in src/corpora, so the
    # Open Science "pinned revision" promise holds for these benchmarks too.
    import random
    rng = random.Random(seed)
    if bench == "mmlu":
        ds = list(X.hf_dataset("cais/mmlu", "all", split="test"))
        rng.shuffle(ds)
        out = []
        for ex in ds[:n]:
            q = ex["question"].strip()
            ch = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", ex["choices"]))
            out.append({"doc": f"{q}\n\n{ch}",
                        "ask": "Answer the multiple-choice question. Reply with the "
                               "letter (A, B, C or D) only.",
                        "gold": "ABCD"[ex["answer"]]})
        return out
    if bench == "gsm8k":
        ds = list(X.hf_dataset("openai/gsm8k", "main", split="test"))
        rng.shuffle(ds)
        return [{"doc": ex["question"].strip(),
                 "ask": "Solve the math problem. End your reply with the final numeric "
                        "answer on its own line.",
                 "gold": ex["answer"].split("####")[-1].strip().replace(",", "")}
                for ex in ds[:n]]
    if bench == "ifeval":
        ds = list(X.hf_dataset("google/IFEval", split="train"))
        rng.shuffle(ds)
        return [{"doc": ex["prompt"].strip(),
                 "ask": "Carry out the instructions in the fetched document exactly.",
                 "gold": None, "ifeval": {"instruction_id_list": ex["instruction_id_list"],
                                          "kwargs": ex["kwargs"],
                                          "prompt": ex["prompt"]}}
                for ex in ds[:n]]
    raise SystemExit(bench)


def final_text(completion):
    for marker in ("<|channel|>final<|message|>", "</think>"):
        if marker in completion:
            t = completion.split(marker, 1)[1]
            for stop in ("<|return|>", "<|end|>", "<|start|>", "<|im_end|>",
                         "<|endoftext|>"):
                t = t.split(stop)[0]
            return t.strip()
    return completion.split("<|endoftext|>")[0].strip()


def score(bench, item, completion):
    t = final_text(completion)
    if bench == "mmlu":
        m = re.search(r"\b([ABCD])\b", t)
        return bool(m and m.group(1) == item["gold"])
    if bench == "gsm8k":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
        return bool(nums and nums[-1].replace(",", "") == item["gold"])
    if bench == "ifeval":
        from lm_eval.tasks.ifeval import instructions_registry
        ok = True
        for iid, kw in zip(item["ifeval"]["instruction_id_list"],
                           item["ifeval"]["kwargs"]):
            cls = instructions_registry.INSTRUCTION_DICT[iid](iid)
            cls.build_description(**{k: v for k, v in (kw or {}).items()
                                     if v is not None})
            args = cls.get_instruction_args()
            if args and "prompt" in args:
                cls.build_description(prompt=item["ifeval"]["prompt"])
            if not t.strip() or not cls.check_following(t):
                ok = False
                break
        return ok
    raise SystemExit(bench)


def build_prompt(tok, item, mode, i):
    if mode == "plain":
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{item['ask']}\n\n{item['doc']}"}]
        text = X.apply_template(tok, msgs, None, gen_prompt=True)
        return text, None
    s = {"system": "You are a helpful assistant with access to tools.",
         "user": f"{item['ask']}\n- The document has been fetched for you.\n"
                 f"- URL: https://x0.at/gub{i:04d}.txt",
         "tools": FETCH_TOOL, "retrieval_tool": "fetch_url",
         "retrieval_args": {"url": f"https://x0.at/gub{i:04d}.txt"},
         "payload": {"content": item["doc"]}, "payload_clean": {"content": item["doc"]},
         "id": f"gub-{i}"}
    text, span = X.prompt_and_span(tok, s, poisoned=False)
    ids, idx = X.token_span(tok, text, span)
    return text, idx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, choices=["mmlu", "gsm8k", "ifeval"])
    ap.add_argument("--mode", default="both", choices=["both", "plain", "tool"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--probe-run", dest="probe_run", default="runs/gpt-oss-20b-userabl")
    ap.add_argument("--direction", default="combo_ovr8_pat1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--steer-layers", dest="steer_layers", default="12,16,20")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", dest="max_new", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    items = load_items(args.bench, args.n)
    layers = [int(x) for x in args.steer_layers.split(",")]
    model, tok = X.load_model_and_tok(args.model, args.device)
    dirs, sig, abl = X.build_dirs(args.probe_run, layers, args.direction, args.device)
    modes = ["plain", "tool"] if args.mode == "both" else [args.mode]

    results = {"bench": args.bench, "n": len(items), "model": args.model,
               "cell": f"{args.direction}@{args.alpha}", "arms": {}}
    for mode in modes:
        prompts, spans = [], []
        for i, it in enumerate(items):
            t, idx = build_prompt(tok, it, mode, i)
            prompts.append(t)
            spans.append(idx)
        for arm in ("baseline", "steered"):
            comps = []
            for k in range(0, len(prompts), args.batch):
                chunk = prompts[k:k + args.batch]
                enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left",
                          add_special_tokens=False).to(model.device)
                st = None
                if arm == "steered":
                    st = X.Steer(model, layers, dirs, args.alpha, "sigma", sig, abl,
                                 "add")
                    # LEFT padding shifts every token index by the row's pad length;
                    # spans were computed on the unpadded prompt
                    plen = enc["input_ids"].shape[1]
                    offs = [plen - len(tok(chunk[j],
                                           add_special_tokens=False)["input_ids"])
                            for j in range(len(chunk))]
                    st.positions = [[q + offs[j] for q in (spans[k + j] or [])]
                                    for j in range(len(chunk))]
                with torch.no_grad():
                    if st:
                        with st:
                            g = model.generate(**enc, max_new_tokens=args.max_new,
                                               do_sample=False,
                                               pad_token_id=tok.eos_token_id)
                    else:
                        g = model.generate(**enc, max_new_tokens=args.max_new,
                                           do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                for j in range(len(chunk)):
                    comps.append(tok.decode(g[j][enc["input_ids"].shape[1]:],
                                            skip_special_tokens=False))
                if (k // args.batch) % 5 == 0:
                    print(f"[bench] {args.bench}/{mode}/{arm}: {k + len(chunk)}"
                          f"/{len(prompts)}", flush=True)
            accs = [score(args.bench, it, c) for it, c in zip(items, comps)]
            key = f"{mode}:{arm}"
            results["arms"][key] = {"acc": sum(accs) / len(accs), "per_item": accs,
                                    "completions_sha": hash(tuple(comps)) & 0xffffffff}
            if mode == "plain":
                results["arms"][key]["completions"] = [c[:2000] for c in comps]
            print(f"[bench] {args.bench} {key}: acc {sum(accs)/len(accs):.4f}",
                  flush=True)
        if mode == "plain" and "plain:baseline" in results["arms"] \
                and "plain:steered" in results["arms"]:
            a = results["arms"]["plain:baseline"]["completions"]
            b = results["arms"]["plain:steered"]["completions"]
            ident = sum(x == y for x, y in zip(a, b)) / len(a)
            results["plain_identity_rate"] = ident
            print(f"[bench] plain-mode identity (structural no-op check): {ident:.3f}",
                  flush=True)

    out = args.out or f"runs/general_utility_{args.bench}.json"
    blob = json.dumps(results, indent=1)
    with open(out + ".tmp", "w") as f:
        f.write(blob)
    json.load(open(out + ".tmp"))
    os.replace(out + ".tmp", out)
    print(f"[bench] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
