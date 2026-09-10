"""WHY does MMLU slip under steering? Two controls that separate position from comprehension.

Variants (tool mode, baseline + steered arms, deterministic letter scoring):
  goldA / goldD      options reordered so the gold answer sits at A (resp. D) for EVERY
                     item. Cost collapsing at goldA and inflating at goldD => the damage
                     is POSITIONAL (late span content under-read). Flat => comprehension
                     noise + first-option prior.
  q_user_opt_tool    question in the (unsteered) user message, options in the steered doc
  q_tool_opt_user    question in the steered doc, options in the (unsteered) user message
                     -> localizes which component's perturbation causes the slip, and
                     whether moving task text out of the span rescues accuracy.

Usage: gub_mmlu_anatomy.py --variant goldA|goldD|q_user_opt_tool|q_tool_opt_user
       [--n 250] [cell flags as general_utility_bench] --out runs/...
"""
import argparse, json, os, re, sys, torch, zlib
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/datadrive/huggingface/")
import xpia_defense as X
from general_utility_bench import FETCH_TOOL, final_text

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True,
                choices=["goldA", "goldB", "goldC", "goldD",
                         "q_user_opt_tool", "q_tool_opt_user"])
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--model", default="openai/gpt-oss-20b")
ap.add_argument("--probe-run", default="runs/gpt-oss-20b-userabl")
ap.add_argument("--direction", default="combo_ovr8_pat1")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--steer-layers", default="12,16,20")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-new", type=int, default=640)
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
a = ap.parse_args()

import random
rng = random.Random(0)
mm = list(X.hf_dataset("cais/mmlu", "all", split="test")); rng.shuffle(mm); mm = mm[:a.n]

items = []
for ex in mm:
    ch = list(ex["choices"]); gold = ex["answer"]
    if a.variant.startswith("gold"):
        tgt = "ABCD".index(a.variant[-1])
        order = list(range(4)); order.remove(gold)
        # zlib.crc32, not hash(): Python's str hash is salted per process, so the
        # distractor order (hence the item set) was not reproducible across runs
        # (adversarial review, 2026-08-28).
        rng2 = random.Random(zlib.crc32(ex["question"].encode()) & 0xffff)
        rng2.shuffle(order); order.insert(tgt, gold)
        ch = [ex["choices"][j] for j in order]
        gold_letter = "ABCD"[tgt]
    else:
        gold_letter = "ABCD"[gold]
    opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", ch))
    items.append({"q": ex["question"].strip(), "opts": opts, "gold": gold_letter})

layers = [int(x) for x in a.steer_layers.split(",")]
model, tok = X.load_model_and_tok(a.model, a.device)
dirs, sig, abl = X.build_dirs(a.probe_run, layers, a.direction, a.device)
ASK = "Answer the multiple-choice question. Reply with the letter (A, B, C or D) only."

def build(i, it):
    url = f"https://x0.at/an{i:04d}.txt"
    if a.variant == "q_user_opt_tool":
        user = f"{ASK}\nThe question: {it['q']}\n- The answer options are in the fetched document.\n- URL: {url}"
        doc = it["opts"]
    elif a.variant == "q_tool_opt_user":
        user = f"{ASK}\nThe question is in the fetched document.\nOptions:\n{it['opts']}\n- URL: {url}"
        doc = it["q"]
    else:
        user = f"{ASK}\n- The document has been fetched for you.\n- URL: {url}"
        doc = f"{it['q']}\n\n{it['opts']}"
    s = {"system": "You are a helpful assistant with access to tools.", "user": user,
         "tools": FETCH_TOOL, "retrieval_tool": "fetch_url",
         "retrieval_args": {"url": url}, "payload": {"content": doc},
         "payload_clean": {"content": doc}, "id": f"an-{i}"}
    text, span = X.prompt_and_span(tok, s, poisoned=False)
    ids, idx = X.token_span(tok, text, span)
    return text, idx

prompts, spans = zip(*[build(i, it) for i, it in enumerate(items)])
res = {"variant": a.variant, "n": len(items), "cell": f"{a.direction}@{a.alpha}", "arms": {}}
for arm in ("baseline", "steered"):
    accs, picks = [], []
    for k in range(0, len(prompts), a.batch):
        chunk = list(prompts[k:k + a.batch])
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=False).to(model.device)
        st = None
        if arm == "steered":
            st = X.Steer(model, layers, dirs, a.alpha, "sigma", sig, abl, "add")
            plen = enc["input_ids"].shape[1]
            offs = [plen - len(tok(chunk[j], add_special_tokens=False)["input_ids"])
                    for j in range(len(chunk))]
            st.positions = [[q + offs[j] for q in (spans[k + j] or [])]
                            for j in range(len(chunk))]
        with torch.no_grad():
            if st:
                with st: g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            else: g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            t = final_text(tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=False))
            m = re.search(r"\b([ABCD])\b", t)
            picks.append(m.group(1) if m else "?")
            accs.append(bool(m and m.group(1) == items[k + j]["gold"]))
        if (k // a.batch) % 10 == 0:
            print(f"[anatomy] {a.variant}/{arm}: {k+len(chunk)}/{len(prompts)}", flush=True)
    from collections import Counter
    res["arms"][arm] = {"acc": sum(accs)/len(accs), "per_item": accs,
                        "pick_dist": dict(Counter(picks))}
    print(f"[anatomy] {a.variant} {arm}: acc {sum(accs)/len(accs):.4f} picks {dict(Counter(picks))}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out, flush=True)
