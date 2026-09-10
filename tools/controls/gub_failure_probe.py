"""Regenerate FLIPPED general-utility items with full completions for failure reading.

The bench artifacts store per-item correctness for tool arms but not completions; this
reruns a given index list (both arms, tool mode) and saves full texts so flips can be
classified (misread vs letter-bias vs garbled copy vs constraint slip).

Usage: gub_failure_probe.py --bench mmlu|ifeval --indices 1,5,9 [--n-max 40] [same cell
flags as general_utility_bench] --out runs/gub_failures_<bench>.json
"""
import argparse, json, os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/datadrive/huggingface/")
import xpia_defense as X
from general_utility_bench import load_items, build_prompt, score, final_text

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--indices", required=True)
ap.add_argument("--n-max", type=int, default=40)
ap.add_argument("--model", default="openai/gpt-oss-20b")
ap.add_argument("--probe-run", default="runs/gpt-oss-20b-userabl")
ap.add_argument("--direction", default="combo_ovr8_pat1")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--steer-layers", default="12,16,20")
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
a = ap.parse_args()
idxs = [int(x) for x in a.indices.split(",")][: a.n_max]
items = load_items(a.bench, 500 if a.bench == "mmlu" else 250)
layers = [int(x) for x in a.steer_layers.split(",")]
model, tok = X.load_model_and_tok(a.model, a.device)
dirs, sig, abl = X.build_dirs(a.probe_run, layers, a.direction, a.device)
rows = []
for i in idxs:
    text, span = build_prompt(tok, items[i], "tool", i)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    outs = {}
    for arm in ("baseline", "steered"):
        st = None
        if arm == "steered":
            st = X.Steer(model, layers, dirs, a.alpha, "sigma", sig, abl, "add")
            st.positions = [span]
        with torch.no_grad():
            if st:
                with st: g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            else: g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        c = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        outs[arm] = {"final": final_text(c), "ok": score(a.bench, items[i], c)}
    rows.append({"i": i, "gold": items[i].get("gold"), "doc_head": items[i]["doc"][:300], **outs})
    print(f"[probe] {i}: base ok={outs['baseline']['ok']} steered ok={outs['steered']['ok']}", flush=True)
json.dump(rows, open(a.out, "w"), indent=1)
print("wrote", a.out)
