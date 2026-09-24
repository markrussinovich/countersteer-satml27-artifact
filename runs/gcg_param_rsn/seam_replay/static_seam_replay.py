#!/usr/bin/env python3
"""Fresh-process static-arm replay — §26.56 static-seam investigation (2026-09-24).

The two param-class GCG runs (runs/gcg_param/pm.shard*.json, --objective forced/auto,
launched 2026-09-14; runs/gcg_param_rsn/rsn.shard*.json, --objective reasoned, launched
2026-09-22; same box <FLEET_HOST_C>, same GPUs, same venv, greedy) disagree on their STATIC
arms: byte-identical within objective mode across processes (run vs smoke), different
between modes, on 48/52 defended-static rows. The static-arm code path in
tools/controls/adaptive_gcg.py takes NO objective-dependent input, so the divergence
must be process-history numerics. This script generates the static arms for named
samples in a FRESH process (no GCG history, no prior generations) via the exact same
loader + X.run_arm call, so the history-free reading can be compared byte-wise against
both runs' stored completions.

Usage (on the run box, <FLEET_HOST_C>):
  CUDA_VISIBLE_DEVICES=<g> .venv/bin/python tmp/static_seam_replay.py \
      --ids nem-119-param,nem-743-param --out tmp/replay_g<g>_p1.json
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools", "controls"))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="comma-separated sample ids")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=4096)
    a = ap.parse_args()

    torch.manual_seed(0)  # as adaptive_gcg.py --seed 0 (greedy: RNG unused, kept for parity)
    layers = [12, 16, 20]
    model, tok = X.load_model_and_tok("openai/gpt-oss-20b", a.device)
    model.requires_grad_(False)
    import __main__ as _m
    if not hasattr(_m, "TorchLogReg"):
        _m.TorchLogReg = X.TorchLogReg
    dirs, sigmas, _ = X.build_dirs(f"{ROOT}/runs/gpt-oss-20b-userabl", layers,
                                   "combo_ovr8_pat1", model.device,
                                   match_sigma_to=None)
    assert all(s_ > 0 for s_ in sigmas), sigmas
    print("[replay] sigmas", [round(s_, 2) for s_ in sigmas], flush=True)

    # the canonical param pool load, byte-for-byte the adaptive_gcg.py path
    S = json.load(open(X.param_corpus_path("dev", "fit", ROOT)))["samples"]
    man = X.param_split_manifest(ROOT, "dev")
    if man:
        by = {s["id"]: s for s in S}
        S = [by[i] for i in man if i in by]
    want = [w for w in a.ids.split(",") if w]
    by_id = {s["id"]: s for s in S}
    missing = [w for w in want if w not in by_id]
    assert not missing, f"ids not in dev/fit pool: {missing}"

    out = {}
    for sid in want:
        s = by_id[sid]
        row = {}
        for label, use_steer in (("defended_static", True), ("undefended_static", False)):
            arm = X.run_arm(model, tok, [json.loads(json.dumps(s))], label=label,
                            batch=1, max_new=a.max_new,
                            layers=layers if use_steer else None,
                            dirs=dirs if use_steer else None,
                            alpha=1.0 if use_steer else 0.0,
                            direction="combo_ovr8_pat1" if use_steer else "-",
                            scale="sigma", sigmas=sigmas if use_steer else None)
            row[label] = arm.completions[0]
            print(f"[replay] {sid} {label}: {len(arm.completions[0])} chars", flush=True)
        out[sid] = row

    body = json.dumps(out, ensure_ascii=False, indent=1)
    json.loads(body)  # parse-before-report
    with open(a.out + ".tmp", "w") as f:
        f.write(body)
    os.replace(a.out + ".tmp", a.out)
    print("REPLAY-DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
