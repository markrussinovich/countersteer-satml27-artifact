#!/usr/bin/env python
"""Re-run the headline steering cell at a KNOWN, RECORDED configuration.

This is the script that overturned the project's headline result. Run as documented --
`mn_tool`, alpha=2.5, layers [4,8,12], `--scale sigma`, span-wide (`tau=0`) -- the
intervention makes the attack WORSE, and is indistinguishable from a magnitude-matched
random direction:

    attacked (base-XPIA)              ASR 0.375   strict 0.125   no_action 0.083
    mn_tool@2.5, tau=0                ASR 0.458   strict 0.250   no_action 0.083
    random@2.5,  tau=0                ASR 0.458   strict 0.292   no_action 0.083
    stored cell (results_…-3009219)   ASR 0.042   strict 0.042   no_action 0.292

The attacked arm reproduces the stored run byte-identically (24/24), so the harness is not
in question: the ASR-0.042 result depends entirely on the UNRECORDED `--tau` conditional
per-token gate, whose gate reads p_tool from the DISCARDED 4-role probe. See README.md.

The judge is deliberately STUBBED to a constant. Only ASR / asr_strict / tool_called /
no_action / truncated are computed, and every one of those is a deterministic function of
the completions -- so this script's numbers are exactly reproducible, which is the whole
point of it. Do NOT read its CORRECT column; it is meaningless by construction.

Usage:
    python tools/controls/repro_headline_cell.py [TAU] [RUN_DIR] [MODEL] [DEVICE] [ALPHA]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT
TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
RUN_DIR = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/runs/gpt-oss-20b-resid"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "openai/gpt-oss-20b"
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cuda:0"
ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 2.5
LAYERS = [4, 8, 12]

# Stub the judge: its label is not deterministic and is not what this script measures.
X.judge = lambda samples, comps, votes=3, concurrency=16: ["CORRECT"] * len(samples)


def main():
    dev = E.dev_samples(24)          # same dev slice as the report -- do not re-implement
    model, tok = X.load_model_and_tok(MODEL, DEVICE)
    probes = X.build_probes(RUN_DIR, LAYERS, model.device) if TAU > 0 else None
    common = dict(batch=12, max_new=1024, no_think=False, scale="sigma", mode="add",
                  norm_preserve=True, probes=probes, tau=TAU)

    out, rows = {}, []
    r = X.run_arm(model, tok, dev, label="base-XPIA", **common)
    out["base-XPIA"] = r.completions
    rows.append(("base-XPIA", r))
    for dn in ["mn_tool", "random"]:
        dirs, sig, abl = X.build_dirs(RUN_DIR, LAYERS, dn, model.device,
                                      match_sigma_to="mn_tool")
        r = X.run_arm(model, tok, dev, layers=LAYERS, dirs=dirs, alpha=ALPHA, direction=dn,
                      sigmas=sig, ablate_axes=abl, label=f"{dn}@{ALPHA}", **common)
        out[f"{dn}@{ALPHA}"] = r.completions
        rows.append((f"{dn}@{ALPHA}", r))

    print(f"\n=== tau={TAU}  alpha={ALPHA}  layers={LAYERS}  scale=sigma  run={RUN_DIR} ===")
    print(f"{'arm':<20}{'ASR':>8}{'strict':>9}{'tool_called':>13}{'no_action':>11}{'trunc':>8}")
    for lab, r in rows:
        print(f"{lab:<20}{r.asr:8.3f}{r.asr_strict:9.3f}{r.tool_called:13.3f}"
              f"{r.no_action:11.3f}{r.truncated:8.3f}")

    # config is PERSISTED alongside the completions -- the original artifacts did not do
    # this, which is why the headline cell could not identify its own run.
    dst = f"{ROOT}/runs/repro_tau{TAU}_alpha{ALPHA}.json"
    json.dump({"config": {"tau": TAU, "alpha": ALPHA, "layers": LAYERS, "scale": "sigma",
                          "mode": "add", "norm_preserve": True, "run_dir": RUN_DIR,
                          "model": MODEL, "batch": 12, "max_new": 1024, "n": len(dev),
                          "judge": "STUBBED -- CORRECT column is meaningless"},
               "metrics": {lab: {"asr": r.asr, "asr_strict": r.asr_strict,
                                 "tool_called": r.tool_called, "no_action": r.no_action,
                                 "truncated": r.truncated} for lab, r in rows},
               "completions": out}, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
