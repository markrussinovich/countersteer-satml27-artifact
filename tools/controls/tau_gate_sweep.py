#!/usr/bin/env python
"""Per-token (tau-gated) steering sweep: several tau values IN ONE PROCESS with shared
clean/base/ungated-anchor arms.

Why this driver exists: `--tau` is process-global in src/cli.py, but a tau comparison is
only valid paired in one process (cross-process baselines rotate on some models, FINDINGS
§23l), and the UNGATED arm doubles as the gate-no-op detector — a gated arm identical to
it means the gate silently did nothing (this project has shipped two silent no-ops).
run_arm takes `tau`/`probes` per call, so the driver runs:

  clean -> base-XPIA -> dir@alpha (ungated anchor) -> CLEAN+ ->
  [dir@alpha tau=T -> CLEAN+ tau=T]  for each T

and writes a score_table-compatible artifact (same _meta contract as src/cli.py's writer,
including arm_flags and clean_sha). Model-agnostic; everything is an argument.

Example (the FINDINGS §23ae.12 registration):
  tau_gate_sweep.py --model Qwen/Qwen3-Next-80B-A3B-Thinking \
      --outdir runs/qwen3next-80b --direction dim_no_override_ac \
      --sigma-ref dim_no_override_both --layers 28,32,40 --alpha 9 \
      --taus 0.5,0.9 --n-eval 52 --max-new 16384 --batch 4
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/datadrive/huggingface/")

from src.corpora import build_splits                           # noqa: E402
from src.probes import build_dirs, build_probes                # noqa: E402
from src.arms import run_arm                                   # noqa: E402
import xpia_defense as X                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--sigma-ref", default="",
                    help="key for --match-sigma-to semantics; empty = own sigma")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--taus", required=True, help="comma list, e.g. 0.5,0.9")
    ap.add_argument("--n-eval", type=int, default=52)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=16384)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    taus = [float(x) for x in a.taus.split(",")]
    outdir = a.outdir if os.path.isabs(a.outdir) else f"{ROOT}/{a.outdir}"

    all_samples = X.build_dataset()
    bins = build_splits(all_samples, n_test=96, n_probe=250, n_eval=a.n_eval, seed=0)
    samples = [all_samples[i] for i in bins["dev"]]
    print(f"[data] {len(samples)} dev samples (shipped), layers {layers}, "
          f"alpha {a.alpha}, taus {taus}, sigma_ref {a.sigma_ref or '(own)'}", flush=True)

    # CPU preflight BEFORE the model load (review hardening): a missing direction/sigma/
    # probe must fail in seconds, not after a 20-40 min model load on a cluster node.
    build_dirs(outdir, layers, a.direction, "cpu",
               match_sigma_to=(a.sigma_ref or None), require_sigma=True)
    build_probes(outdir, layers, "cpu")
    print("[preflight] direction + sigma + mn probe present at all steer layers", flush=True)

    model, tok = X.load_model_and_tok(a.model, a.device)
    dirs, sig, abl = build_dirs(outdir, layers, a.direction, model.device,
                                match_sigma_to=(a.sigma_ref or None),
                                require_sigma=True)
    probes = build_probes(outdir, layers, model.device)
    common = dict(batch=a.batch, max_new=a.max_new, scale="sigma", mode="add")
    steer_kw = dict(layers=layers, dirs=dirs, alpha=a.alpha, direction=a.direction,
                    sigmas=sig, ablate_axes=abl)

    results = []
    clean = run_arm(model, tok, samples, clean=True, label="clean",
                    early_abort_trunc=0, **common)
    clean.correct = 1.0
    common["ref_completions"] = clean.completions
    results.append(clean)
    results.append(run_arm(model, tok, samples, label="base-XPIA", **common))
    base_label = f"{a.direction}@{a.alpha}"
    results.append(run_arm(model, tok, samples, label=base_label, **steer_kw, **common))
    results.append(run_arm(model, tok, samples, clean=True,
                           label=f"CLEAN+{base_label}", **steer_kw, **common))
    for t in taus:
        lb = f"{a.direction}@{a.alpha}+tau{t}"
        results.append(run_arm(model, tok, samples, probes=probes, tau=t,
                               label=lb, **steer_kw, **common))
        results.append(run_arm(model, tok, samples, clean=True, probes=probes, tau=t,
                               label=f"CLEAN+{lb}", **steer_kw, **common))

    stem = (f"results_add-{a.direction.replace('_', '-')}-taugate-{os.getpid()}")
    json.dump({"model": a.model, "stage": "sweep", "steer_layers": layers,
               "driver": "tools/controls/tau_gate_sweep.py (one-process tau comparison)",
               "config": dict(vars(a), steer_layers=layers, taus=taus, scale="sigma",
                              mode="add", steer_span="payload"),
               "results": [{k: v for k, v in r.__dict__.items() if k != "completions"}
                           for r in results]},
              open(f"{outdir}/{stem}.json", "w"), indent=2, default=str)
    meta = {"corpus": "shipped", "stage": "sweep", "template_set": None,
            "steer_span": "payload", "batch": a.batch, "max_new": a.max_new,
            "n_eval": a.n_eval, "n_samples": len(samples), "shard": 0, "nshard": 1,
            "corpus_sha": None, "corpus_file": None,
            "sample_ids": [s["id"] for s in samples],
            "arm_flags": {r.label: {"aborted": r.aborted, "truncated": r.truncated}
                          for r in results},
            "steer_layers": layers, "sigma_ref": (a.sigma_ref or a.direction),
            "effective_sigmas": {a.direction: sig},
            "clean_sha": hashlib.sha256(
                json.dumps(clean.completions).encode()).hexdigest()[:16]}
    blob = json.dumps({"_meta": meta, **{r.label: r.completions for r in results}})
    cp = f"{outdir}/{stem}_completions.json"
    with open(cp + ".tmp", "w") as f:
        f.write(blob)
    os.replace(cp + ".tmp", cp)
    json.load(open(cp))
    print(f"wrote {outdir}/{stem}.json and {cp}")

    # GATE-NO-OP DETECTOR: a tau arm byte-identical to the ungated anchor means the gate
    # never modulated anything. Report; do not fail the run (identical BEHAVIOUR at a
    # weak tau is possible) -- but identical COMPLETIONS across ALL taus is a no-op.
    anchor = next(r.completions for r in results if r.label == base_label)
    ident = {r.label: sum(x == y for x, y in zip(anchor, r.completions))
             for r in results if "+tau" in r.label and not r.label.startswith("CLEAN+")}
    print(f"[gate-evidence] completions identical to ungated anchor: {ident}")
    if ident and all(v == len(samples) for v in ident.values()):
        print("TAU_GATE_NOOP_WARNING: every tau arm is byte-identical to the ungated "
              "anchor -- the gate did not modulate anything; treat tau rows as NO-OP")
    print("TAU_SWEEP_DONE rc=0")


if __name__ == "__main__":
    main()
