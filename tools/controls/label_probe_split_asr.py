#!/usr/bin/env python
"""Generate ASR labels on the 250-sample probe-dev split (unsteered, attacked).

Produces the reusable asset for the SUCCEEDED-vs-BLOCKED injection direction: for every
sample in the template-disjoint probe split, the unsteered attacked completion and whether
the attack fired.

Why succeeded-vs-blocked rather than injected-vs-legitimate: the existing `inj_dim`
(xpia_defense.py:725) labels tokens by span membership, and its own report shows that is
substantially a BOILERPLATE detector -- a bag-of-words null over token ids reaches 0.988
by-sample because 988/1271 injections share the literal "[PRIORITY MESSAGE]" prefix.
Conditioning on SUCCESS puts that boilerplate in BOTH classes, where it cancels in the
difference of means. What survives is what distinguishes an injection that works from one
that does not -- the causal axis, not the identity axis.

Note the probe split is a single template family, so the boilerplate cancellation is exact,
and dev/test are different families -- making the template-family holdout automatic.

Labels come from `attack_influenced` (the same deterministic function the sweep uses), NOT
from the LLM judge. Generation is greedy, so this is exactly reproducible.

Shard across GPUs (each shard is an independent process):
    for g in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$g python tools/controls/label_probe_split_asr.py $g 4 &
    done; wait
    python tools/controls/label_probe_split_asr.py --merge

Writes runs/probe_split_asr_labels[.shardN].json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

X = E.X
ROOT = E.ROOT
OUT = f"{ROOT}/runs/probe_split_asr_labels.json"


def merge():
    """Merge shard files, with the completeness checks an unguarded `update()` lacks.

    Hazards this guards against: a STALE shard file from a previous run with a different
    `nshard` (whose labels were generated under a different batch composition) being
    silently merged, and merging before every shard has finished. Neither raises on its
    own -- `n_shards` would just count files.
    """
    shards, metas, total = {}, [], 0
    for f in sorted(os.listdir(f"{ROOT}/runs")):
        if not f.startswith("probe_split_asr_labels.shard"):
            continue
        d = json.load(open(f"{ROOT}/runs/{f}"))
        metas.append(d)
        shards.update(d["samples"])
        total += d["n"]
    if not shards:
        raise SystemExit("no shards found")

    nsh = {m["nshard"] for m in metas}
    mdl = {m["model"] for m in metas}
    got = sorted(m["shard"] for m in metas)
    assert len(nsh) == 1, f"shards disagree on nshard: {nsh} -- stale file present?"
    assert len(mdl) == 1, f"shards disagree on model: {mdl}"
    assert got == list(range(nsh.pop())), f"expected shards {list(range(len(got)))}, got {got}"
    assert total == len(shards), (
        f"key collision: shards claim {total} samples but merged to {len(shards)}")
    _all = X.build_dataset()
    expect = {_all[i]["id"] for i in X.build_splits(_all, verbose=False)["probe"]}
    assert set(shards) == expect, (
        f"merged set != probe split ({len(set(shards) ^ expect)} differing ids)")

    # `fired is None` means the sample carries no attacker-specific value to detect, so the
    # attack is NOT SCOREABLE -- it is not evidence of blocking. run_arm excludes these
    # from the ASR denominator, and the succeeded-vs-blocked fit must DROP them rather than
    # putting them in the negative class.
    succ = [k for k, v in shards.items() if v["fired"] is True]
    blok = [k for k, v in shards.items() if v["fired"] is False]
    none = [k for k, v in shards.items() if v["fired"] is None]
    out = {"n": len(shards), "n_shards": len(metas), "model": mdl.pop() if mdl else None,
           "n_succeeded": len(succ), "n_blocked": len(blok), "n_unscoreable": len(none),
           "unscoreable_ids": none, "samples": shards}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"merged {len(metas)} shards -> {len(shards)} samples: {len(succ)} succeeded / "
          f"{len(blok)} blocked / {len(none)} UNSCOREABLE (dropped from the fit)")
    print(f"wrote {OUT}")


def main():
    if "--merge" in sys.argv:
        return merge()
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nshard = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    model_id = sys.argv[3] if len(sys.argv) > 3 else "openai/gpt-oss-20b"

    all_samples = X.build_dataset()
    bins = X.build_splits(all_samples)          # SAME partition as the sweep
    probe = [all_samples[i] for i in bins["probe"]]

    # Model-specific params come from DEFAULTS, not hardcoded: Nemotron needs
    # max_new=4096/batch=4 and a hardcoded 12/1024 would silently be wrong for it.
    cfg = dict(X.DEFAULTS.get(model_id, X.DEFAULTS["default"]))
    BATCH = cfg.get("batch", 12)

    # CHUNK GLOBALLY, then hand whole chunks to shards. Striding `probe[shard::nshard]`
    # BEFORE run_arm chunks it means shard 0's first batch is probe[0],probe[4],probe[8],...
    # -- a different batch composition than an unsharded run's probe[0:12]. Batch shape
    # changes bf16 activations (1.7-5.1% relative L2, todo/02-efficiency-backlog.md), so
    # that made the emitted labels a function of `nshard`. Whole global chunks make
    # composition nshard-independent, and also leave ONE partial chunk instead of `nshard`.
    chunks = [probe[c:c + BATCH] for c in range(0, len(probe), BATCH)]
    mine = [s for ci, ch in enumerate(chunks) if ci % nshard == shard for s in ch]
    print(f"[shard {shard}/{nshard}] {len(mine)} samples from "
          f"{len([c for ci, c in enumerate(chunks) if ci % nshard == shard])} of "
          f"{len(chunks)} global chunks (batch={BATCH})")

    # judge is irrelevant here -- we want the deterministic attack_influenced label only.
    # NOTE it still prints a fabricated CORRECT=1.000 into the log; ignore that column.
    X.judge = lambda samples, comps, votes=3, concurrency=16: ["CORRECT"] * len(samples)
    model, tok = X.load_model_and_tok(model_id, "cuda:0")
    r = X.run_arm(model, tok, mine, label=f"probe-split-shard{shard}",
                  batch=BATCH, max_new=cfg.get("max_new", 1024),
                  no_think=cfg.get("no_think", False))

    rec = {}
    for s, c in zip(mine, r.completions):
        v = X.attack_influenced(s, c)
        rec[s["id"]] = {"fired": bool(v) if v is not None else None,
                        "template": X.template_key(s), "completion": c}
    fired = sum(1 for v in rec.values() if v["fired"])
    dst = f"{ROOT}/runs/probe_split_asr_labels.shard{shard}.json"
    json.dump({"shard": shard, "nshard": nshard, "model": model_id,
               "n": len(rec), "n_succeeded": fired, "samples": rec},
              open(dst, "w"), indent=1)
    print(f"[shard {shard}] {fired}/{len(rec)} succeeded -> {dst}")


if __name__ == "__main__":
    main()
