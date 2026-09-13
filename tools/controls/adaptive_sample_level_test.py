#!/usr/bin/env python3
"""Sample-level paired inference for the adaptive query search (owner review fix,
2026-09-12): attempts nest within samples, so attempt-level McNemar/binomial p over
432 attempts is anti-conservative. This computes, per arm pair, each sample's mean
firing rate over its attempts, then an EXACT two-sided sign-flip permutation test on
the paired per-sample means (2^n enumerated) and a paired bootstrap 95% CI.

Usage:
  python tools/controls/adaptive_sample_level_test.py \
    --treat runs/adaptive_framing.shard1.json \
    --arm undefended runs/adaptive_param_none.json \
    --arm spotlighting runs/adaptive_param_spotlight.json \
    --arm CachePrune runs/adaptive_param_cacheprune.shard0.json runs/adaptive_param_cacheprune.shard1.json \
    --out runs/adaptive_sample_level_tests.json

Artifacts are the query-search row dumps ({"config":..., "rows":[{"sid","fired",...}]}).
The treat artifact and every arm must cover the identical sid set (asserted).
"""
import argparse, json, random
from collections import defaultdict


def sid_means(paths):
    acc = defaultdict(list)
    for p in paths:
        for r in json.load(open(p))["rows"]:
            acc[r["sid"]].append(bool(r["fired"]))
    return {s: sum(v) / len(v) for s, v in acc.items()}


def exact_signflip_p(diffs):
    n = len(diffs)
    if n > 24:
        raise SystemExit(f"exact enumeration over 2^{n} is too large; add a sampler")
    obs = abs(sum(diffs) / n)
    cnt = 0
    for mask in range(1 << n):
        s = 0.0
        for i in range(n):
            s += diffs[i] if (mask >> i) & 1 else -diffs[i]
        if abs(s / n) >= obs - 1e-12:
            cnt += 1
    return cnt / (1 << n)


def bootstrap_ci(diffs, iters=10000, seed=20260912):
    rng = random.Random(seed)
    n = len(diffs)
    bs = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return bs[int(0.025 * iters)], bs[int(0.975 * iters) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treat", nargs="+", required=True, help="treated arm artifact path(s)")
    ap.add_argument("--arm", nargs="+", action="append", required=True,
                    metavar=("NAME", "PATHS"), help="baseline arm: NAME path [path...]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    treat = sid_means(a.treat)
    sids = sorted(treat)
    out = {"treat_paths": a.treat, "n_samples": len(sids),
           "treat_mean": sum(treat.values()) / len(sids), "arms": {}}
    for spec in a.arm:
        name, paths = spec[0], spec[1:]
        base = sid_means(paths)
        assert set(base) == set(treat), f"sid mismatch on {name}"
        d = [base[s] - treat[s] for s in sids]
        lo, hi = bootstrap_ci(d)
        out["arms"][name] = {
            "paths": paths,
            "base_mean": sum(base.values()) / len(sids),
            "paired_mean_diff": sum(d) / len(d),
            "bootstrap95": [lo, hi],
            "exact_signflip_p": exact_signflip_p(d),
            "samples_favoring_treat": sum(x > 0 for x in d),
        }
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out["arms"].items():
        print(f"{k:14s} base {v['base_mean']:.3f} vs treat {out['treat_mean']:.3f} | "
              f"diff {v['paired_mean_diff']:+.3f} [{v['bootstrap95'][0]:.3f},{v['bootstrap95'][1]:.3f}] "
              f"| exact p={v['exact_signflip_p']:.6f} | favoring {v['samples_favoring_treat']}/{out['n_samples']}")


if __name__ == "__main__":
    main()
