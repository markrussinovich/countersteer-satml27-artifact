#!/usr/bin/env python
"""ROUTER-NULL DIRECTION SURGERY: project the steering direction out of the subspace the
downstream MoE routers can see, and merge the result into the probe pickles as a new
direction so the existing sweep can run it with `--directions <name>` and no other change.

WHY (FINDINGS section 23, finding 3). Direction QUALITY does not discriminate the models
where steering works from the ones where it fails -- fired-vs-not separation is 0.8-1.0
sigma on all four. What discriminates them is what the dose does to MoE routing on the way
to moving behaviour: the working gpt-oss cell leaves ~90% of top-k routing intact, whereas
on Qwen3-Next-80B and GLM-4.5-Air every behaviourally effective dose moves 55-90% of the
routing mass and the model degenerates (rumination, repetition loops, tool-call spam) --
the phenotype of computing whole spans on the wrong experts.

THE SURGERY. A router's decision depends on the residual only through W_g @ RMSNorm(h),
so a residual displacement that lies in the NULL SPACE of the stacked downstream router
gates is invisible to those routers to first order, while still moving the token along
whatever remains of the steering axis. Take the stacked gate matrices of the next
`--n-downstream` MoE layers, take their top-r right singular subspace V (the directions
those routers are most sensitive to), and steer along

    d_rnull = normalise( d - V^T V d )

CPU PRE-SCREEN ALREADY RUN (runs/router_sensitivity_extra.json,
`routernull_projection_qwen3next`): on Qwen3-Next this retains 0.70-0.91 of the
fired-vs-not separation while cutting the router-logit response to 0.14-0.43x, at
r = 512 down to 64. That is a Pareto knob, measured at first order on captured span-mean
activations; it is NOT a behavioural result and this script does not make it one. Run the
smoke.

SIGMA IS NOT OPTIONAL. `probes.build_dirs` reads `sigmas[name]` with a `.get(name, 0.0)`
fallback, and `--scale sigma` then makes the step alpha*0 = 0: a no-op arm that reports as
a defense. This script computes the sigma the same way build_override_direction.py does --
the standard deviation of the UNCENTERED capture-row projections onto the unit direction --
asserts it is positive, and re-reads every pickle it wrote to prove the direction AND its
sigma are both there.

Usage:
    python tools/controls/build_routernull_direction.py \
        --run runs/qwen3next-80b/probe \
        --capture runs/qwen3next-80b/override_slope_qwen3next-80b.json \
        --layers 28,32,40 --model Qwen/Qwen3-Next-80B-A3B-Thinking \
        --ranks 64,128,256,512 [--n-downstream 8] [--dry-run]

    # GLM lives under a different HF root on this fleet:
    ... --model zai-org/GLM-4.5-Air --hf-home /datadrive2/huggingface
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402

from src.moe import checkpoint_router_weights, resolve_snapshot  # noqa: E402

X = E.X
ROOT = E.ROOT


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="probe directory holding probe_L*.pkl (written in place)")
    ap.add_argument("--capture", required=True,
                    help="the override_slope_*.json this model's direction was fit from; "
                         "its activations define sigma and the fired-vs-not separation")
    ap.add_argument("--layers", required=True, help="comma list of STEERED layers")
    ap.add_argument("--model", default=None,
                    help="hub id, used only to locate the local snapshot for the router "
                         "weights -- the model is never loaded and no GPU is touched")
    ap.add_argument("--snapshot", default=None, help="explicit snapshot dir; beats --model")
    ap.add_argument("--hf-home", dest="hf_home", default=None,
                    help="HF cache root to search when the model is not in $HF_HOME "
                         "(GLM-4.5-Air sits under /datadrive2/huggingface on this fleet)")
    ap.add_argument("--direction", default="dim_no_override_both",
                    help="source direction key in the probe pickles")
    ap.add_argument("--ranks", default="64,128,256,512",
                    help="comma list of r; one direction `<direction>_rnull<r>` per value")
    ap.add_argument("--n-downstream", dest="n_downstream", type=int, default=8,
                    help="how many MoE layers below each steered layer to stack into the "
                         "router subspace. 0 = every downstream MoE layer. Default 8 is "
                         "what the runs/router_sensitivity_extra.json pre-screen used, so "
                         "its numbers are directly comparable to this script's report.")
    ap.add_argument("--report", default=None, help="output JSON path (default derived)")
    ap.add_argument("--force", action="store_true",
                    help="replace a direction key that already exists. Without it the "
                         "script refuses, because the key carries no discriminator for "
                         "--n-downstream / --capture / --snapshot and a silent overwrite "
                         "makes the previous fit unreproducible.")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="compute and print everything, write nothing. The pickles are read "
                         "by build_dirs at the start of every steered arm, so rewriting them "
                         "under a running sweep is a torn read.")
    return ap.parse_args(argv)


def group_centered(A, rows):
    """Within-(sid x delegation) centering -- byte-identical to build_override_direction.py,
    which is what makes `sep_retained` comparable to the source direction's own fit."""
    grp = np.array([f'{r["sid"]}::{r.get("delegation", "none")}' for r in rows])
    Ac = A.copy()
    for g in sorted(set(grp)):
        m = grp == g
        Ac[m] -= Ac[m].mean(0)
    return Ac, ("sid x delegation" if any("delegation" in r for r in rows) else "sid")


def main(argv=None):
    a = parse_args(argv)
    layers = [int(x) for x in a.layers.split(",")]
    ranks = sorted({int(x) for x in a.ranks.split(",")})
    if not a.model and not a.snapshot:
        raise SystemExit("pass --model (hub id) or --snapshot (local dir)")
    snap = resolve_snapshot(a.model or "", snapshot=a.snapshot, hf_home=a.hf_home)
    print(f"[snapshot] {snap}")

    # Router weights FIRST: "the checkpoint is not downloaded on this host" is the most
    # likely failure and the most expensive one to discover late. Layer indices present here
    # are exactly the MoE layers; dense layers simply have no router key.
    Wg = checkpoint_router_weights(snap)

    if not os.path.exists(a.capture):
        raise SystemExit(f"--capture {a.capture} does not exist; it must be the "
                         f"override_slope_*.json this model's direction was fit from "
                         f"(sigma and the fired-vs-not separation are computed from it)")
    cap = json.load(open(a.capture))
    rows = cap["rows"]
    fired = np.array([bool(r["fired"]) for r in rows])
    if fired.std() == 0:
        raise SystemExit(f"{a.capture}: `fired` has zero variance -- there is no separation "
                         f"to retain, so router-null surgery cannot be evaluated here.")
    moe_layers = sorted(Wg)
    n_layers = moe_layers[-1] + 1
    print(f"[routers] {len(moe_layers)} MoE layers in {os.path.basename(snap)}: "
          f"{moe_layers[:6]}{'...' if len(moe_layers) > 6 else ''} "
          f"(n_experts x hidden = {tuple(Wg[moe_layers[0]].shape)})")

    report = {"run": a.run, "capture": a.capture, "snapshot": snap,
              "source_direction": a.direction, "ranks": ranks,
              "n_downstream": a.n_downstream, "per_layer": {}}
    written = []
    for L in layers:
        if str(L) not in {str(k) for k in cap["layers"]}:
            raise SystemExit(f"{a.capture} has no activations for layer {L}")
        A = np.array([r["act"][str(L)] for r in rows], dtype=np.float32)
        Ac, how = group_centered(A, rows)
        p = X.load_probe(f"{a.run}/probe_L{L}.pkl")
        if a.direction not in p["dirs"]:
            raise SystemExit(f"L{L}: no direction {a.direction}; have {list(p['dirs'])}")
        d = np.asarray(p["dirs"][a.direction], dtype=np.float32)
        u = d / (np.linalg.norm(d) + 1e-12)
        # ROW-SET AGREEMENT. build_override_direction.py can fit on a FILTERED subset
        # (--delegated-only). If the source direction was fitted on fewer rows than this
        # script uses for sigma, then (a) `sep_retained` is not comparable to "the source
        # direction's own fit" and (b) the _rnull direction's sigma sits on a different row
        # set from its source's -- so an _rnull arm and its source arm at the same --alpha
        # are NOT dose-matched (adversarial review 2026-09-01, defect 10).
        src_prov = p.get("override_provenance", {}).get(a.direction, {})
        src_rows = src_prov.get("n_rows")
        if src_rows is not None and int(src_rows) != len(rows):
            raise SystemExit(
                f"L{L}: `{a.direction}` was fitted on {src_rows} rows but --capture "
                f"{a.capture} has {len(rows)}. sigma and sep_retained would be computed on a "
                f"different row set from the source direction's own fit, so the two "
                f"directions would NOT be dose-matched at the same --alpha. Pass the capture "
                f"(and any row filter) the source direction was actually fitted on.")
        if u.shape[0] != Wg[moe_layers[0]].shape[1]:
            raise SystemExit(f"L{L}: direction has {u.shape[0]} dims but the router weights "
                             f"have {Wg[moe_layers[0]].shape[1]} -- wrong snapshot for this "
                             f"probe directory.")
        down = [Lc for Lc in moe_layers if Lc > L]
        if a.n_downstream > 0:
            down = [Lc for Lc in down if Lc < L + 1 + a.n_downstream][:a.n_downstream]
        if not down:
            raise SystemExit(f"L{L}: no MoE router below it (model has {n_layers} layers); "
                             f"there is nothing for the surgery to hide from.")
        Wall = np.concatenate([Wg[Lc].numpy() for Lc in down], 0)
        # Right singular subspace via eigh(W^T W), NOT svd(W). We never use U, and with
        # --n-downstream 0 on Qwen3-Next `Wall` is (48*512, 2048) = 24576 x 2048, whose U
        # alone is ~200 MB that svd computes and this script throws away -- once per steered
        # layer (adversarial review 2026-09-01, defect 11). eigh returns eigenvalues
        # ASCENDING, so reverse to get the top-r subspace first. float64 for the Gram matrix:
        # squaring the singular values halves the effective precision, and 2048x2048 in
        # float64 is 32 MB.
        evals, evecs = np.linalg.eigh(Wall.astype(np.float64).T @ Wall.astype(np.float64))
        Vt = evecs[:, ::-1].T.astype(np.float32)     # rows = right singular vectors, desc
        # TRUNCATE to min(Wall.shape), exactly what svd(full_matrices=False) returned.
        # eigh on the (H, H) Gram matrix yields H eigenvectors, and when the stacked router
        # matrix has fewer ROWS than H (GLM-4.5-Air: 8*128 = 1024 rows vs hidden 4096) the
        # extra ones sit in the NULL SPACE of W -- projecting the direction out of those
        # would destroy steering signal while buying no router invisibility at all. The rank
        # guard below then skips r beyond this, unchanged.
        Vt = Vt[:int(min(Wall.shape))]
        sep0 = float((Ac[fired] @ u).mean() - (Ac[~fired] @ u).mean())
        base_logit = float(np.linalg.norm(Wall @ u))
        per = {"downstream_routers": down, "stacked_rows": int(Wall.shape[0]),
               "sep_raw_source": sep0, "centering": how, "ranks": {}}
        print(f"\nL{L}: stacking routers {down} -> {Wall.shape} | "
              f"sep(source, {how}-centered) = {sep0:+.4f}")
        for r in ranks:
            if r > Vt.shape[0]:
                print(f"  r={r:<4} SKIPPED: only {Vt.shape[0]} right singular vectors exist "
                      f"(stacked router matrix has rank <= {Vt.shape[0]})")
                continue
            V = Vt[:r]
            d2 = u - V.T @ (V @ u)
            n2 = float(np.linalg.norm(d2))
            if n2 < 1e-6:
                print(f"  r={r:<4} SKIPPED: the direction lies entirely inside the top-{r} "
                      f"router subspace (residual norm {n2:.2e}) -- nothing survives")
                continue
            d2 = (d2 / n2).astype(np.float32)
            sep = float((Ac[fired] @ d2).mean() - (Ac[~fired] @ d2).mean())
            # sigma EXACTLY as build_override_direction.py defines it: spread of the
            # UNCENTERED capture-row projections onto the unit axis (the ITI convention
            # `--scale sigma` multiplies alpha by).
            sigma = float((A @ d2).std())
            ratio = float(np.linalg.norm(Wall @ d2) / max(base_logit, 1e-12))
            key = f"{a.direction}_rnull{r}"
            cell = dict(cos_to_source=float(d2 @ u), residual_norm=n2,
                        sep=sep, sep_retained=(sep / sep0 if sep0 else float("nan")),
                        routerlogit_ratio=ratio, sigma=sigma, key=key)
            per["ranks"][str(r)] = cell
            print(f"  r={r:<4} -> {key:<34} cos={cell['cos_to_source']:+.3f}  "
                  f"sep_retained={cell['sep_retained']:.3f}  "
                  f"routerlogit={ratio:.3f}x  sigma={sigma:.4f}")
            if sigma <= 0:
                raise SystemExit(
                    f"L{L} r={r}: sigma computed as {sigma} -- a zero sigma makes "
                    f"--scale sigma apply a step of alpha*0 and the arm becomes a silent "
                    f"no-op that still prints as a defense. Refusing to write it.")
            if key in p["dirs"] and not a.force:
                # A silent overwrite makes the PREVIOUS fit unreproducible: the key carries
                # no discriminator for --n-downstream / --capture / --snapshot, and
                # `routernull_provenance` records only the newest fit. A locked cell naming
                # this direction would change meaning with no trace (review defect 9).
                prev = p.get("routernull_provenance", {}).get(key, {})
                raise SystemExit(
                    f"L{L}: {key} already exists in {a.run}/probe_L{L}.pkl "
                    f"(previous fit: n_downstream over routers {prev.get('downstream_routers')}, "
                    f"capture {prev.get('capture')}, snapshot {prev.get('snapshot')}). "
                    f"Overwriting would make that number unreproducible. Pass --force to "
                    f"replace it deliberately, or use a different --ranks / rename the key.")
            p["dirs"][key] = d2
            p["sigmas"][key] = sigma
            p.setdefault("routernull_provenance", {})[key] = {
                "source_direction": a.direction, "rank": r,
                "downstream_routers": down, "snapshot": snap,
                "capture": a.capture, "sep_retained": cell["sep_retained"],
                "routerlogit_ratio": ratio, "centering": how,
                "n_rows": len(rows), "source_provenance": src_prov,
                "note": "d - V^T V d over the top-r right singular subspace of the stacked "
                        "downstream router gate weights, renormalised to unit norm; sigma "
                        "= std of uncentered capture projections onto it",
            }
        if not per["ranks"]:
            raise SystemExit(f"L{L}: every requested rank was skipped; nothing to write.")
        report["per_layer"][str(L)] = per
        if not a.dry_run:
            # ATOMIC. `pickle.dump` streams into the handle, so a serialisation error part
            # way through leaves a TRUNCATED pickle with a fresh mtime -- and this file
            # carries every OTHER direction for this layer, including the ones a locked cell
            # depends on. Serialise to bytes, write, then os.replace, which is also what
            # makes a concurrent `build_dirs` read either the old file or the new one and
            # never a half-written one (adversarial review 2026-09-01, defect 8).
            blob = pickle.dumps(p)
            dst_pkl = f"{a.run}/probe_L{L}.pkl"
            with open(dst_pkl + ".tmp", "wb") as f:
                f.write(blob)
            os.replace(dst_pkl + ".tmp", dst_pkl)
            written.append(L)

    if a.dry_run:
        print("\n[dry-run] nothing written -- no pickle touched, no report written")
        return 0

    # READ BACK. Existence is not success; a direction without its sigma is worse than a
    # missing direction because the sweep runs it and calls the result a defense.
    for L in written:
        q = X.load_probe(f"{a.run}/probe_L{L}.pkl")
        for r, cell in report["per_layer"][str(L)]["ranks"].items():
            k = cell["key"]
            assert k in q["dirs"], f"L{L}: {k} missing from the pickle after write"
            got = float(q.get("sigmas", {}).get(k, 0.0))
            assert got > 0, f"L{L}: {k} written with sigma {got} -- step would be alpha*0"
            assert abs(got - cell["sigma"]) < 1e-6, f"L{L}: {k} sigma round-trip mismatch"
    print(f"\nverified: direction + positive sigma present for every rank at layers "
          f"{written}")

    tag = os.path.basename(os.path.normpath(a.run))
    if tag == "probe":
        tag = os.path.basename(os.path.dirname(os.path.normpath(a.run)))
    dst = a.report or f"{ROOT}/runs/routernull_direction_{tag}.json"
    blob = json.dumps(report, indent=1)
    with open(dst + ".tmp", "w") as f:
        f.write(blob)
    os.replace(dst + ".tmp", dst)
    json.load(open(dst))
    print(f"merged {len(ranks)} router-null directions into {a.run}/probe_L*.pkl")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
