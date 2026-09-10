#!/usr/bin/env python
"""Cross-MODEL utility scoring: score arms of run B against run A's clean arm.

WHY. `score_table.py --compare` refuses on a clean_sha mismatch by design — two runs of
the SAME model must share a reference. A weight-modification defense (SecAlign DPO merge)
IS a different model, so its clean arm can never share a sha with the base model's. But
the deployment-cost question ("what does the defense cost clean traffic") is defined
against the UNMODIFIED model's behaviour (CLAUDE.md: correctness = same tool call, same
parameters, as the unattacked model). This scores candidate-run arms with
`src.scoring.behavioural_score` against the REFERENCE run's clean completions, joined by
sample id — the same join rule and scorer as score_table, no LLM, deterministic.

Prints, per candidate arm: corr (struct_exact vs base clean), plus the same arm scored
against the candidate run's OWN clean (the within-model reading score_table would give).
Both are brackets of "utility"; report both, never one alone.

Usage:
  score_ref_swap.py REF_completions.json CAND_completions.json [--arms a,b,...]

Requires both runs to be on the same corpus/split (asserted via corpus_sha + sample_ids).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.scoring import behavioural_score  # noqa: E402


def by_sid(arm_rows, sids):
    assert len(arm_rows) == len(sids), (len(arm_rows), len(sids))
    return dict(zip(sids, arm_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--arms", default=None,
                    help="comma list of candidate arms to score (default: all non-clean)")
    a = ap.parse_args()

    R, C = json.load(open(a.ref)), json.load(open(a.cand))
    rm, cm = R["_meta"], C["_meta"]
    assert rm["corpus"] == cm["corpus"], f"corpus name mismatch: {rm['corpus']} vs {cm['corpus']}"
    assert rm["corpus_sha"] == cm["corpus_sha"], (
        f"corpus sha mismatch: {rm['corpus_sha']} vs {cm['corpus_sha']}")
    if rm["corpus_sha"] is None:
        # shipped/param_abuse publish no content hash, so the sha assert above is VACUOUS
        # (None==None). The order-equal sample_id list below is the real join guard; an
        # in-place corpus rebuild with stable ids would NOT be caught here (review
        # 2026-09-01) -- provenance rests on the FINDINGS 21 audit trail.
        print("WARNING: corpus_sha is None for this corpus -- sha check is vacuous; "
              "join safety rests on the order-equal sample_id lists only")
    assert rm["sample_ids"] == cm["sample_ids"], "sample id lists differ"
    sids = rm["sample_ids"]
    assert len(set(sids)) == len(sids), "duplicate sample ids -- positional join unsafe"

    ref_clean = by_sid(R["clean"], sids)
    own_clean = by_sid(C["clean"], sids)
    arms = (a.arms.split(",") if a.arms
            else [k for k in C if k not in ("_meta", "clean")])

    print(f"ref run:  {a.ref} (clean_sha {rm['clean_sha']})")
    print(f"cand run: {a.cand} (clean_sha {cm['clean_sha']})")
    print(f"corpus {cm['corpus']} n={len(sids)}")
    print(f"{'arm':<38} {'corr_vs_REF_clean ^':>20} {'corr_vs_OWN_clean ^':>20}")

    def corr(ref, rows):
        # behavioural_score returns a dict; the pipeline's CORRECT is its `struct_exact`
        # key, valid only where `scoreable` (the reference made >=1 parseable call --
        # same denominator rule as score_table / the sweep's own report).
        n_ok = n_sc = 0
        for sid in sids:
            v = behavioural_score(ref[sid], rows[sid])
            if not v.get("scoreable"):
                continue
            n_sc += 1
            n_ok += bool(v.get("struct_exact"))
        return f"{n_ok}/{n_sc} = {n_ok / max(n_sc, 1):.3f}"

    for arm in arms:
        rows = by_sid(C[arm], sids)
        print(f"{arm:<38} {corr(ref_clean, rows):>20} {corr(own_clean, rows):>20}")
    # the cand CLEAN arm vs REF clean is the deployment-cost number:
    print(f"{'clean (cand model, no attack)':<38} {corr(ref_clean, own_clean):>20} {'1.000 (identity)':>20}")


if __name__ == "__main__":
    main()
