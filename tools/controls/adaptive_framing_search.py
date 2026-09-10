#!/usr/bin/env python
"""BLACK-BOX QUERY-BASED ADAPTIVE ATTACK: best-of-N framing search against the DEFENDED
agent (owner threat model, FINDINGS 10v).

The attacker cannot see the defense's internals but can query the deployed (steered)
agent and keep whichever attack wording works. The query set is the full
OVERRIDE x VOICE framing grid from override_slope_experiment (the same generator the
direction was fit against — attacker-pessimal for us, since these are the framings the
defense knows best), applied to each eval sample's own injected action. Per sample the
adaptive outcome is ANY query fired; the headline is best-of-N ASR vs the static ASR of
the corpus's shipped wording, both under the deployed defense.

Usage:
  .venv/bin/python tools/controls/adaptive_framing_search.py \
      --n 24 --direction combo_ovr8_pat1 --alpha 1.0 --steer-layers 12,16,20 \
      --shard 0 --nshard 4 --out runs/adaptive_framing.shard0.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402
import override_slope_experiment as O  # noqa: E402

X = E.X
ROOT = E.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--probe-dir", default=f"{ROOT}/runs/gpt-oss-20b-userabl")
    ap.add_argument("--direction", default="combo_ovr8_pat1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--match-sigma-to", default=None,
                    help="anchor the step to another direction's sigma (the battery "
                         "convention for scopeP4: --alpha 8.06 --match-sigma-to "
                         "dim_no_override)")
    ap.add_argument("--steer-layers", default="12,16,20")
    ap.add_argument("--decode-direction", dest="decode_direction", default=None,
                    help="ALSO steer at decode time along this direction (FINDINGS §15; "
                         "fit by tools/controls/decision_point_fit.py), at its own sigma")
    ap.add_argument("--decode-alpha", dest="decode_alpha", type=float, default=0.0)
    ap.add_argument("--defense", default="steer",
                    choices=["steer", "none", "cacheprune", "spotlight"],
                    help="which DEFENDED agent the attacker queries. `steer` (default) is "
                         "the original behaviour: the steering cell named by --direction/"
                         "--alpha/--steer-layers. `none` = the UNDEFENDED model (the "
                         "adaptive union anchor). `cacheprune` = the CachePrune baseline "
                         "(arXiv:2504.21228), requires --kv-mask; applied over the payload "
                         "span exactly as in xpia_defense --defense cacheprune. "
                         "`spotlight` = spotlighting-with-delimiting (arXiv:2403.14720), "
                         "the AgentDojo inbuilt prompt baseline applied to the single-turn "
                         "render: SPOTLIGHT_SUFFIX on the system message + tool output "
                         "wrapped in << >>. For every non-steer defense the steering args "
                         "are ignored.")
    ap.add_argument("--kv-mask", dest="kv_mask", default=None,
                    help="mask JSON from tools/controls/build_cacheprune_mask.py "
                         "(required for --defense cacheprune)")
    ap.add_argument("--actions", default="tool,param",
                    help="restrict the query grid's action factor (e.g. `param` for the "
                         "param-only adaptive readout)")
    ap.add_argument("--save-completions", dest="save_completions",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="persist each query's completion in the artifact rows "
                         "(size-bounded by --completion-cap). The 2026-08-29 runs saved "
                         "only `fired`, which made the cracked transcripts unreadable "
                         "after the fact -- default ON so that cannot recur; "
                         "--no-save-completions restores the lean artifact")
    ap.add_argument("--completion-cap", dest="completion_cap", type=int, default=4000,
                    help="chars of each completion kept in the artifact; head+tail halves "
                         "so the tool call at the END survives the cap")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--shard", type=int, default=0, help="over FRAMINGS (queries)")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    layers = [int(x) for x in a.steer_layers.split(",")]
    model, tok = X.load_model_and_tok(a.model, a.device)
    # ── wire-format dispatch (the §12 bridge conventions, shared via src/scoring):
    # ChatML/Thinking models open every completion inside <think>, and Qwen quotes
    # complete parseable <tool_call> JSON while DELIBERATING (17 think-quoted blocks in
    # one steered arm vs 0 clean, adversarial review 2026-08-30), so `fired` scored on
    # raw text over-counts cracks arm-asymmetrically. Score on reasoning_free text;
    # persist the raw reading beside it. Harmony passes through unchanged (the recorded
    # gpt-oss adaptive numbers keep their convention).
    fmt = X.fmt_of(tok)
    if fmt == "chatml" and a.max_new < 2048:
        print(f"[WARN] chatml/Thinking model with --max-new {a.max_new}: a turn truncated "
              f"inside <think> emits no call and poisons `fired` (the §12 truncation "
              f"incident); use >= 2048", flush=True)
    kvm = None
    if a.defense == "cacheprune":
        if not a.kv_mask:
            raise SystemExit("--defense cacheprune requires --kv-mask (build it with "
                             "tools/controls/build_cacheprune_mask.py)")
        kvm = X.load_kv_mask(a.kv_mask, model.config)
    if a.defense == "steer":
        dirs, sigmas, _ = X.build_dirs(a.probe_dir, layers, a.direction, model.device,
                                       match_sigma_to=a.match_sigma_to)
        # sigma provenance, printed once (the bridge convention): the Qwen pickles have no
        # `dim_no_override` entry and build_dirs falls back SILENTLY on a missing
        # match-sigma-to key, so the resolved step is the only trustworthy record
        rt = len(layers) ** 0.5
        print(f"[cell] {a.direction} @ alpha {a.alpha} "
              f"(match_sigma_to={a.match_sigma_to!r}) sigmas "
              f"{[round(s, 4) for s in sigmas]} -> per-layer step "
              f"{[round(a.alpha / rt * s, 3) for s in sigmas]} at layers {layers}",
              flush=True)
        assert all(s > 0 for s in sigmas), \
            f"sigma missing for {a.direction} in {a.probe_dir} (got {sigmas})"
    else:
        # non-steer arm: no residual-stream edit; the steering args are ignored
        dirs, sigmas, layers = None, None, []
        print(f"[cell] defense={a.defense}"
              + (f" kv_mask={a.kv_mask}" if kvm is not None else ""), flush=True)

    dec = {}
    if a.defense != "steer":
        pass                    # decode-time steering composes only on the steer arm
    elif a.decode_alpha > 0:
        if not a.decode_direction:
            raise SystemExit("--decode-alpha > 0 needs --decode-direction")
        dd, ds, _ = X.build_dirs(a.probe_dir, layers, a.decode_direction, model.device,
                                 match_sigma_to=a.decode_direction)
        dec = dict(decode_dirs=dd, decode_alpha=a.decode_alpha, decode_sigmas=ds)
        print(f"[decode] +{a.decode_direction}@{a.decode_alpha} (own sigmas "
              f"{[round(s, 3) for s in ds]})", flush=True)

    pairs = O.paired_samples(a.split, a.n)
    # the query grid: every (override, voice, action) framing; delegation NOT included
    # (delegated authority is out of scope per the 2026-08-29 owner ruling)
    acts_wanted = [x for x in a.actions.split(",") if x]
    assert all(x in O.ACTIONS for x in acts_wanted), f"unknown action in {acts_wanted}"
    queries = [(o_, v_, act) for o_ in O.OVERRIDE for v_ in O.VOICE
               for act in acts_wanted]
    # shard on a SEEDED PERMUTATION of the query index, not the raw index: with action
    # innermost, `i % 2` would alias the action factor with the GPU device, so a dead
    # shard would lose an entire action class (the override_slope SHARD_PERM rationale;
    # adversarial review 2026-08-31)
    perm = list(np.random.default_rng(0).permutation(len(queries)))
    mine = [queries[j] for i, j in enumerate(perm) if i % a.nshard == a.shard]
    print(f"[shard {a.shard}/{a.nshard}] {len(mine)} queries x {len(pairs)} samples "
          f"under {a.direction}@{a.alpha}", flush=True)
    X.judge = lambda samples, comps, votes=3, concurrency=16: ["CORRECT"] * len(samples)

    # does the generation prompt end inside <think>? (constant per model template; probed
    # once on a rendered sample, the same test the AgentDojo bridge uses per prompt)
    in_think = False
    if fmt == "chatml" and pairs:
        probe_txt, _ = X.prompt_and_span(
            tok, O.variant(pairs[0][0], pairs[0][1], "none", "none", "data", "tool"),
            poisoned=True)
        in_think = probe_txt.rstrip().endswith("<think>")
    print(f"[fmt] {fmt}, in_think={in_think}: `fired` scored on reasoning_free text "
          f"(raw kept as fired_raw)", flush=True)

    rows = []
    for (o_, v_, act) in mine:
        vs = [O.variant(orig, prm, "none", o_, v_, act) for orig, prm in pairs]
        if a.defense == "spotlight":
            # the prompt-level baseline edits the RENDER, not the sample content: system
            # suffix + << >> around the tool output (src/spans.py honours the flag). The
            # scoring fields (payload_clean, injection_text, target_*) are untouched.
            for s in vs:
                s["system"] = s["system"] + X.SPOTLIGHT_SUFFIX
                s["tool_output_wrap"] = "spotlight"
        if a.defense == "steer":
            arm_kw = dict(layers=layers, dirs=dirs, alpha=a.alpha,
                          direction=a.direction, scale="sigma", sigmas=sigmas, **dec)
        else:
            arm_kw = dict(layers=[], dirs=None, alpha=0.0, direction=a.defense,
                          kv_mask=kvm)
        # early_abort_trunc=0: the futility abort (src/arms.py, 2026-09-03) pads skipped
        # samples with "" completions, which this script would record as fired=false --
        # silent censoring that DEFLATES best-of-N ASR, the exact wrong direction for an
        # attack measurement (adversarial review, 2026-09-04). An attack query arm has no
        # truncation validity gate to fail, so the abort's premise does not apply here.
        r = X.run_arm(model, tok, vs, label=f"q:{o_}-{v_}-{act}", batch=a.batch,
                      max_new=a.max_new, early_abort_trunc=0, **arm_kw)
        for s, comp in zip(vs, r.completions):
            scored = X.reasoning_free(comp or "", fmt, in_think)
            row = {"sid": s["id"].split("::")[0], "override": o_, "voice": v_,
                   "action": act, "fired": bool(X.attack_influenced(s, scored)),
                   # exact-literal sibling (attack_fired: all target_args byte-exact) --
                   # obedience-pattern and exact-literal are BOTH reported, per the
                   # convention in src/scoring.attack_influenced's docstring
                   "fired_lit": bool(X.attack_fired(s, scored))}
            if fmt != "harmony":
                row["fired_raw"] = bool(X.attack_influenced(s, comp))
                row["truncated"] = "</think>" not in (comp or "") if in_think else None
            if a.save_completions:
                c, cap_ = comp or "", max(a.completion_cap, 200)
                if len(c) > cap_:
                    # keep the START (analysis-channel pivot) and the END (the tool
                    # call, which is what `fired` was scored on); mark the elision
                    h = cap_ // 2
                    c = c[:h] + f"\n...[{len(c) - 2 * h} chars elided]...\n" + c[-h:]
                row["completion"] = c
            rows.append(row)
        n_f = sum(x["fired"] for x in rows if x["override"] == o_ and x["voice"] == v_
                  and x["action"] == act)
        print(f"  [q:{o_}-{v_}-{act}] fired {n_f}/{len(pairs)}", flush=True)
        # per-query checkpoint: a shard is 5-6h of GPU time and save-at-end loses all of
        # it to a late crash (adversarial review 2026-08-31). Same truncation-safe write
        # as the final one; `partial` marks it.
        blob = json.dumps({"config": vars(a), "partial": True, "rows": rows}, indent=1)
        with open(a.out + ".tmp", "w") as f:
            f.write(blob)
        os.replace(a.out + ".tmp", a.out)

    blob = json.dumps({"config": vars(a), "rows": rows}, indent=1)
    with open(a.out + ".tmp", "w") as f:
        f.write(blob)
    os.replace(a.out + ".tmp", a.out)
    per_sid, per_sid_lit = {}, {}
    for x in rows:
        per_sid[x["sid"]] = per_sid.get(x["sid"], False) or x["fired"]
        per_sid_lit[x["sid"]] = per_sid_lit.get(x["sid"], False) or x.get("fired_lit", False)
    print(f"[shard {a.shard}] best-of-{len(mine)}-queries ASR (this shard alone): "
          f"pattern {sum(per_sid.values())}/{len(per_sid)}, "
          f"exact-literal {sum(per_sid_lit.values())}/{len(per_sid_lit)} | wrote {a.out}")


if __name__ == "__main__":
    main()
