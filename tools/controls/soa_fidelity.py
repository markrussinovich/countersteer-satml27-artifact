#!/usr/bin/env python
"""Benign composition-fidelity of the SoA baseline defenses, apples-to-apples with §24d.

WHY (owner directive 2026-09-04): FINDINGS §24d found the gpt-5.4 blinded pairwise judge
reads steering's benign cost on composition-heavy corpora far below the byte-exact bracket
(e.g. gpt-oss webpage-param CLEAN+steer judgeGoalOK 38.5 vs strict/lenient 75/75). That
finding cannot go in the paper without the SYMMETRIC measurement on the other defenses:
prompt-level defenses inject markup/instructions and filters delete content — both may
degrade composed summaries too, or may not. Either answer is publishable; the unmeasured
comparison is not.

WHAT THIS RUNS. For one source completions artifact (a §24d-scored steering cell), this
generates the defense-on-CLEAN arm (no injection anywhere) for each requested baseline
defense on the SAME samples, and writes a completions artifact whose `clean` list is the
SOURCE run's clean reference — so `judge_utility.py score` judges every defense arm
against the exact reference completions the steering cell was judged against, with the
same denominators and the strict/lenient bracket computed in the same pass.

Defense arms (all defense-on-clean; injection never present):
  spotlight    system suffix + << >> tool-output wrap (src/spans.py render, the same
               transform adaptive_framing_search.py's spotlight arm uses)
  sandwich     the user request re-asserted after the tool output (repeat_user_prompt)
  reminder     REMINDER_SENTENCE prepended to the tool output (AutoDojo, arXiv:2606.15057)
  cacheprune   KV mask over the payload span (arXiv:2504.21228; needs --kv-mask; the mask
               is model-specific — only pass it where one was fit)
  pi_protectai / pi_promptguard / pi_piguard
               the classifier filter applied to the CLEAN payload exactly as the
               multi-turn port applies it to tool messages (chunked scan, threshold 0.5,
               agentdojo_smoke.LocalPIDetector): a flagged payload is replaced by the
               upstream omission marker, then the UNDEFENDED model runs. On clean traffic
               every flag is a false positive — that deletion cost is the measurement.

Usage:
    XPIA_JUDGE... .venv/bin/python tools/controls/soa_fidelity.py \
        --source runs/gpt-oss-20b-userabl/results_confirm_..._completions.json \
        --model openai/gpt-oss-20b [--device cuda:0] \
        --defenses spotlight,sandwich,reminder,cacheprune,pi_protectai,pi_promptguard,pi_piguard \
        [--kv-mask runs/cacheprune_mask.json] --out runs/soa_fidelity/<cell>.json
Then:
    .venv/bin/python tools/controls/judge_utility.py score <out> [--votes 3]
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402
from score_table import load_corpus  # noqa: E402

X = E.X
ROOT = E.ROOT

OMIT = "<Data omitted because a prompt injection was detected>"
PI_ARMS = {"pi_protectai": "pi_detector",
           "pi_promptguard": "pi_detector_promptguard",
           "pi_piguard": "pi_detector_piguard"}
DEFAULT_DEFENSES = "spotlight,sandwich,reminder,cacheprune,pi_protectai,pi_promptguard,pi_piguard"


def mutate(samples, defense):
    """Defense-on-clean sample transform. Returns (mutated deep copy, engagement dict)."""
    S = copy.deepcopy(samples)
    eng = {}
    if defense == "spotlight":
        for s in S:
            s["system"] = s["system"] + X.SPOTLIGHT_SUFFIX
            s["tool_output_wrap"] = "spotlight"
    elif defense == "reminder":
        for s in S:
            s["tool_output_wrap"] = "reminder"
    elif defense == "sandwich":
        for s in S:
            s["repeat_user_prompt"] = True
    elif defense in PI_ARMS:
        # the filter preprocesses the payload the model would read (its rendered JSON,
        # exactly the text prompt_and_span puts in the tool message), then the UNDEFENDED
        # model runs; a flagged payload is replaced wholesale, mirroring the multi-turn
        # port's message-level transform
        import agentdojo_smoke as AS
        det = AS.LocalPIDetector(PI_ARMS[defense])
        flagged = []
        for s in S:
            text = json.dumps(s["payload_clean"], ensure_ascii=False)
            is_inj, score = det.detect(text)
            if is_inj:
                flagged.append(s["id"])
                s["payload_clean"] = OMIT
        eng = {"checked": det.n_checked, "flagged": det.n_flagged,
               "chunked": det.n_chunked, "flagged_ids": flagged}
        print(f"[fidelity] {defense}: flagged {det.n_flagged}/{det.n_checked} clean "
              f"payloads (false positives on benign traffic), {det.n_chunked} chunked")
    elif defense == "cacheprune":
        pass                       # applied via kv_mask at run time, not a render change
    else:
        raise SystemExit(f"unknown defense {defense!r}")
    return S, eng


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="the SCORED steering cell's completions artifact; its clean arm "
                         "becomes this artifact's judge reference")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--defenses", default=DEFAULT_DEFENSES)
    ap.add_argument("--kv-mask", dest="kv_mask", default=None,
                    help="CachePrune mask JSON (model-specific); the cacheprune arm is "
                         "skipped with a warning if unset")
    ap.add_argument("--batch", type=int, default=None,
                    help="default: the source run's batch")
    ap.add_argument("--n", type=int, default=0,
                    help="SMOKE: run only the first N samples (clean reference and "
                         "_meta.sample_ids are sliced consistently so judge_utility "
                         "still aligns). 0 = all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.load(open(a.source))
    meta = d["_meta"]
    S0 = load_corpus(meta)
    assert len(S0) == len(d["clean"]), (
        f"corpus/clean length mismatch {len(S0)} vs {len(d['clean'])} — the loader "
        f"aligns by _meta.sample_ids; a mismatch means the wrong corpus or split")
    if a.n:
        S0 = S0[:a.n]
        d["clean"] = d["clean"][:a.n]
        meta = {**meta, "sample_ids": [s["id"] for s in S0], "fidelity_smoke_n": a.n}
    batch = a.batch or meta.get("batch", 12)
    max_new = meta["max_new"]
    defenses = [x for x in a.defenses.split(",") if x]
    if "cacheprune" in defenses and not a.kv_mask:
        print("[fidelity] WARNING: no --kv-mask; skipping the cacheprune arm")
        defenses = [x for x in defenses if x != "cacheprune"]

    model, tok = X.load_model_and_tok(a.model, a.device)
    out = {"_meta": {**meta,
                     "fidelity_source": a.source,
                     "fidelity_model": a.model,
                     "fidelity_defenses": defenses,
                     "fidelity_engagement": {},
                     "fidelity_note": ("clean = the SOURCE run's reference completions; "
                                       "every CLEAN+<defense> arm generated by "
                                       "soa_fidelity.py on the same samples, no "
                                       "injection anywhere")},
           "clean": d["clean"]}

    for defense in defenses:
        Sm, eng = mutate(S0, defense)
        if eng:
            out["_meta"]["fidelity_engagement"][defense] = eng
        kw = {}
        if defense == "cacheprune":
            spec = X.load_kv_mask(a.kv_mask, model.config)
            kw["kv_mask"] = spec
        r = X.run_arm(model, tok, Sm, clean=True, batch=batch, max_new=max_new,
                      label=f"CLEAN+{defense}", direction=defense,
                      early_abort_trunc=0, **kw)
        out[f"CLEAN+{defense}"] = r.completions
        print(f"[fidelity] CLEAN+{defense}: {len(r.completions)} completions, "
              f"truncated {r.truncated:.3f}, no_call {r.no_call:.3f}", flush=True)
        # write-then-replace after every arm: an 8-arm run must not lose everything late
        blob = json.dumps(out, ensure_ascii=False)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out + ".tmp", "w") as f:
            f.write(blob)
        os.replace(a.out + ".tmp", a.out)

    json.load(open(a.out))   # an artifact is not written until it parses
    print(f"[fidelity] wrote {a.out} arms={[k for k in out if k != '_meta']}")


if __name__ == "__main__":
    main()
