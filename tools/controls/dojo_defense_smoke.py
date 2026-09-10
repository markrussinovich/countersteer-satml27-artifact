#!/usr/bin/env python
"""Engagement smoke for the --dojo-defense baselines: prove each defense element ENGAGES
in the rendered prompt, on any supported model (harmony gpt-oss / ChatML Qwen3).

One attacked cell per defense, rendered-prompt evidence per defense:
  spotlighting_with_delimiting  system suffix present; every recorded tool span is
                                << >>-wrapped; the injection text sits INSIDE the wrapping
  repeat_user_prompt            the user prompt appears >=2x in the last rendered prompt
  tool_filter                   kept_log shows the toolset shrank; the last prompt names
                                only the kept tools
Plus the arm-equality check: the attack's injection dict is byte-identical between the raw
undefended pipeline and every defense pipeline.

Usage:
    python tools/controls/dojo_defense_smoke.py [--model openai/gpt-oss-20b]
        [--cell suite:user_task:injection_task:attack]
"""
import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import _probe_eval as E  # noqa: E402
import agentdojo_smoke as S  # noqa: E402

X = E.X

from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.benchmark import run_task_with_injection_tasks  # noqa: E402
from agentdojo.logging import OutputLogger  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probe-dir", default=f"{E.ROOT}/runs/gpt-oss-20b-userabl")
    ap.add_argument("--layers", default="12,16,20")
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--match-sigma-to", default="dim_no_override")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--cell", default="banking:user_task_14:injection_task_4:important_instructions",
                    help="suite:user_task:injection_task:attack -- pick one the model is "
                         "COMPROMISED on undefended, so engagement is informative")
    a = ap.parse_args()
    a.dojo_defense = None
    a.direction = None

    su, ut_id, it_id, attack = a.cell.split(":")
    model, tok = X.load_model_and_tok(a.model, a.device)
    suite = get_suite(S.VERSION, su)
    ut = suite.get_user_task_by_id(ut_id)
    it = suite.get_injection_task_by_id(it_id)

    raw_pipe, _ = S.build_pipeline(model, tok, a, None)
    raw_atk = load_attack(attack, suite, raw_pipe)
    ref_inj = raw_atk.attack(ut, it)
    print(f"[ref] cell={a.cell} injection keys={list(ref_inj)} "
          f"n_suite_tools={len(suite.tools)}", flush=True)
    inj_text = next(iter(ref_inj.values()))

    ok_all = True
    for d in S.DOJO_DEFENSES:
        a.dojo_defense = d
        pipe, llm = S.build_pipeline(model, tok, a, "DEFENSE_ON")
        assert llm.direction is None, "dojo baseline must not steer"
        atk = load_attack(attack, suite, pipe)
        same = atk.attack(ut, it) == ref_inj
        print(f"\n=== {d} === injection identical to raw arm: {same}", flush=True)
        assert same, f"{d}: injection rendering differs from the raw arm"
        llm.transcript = []
        with OutputLogger(None):
            u_, s_ = run_task_with_injection_tasks(
                suite, pipe, ut, atk, None, True,
                injection_tasks=[it_id], benchmark_version=S.VERSION)
        u = sum(u_.values()) / max(1, len(u_))
        sec = sum(s_.values()) / max(1, len(s_))
        last = llm.transcript[-1]["prompt"]
        print(f"utility={u:.2f} security={sec:.2f} llm_calls={llm.n_calls} "
              f"steered_tokens={llm.n_steered_tokens} turns={len(llm.transcript)}",
              flush=True)

        if d == "spotlighting_with_delimiting":
            ok_sys = S.SPOTLIGHT_SUFFIX.strip() in last
            spans = llm.transcript[-1]["spans"]
            bodies = [last[lo:hi] for lo, hi in spans]
            wrapped_ok = bool(bodies) and all(
                b.startswith("<<") and b.endswith(">>") for b in bodies)
            # harmony JSON-encodes tool content (newlines render as literal \n), ChatML
            # renders it raw -- match on the injection's longest single LINE, unescaped
            frag = max(inj_text.splitlines(), key=len).strip()[:50]
            inj_inside = any(frag in b.replace("\\'", "'").replace('\\"', '"')
                             for b in bodies)
            hit = next((b for b in bodies if frag in b.replace("\\'", "'")
                        .replace('\\"', '"')), bodies[0] if bodies else "")
            print(f"  system suffix present: {ok_sys}; tool bodies << >>-wrapped: "
                  f"{wrapped_ok} ({len(bodies)} bodies); injection inside: {inj_inside}")
            print(f"  DELIMITED TOOL BODY: {hit[:120]!r} ... {hit[-80:]!r}")
            ok_all &= ok_sys and wrapped_ok and inj_inside
        elif d == "repeat_user_prompt":
            n = last.count(ut.PROMPT[:80])
            print(f"  user prompt occurrences in last rendered prompt: {n} (need >=2)")
            print(f"  PROMPT TAIL: ...{last[-350:]!r}")
            ok_all &= n >= 2
        elif d == "tool_filter":
            tf = pipe.tool_filter
            names = [getattr(t, "__name__", getattr(t, "name", str(t)))
                     for t in suite.tools]
            present = [n for n in names if n in last]
            print(f"  kept_log={tf.kept_log} fallbacks={tf.n_fallback}")
            print(f"  tools named in LAST prompt: {len(present)}/{len(names)}: {present}")
            print(f"  FILTER REPLY tail: {llm.transcript[0]['completion'][-350:]!r}")
            ok_all &= bool(tf.kept_log) and tf.kept_log[0][1] < tf.kept_log[0][0]

    if not ok_all:
        raise SystemExit("\nSMOKE FAILED: at least one engagement check did not hold")
    print("\nSMOKE OK: all three defenses engage, injections identical across arms")


if __name__ == "__main__":
    main()
