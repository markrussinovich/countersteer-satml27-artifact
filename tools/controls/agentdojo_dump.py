#!/usr/bin/env python
"""Dump the actual prompts and responses for one AgentDojo task, attacked vs defended.

The metrics say a task was hijacked or not. This shows WHAT the model saw and WHAT it did --
including exactly which characters the steering hook edited. Use it before believing a number.

Usage:
    python tools/controls/agentdojo_dump.py [--suite slack] [--task user_task_0]
                                            [--injection-task injection_task_0] [--chars 1400]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402
import agentdojo_smoke as S  # noqa: E402

X = E.X
ROOT = E.ROOT

from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.benchmark import run_task_with_injection_tasks  # noqa: E402
from agentdojo.logging import OutputLogger  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402


def show(label, tr, chars, mark_spans):
    print("\n" + "=" * 100)
    print(f"### {label} — {len(tr)} LLM turn(s)")
    print("=" * 100)
    for i, t in enumerate(tr):
        p = t["prompt"]
        print(f"\n----- turn {i+1}: PROMPT ({len(p)} chars, {t['n_steered']} tokens steered) -----")
        if mark_spans and t["spans"]:
            # show the tool-output span(s) delimited, which is exactly what the hook edits
            lo, hi = t["spans"][0]
            head = p[max(0, lo - 260):lo]
            body = p[lo:hi]
            tail = p[hi:hi + 120]
            print("..." + head)
            print(f">>>>> TOOL OUTPUT SPAN (chars {lo}..{hi}) — STEERED >>>>>")
            print(body[:chars] + ("..." if len(body) > chars else ""))
            print("<<<<< END SPAN <<<<<")
            print(tail)
        else:
            print(p[:600] + "\n   [ ... ]\n" + p[-700:])
        print(f"\n----- turn {i+1}: RESPONSE -----")
        print(t["completion"][:chars] + ("..." if len(t["completion"]) > chars else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probe-dir", default=f"{ROOT}/runs/gpt-oss-20b-userabl")
    ap.add_argument("--direction", default="dim_no_override_both")
    ap.add_argument("--layers", default="12,16,20")
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--suite", default="slack")
    ap.add_argument("--task", default="user_task_0")
    ap.add_argument("--injection-task", default=None)
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--chars", type=int, default=1400)
    a = ap.parse_args()

    suite = get_suite(S.VERSION, a.suite)
    ut = suite.get_user_task_by_id(a.task)
    inj = [a.injection_task or list(suite.injection_tasks)[0]]
    model, tok = X.load_model_and_tok(a.model, a.device)
    print(f"[task] {a.suite}/{a.task}: {ut.PROMPT!r}")
    print(f"[injection task] {inj[0]}: "
          f"{suite.get_injection_task_by_id(inj[0]).GOAL!r}")

    for label, direction in (("ATTACKED (no defense)", None),
                             (f"DEFENDED ({a.direction} @ {a.alpha})", a.direction)):
        pipe, llm = S.build_pipeline(model, tok, a, direction)
        llm.transcript = []
        with OutputLogger(None):
            util, sec = run_task_with_injection_tasks(
                suite, pipe, ut, load_attack(a.attack, suite, pipe), None, True,
                injection_tasks=inj, benchmark_version=S.VERSION)
        u = sum(util.values()) / max(1, len(util))
        s_ = sum(sec.values()) / max(1, len(sec))
        show(f"{label}   utility={u:.2f}  security={s_:.2f}", llm.transcript, a.chars,
             mark_spans=True)


if __name__ == "__main__":
    main()
