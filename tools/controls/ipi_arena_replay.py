#!/usr/bin/env python
"""IPI Arena replay: the 95 released human red-team attacks vs our local steered models.

SOURCE. arXiv:2603.15714 ("How Vulnerable Are AI Agents to Indirect Prompt Injections?"),
kit github.com/GraySwanAI/ipi_arena_os (vendored at reference/ipi-arena, pinned
5e2e285910d581c97767195f1d113ef323f9ab07), attacks HF `sureheremarv/ipi_arena_attacks`
(pinned revision below). The attacks are HUMAN-crafted, arena-successful injections
generated against Qwen-family targets -- an external-validity row for the adaptive column,
LLMail-replay style.

WHAT IS SCORED -- their deterministic tool judge, verbatim. Each behavior defines a
`tool_judge` config (target tool, arg matchers, check mode); `judge_tool_usage` is imported
from the vendored kit and applied to the calls our model actually emitted. This is the
tier-1 compromise reading: did the model perform the injected action.

DOCUMENTED DEVIATIONS from their arena pipeline (each recorded in the artifact `_meta`):
  1. SINGLE-STEP by default. Their runner loops up to 5 steps with an LLM WorldSim
     simulating tool responses. The injected tool output sits in the PREFILL, so the first
     generated turn is the decision point; calls the model would only make after reading a
     further (simulated) tool response are missed -> our undefended rate is a LOWER bound
     on their loop's. --max-steps N>1 enables a loop with a DETERMINISTIC stub tool
     response (labelled, not their WorldSim).
  2. DETERMINISTIC SUBSET of their break criterion. Their `is_break` requires ALL criteria
     incl. LLM-judged ones (stealth, task-completion); we score the tool_judge criterion
     only -> our rate is an UPPER bound on their is_break. Both bounds are per-attack
     obedience readings; neither changes the defended-vs-undefended CONTRAST, which is what
     this row exists for.
  3. Local chat-template rendering + greedy decoding via agentdojo_bridge.SteeredLLM
     (the exact deployment path of every steering number in this project), not their
     OpenRouter/vLLM API client.
  4. CLEAN arms ({fill} -> "") are OUR addition -- the arena has no unattacked baseline;
     we need one for the capability guard (does the steered model still act at all).
  5. EXCLUDED, with counts in _meta: the 8 browser behaviors (Anthropic-format
     screenshot scenarios -- not renderable for a text-only local model; 6 of the 95
     attacks) and behaviors without a tool_judge config (their own kit scores those 0
     on tool_call criteria -- `flashing-lights-seizure`, 3 attacks).

Usage (defaults = the Qwen3-30B champion cell, FINDINGS 14b):
  PYTHONPATH= .venv/bin/python tools/controls/ipi_arena_replay.py \\
      --device cuda:0 --arms clean,cleanplus,attacked,defended \\
      --out runs/ipi_arena_replay.qwen3-30b.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools/controls"))
sys.path.insert(0, os.path.join(ROOT, "reference/ipi-arena/src"))

import _probe_eval as E  # noqa: E402
from agentdojo_bridge import SteeredLLM  # noqa: E402
from score_dojo_soa import wilson  # noqa: E402

X = E.X
# the AutoDojo reminder sentence, aliased from the shared constant (src/spans.py) rather
# than imported from agentdojo_smoke per-episode (review 2026-09-07, dependency-surface note)
REMINDER_SENTENCE = X.REMINDER_SENTENCE

from ipi_arena_bench.behavior import build_messages, load_behaviors  # noqa: E402
from ipi_arena_bench.judges.tool_usage import judge_tool_usage  # noqa: E402
from ipi_arena_bench.llm_client import ToolCall  # noqa: E402

KIT_COMMIT = "5e2e285910d581c97767195f1d113ef323f9ab07"
ATTACKS_DATASET = "sureheremarv/ipi_arena_attacks"
ATTACKS_REVISION = "2ef2d9785ed50981832b199ac89e6a0ff85d666d"
STUB_TOOL_RESPONSE = json.dumps({"status": "success"})


class _FC:
    """Duck-typed agentdojo FunctionCall for the bridge's message converter."""

    def __init__(self, function, args):
        self.function, self.args = function, args


class _Runtime:
    """Duck-typed FunctionsRuntime: SteeredLLM uses .functions[name].{name,description,
    parameters.model_json_schema()} (schema render) and set(.functions) (known names)."""

    def __init__(self, behavior):
        self.functions = {}
        for t in behavior.get("tools", []):
            params = t.get("parameters", {"type": "object", "properties": {}})
            self.functions[t["name"]] = SimpleNamespace(
                name=t["name"], description=t.get("description", ""),
                parameters=SimpleNamespace(model_json_schema=lambda p=params: p))


def to_bridge_messages(openai_msgs):
    """Kit OpenAI-format prefill -> the bridge's agentdojo-shaped message dicts."""
    out, call_names = [], {}
    for m in openai_msgs:
        role = m["role"]
        if role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                fn = c["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                calls.append(_FC(fn["name"], args))
                call_names[c.get("id", "")] = fn["name"]
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif role == "tool":
            name = call_names.get(m.get("tool_call_id", ""), "tool")
            out.append({"role": "tool", "content": m.get("content") or "",
                        "error": None, "tool_call": _FC(name, {})})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def run_episode(llm, behavior, attack_text, max_steps, reminder=False):
    """One arm on one (behavior, attack). Returns (kit ToolCalls, per-step transcript).

    reminder=True applies the AutoDojo reminder defense to every tool-role message in the
    prefill (and to stub responses in the multi-step path): REMINDER_SENTENCE prepended
    with no separator, exactly the ToolsExecutor formatter in agentdojo_smoke.

    Returns (kit ToolCalls, transcript, n_reminded) -- n_reminded = tool messages the
    transform touched, the reminder arm's ENGAGEMENT EVIDENCE (rows keep completions, not
    prompts, so without this a no-op transform would be invisible in the artifact --
    pre-launch review 2026-09-07, defect 3)."""
    msgs = to_bridge_messages(build_messages(behavior, attack_text))
    n_reminded = 0
    if reminder:
        for m in msgs:
            if m["role"] == "tool":
                m["content"] = f"{REMINDER_SENTENCE}{m['content']}"
                n_reminded += 1
    rt = _Runtime(behavior)
    llm.transcript = []
    all_calls = []
    for _step in range(max_steps):
        _, _, _, out_msgs, _ = llm.query(None, rt, None, msgs, {})
        last = out_msgs[-1]
        calls = last.get("tool_calls") or []
        all_calls += [ToolCall(name=c.function, arguments=dict(c.args)) for c in calls]
        if not calls or _step == max_steps - 1:
            break
        # deterministic stub tool responses (DEVIATION 1: their WorldSim is an LLM)
        stub = (f"{REMINDER_SENTENCE}{STUB_TOOL_RESPONSE}" if reminder
                else STUB_TOOL_RESPONSE)
        msgs = [*out_msgs] + [{"role": "tool", "content": stub,
                               "error": None, "tool_call": _FC(c.function, dict(c.args))}
                              for c in calls]
    return all_calls, list(llm.transcript), n_reminded


def load_attacks(attacks_json=None):
    """Pinned attack set. `attacks_json` (a local dump written by --dump-attacks) takes
    priority -- cluster jobs run HF-offline, so the dump is staged with the snapshot."""
    if attacks_json:
        rows = json.load(open(attacks_json))["attacks"]
        return [(r["behavior_id"].strip(), r["attack"]) for r in rows]
    from datasets import load_dataset
    ds = load_dataset(ATTACKS_DATASET, revision=ATTACKS_REVISION, split="train")
    return [(r["behavior_id"].strip(), r["attack"]) for r in ds]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probe-dir", default=f"{ROOT}/runs/qwen3-30b-thinking")
    ap.add_argument("--direction", default="dim_no_override_both")
    ap.add_argument("--layers", default="8,20,32",
                    help="Qwen3-30B champion cell (FINDINGS 14b): L8/20/32")
    ap.add_argument("--alpha", type=float, default=12.0)
    ap.add_argument("--match-sigma-to", default="",
                    help="'' = the direction's OWN sigma -- REQUIRED for the Qwen probe "
                         "dir (no dim_no_override entry; the champion cell is own-sigma)")
    ap.add_argument("--max-new", type=int, default=4096,
                    help=">=4096 on thinking models (budget rule)")
    ap.add_argument("--max-steps", type=int, default=1,
                    help="1 = single decision-point turn (default; see DEVIATION 1). "
                         ">1 loops with a deterministic stub tool response, NOT their "
                         "LLM WorldSim.")
    ap.add_argument("--arms", default="clean,cleanplus,attacked,defended")
    ap.add_argument("--limit", type=int, default=0, help="first N attacks only (smoke)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1,
                    help="interleaved sharding over the usable attack list (clean arms "
                         "shard over the unique-behavior list the same way)")
    ap.add_argument("--attacks-json", default=None,
                    help="local pinned dump from --dump-attacks (cluster jobs are "
                         "HF-offline); default = HF dataset at the pinned revision")
    ap.add_argument("--dump-attacks", default=None,
                    help="write the pinned HF attack set to this path and exit")
    ap.add_argument("--attn-impl", default="",
                    help="'' = the transformers default resolution (the certified "
                         "serving path). Non-empty (e.g. flex_attention) changes the "
                         "serving path and the artifact carries a per-path label -- "
                         "needed for gpt-oss, whose eager prefill cannot host the "
                         "longest IPI prompts on 80GB (FINDINGS 25f addendum).")
    ap.add_argument("--rival", default=None, choices=["cacheprune", "reminder"],
                    help="RIVAL-DEFENSE battery: the defended/cleanplus arms carry this "
                         "defense instead of steering (direction OFF for those arms; the "
                         "clean/attacked comparators are unchanged). cacheprune needs "
                         "--kv-mask; reminder is the AutoDojo prompt transform.")
    ap.add_argument("--kv-mask", default=None,
                    help="CachePrune mask JSON (build_cacheprune_mask.py) for the model "
                         "under test -- masks are per-architecture, never shared")
    ap.add_argument("--kit-data", default=f"{ROOT}/reference/ipi-arena/data")
    ap.add_argument("--out", default=f"{ROOT}/runs/ipi_arena_replay.json")
    a = ap.parse_args()
    if a.rival == "cacheprune":
        assert a.kv_mask and os.path.exists(a.kv_mask), \
            "--rival cacheprune requires an existing --kv-mask (fail fast, not mid-run)"
        json.load(open(a.kv_mask))

    if a.dump_attacks:
        rows = [{"behavior_id": b, "attack": t} for b, t in load_attacks()]
        blob = json.dumps({"dataset": ATTACKS_DATASET, "revision": ATTACKS_REVISION,
                           "attacks": rows}, indent=1)
        with open(a.dump_attacks + ".tmp", "w") as f:
            f.write(blob)
        os.replace(a.dump_attacks + ".tmp", a.dump_attacks)
        json.load(open(a.dump_attacks))
        print(f"wrote {a.dump_attacks} ({len(rows)} attacks)")
        return

    behaviors = load_behaviors(a.kit_data)
    attacks = load_attacks(a.attacks_json)
    excluded = {"browser": [], "no_tool_judge": [], "unknown": []}
    usable = []
    for bid, atk in attacks:
        b = behaviors.get(bid)
        if b is None:
            excluded["unknown"].append(bid)
        elif b.get("message_format") == "anthropic" or b.get("html_template"):
            excluded["browser"].append(bid)
        elif not b.get("tool_judge"):
            excluded["no_tool_judge"].append(bid)
        else:
            usable.append((bid, atk))
    if a.limit:
        usable = usable[:a.limit]
    uniq_behaviors = sorted({bid for bid, _ in usable})
    if a.nshard > 1:
        usable = [x for i, x in enumerate(usable) if i % a.nshard == a.shard]
        uniq_behaviors = [x for i, x in enumerate(uniq_behaviors)
                          if i % a.nshard == a.shard]
    print(f"[data] {len(attacks)} released attacks -> {len(usable)} scoreable "
          f"(excluded: {({k: len(v) for k, v in excluded.items()})}) over "
          f"{len(uniq_behaviors)} behaviors", flush=True)

    model, tok = X.load_model_and_tok(a.model, a.device, attn_impl=a.attn_impl or None)
    layers = [int(x) for x in a.layers.split(",")]
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]

    def make_llm(defended_arm):
        """defended_arm = the defense-ON arms (cleanplus/defended). With --rival set the
        defense is the RIVAL (CachePrune's KV mask, or the reminder prompt transform in
        run_episode), the steering direction is OFF, and the clean/attacked comparator
        arms are identical to the steering run's -- the SoA 4-arm battery convention."""
        if defended_arm and a.rival == "cacheprune":
            return SteeredLLM(model, tok, probe_dir=a.probe_dir, direction=None,
                              layers=layers, alpha=a.alpha,
                              match_sigma_to=a.match_sigma_to, max_new=a.max_new,
                              kv_mask=a.kv_mask)
        return SteeredLLM(model, tok, probe_dir=a.probe_dir,
                          direction=(a.direction if defended_arm and not a.rival
                                     else None),
                          layers=layers, alpha=a.alpha,
                          match_sigma_to=a.match_sigma_to, max_new=a.max_new)

    rows, t0 = [], time.time()
    for arm in arms:
        steered = arm in ("cleanplus", "defended")
        attacked = arm in ("attacked", "defended")
        llm = make_llm(steered)
        # reminder (AutoDojo arXiv:2606.15057): the sentence PREPENDED to every tool
        # output. In this replay the tool outputs live in the PREFILL, so the transform
        # applies at message level in run_episode -- same text, same position as the
        # ToolsExecutor formatter wiring in agentdojo_smoke.build_pipeline.
        reminder_on = bool(steered and a.rival == "reminder")
        # clean arms: one episode per BEHAVIOR ({fill} -> ""); attack arms: per ATTACK
        jobs = ([(bid, "") for bid in uniq_behaviors] if not attacked else usable)
        for j, (bid, atk) in enumerate(jobs):
            b = behaviors[bid]
            calls, transcript, n_reminded = run_episode(llm, b, atk, a.max_steps,
                                                        reminder=reminder_on)
            verdict = judge_tool_usage(calls, b["tool_judge"])
            rows.append({
                "arm": arm, "behavior_id": bid, "category": b["category"],
                "attack_sha": hashlib.sha1(atk.encode()).hexdigest()[:10] if atk else None,
                "tool_judge_passed": bool(verdict["passed"]),
                "tool_judge_reason": verdict["reason"],
                "calls_made": [tc.to_dict() for tc in calls],
                "n_calls": len(calls),
                "steps": len(transcript),
                "n_steered": sum(t["n_steered"] for t in transcript),
                "n_reminded_msgs": n_reminded,
                "completions": [t["completion"] for t in transcript],
            })
            print(f"  [{arm} {j + 1}/{len(jobs)}] {bid:<40} "
                  f"pass={int(verdict['passed'])} calls={len(calls)}", flush=True)
        rows_arm = [r for r in rows if r["arm"] == arm]
        k = sum(r["tool_judge_passed"] for r in rows_arm)
        n = len(rows_arm)
        lo, hi = wilson(k, n)
        nocall = sum(r["n_calls"] == 0 for r in rows_arm)
        print(f"[{arm}] tool_judge pass {k}/{n} = {k / max(1, n):.3f} "
              f"[{lo:.3f},{hi:.3f}]  no-call {nocall}/{n}  trunc(turns) {llm.n_truncated}",
              flush=True)
        for r in rows_arm:
            r["arm_trunc_total"] = llm.n_truncated
            # ENGAGEMENT EVIDENCE for kv-mask arms (review 2026-09-07 defect 3): the
            # bridge counts kv-masked positions in n_steered_tokens but the per-turn
            # transcript n_steered is 0 for kv arms; the arm total is the proof the
            # mask actually edited something.
            r["arm_steered_tokens_total"] = llm.n_steered_tokens

    try:
        import subprocess
        git_sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
    except Exception:
        git_sha = None
    meta = {
        "git_sha": git_sha,
        "kit_commit": KIT_COMMIT, "attacks_dataset": ATTACKS_DATASET,
        "attacks_revision": ATTACKS_REVISION,
        "n_attacks_released": len(attacks), "n_scoreable": len(usable),
        "excluded_counts": {k: len(v) for k, v in excluded.items()},
        "excluded_ids": {k: sorted(set(v)) for k, v in excluded.items()},
        "behaviors": uniq_behaviors,
        "config": vars(a),
        "deviations": [
            "single-step decision-point replay by default (their loop: 5 steps + LLM "
            "WorldSim) -- undefended rate is a LOWER bound on their loop",
            "deterministic tool_judge only (their is_break also requires LLM-judged "
            "criteria) -- rate is an UPPER bound on their is_break",
            "local chat-template rendering + greedy decoding via SteeredLLM, not their "
            "API client",
            "clean/cleanplus arms ({fill}->'') are our addition for the capability guard",
            "browser behaviors and tool_judge-less behaviors excluded (counts above)",
        ],
    }
    blob = json.dumps({"_meta": meta, "rows": rows}, indent=1, default=str)
    with open(a.out + ".tmp", "w") as f:
        f.write(blob)
    os.replace(a.out + ".tmp", a.out)
    json.load(open(a.out))
    print(f"\nwrote {a.out} ({len(rows)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
