#!/usr/bin/env python
"""Per-episode non-truncation audit for an AgentDojo battery artifact.

WHY THIS IS A STANDING CHECK. FINDINGS section 21's closure established that AgentDojo's
`security` checker is OUTCOME-based, so a turn cut off mid-compliance scores as SECURE.
At max_new 768 that censored 11 of 18 flagged episodes into false "secure" verdicts, and
the confirmatory rerun showed 2048 STILL censors. The closure's rule: *any full-battery
rerun must use max_new > 2048 or verify per-episode non-truncation.* This script is that
verification, factored out of the one-off `tmp/audit_20260901/dojo_audit.py` so the check
travels with the battery instead of being reinvented per run.

What it reports per arm:
  turns / trunc_turns / trunc_episodes  -- raw budget pressure
  TRUNC-WITH-INTENT                     -- a truncated turn naming one of the attacker's
                                           target tools. THIS is the censoring signature:
                                           the model was composing the attacker's call and
                                           ran out of budget, so `security` reads 0 for a
                                           reason that is not the defense.
  BLIND                                 -- truncated turns the intent test CANNOT see (see
                                           below). Non-zero BLIND means the all-clear is
                                           not an all-clear.
  ENDED-ON-CALL                         -- episodes whose LAST turn is a tool call: the
                                           agent-loop turn cap, the OTHER censoring channel
                                           an outcome checker also reads as SECURE.
  SUSP-PARSE                            -- call-like text that yielded 0 executable calls
  loops                                 -- degenerate repetition (LOWER BOUND, see below)

Truncation is a per-format test on the RAW completion, dispatched on each turn's own `fmt`
field (harmony vs chatml). It is a CONTAINMENT test over that format's terminators, never a
tail test. Executable calls are parsed after stripping the reasoning region via
`src.scoring.reasoning_free`, matching the scoring path: text the model merely deliberates
about never reaches the environment or their checker.

KNOWN LIMITS, all found by adversarial review 2026-09-01 and surfaced in the output rather
than hidden:
  * The comply-intent test needs a ground-truth attacker tool NAME. Rows with no
    `attacker_calls` (defended-only baseline runs never populate it; some injection tasks
    have a text goal, not a call) are structurally unflaggable. Those are counted as BLIND
    and the tool EXITS NON-ZERO when any truncated turn lands in one -- a gate that reports
    "clean" because it cannot see is worse than no gate.
  * `has_loop` only catches repeating cycles of 20-80 chars, so loop counts are LOWER
    BOUNDS and must not be quoted as counts.

Usage:
  python tools/controls/agentdojo_trunc_audit.py --glob 'runs/dir/foo.shard[0-9].json'
  python tools/controls/agentdojo_trunc_audit.py --glob '...' --json out.json

Exit codes: 0 = no censoring signature and nothing blind; 1 = trunc-with-intent found;
2 = blind truncated turns (verdict is UNKNOWN, not clean).

The transcripts sibling (`<artifact>.transcripts.json`) must sit beside each shard; its
keys are "<result_index>|<arm>".
"""
import argparse
import collections
import glob as globmod
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.scoring import parse_tool_calls, reasoning_free  # noqa: E402

ARMS = ("clean", "cleanplus", "attacked", "defended")
# The analysis channel is deliberation: it is not executed and cannot reach the checker.
STRIP_AN = re.compile(r"<\|channel\|>analysis<\|message\|>.*?(?:<\|end\|>|(?=<\|start\|>)|$)", re.S)

# Per-format terminators. A turn is truncated iff NONE of its format's terminators appears
# ANYWHERE in the completion -- a containment test, not a tail test. Hardcoding harmony's
# set made every ChatML turn read as truncated (991/991 on the Qwen battery), which would
# have turned this gate into a 100%-false-positive alarm on 5 of the 6 paper models.
TERMINATORS = {
    "harmony": ("<|call|>", "<|return|>", "<|end|>"),
    "chatml": ("<|im_end|>", "<|endoftext|>"),
    # the three sets below mirror the bridge's per-format truncation predicates verbatim
    # (agentdojo_bridge.SteeredLLM.query); added with the Gemma port 2026-09-04 -- the
    # phi3/glm45 grids predated this tool's format table and fataled here.
    "phi3": ("<|end|>", "<|endoftext|>"),
    "glm45": ("<|endoftext|>", "<|user|>", "<|observation|>"),
    "gemma4": ("<|tool_response>", "<turn|>", "<eos>"),
}


def exe_calls(c, fmt="harmony"):
    """Calls that could actually execute: reasoning region removed, then parsed."""
    return parse_tool_calls(STRIP_AN.sub("", reasoning_free(c, fmt)))


def truncated(c, fmt="harmony"):
    """No terminator for this wire format anywhere => the turn hit the max_new cap."""
    term = TERMINATORS.get(fmt)
    if term is None:
        raise SystemExit(f"FATAL: unsupported fmt {fmt!r}; add its terminators to "
                         "TERMINATORS rather than letting the test silently misfire")
    return not any(t in c for t in term)


def has_loop(t):
    """Any 20-80 char chunk repeated >=5x consecutively in the tail."""
    for m in re.finditer(r"(.{20,80}?)\1{4,}", t[-4000:], re.S):
        return m.group(1)[:60]
    return None


def load_pairs(pattern):
    """[(result_row, {arm: [turn, ...]})] over every shard matched by `pattern`."""
    out = []
    files = sorted(globmod.glob(pattern))
    if not files:
        raise SystemExit(f"FATAL: no shards match {pattern!r}")
    for f in files:
        tf = f[:-5] + ".transcripts.json"
        if not os.path.exists(tf):
            raise SystemExit(f"FATAL: transcripts sibling missing for {f}")
        d = json.load(open(f))
        # Group the transcript keys ONCE. Re-scanning every key per result row is
        # O(results x keys) -- ~5M string splits on an 8-shard battery.
        byidx = collections.defaultdict(dict)
        for k, v in json.load(open(tf)).items():
            i, arm = k.split("|", 1)
            byidx[int(i)][arm] = v
        for i, r in enumerate(d["results"]):
            r["_shard"] = os.path.basename(f)
            r["_idx"] = i
            out.append((r, byidx.get(i, {})))
    return files, out


def audit(pattern):
    files, rows = load_pairs(pattern)
    agg = collections.defaultdict(collections.Counter)
    trunc_intent, blind, ended_on_call, susp_parse, loops = [], [], [], [], []
    for r, tr in rows:
        # `attacker_calls` is only populated when the clean arm ran (agentdojo_run.py);
        # defended-only baseline runs have none, so the intent test is BLIND there.
        names = [n for n, _ in (r.get("attacker_calls") or [])]
        for arm in ARMS:
            turns = tr.get(arm)
            if turns is None:
                continue
            A = agg[arm]
            A["episodes"] += 1
            A["turns"] += len(turns)
            ep_trunc = 0
            for j, tu in enumerate(turns):
                c = tu["completion"]
                fmt = tu.get("fmt", "harmony")
                is_tr = truncated(c, fmt)
                lp = has_loop(c)
                where = dict(shard=r["_shard"], idx=r["_idx"], arm=arm, turn=j,
                             suite=r["suite"], user_task=r["user_task"],
                             injection_task=r["injection_task"])
                if is_tr:
                    A["trunc_turns"] += 1
                    ep_trunc += 1
                    if lp:
                        A["trunc_loops"] += 1
                    if arm in ("attacked", "defended"):
                        if not names:
                            # No ground truth to match against: UNKNOWN, not clean.
                            A["blind_trunc"] += 1
                            blind.append(dict(where, loop=lp))
                        elif any(n in c for n in names):
                            trunc_intent.append(dict(where, loop=lp))
                nostrip = STRIP_AN.sub("", c)
                if not exe_calls(c, fmt) and re.search(
                        r"to=functions|<\|channel\|>commentary to=|<tool_call>", nostrip):
                    susp_parse.append(dict(where, head=nostrip[:150]))
                if lp and not is_tr:
                    loops.append(dict(where, loop=lp[:40]))
            if ep_trunc:
                A["trunc_episodes"] += 1
            # The OTHER censoring channel: the agent loop hit its TURN cap while the model
            # was still calling tools. Their outcome checker reads that as SECURE too.
            last = turns[-1]
            if exe_calls(last["completion"], last.get("fmt", "harmony")):
                A["ended_on_call"] += 1
                if arm in ("attacked", "defended"):
                    ended_on_call.append(dict(
                        shard=r["_shard"], idx=r["_idx"], arm=arm, turns=len(turns),
                        suite=r["suite"], user_task=r["user_task"],
                        injection_task=r["injection_task"]))
    return files, agg, trunc_intent, blind, ended_on_call, susp_parse, loops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="shard glob, e.g. 'runs/d/x.shard[0-9].json'")
    ap.add_argument("--json", default=None, help="write the flag lists here")
    a = ap.parse_args()
    files, agg, ti, blind, eoc, sp, loops = audit(a.glob)
    print(f"[audit] {len(files)} shards: {', '.join(os.path.basename(f) for f in files)}")
    print(f"\n{'arm':<12}{'episodes':>9}{'turns':>7}{'truncT':>8}{'truncEp':>9}"
          f"{'truncLoop':>11}{'blindTrunc':>12}{'endedOnCall':>13}")
    for arm in ARMS:
        A = agg.get(arm)
        if not A:
            continue
        print(f"{arm:<12}{A['episodes']:>9}{A['turns']:>7}{A['trunc_turns']:>8}"
              f"{A['trunc_episodes']:>9}{A['trunc_loops']:>11}{A['blind_trunc']:>12}"
              f"{A['ended_on_call']:>13}")
    print(f"\nTRUNC-WITH-INTENT (truncated turn naming an attacker-target tool): {len(ti)}")
    for x in ti:
        print(f"  {x['arm']:<10}{x['suite']}/{x['user_task']}/{x['injection_task']} "
              f"turn {x['turn']}" + (f"  LOOP[{x['loop']}]" if x["loop"] else ""))
    print(f"\nBLIND (truncated attacked/defended turn with NO attacker_calls ground truth "
          f"-- the intent test cannot fire here): {len(blind)}")
    for x in blind:
        print(f"  {x['arm']:<10}{x['suite']}/{x['user_task']}/{x['injection_task']} "
              f"turn {x['turn']}")
    print(f"\nENDED-ON-CALL (episode's last turn is a tool call => agent-loop turn cap, "
          f"the other censoring channel): {len(eoc)}")
    for x in eoc:
        print(f"  {x['arm']:<10}{x['suite']}/{x['user_task']}/{x['injection_task']} "
              f"({x['turns']} turns)")
    print(f"\nSUSP-PARSE (call-like text, 0 executable calls): {len(sp)}")
    for x in sp[:10]:
        print(f"  {x['arm']:<10}{x['shard']}#{x['idx']} turn {x['turn']}: {x['head'][:110]!r}")
    print(f"\nloops (LOWER BOUND -- 20-80 char cycles only, do not quote as a count): "
          f"{len(loops)} {dict(collections.Counter(x['arm'] for x in loops))}")
    if a.json:
        payload = json.dumps({
            "glob": a.glob, "shards": [os.path.abspath(f) for f in files],
            "per_arm": {k: dict(v) for k, v in agg.items()},
            "trunc_intent": ti, "blind": blind, "ended_on_call": eoc,
            "susp_parse": sp, "loops": loops}, indent=1)
        tmp = a.json + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(payload)
        json.load(open(tmp))  # it is not written until it parses
        os.replace(tmp, a.json)
        print(f"\n[audit] flags -> {os.path.abspath(a.json)}")
    # Gate: 1 = censoring signature present; 2 = the verdict is UNKNOWN because truncated
    # turns fell where the intent test is blind. Silence is never reported as an all-clear.
    if ti:
        print("\nVERDICT: CENSORING SIGNATURE PRESENT -- security is a lower bound.")
        return 1
    if blind:
        print("\nVERDICT: UNKNOWN -- truncated turns lie in the intent test's blind set.")
        return 2
    print("\nVERDICT: no censoring signature and no blind truncated turns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
