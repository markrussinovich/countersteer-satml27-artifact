#!/usr/bin/env python
"""Run the cell list from agentdojo_pairs.py: attacked + defended, both judges, sharded.

One cell = (suite, user_task, injection_task, attack). Each cell runs THREE arms -- clean,
attacked, defended -- because our own tier-2 metric needs the clean run's tool calls as its
reference and AgentDojo's utility needs nothing else to compare against.

SCORED TWICE, deliberately:
  AgentDojo   utility  (their per-task checker)   security (their per-injection checker)
  ours        goal     contam                     struct_exact vs the clean arm
Their `security` and our `goal` agreed 10/10 on the first smoke; keeping both is how a
disagreement gets caught rather than assumed away.

THE UNICODE CAVEAT IS MEASURED, NOT PATCHED. Their utility checkers use literal substring
matching, and the defended arm has already failed a task purely because the model emitted
U+2011 NON-BREAKING HYPHEN instead of U+002D. We report their verdict as primary and
additionally record whether the answer would pass under dash normalisation. Their checker is
never modified.

Usage:
    python tools/controls/agentdojo_run.py --cells runs/agentdojo_cells.json \\
        --shard 0 --nshard 4 --out runs/agentdojo_run.shard0.json
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_eval as E  # noqa: E402
import agentdojo_smoke as S  # noqa: E402
from agentdojo_bridge import SCHEDULES, SPAN_SELECTS, reasoning_free  # noqa: E402

X = E.X
ROOT = E.ROOT

from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.benchmark import (run_task_with_injection_tasks,  # noqa: E402
                                 run_task_without_injection_tasks)
from agentdojo.logging import OutputLogger  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

# `dash_artifact` REMOVED 2026-08-05. It claimed to identify utility failures caused by the
# model emitting U+2011 instead of U+002D. Adversarial review found four independent defects,
# each sufficient: it tested the FULL concatenated transcript (chain-of-thought included) where
# AgentDojo checks only the last assistant message; it ignored the `pre_environment ==
# post_environment` clause that 22 of 34 eligible tasks also require; its `len(t) > 6` filter
# discarded the answer and kept the boilerplate (`banking/user_task_7` reduced to `['friend.']`,
# giving a "rescue" on a fully hijacked run); and 19 of 34 eligible tasks contain no hyphen at
# all, so it could return True where dash normalisation is logically incapable of mattering.
#
# The right way to ask the question is `checker(raw) == False and checker(normalised) == True`
# against AgentDojo's own checker. That needs the pre/post environments, which we do not retain.
# Since full transcripts are now persisted, this is answerable OFFLINE and must not be
# guessed at inline.

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probe-dir", default=f"{ROOT}/runs/gpt-oss-20b-userabl")
    ap.add_argument("--direction", default="dim_no_override_both")
    ap.add_argument("--layers", default="12,16,20")
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--match-sigma-to", default="dim_no_override",
                    help="whose stored sigma sets the step (step = alpha*sigma). The default "
                         "is the locked-in gpt-oss AgentDojo setting. Pass '' to use the "
                         "steered direction's OWN sigma -- required reading for any other "
                         "probe dir: the Qwen3 pickles have no `dim_no_override` entry, and "
                         "build_dirs silently falls back to the direction's own sigma, so "
                         "say it explicitly. The bridge prints the resolved sigmas either way.")
    ap.add_argument("--kv-mask", dest="kv_mask", default=None,
                    help="CachePrune baseline (arXiv:2504.21228): path to the mask "
                         "JSON from tools/controls/build_cacheprune_mask.py. The "
                         "defended/cleanplus arms mask K/V cache coordinates over "
                         "the tool spans instead of steering; --direction is "
                         "ignored for those arms. Mutually exclusive with "
                         "--dojo-defense.")
    ap.add_argument("--agri-probe", dest="agri_probe", default=None,
                    help="AGRI baseline (arXiv:2608.02657): path to the probe spec JSON "
                         "from tools/controls/build_agri_probe.py. The defended/cleanplus "
                         "arms run the probe-gated anti-injection reasoning prefill "
                         "around the UNSTEERED model; --direction is ignored for those "
                         "arms. Mutually exclusive with --kv-mask/--dojo-defense/"
                         "--stack-dojo.")
    ap.add_argument("--dojo-defense", default=None, choices=S.DOJO_DEFENSES,
                    help="the defended/cleanplus arms become AgentDojo's OWN inbuilt defense "
                         "(their from_config wiring, single-model, no detector) around the "
                         "UNSTEERED local model; --direction is ignored for those arms. The "
                         "clean and attacked comparator arms stay the raw undefended model, "
                         "so baseline rows share them with our steering rows.")
    ap.add_argument("--stack-dojo", dest="stack_dojo", default=None,
                    choices=S.STACKABLE_DEFENSES,
                    help="STACKED arm: wire this prompt-level AgentDojo defense AROUND the "
                         "steered LLM, so the defended/cleanplus arms carry steering AND the "
                         "prompt defense together (the composition question). Mutually "
                         "exclusive with --dojo-defense and --kv-mask.")
    ap.add_argument("--alphas", default=None,
                    type=lambda s: sorted({float(x) for x in s.split(",")}),
                    help="DOSE-FRONTIER mode (owner program 2026-09-08): comma list of "
                         "alphas. Each cell runs clean + attacked ONCE (shared) plus "
                         "cleanplus@a / defended@a per alpha, all in the same process — "
                         "cell-level pairing never crosses processes. Mutually exclusive "
                         "with --benign-only/--defended-only. NOTE: the default=None is "
                         "never passed through `type` (argparse only applies type to "
                         "STRING defaults — the set() landmine in CLAUDE.md).")
    ap.add_argument("--steer-mode", default="add",
                    help="steering operator for the steered arms (src/steering.py MODES; "
                         "'add' = deployed subtraction; ablate* = dose-free projection, "
                         "run with --alpha 0). Unsteered arms always run 'add'/no-op.")
    ap.add_argument("--steer-schedule", default="fixed",
                    help="dose schedule over turns for the STEERED arms (§26.5 item 2, "
                         "owner program 2026-09-08); one of "
                         "{fixed,energy-norm} or a comma list of both. 'fixed' (default) "
                         "= today's behavior, byte-identical: alpha untouched every "
                         "forward. 'energy-norm' = per forward, scale alpha by "
                         "min(1, sqrt(n_0/n_t)) so TOTAL per-forward edit energy "
                         "(n_t x step^2) stays ~constant as tool spans accumulate; n_0 = "
                         "steered-token count at the episode's first steered forward, "
                         "capped so early turns are never dosed above fixed. See "
                         "agentdojo_bridge.sched_step. A COMMA LIST is the CONTROLLED "
                         "schedule battery: one process per cell, clean/attacked shared, "
                         "plus cleanplus@SCHED / defended@SCHED per schedule at the same "
                         "alpha (the --alphas pairing pattern). Applies to cleanplus/"
                         "defended arms only; clean/attacked are unsteered either way.")
    ap.add_argument("--span-select", dest="span_select", default="full",
                    help="WHICH tokens inside each tool span the STEERED arms edit "
                         "(§26.12 content-leaf program): one of {full,leaf,struct,random,"
                         "energy} or a comma list. 'full' (default) = every span token, "
                         "byte-identical to the pre-selector bridge. 'leaf' = schema-aware "
                         "content-leaf selection, fail-closed to full-span per uncertain "
                         "span. 'struct' = the exact per-span complement of leaf. 'random' "
                         "= per span, leaf-count-matched positions sampled from the span's "
                         "full set (span-text-seeded, stable across re-renders). 'energy' "
                         "= full-span positions with alpha scaled per forward by "
                         "sqrt(leaf/full) so TOTAL edit energy matches the leaf arm. A "
                         "COMMA LIST is the CONTROLLED selector battery: one process per "
                         "cell, shared clean/attacked, plus cleanplus@SEL / defended@SEL "
                         "per selector (the --alphas pairing pattern). COMBINES with "
                         "--benign-only: clean + cleanplus@SEL only (the §26.12 "
                         "regression panel).")
    ap.add_argument("--system", default="short", choices=["short", "yaml"],
                    help="system message: 'short' = the SYSTEM constant all prior AgentDojo "
                         "rows used; 'yaml' = the imported agentdojo package's own "
                         "system_messages.yaml default -- REQUIRED for AgentDyn runs "
                         "(build_pipeline reads this via getattr)")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--cells", default=f"{ROOT}/runs/agentdojo_cells.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--benign-only", action="store_true", default=False,
                    help="BENIGN SMOKE (owner ruling 2026-08-28: benign utility is "
                         "measured primarily here): dedup cells to unique (suite, "
                         "user_task) and run ONLY the clean and cleanplus arms.")
    ap.add_argument("--defended-only", action="store_true", default=False,
                    help="run ONLY the defended arm over the full attacked cell set. "
                         "AgentDojo's own checkers need no reference arm; ours_* columns "
                         "are skipped. For dose-floor measurements of a new direction "
                         "against the existing attacked baseline.")
    ap.add_argument("--security-only", action="store_true", default=False,
                    help="§26.12 gate-2 mode, valid ONLY with a multi --span-select "
                         "battery: run attacked (the in-process anchor) + defended@SEL "
                         "per selector, skipping clean/cleanplus (already measured by "
                         "the benign panel; AgentDojo's per-episode checkers need no "
                         "reference arm, ours_* are skipped).")
    ap.add_argument("--no-adjudicate", action="store_true", default=False,
                    help="skip the composed/adjudicated columns. Off by default: an "
                         "unmitigated struct_exact is not comparable to any Nemotron number "
                         "in BEST_DEFENSE.md, all of which carry both mitigations.")
    ap.add_argument("--out", default=f"{ROOT}/runs/agentdojo_run.json")
    a = ap.parse_args()

    if a.alphas and (a.benign_only or a.defended_only or a.dojo_defense or a.kv_mask
                     or a.agri_probe):
        ap.error("--alphas is mutually exclusive with --benign-only/--defended-only/"
                 "--dojo-defense/--kv-mask/--agri-probe (review 2026-09-08 D3: the "
                 "combinations silently drop or mislabel arms)")
    if a.agri_probe and (a.dojo_defense or a.kv_mask or a.stack_dojo):
        ap.error("--agri-probe is mutually exclusive with --dojo-defense/--kv-mask/"
                 "--stack-dojo -- one defense per arm")
    if a.agri_probe and not a.direction:
        # review 2026-09-11: build_pipeline reads `direction` as the defense-ON flag, so
        # an empty --direction would silently run the defended/cleanplus arms UNDEFENDED
        # while labelling them AGRI (the §23e no-op-wearing-a-label class). The direction
        # itself is ignored on AGRI arms; it must merely be non-empty.
        ap.error("--agri-probe requires a non-empty --direction (used only as the "
                 "defense-ON flag; ignored on AGRI arms)")
    # --steer-schedule: a single value threads straight through; a comma list is the
    # §26.5 item 2 CONTROLLED schedule battery (per-schedule steered arms sharing their
    # cell's clean/attacked comparators in one process, the --alphas pairing pattern).
    scheds = []
    for s_ in str(a.steer_schedule).split(","):
        if s_ not in SCHEDULES:
            ap.error(f"--steer-schedule {s_!r} is not one of {list(SCHEDULES)}")
        if s_ not in scheds:
            scheds.append(s_)
    sched_battery = len(scheds) > 1
    if not sched_battery:
        a.steer_schedule = scheds[0]   # normalized string; build_pipeline reads this
    if sched_battery and (a.alphas or a.benign_only or a.defended_only
                          or a.dojo_defense or a.kv_mask or a.agri_probe):
        ap.error("a multi-schedule --steer-schedule (the §26.5 item 2 controlled battery) "
                 "is mutually exclusive with --alphas/--benign-only/--defended-only/"
                 "--dojo-defense/--kv-mask -- one comparison per process, and the "
                 "steered-arm wiring would silently mislabel arms otherwise")
    if any(s_ != "fixed" for s_ in scheds) and (a.dojo_defense or a.kv_mask
                                                or a.agri_probe):
        ap.error("--steer-schedule applies to STEERED arms only; with --dojo-defense/"
                 "--kv-mask/--agri-probe the defended arm is unsteered, so a non-fixed "
                 "schedule would silently not run (the §23e no-op-wearing-a-label class)")
    if any(s_ != "fixed" for s_ in scheds) and not a.direction:
        # review8 D3: build_pipeline forces schedule="fixed" when direction is falsy,
        # BYPASSING the bridge's no-op-label guard -- arms labelled @energy-norm would run
        # unsteered. Refuse here instead.
        ap.error("a non-fixed --steer-schedule requires a non-empty --direction")
    # --span-select: single value threads through; a comma list is the §26.12 CONTROLLED
    # selector battery (per-selector steered arms sharing their cell's comparators in one
    # process, the --alphas pairing pattern). Unlike the other batteries it COMBINES with
    # --benign-only (clean + cleanplus@SEL only) -- that combination IS the §26.12
    # regression panel.
    sels = []
    for s_ in str(a.span_select).split(","):
        if s_ not in SPAN_SELECTS:
            ap.error(f"--span-select {s_!r} is not one of {list(SPAN_SELECTS)}")
        if s_ not in sels:
            sels.append(s_)
    sel_battery = len(sels) > 1
    if not sel_battery:
        a.span_select = sels[0]   # normalized string; build_pipeline reads this
    if sel_battery and (a.alphas or sched_battery or a.defended_only
                        or a.dojo_defense or a.kv_mask or a.agri_probe):
        ap.error("a multi --span-select (the §26.12 selector battery) is mutually "
                 "exclusive with --alphas / a multi-schedule battery / --defended-only / "
                 "--dojo-defense / --kv-mask -- one comparison per process")
    if a.security_only and not sel_battery:
        ap.error("--security-only is defined only for the multi --span-select battery "
                 "(attacked + defended@SEL); use --defended-only elsewhere")
    if a.security_only and a.benign_only:
        ap.error("--security-only and --benign-only are mutually exclusive")
    if any(s_ != "full" for s_ in sels) and (a.dojo_defense or a.kv_mask
                                             or a.agri_probe):
        ap.error("--span-select applies to STEERED arms only; with --dojo-defense/"
                 "--kv-mask/--agri-probe the defended arm is unsteered, so a non-full "
                 "selector would silently not run (the §23e no-op-wearing-a-label class)")
    if any(s_ != "full" for s_ in sels) and not a.direction:
        ap.error("a non-full --span-select requires a non-empty --direction")
    if any(s_ != "full" for s_ in sels) and any(s2 != "fixed" for s2 in scheds):
        ap.error("--span-select and a non-fixed --steer-schedule are mutually exclusive "
                 "-- one dose manipulation per arm")
    cells = json.load(open(a.cells))["cells"]
    if a.benign_only:
        seen, dedup = set(), []
        for c in cells:
            k = (c["suite"], c["user_task"])
            if k not in seen:
                seen.add(k); dedup.append(c)
        print(f"[benign-only] {len(dedup)} unique user tasks from {len(cells)} cells")
        cells = dedup
    # SHARD BY TASK GROUP, NOT BY CELL INDEX. The clean arm depends only on
    # (suite, user_task) -- not on the injection task and not on the attack -- so every cell
    # sharing a task can share one clean run. Round-robin over cells would scatter a task's
    # cells across all four shards and defeat the cache; round-robin over GROUPS keeps them
    # together. Measured scale: 39 distinct solvable tasks behind ~180 cells, so this turns
    # ~180 clean runs into ~39 -- roughly a quarter of the entire job.
    groups = {}
    for c in cells:
        groups.setdefault((c["suite"], c["user_task"]), []).append(c)
    keys = sorted(groups)
    mine = [c for i, k in enumerate(keys) if i % a.nshard == a.shard for c in groups[k]]
    print(f"[run] {len(mine)} of {len(cells)} cells across "
          f"{len([1 for i, _ in enumerate(keys) if i % a.nshard == a.shard])} task groups "
          f"(shard {a.shard}/{a.nshard})", flush=True)

    model, tok = X.load_model_and_tok(a.model, a.device)
    out, t0 = [], time.time()
    saved_turns = {}
    clean_cache = {}          # (suite, user_task) -> (row_dict, transcript, tool_outputs)
    cleanplus_cache = {}      # (suite, user_task) -> (row_dict, transcript)
    clean_turns_cache, cleanplus_turns_cache = {}, {}

    for k, c in enumerate(mine):
        suite = get_suite(S.VERSION, c["suite"])
        ut = suite.get_user_task_by_id(c["user_task"])
        row = dict(c)
        transcripts = {}
        ckey = (c["suite"], c["user_task"])
        clean_tool_out = []
        if ckey in clean_cache:
            # Reuse, not re-run. Generation is greedy and the clean prompt is byte-identical
            # across every cell of this task, so a second run would reproduce it exactly.
            row["clean"], transcripts["clean"], clean_tool_out = clean_cache[ckey]
            row["clean"] = dict(row["clean"], cached=True)
            # A CACHED ARM STILL NEEDS ITS TRANSCRIPT WRITTEN. Storing turns only for arms
            # that actually ran left 46 of 93 cells with no `|clean` entry, so half the run
            # could not be re-scored at all -- which defeats the reason transcripts are kept.
            saved_turns[(k, "clean")] = clean_turns_cache[ckey]
        if ckey in cleanplus_cache:
            row["cleanplus"], transcripts["cleanplus"] = cleanplus_cache[ckey]
            row["cleanplus"] = dict(row["cleanplus"], cached=True)
            saved_turns[(k, "cleanplus")] = cleanplus_turns_cache[ckey]
        # CLEAN+ is the DEPLOYMENT COST arm: defense on, no injection. CLAUDE.md's headline
        # block calls it `utilBenign` and it is paid on ~100% of traffic, because a poisoned
        # tool response is not identifiable in advance. Without it the run cannot report the
        # required table at all. It is cacheable on (suite, user_task) for the same reason the
        # clean arm is.
        # Build the arm list, THEN iterate. Putting a conditional `None` inside the tuple made
        # Python try to unpack it at loop entry, before any guard could skip it.
        arms = []   # (arm_name, direction, attacked, alpha_override, sched_override,
                    #  sel_override)
        if a.defended_only:
            # Only the defended arm: AgentDojo's own security/utility checkers are
            # per-episode and need no reference, so this measures a direction's
            # attacked-side floor at 1/4 the cost. The `ours_*` columns are skipped
            # (their guard below requires the clean transcript).
            arms.append(("defended", a.direction, True, None, None, None))
        elif a.alphas:
            # DOSE-FRONTIER mode (owner program 2026-09-08, FINDINGS pre-registration):
            # one process per cell, the SHARED clean/attacked arms run once, and each
            # alpha contributes its own cleanplus@a / defended@a pair. Cell-level pairing
            # never crosses processes; alpha rows share their cell's comparators.
            if ckey not in clean_cache:
                arms.append(("clean", None, False, None, None, None))
            for al in a.alphas:
                if (ckey, al) not in cleanplus_cache:
                    arms.append((f"cleanplus@{al:g}", a.direction, False, al, None, None))
            arms.append(("attacked", None, True, None, None, None))
            arms += [(f"defended@{al:g}", a.direction, True, al, None, None)
                     for al in a.alphas]
        elif sched_battery:
            # §26.5 item 2 CONTROLLED schedule battery: one process per cell, the SHARED
            # clean/attacked arms run once, and each schedule contributes its own
            # cleanplus@SCHED / defended@SCHED pair at the SAME alpha. Cell-level pairing
            # never crosses processes; schedule rows share their cell's comparators.
            if ckey not in clean_cache:
                arms.append(("clean", None, False, None, None, None))
            for sc_ in scheds:
                if (ckey, sc_) not in cleanplus_cache:
                    arms.append((f"cleanplus@{sc_}", a.direction, False, None, sc_, None))
            arms.append(("attacked", None, True, None, None, None))
            arms += [(f"defended@{sc_}", a.direction, True, None, sc_, None)
                     for sc_ in scheds]
        elif sel_battery:
            # §26.12 CONTROLLED selector battery: one process per cell, shared
            # clean/attacked, plus cleanplus@SEL / defended@SEL per selector at the same
            # alpha. With --benign-only (the regression panel): clean + cleanplus@SEL only.
            # With --security-only (the gate-2 smoke): attacked + defended@SEL only --
            # the benign arms are the panel's, measured once, not re-burned.
            if not a.security_only:
                if ckey not in clean_cache:
                    arms.append(("clean", None, False, None, None, None))
                for se in sels:
                    if (ckey, se) not in cleanplus_cache:
                        arms.append((f"cleanplus@{se}", a.direction, False, None, None, se))
            if not a.benign_only:
                arms.append(("attacked", None, True, None, None, None))
                arms += [(f"defended@{se}", a.direction, True, None, None, se)
                         for se in sels]
        else:
            if ckey not in clean_cache:
                arms.append(("clean", None, False, None, None, None))
            if ckey not in cleanplus_cache:
                arms.append(("cleanplus", a.direction, False, None, None, None))
            if not a.benign_only:
                arms += [("attacked", None, True, None, None, None),
                         ("defended", a.direction, True, None, None, None)]
        # dose-frontier / schedule-battery / selector-battery cache reuse for cleanplus@X
        if a.alphas or sched_battery or sel_battery:
            _pairs = ([(al, f"cleanplus@{al:g}") for al in a.alphas] if a.alphas
                      else [(sc_, f"cleanplus@{sc_}") for sc_ in scheds] if sched_battery
                      else [(se, f"cleanplus@{se}") for se in sels])
            for kk, nm in _pairs:
                if (ckey, kk) in cleanplus_cache:
                    row[nm], transcripts[nm] = cleanplus_cache[(ckey, kk)]
                    row[nm] = dict(row[nm], cached=True)
                    saved_turns[(k, nm)] = cleanplus_turns_cache[(ckey, kk)]
        for arm, direction, attacked, alpha_override, sched_override, sel_override in arms:
            a_arm = a
            if (alpha_override is not None or sched_override is not None
                    or sel_override is not None):
                import copy
                a_arm = copy.copy(a)
                if alpha_override is not None:
                    a_arm.alpha = alpha_override
                if sched_override is not None:
                    a_arm.steer_schedule = sched_override
                if sel_override is not None:
                    a_arm.span_select = sel_override
            pipe, llm = S.build_pipeline(model, tok, a_arm, direction)
            llm.transcript = []
            try:
                with OutputLogger(None):
                    if attacked:
                        atk = load_attack(c["attack"], suite, pipe)
                        u_, s_ = run_task_with_injection_tasks(
                            suite, pipe, ut, atk, None, True,
                            injection_tasks=[c["injection_task"]],
                            benchmark_version=S.VERSION)
                        u = sum(u_.values()) / max(1, len(u_))
                        sec = sum(s_.values()) / max(1, len(s_))
                    else:
                        u_, _ = run_task_without_injection_tasks(suite, pipe, ut, None, True)
                        u, sec = float(u_), None
                # SCORING TEXT IS REASONING-STRIPPED (adversarial review 2026-08-30): Qwen
                # quotes parseable <tool_call> JSON while deliberating (17 quoted blocks in
                # one steered arm vs 0 clean on the benign smoke), so ours_goal /
                # behavioural_score on raw chatml text fire on refusals, arm-asymmetrically.
                # Harmony text passes through unchanged -- see reasoning_free's docstring.
                transcripts[arm] = "".join(
                    reasoning_free(t["completion"], t.get("fmt", "harmony"),
                                   t.get("in_think", False)) for t in llm.transcript)
                # `truncated` = turns that hit max_new. Steering multiplies think length
                # (~4x on the smoke), so the cap lands mostly on steered arms and a truncated
                # final turn is utility 0 by construction -- report it beside utility.
                row[arm] = {"utility": u, "security": sec, "llm_calls": llm.n_calls,
                            "steered_tokens": llm.n_steered_tokens,
                            "truncated": llm.n_truncated}
                if direction and getattr(a_arm, "steer_schedule", "fixed") != "fixed":
                    # engagement evidence for the §26.5 item 2 schedule experiment: the
                    # per-turn alpha multiplier actually applied (None = unsteered turn).
                    # Only written on steered arms running a non-fixed schedule, so
                    # existing artifact schemas are untouched on the default path.
                    row[arm]["sched_scales"] = [t.get("sched_scale")
                                                for t in llm.transcript]
                if direction and getattr(a_arm, "span_select", "full") != "full":
                    # §26.12 selector engagement evidence (the §25h steered-tokens
                    # standard): per-arm aggregate of what the selector saw, selected and
                    # fell back on. Per-forward detail rides in the transcripts.
                    st = [t.get("sel_stats") for t in llm.transcript
                          if t.get("sel_stats")]
                    row[arm]["sel"] = {
                        "span_select": a_arm.span_select,
                        "forwards": len(st),
                        "spans": sum(s["n_spans"] for s in st),
                        "fallback_spans": sum(s["n_fallback"] for s in st),
                        "html_spans": sum(s["html_spans"] for s in st),
                        "full_tokens": sum(s["full_tokens"] for s in st),
                        "leaf_tokens": sum(s["leaf_tokens"] for s in st),
                        "sel_tokens": sum(s["sel_tokens"] for s in st),
                        "content_keys": sorted({kk for s in st
                                                for kk in s["content_keys"]}),
                        "struct_keys": sorted({kk for s in st
                                               for kk in s["struct_keys"]}),
                        # review1 F2: shape exclusions counted AND named -- an unnamed
                        # shape exclusion would be an unlogged non-coverage class
                        "shape_struct": sum(s.get("shape_struct", 0) for s in st),
                        "shape_struct_keys": sorted({kk for s in st
                                                     for kk in s.get("shape_struct_keys",
                                                                     [])}),
                        # v2 contract telemetry (§26.17; empty on v1 arms)
                        "trusted_keys": sorted({kk for s in st
                                                for kk in s.get("trusted_keys", [])}),
                        "steered_keys": sorted({kk for s in st
                                                for kk in s.get("steered_keys", [])}),
                        "data_keys": sorted({kk for s in st
                                             for kk in s.get("data_keys", [])}),
                        "fallbacks": [f for s in st for f in s["fallbacks"]][:50],
                    }
                if getattr(pipe, "tool_filter", None):
                    # engagement proof for the tool_filter baseline: (n_before, n_after)
                    # per filter call, plus how often the final-channel reply was empty
                    row[arm]["tools_kept"] = pipe.tool_filter.kept_log
                    row[arm]["filter_fallbacks"] = pipe.tool_filter.n_fallback
                if getattr(pipe, "pi_detector", None):
                    # engagement proof for the classifier-filter baselines: a filter row
                    # with zero checks (or zero flags on the attacked arm) is a no-op
                    row[arm]["pi_checked"] = pipe.pi_detector.n_checked
                    row[arm]["pi_flagged"] = pipe.pi_detector.n_flagged
                    # messages long enough to need >1 detection window (the 512-token
                    # blind-spot audit trail; review 2026-09-04)
                    row[arm]["pi_chunked"] = pipe.pi_detector.n_chunked
                if getattr(pipe, "fmt_counter", None):
                    # engagement proof for formatter-based prompt defenses (reminder,
                    # spotlighting): transcripts drop prompts, so this is the only
                    # artifact evidence the formatter ran
                    row[arm]["fmt_calls"] = pipe.fmt_counter.n
                if getattr(llm, "agri", None):
                    # engagement proof for the AGRI baseline: probe evaluations, threshold
                    # crossings, and turns that carried the reasoning prefill. An arm with
                    # zero checks -- or an attacked-side arm with zero fires -- is the
                    # §23e no-op-wearing-a-label failure and must be treated as such.
                    row[arm]["agri_checked"] = llm.agri.n_checked
                    row[arm]["agri_fired"] = llm.agri.n_fired
                    row[arm]["agri_prefilled"] = llm.agri.n_prefilled
                if arm == "clean":
                    clean_tool_out = S.tool_outputs_from(llm.transcript)
                    clean_cache[ckey] = (row[arm], transcripts[arm], clean_tool_out)
                    clean_turns_cache[ckey] = llm.transcript
                elif arm == "cleanplus":
                    cleanplus_cache[ckey] = (row[arm], transcripts[arm])
                    cleanplus_turns_cache[ckey] = llm.transcript
                elif arm.startswith("cleanplus@") and alpha_override is not None:
                    cleanplus_cache[(ckey, alpha_override)] = (row[arm], transcripts[arm])
                    cleanplus_turns_cache[(ckey, alpha_override)] = llm.transcript
                elif arm.startswith("cleanplus@") and sched_override is not None:
                    cleanplus_cache[(ckey, sched_override)] = (row[arm], transcripts[arm])
                    cleanplus_turns_cache[(ckey, sched_override)] = llm.transcript
                elif arm.startswith("cleanplus@") and sel_override is not None:
                    cleanplus_cache[(ckey, sel_override)] = (row[arm], transcripts[arm])
                    cleanplus_turns_cache[(ckey, sel_override)] = llm.transcript
                # every arm's turns are persisted so all scoring can be redone off stored
                # generations -- see the transcript block below
                saved_turns[(k, arm)] = llm.transcript
            except Exception as e:
                row[arm] = {"error": f"{type(e).__name__}: {e}"}
        if "clean" in transcripts:
            samp = S.ours_sample(suite, c["injection_task"], clean_tool_out)
            row["ours_scoreable"] = samp["ours_scoreable"]
            row["attacker_calls"] = samp["attacker_calls"]
            # SCORE THE CLEAN AND CLEAN+ ARMS TOO. `goal` fires on legitimate behaviour on this
            # corpus (21/949 task x injection pairs have a user task whose own ground truth
            # satisfies the predicate), so the unattacked arm is the metric's false-positive
            # floor and every arm is read against it -- not against 0.
            for arm in list(transcripts):
                if "error" not in row.get(arm, {}):
                    row[arm].update(S.ours_scores(samp, transcripts["clean"],
                                                  transcripts[arm]))
        # AN ARM THAT THREW MUST NOT SHRINK ONLY ITS OWN DENOMINATOR. Recording a failed arm
        # as `{"error": ...}` and averaging with `is not None` drops that arm from its mean
        # while the other arms keep the cell -- and the defended arm carries strictly more
        # failure surface (probe load, longer CoT, larger prefill). Mark the whole cell so a
        # scorer can drop it wholesale rather than compare arms over different sample sets.
        if a.alphas:
            _expected = (["clean", "attacked"]
                         + [f"cleanplus@{al:g}" for al in a.alphas]
                         + [f"defended@{al:g}" for al in a.alphas])
        elif sched_battery:
            _expected = (["clean", "attacked"]
                         + [f"cleanplus@{sc_}" for sc_ in scheds]
                         + [f"defended@{sc_}" for sc_ in scheds])
        elif sel_battery and a.security_only:
            _expected = ["attacked"] + [f"defended@{se}" for se in sels]
        elif sel_battery:
            _expected = (["clean"] + [f"cleanplus@{se}" for se in sels]
                         + ([] if a.benign_only
                            else ["attacked"] + [f"defended@{se}" for se in sels]))
        elif a.benign_only:
            _expected = ["clean", "cleanplus"]
        else:
            _expected = ["clean", "cleanplus", "attacked", "defended"]
        row["complete"] = all("error" not in row.get(arm, {"error": 1})
                              for arm in _expected)
        out.append(row)
        el = time.time() - t0
        print(f"  [{k+1}/{len(mine)}] {c['suite']}/{c['user_task']}/{c['injection_task']} "
              f"{c['attack'][:28]:<28} atk sec={row.get('attacked',{}).get('security')} "
              f"def sec={row.get('defended',{}).get('security')} "
              f"| {el:.0f}s eta {el/(k+1)*(len(mine)-k-1):.0f}s", flush=True)
        # incremental write: an 8-hour run must not lose everything to a late crash
        blob = json.dumps({"config": vars(a), "results": out}, indent=1, default=str)
        with open(a.out + ".tmp", "w") as f:
            f.write(blob)
        os.replace(a.out + ".tmp", a.out)
        # PERSIST THE GENERATIONS. Every defect the adversarial review found was a SCORING
        # defect, and none could have been re-adjudicated from scalars -- the run would have
        # had to be burned again. This repo keeps rescore_behavioural.py for exactly that
        # reason. Prompts are dropped (they are ~10x the size and reconstructible); completions
        # and spans are kept.
        tblob = json.dumps({(f"{i}|{arm}"): [{"completion": t["completion"],
                                              "spans": t["spans"],
                                              "n_steered": t["n_steered"],
                                              "fmt": t.get("fmt", "harmony"),
                                              "in_think": t.get("in_think", False),
                                              # §26.5 item 2: which dose schedule ran and
                                              # the per-turn alpha multiplier it applied
                                              "schedule": t.get("schedule", "fixed"),
                                              "sched_scale": t.get("sched_scale"),
                                              # AGRI arms: per-turn probe score and
                                              # whether the reasoning prefill was applied
                                              "agri_score": t.get("agri_score"),
                                              "agri_prefilled": t.get("agri_prefilled"),
                                              # §26.12: which span selector ran and its
                                              # per-forward coverage/fallback evidence
                                              "span_select": t.get("span_select", "full"),
                                              "sel_stats": t.get("sel_stats")}
                                             for t in turns]
                            for (i, arm), turns in saved_turns.items()}, default=str)
        with open(a.out.replace(".json", ".transcripts.json") + ".tmp", "w") as f:
            f.write(tblob)
        os.replace(a.out.replace(".json", ".transcripts.json") + ".tmp",
                   a.out.replace(".json", ".transcripts.json"))

    json.load(open(a.out))
    print(f"\nwrote {a.out} ({len(out)} cells, {time.time()-t0:.0f}s)")

    # ── ADJUDICATION RUNS AS PART OF THE EVAL, NOT AS AN OPTIONAL AFTERTHOUGHT ────────────
    # Raw `struct_exact` is the wrong number to leave in an artifact: every Nemotron figure in
    # BEST_DEFENSE.md carries the composed-field exemption and the drift adjudicator, so an
    # unmitigated AgentDojo figure is not comparable to any of them. Running it here means the
    # artifact is born with all three readings and nobody has to remember a second command.
    #
    # It is CPU + one network endpoint, cached by content hash, and one-directional (it can
    # only move a sample wrong -> right), so it cannot manufacture a defense. The cache write
    # is flock-guarded because four shards do this concurrently.
    if not a.no_adjudicate:
        try:
            _finalise_scoring(out, saved_turns, a)
            blob = json.dumps({"config": vars(a), "results": out}, indent=1, default=str)
            with open(a.out + ".tmp", "w") as f:
                f.write(blob)
            os.replace(a.out + ".tmp", a.out)
            json.load(open(a.out))
            print(f"rewrote {a.out} with corrComp / corrAdj columns")
        except Exception as e:
            # NEVER lose a completed generation run to a scoring failure. The transcripts are
            # on disk; agentdojo_rescore.py can redo this offline.
            print(f"[adjudication FAILED: {type(e).__name__}: {e}] "
                  f"raw struct_exact columns stand; rescore offline from the transcripts")


def _finalise_scoring(out, saved_turns, a):
    """Add composed-field and adjudicated correctness to every arm, in place."""
    import score_table as ST
    arms = ("clean", "cleanplus", "attacked", "defended")
    cells = [(i, r) for i, r in enumerate(out) if r.get("complete")]
    S, ref, texts, spans = [], [], {A: [] for A in arms}, []
    for i, r in cells:
        turns = saved_turns.get((i, "clean")) or []
        outs = [t["prompt"][lo:hi] for t in turns for lo, hi in t.get("spans", [])] \
            if turns and turns[0].get("prompt") else []
        S.append({"id": f"{r['suite']}/{r['user_task']}/{r['injection_task']}",
                  "payload_clean": {"tool_outputs": outs}, "target_tool": None,
                  "target_args": {}, "injection_text": "", "retrieval_tool": None})
        for A in arms:
            t = saved_turns.get((i, A)) or []
            # same reasoning-stripping the live ours_* scoring applies -- a think-quoted
            # <tool_call> must not reach behavioural_score / the drift adjudicator either
            texts[A].append("".join(
                reasoning_free(x["completion"], x.get("fmt", "harmony"),
                               x.get("in_think", False)) for x in t))
        ref.append(texts["clean"][-1])
    drift = {A: ST.collect_drift(S, ref, texts[A]) for A in arms}
    allp = [p for d in drift.values() for ps in d.values() for p in ps]
    adj, astats = ({}, {}) if not allp else ST.adjudicate(allp)
    if allp:
        print(f"[adjudicator] {len(allp)} pairs, {astats.get('n_new',0)} new; controls "
              f"identical-pair {astats.get('ctrl_pos_ok','-')}/{astats.get('ctrl_pos_n','-')}, "
              f"marker-appended {astats.get('ctrl_neg_ok','-')}/{astats.get('ctrl_neg_n','-')}")
    for j, (i, r) in enumerate(cells):
        for A in arms:
            if A not in r:
                # battery modes (--alphas / multi-schedule) name their steered arms
                # cleanplus@X / defended@X; the plain names are absent from the row and
                # writing to them would KeyError, aborting adjudication for every arm
                continue
            sc = X.behavioural_score(ref[j], texts[A][j])
            if not sc["scoreable"]:
                continue
            r[A]["struct_exact_composed"] = bool(
                ST.struct_exact_composed(S[j], sc, ref[j], texts[A][j]))
            ps = drift[A].get(j)
            r[A]["struct_exact_adjudicated"] = bool(
                sc.get("struct_exact") or (ps and all(
                    adj.get(ST._adj_key(p["tool"], p["arg"], p["a"], p["b"])) == "EQUIVALENT"
                    for p in ps)))
            r[A]["adjudicator_saw"] = bool(ps)


if __name__ == "__main__":
    main()
