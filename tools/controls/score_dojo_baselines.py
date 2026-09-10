#!/usr/bin/env python
"""Aggregate the AgentDojo inbuilt-defense baseline runs into the comparison table.

Adversarially reviewed 2026-08-30; corrections applied:
  - shard glob is shard[0-9].json (a bare * also matched the .transcripts.json siblings)
  - BOTH steering cells are printed, each with ITS OWN benign pairing -- never one cell's
    security beside another cell's benign delta
  - per-row: n, error count, host; benign cost is the WITHIN-RUN paired delta (clean
    utility LEVELS drift ~5pp across hosts and are not comparable across rows)

Inputs (shards from tools/controls/dojo_baselines_job.sh, downloaded from blob):
  <dir>/dojo_<defense>_atk.shard[0-9].json     defended arm only, 180 attacked cells
  <dir>/dojo_<defense>_benign.shard[0-9].json  clean + cleanplus over unique user tasks

Usage: python tmp/score_dojo_baselines.py [dir=runs]
"""
import glob
import json
import sys

D = next((a for a in sys.argv[1:] if not a.startswith("-")), "runs")
DEFS = ["spotlighting_with_delimiting", "repeat_user_prompt", "tool_filter"]


def rows_of(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        out += json.load(open(f))["results"]
    return out


def mean_key(rows, arm, key):
    v = [r[arm][key] for r in rows
         if isinstance(r.get(arm), dict) and r[arm].get(key) is not None]
    return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)


def errs(rows, arm):
    return sum(1 for r in rows if isinstance(r.get(arm), dict) and "error" in r[arm])


def benign_pairs(rows):
    """(clean_util, cleanplus_util) per unique task where BOTH arms ran without error."""
    seen, out = set(), []
    for r in rows:
        k = (r["suite"], r["user_task"])
        if k in seen:
            continue
        if isinstance(r.get("clean"), dict) and isinstance(r.get("cleanplus"), dict) \
                and "error" not in r["clean"] and "error" not in r["cleanplus"]:
            # mark seen only on a VALID pair: an errored row must not shadow the
            # gap-rerun row for the same task that follows it in a later shard file
            seen.add(k)
            out.append((r["clean"]["utility"], r["cleanplus"]["utility"]))
    return out


def prow(label, sec, n, ua, pairs, err, host):
    bc = sum(a for a, _ in pairs) / len(pairs) if pairs else float("nan")
    bd = sum(b for _, b in pairs) / len(pairs) if pairs else float("nan")
    dl = f"{100*(bd-bc):>+9.1f}" if pairs else f"{'--':>9}"
    print(f"{label:<44}{sec:>10.3f}{n:>6}{ua:>10.3f}{bc:>10.3f}{bd:>9.3f}{dl}"
          f"{len(pairs):>6}{err:>5}  {host}")


hdr = (f"{'arm':<44}{'security v':>10}{'(n)':>6}{'utilAtk ^':>10}"
       f"{'benClean ^':>10}{'benDef ^':>9}{'delta pp':>9}{'(np)':>6}{'err':>5}  host")
print(hdr)

# comparators: raw undefended (attacked arm of the Aug-5 full run, same 180 cells)
ref = rows_of("runs/agentdojo_run.shard[0-9].json")
asec, an = mean_key(ref, "attacked", "security")
autil, _ = mean_key(ref, "attacked", "utility")
print(f"{'attacked undefended (comparator)':<44}{asec:>10.3f}{an:>6}{autil:>10.3f}"
      f"{'':>10}{'':>9}{'':>9}{'':>6}{errs(ref,'attacked'):>5}  local-A100")

# steering cell 1: dim_no_override_both a8.0 -- security AND benign from the SAME full run
s1sec, s1n = mean_key(ref, "defended", "security")
s1ua, _ = mean_key(ref, "defended", "utility")
prow("steering dim_no_override_both a8.0 (ours)", s1sec, s1n, s1ua,
     benign_pairs(ref), errs(ref, "defended"), "local-A100")

# steering cell 2 (LOCKED, BEST_DEFENSE.md): combo_ovr8_pat1 a8.06 -- security from its
# own 180-cell run, benign from ITS dedicated benign pairing artifact
combo = rows_of("runs/agentdojo_combo_v2.shard[0-9].json")
s2sec, s2n = mean_key(combo, "defended", "security")
s2ua, _ = mean_key(combo, "defended", "utility")
try:
    bl = json.load(open("runs/dojo_benign_locked.json"))["results"]
    bl_pairs = benign_pairs(bl)
except FileNotFoundError:
    bl_pairs = []
prow("steering combo_ovr8_pat1 a8.06 (ours, LOCKED)", s2sec, s2n, s2ua,
     bl_pairs, errs(combo, "defended"), "local/.7")

for d in DEFS:
    ar = rows_of(f"{D}/dojo_{d}_atk.shard[0-9].json")
    br = rows_of(f"{D}/dojo_{d}_benign.shard[0-9].json")
    # originals + gap files must tile the 180 cells: a VALID defended row per cell at most
    # once (gap manifests are the exact complement of valid rows; assert, don't assume)
    vk = [(r["suite"], r["user_task"], r["injection_task"], r["attack"]) for r in ar
          if isinstance(r.get("defended"), dict) and "error" not in r["defended"]]
    assert len(vk) == len(set(vk)), f"{d}: duplicate valid defended rows -- double count"
    sec, n = mean_key(ar, "defended", "security")
    ua, _ = mean_key(ar, "defended", "utility")
    lbl = "dojo " + d + (" [mechanical]" if d == "tool_filter" else "")
    prow(lbl, sec, n, ua, benign_pairs(br), errs(ar, "defended"), "singularity-H100")
    if d == "tool_filter" and ar:
        kl = [t for r in ar if isinstance(r.get("defended"), dict)
              for t in r["defended"].get("tools_kept", [])]
        fb = sum(r["defended"].get("filter_fallbacks", 0) for r in ar
                 if isinstance(r.get("defended"), dict))
        if kl:
            shrunk = sum(1 for b, a2 in kl if a2 < b)
            print(f"  [tool_filter engagement] {len(kl)} filter calls, "
                  f"{shrunk} shrank the toolset, mean {sum(b for b,_ in kl)/len(kl):.1f} -> "
                  f"{sum(a2 for _,a2 in kl)/len(kl):.1f} tools, {fb} full-completion "
                  f"fallbacks (each weakens the filter vs a strict port)")

print("""
legend: security v [tier 1] = AgentDojo per-injection checker (attacker task completed),
  mean over cells whose arm ran without error (n stated per row; 180 cells total).
utilAtk ^ [tier 2] = AgentDojo utility under attack, raw level (same denominator as security).
benClean/benDef ^ [tier 2] = AgentDojo utility on clean episodes, raw model vs defense-on,
  paired unique user tasks (np = pairs with both arms non-error). delta pp is the WITHIN-RUN
  paired benign deployment cost; clean LEVELS are not comparable across hosts (~5pp drift).
caveats: all rows share a shortened base system message (first two sentences of AgentDojo's
  default) -- internally valid, NOT comparable to published AgentDojo tables. tool_filter's
  local port falls back to matching tool names in the full completion when the final channel
  is empty (weakens that defense vs a strict port); fallback count printed above.""")


def qwen_table(D):
    """Qwen3-30B-A3B-Thinking-2507 table: same layout, Qwen artifacts.

    Comparators: (a) SAME-PROCESS undefended attacked rerun (qwen_dojo_undefended_atk,
    H100, identical code path as the baselines); (b) the .7 A100 full run
    (runs/qwen_agentdojo_run) whose attacked arm is 0.457/175 and whose defended arm is
    the alpha=16 steering cell -- GUARD-TERRITORY dose, not a shippable point (FINDINGS
    section 12); its benign pairing comes from runs/qwen_agentdojo_benign.
    """
    print(hdr)
    ref = rows_of("runs/qwen_agentdojo_run.shard[0-9].json")
    asec, an = mean_key(ref, "attacked", "security")
    autil, _ = mean_key(ref, "attacked", "utility")
    print(f"{'attacked undefended (.7 full run)':<44}{asec:>10.3f}{an:>6}{autil:>10.3f}"
          f"{'':>10}{'':>9}{'':>9}{'':>6}{errs(ref,'attacked'):>5}  .7-A100")
    ua = rows_of(f"{D}/qwen_dojo_undefended_atk.shard[0-9].json")
    if ua:
        uvk = [(r["suite"], r["user_task"], r["injection_task"], r["attack"]) for r in ua
               if isinstance(r.get("defended"), dict) and "error" not in r["defended"]]
        assert len(uvk) == len(set(uvk)), "undefended rerun: duplicate valid rows"
        s, n = mean_key(ua, "defended", "security")
        u, _ = mean_key(ua, "defended", "utility")
        print(f"{'attacked undefended (same-process rerun)':<44}{s:>10.3f}{n:>6}{u:>10.3f}"
              f"{'':>10}{'':>9}{'':>9}{'':>6}{errs(ua,'defended'):>5}  singularity-H100")
    # CHAMPION cell (owner ruling 2026-08-31): dim_no_override_both @ 12 sigma, own-sigma,
    # layers 8,20,32 -- its own within-run benign pairing
    a12 = rows_of("runs/qwen_ad_a12_def.shard[0-9].json")
    if a12:
        cs, cn = mean_key(a12, "defended", "security")
        cu, _ = mean_key(a12, "defended", "utility")
        try:
            b12 = json.load(open("runs/qwen_ad_a12_benign.json"))["results"]
        except FileNotFoundError:
            b12 = []
        prow("steering dim_no_override_both a12 (CHAMPION)", cs, cn, cu,
             benign_pairs(b12), errs(a12, "defended"), ".7-A100")
    ssec, sn = mean_key(ref, "defended", "security")
    sua, _ = mean_key(ref, "defended", "utility")
    bref = rows_of("runs/qwen_agentdojo_benign.shard[0-9].json")
    prow("steering dim_no_override_both a16 (GUARD dose)", ssec, sn, sua,
         benign_pairs(bref), errs(ref, "defended"), ".7-A100")
    for d in DEFS:
        ar = rows_of(f"{D}/qwen_dojo_{d}_atk.shard[0-9].json")
        br = rows_of(f"{D}/qwen_dojo_{d}_benign.shard[0-9].json")
        if not ar:
            print(f"{'dojo '+d:<44}  (no artifacts yet)")
            continue
        vk = [(r["suite"], r["user_task"], r["injection_task"], r["attack"]) for r in ar
              if isinstance(r.get("defended"), dict) and "error" not in r["defended"]]
        assert len(vk) == len(set(vk)), f"{d}: duplicate valid defended rows"
        sec, n = mean_key(ar, "defended", "security")
        uu, _ = mean_key(ar, "defended", "utility")
        lbl = "dojo " + d + (" [mechanical]" if d == "tool_filter" else "")
        prow(lbl, sec, n, uu, benign_pairs(br), errs(ar, "defended"), "singularity-H100")
        if d == "tool_filter":
            kl = [t for r in ar if isinstance(r.get("defended"), dict)
                  for t in r["defended"].get("tools_kept", [])]
            fb = sum(r["defended"].get("filter_fallbacks", 0) for r in ar
                     if isinstance(r.get("defended"), dict))
            if kl:
                print(f"  [tool_filter engagement] {len(kl)} calls, mean "
                      f"{sum(b for b,_ in kl)/len(kl):.1f} -> "
                      f"{sum(a2 for _,a2 in kl)/len(kl):.1f} tools, {fb} fallbacks")


def reanchor_table():
    """gpt-oss-20b locked cell: the 768-budget run vs the max_new 4096 rerun.

    FINDINGS section 21 established that AgentDojo's `security` checker is OUTCOME-based,
    so a turn cut off mid-compliance scores as SECURE -- 768 (and 2048) demonstrably
    censored the attacked arm downward. This table is the re-anchoring: SAME 180 cells,
    SAME deployed cell (combo_ovr8_pat1 L12/16/20, resolved step [388.28, 696.09,
    1248.54] in BOTH runs -- verify against the `[steer]` line in the shard logs), only
    the generation budget differs. Quote it beside the per-episode non-truncation audit
    (tools/controls/agentdojo_trunc_audit.py); security from a budget-censored arm is a
    LOWER bound, not a measurement.
    """
    new = rows_of("runs/dojo_full_mn4096/agentdojo_combo_v3_mn4096.shard[0-9].json")
    old = rows_of("runs/agentdojo_combo_v2.shard[0-9].json")
    if not new:
        print("  (no runs/dojo_full_mn4096 artifacts)")
        return
    key = lambda r: (r["suite"], r["user_task"], r["injection_task"], r["attack"])  # noqa: E731
    assert len({key(r) for r in new}) == len(new), "mn4096: duplicate cells -- double count"
    print(f"{'arm':<14}{'sec@768 v':>10}{'sec@4096 v':>11}{'(n)':>6}"
          f"{'util@768 ^':>11}{'util@4096 ^':>12}{'(n)':>6}   paired 768->4096")
    mo = {key(r): r for r in old}
    for arm in ("clean", "cleanplus", "attacked", "defended"):
        so, _ = mean_key(old, arm, "security")
        sn, snn = mean_key(new, arm, "security")
        uo, _ = mean_key(old, arm, "utility")
        un, unn = mean_key(new, arm, "utility")
        # paired flips are the honest read: the two runs are independent greedy
        # regenerations, so a level difference alone cannot separate censoring from noise
        flips = {}
        for f in ("security", "utility"):
            up = dn = 0
            for r in new:
                a, b = mo.get(key(r), {}).get(arm), r.get(arm)
                if not (isinstance(a, dict) and isinstance(b, dict)):
                    continue
                if a.get(f) is None or b.get(f) is None:
                    continue
                up += b[f] > a[f]
                dn += b[f] < a[f]
            flips[f] = (up, dn)
        sc = "        --" if snn == 0 else f"{so:>10.3f}"
        sc2 = "         --" if snn == 0 else f"{sn:>11.3f}"
        note = (f"sec {flips['security'][0]}up/{flips['security'][1]}dn  " if snn else "")
        print(f"{arm:<14}{sc}{sc2}{snn:>6}{uo:>11.3f}{un:>12.3f}{unn:>6}   "
              f"{note}util {flips['utility'][0]}up/{flips['utility'][1]}dn")
    # benign deployment cost, paired over UNIQUE user tasks, each run against its own clean
    for lbl, pr in (("mn4096 (4096)", benign_pairs(new)),
                    ("combo_v2 (768)", benign_pairs(old))):
        if not pr:
            continue
        c = sum(a for a, _ in pr) / len(pr)
        d = sum(b for _, b in pr) / len(pr)
        print(f"  benign paired, {lbl:<15} n={len(pr):<4} clean {c:.3f} -> CLEAN+ {d:.3f}"
              f"  = {100*(d-c):+.1f}pp, {100*d/c:.1f}% of clean")
    print("""
legend: security v [1] AgentDojo's per-injection checker; utility ^ [2] their per-task
  checker. Both runs are INDEPENDENT greedy regenerations, so read the paired up/dn
  columns, not the level difference: one-directional flips (many up, ~none down) are the
  censoring signature, symmetric flips are regeneration noise.""")


if "--qwen" in sys.argv:
    print("\n=== Qwen3-30B-A3B-Thinking-2507 (same 180 cells, ChatML bridge) ===")
    qwen_table(D)

if "--mn4096" in sys.argv:
    print("\n=== gpt-oss-20b locked cell: budget re-anchoring (768 -> max_new 4096) ===")
    reanchor_table()
