"""Figures for the 2026-08-26 cross-corpus defense report.

Numbers are transcribed from the canonical scorer's output on the named artifacts (every
value's source artifact is in the DATA comments); the report text carries the same tables.
Palette: the dataviz skill's validated reference instance (light mode), categorical slots
in fixed order by ENTITY: no-defense=blue, deployed=orange, composed=aqua, role-solo=yellow.

Usage: .venv/bin/python tools/controls/make_report_figs.py   -> runs/figs/report_*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"attacked": "#2a78d6", "deployed": "#eb6834", "combo": "#1baf7a",
     "solo": "#eda100", "text": "#0b0b0b", "sub": "#52514e", "surf": "#fcfcfb",
     "grid": "#e5e4e0"}
plt.rcParams.update({
    "figure.facecolor": C["surf"], "axes.facecolor": C["surf"],
    "text.color": C["text"], "axes.edgecolor": C["sub"],
    "axes.labelcolor": C["text"], "xtick.color": C["sub"], "ytick.color": C["sub"],
    "font.size": 11, "axes.titlesize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.axisbelow": True})
os.makedirs("runs/figs", exist_ok=True)


def bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2,
                    b.get_height()), ha="center", va="bottom", fontsize=9,
                    color=C["text"])


# ── FIG 1: attacker goal rate, per corpus, per arm ─────────────────────────────
# sources: shipped head-to-head 3602037 (same process); JSON param 3644432 vs
# BEST_DEFENSE locked 0.138 (cross-process, dagger in caption); paper_param 3594412
# vs 3516200; paper_disjoint smoke + 3516111.
corpora = ["Nemotron JSON\n(extra tool)", "JSON\nparam-abuse",
           "webpage\nparam-abuse", "webpage\nextra tool"]
# TEST-RUNG numbers (FINDINGS 10k lock; confirm artifacts 3872377-80). Webpage
# extra-tool column is MULTI-TURN goal (deployed 3/52; combo 0/49 measured).
goal = {"attacked": [0.675, 0.591, 0.962, 0.981],
        "deployed": [0.037, 0.106, 0.500, 0.058],
        "combo":    [0.013, 0.045, 0.135, 0.000]}
labels = {"attacked": "no defense", "deployed": "deployed defense (override only)",
          "combo": "composed defense (override + role axis)"}
fig, ax = plt.subplots(figsize=(9, 4.4))
x, w = np.arange(len(corpora)), 0.26
for i, k in enumerate(("attacked", "deployed", "combo")):
    b = ax.bar(x + (i - 1) * (w + 0.02), goal[k], w, color=C[k], label=labels[k],
               zorder=3)
    bar_labels(ax, b)
ax.set_xticks(x, corpora)
ax.set_ylabel("attacker goal-success rate  (lower is better)")
ax.set_ylim(0, 1.08)
ax.set_title("Attack success by corpus and defense -- HELD-OUT TEST SPLITS\n"
             "(webpage extra-tool column is multi-turn goal)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("runs/figs/report_goal_by_corpus.png", dpi=160)
plt.close(fig)

# ── FIG 2: the benign-utility bracket, three instruments per cell ──────────────
# strict = struct_exact CLEAN+; judge = GPT-4.1 pairwise at-most-minor (EQUIV+MINOR);
# lenient = corrCompGoalOK CLEAN+. sources: score_table + utility_judge artifacts.
cells = ["deployed\nshipped JSON", "combo\nshipped JSON",
         "deployed\nwebpage param", "combo\nwebpage param"]
strict = [0.605, 0.605, 0.635, 0.686]
# judge = GPT-5.4 v3 protocol (forward-primary, HTML-stripped record), CLEAN+ arms,
# at-most-minor (EQUIV+MINOR); sources: runs/utility_judge_v3_*.json
judge = [0.802, 0.813, 0.924, 0.912]
lenient = [0.88, 0.86, 0.64, 0.69]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
y = np.arange(len(cells))
for i in range(len(cells)):
    lo = min(strict[i], lenient[i], judge[i])
    hi = max(strict[i], lenient[i], judge[i])
    ax.hlines(y[i], lo, hi, color=C["grid"], lw=6, zorder=2)
ax.scatter(strict, y, s=90, color=C["attacked"], zorder=3, label="strict (byte-exact fields)")
ax.scatter(lenient, y, s=90, color=C["solo"], zorder=3, label="lenient (composed prose exempt)")
ax.scatter(judge, y, s=130, color=C["combo"], zorder=4, marker="D",
           label="GPT-5.4 pairwise judge, at-most-minor (flagged rates)")
ax.set_yticks(y, cells)
ax.invert_yaxis()
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("benign utility: fraction of un-attacked traffic still served  (higher is better)")
ax.set_title("Benign utility under three yardsticks — mechanical readings misfire in\n"
             "opposite directions; the judge reads content")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
          fontsize=9)
fig.tight_layout()
fig.savefig("runs/figs/report_utility_bracket.png", dpi=160)
plt.close(fig)

# ── FIG 3: dose-response of the role component (webpage param corpus, n=24) ────
# sources: smokes 8:0(=deployed n=24 3516200-era 0.542), 8:1, 8:1.5, 8:2, 8:4, 0:4.
wts = [0, 1, 1.5, 2, 4]
goal_w = [0.542, 0.083, 0.083, 0.000, 0.000]
noact_w = [0.125, 0.080, 0.040, 0.540, 0.790]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
a1.plot(wts, goal_w, "-o", color=C["combo"], lw=2, ms=8, zorder=3)
a1.set_ylabel("attacker goal rate")
a1.set_xlabel("role-axis weight (override fixed at 8)")
a1.set_title("attack suppression")
a1.set_ylim(-0.03, 0.85)
a2.plot(wts, noact_w, "-o", color=C["deployed"], lw=2, ms=8, zorder=3)
a2.axhspan(0.5, 0.85, color="#f6e2dc", zorder=1)
a2.annotate("capability collapse", (2.15, 0.66), color=C["sub"], fontsize=9)
a2.set_ylabel("no-action rate (guard)")
a2.set_xlabel("role-axis weight (override fixed at 8)")
a2.set_title("capability cost")
a2.set_ylim(-0.03, 0.85)
fig.suptitle("The role component's dose window (webpage param-abuse, n=24 smokes)", y=1.02)
fig.tight_layout()
fig.savefig("runs/figs/report_dose_window.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote runs/figs/report_goal_by_corpus.png, report_utility_bracket.png, "
      "report_dose_window.png")


# ── FIG 4: standard utility vs utility under attack (GPT-5.4 judge, goal-conditioned) ──
# benign = CLEAN+ arm at-most-minor; under attack = defended arm at-most-minor over
# NOT-goal-compromised samples only (tier-1 never double-counted as tier-2).
# sources: runs/utility_judge_v3_{combo-ovr8-pat0-combo-ovr8-pat1-3602037,
# combo-ovr8-pat1-3594412, dim-no-override-both-3516200, combo-ovr8-pat1-3644432}.json
cells4 = ["deployed\nshipped JSON", "combo\nshipped JSON",
          "deployed\nwebpage param", "combo\nwebpage param", "combo\nJSON param"]
benign4 = [0.802, 0.813, 0.924, 0.912, 0.773]
attack4 = [0.633, 0.572, 0.816, 0.838, 0.793]
fig, ax = plt.subplots(figsize=(9, 4.2))
x4, w4 = np.arange(len(cells4)), 0.34
b1 = ax.bar(x4 - w4 / 2 - 0.01, benign4, w4, color=C["combo"], zorder=3,
            label="standard utility (defense on, no attack)")
b2 = ax.bar(x4 + w4 / 2 + 0.01, attack4, w4, color=C["solo"], zorder=3,
            label="utility under attack (uncompromised samples only)")
bar_labels(ax, b1)
bar_labels(ax, b2)
ax.set_xticks(x4, cells4)
ax.set_ylabel("at-most-minor loss vs clean reference  (higher is better)")
ax.set_ylim(0, 1.12)
ax.set_title("Standard utility vs utility under attack -- GPT-5.4 judge, "
             "goal-conditioned (flagged rates)")
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("runs/figs/report_utility_split.png", dpi=160)
plt.close(fig)
print("wrote runs/figs/report_utility_split.png")
