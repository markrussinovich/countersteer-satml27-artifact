# Release verification — clean-install run-through (2026-09-10)

Verified the exported public release at `/datadrive/countersteer` from a fresh install,
following only its `README.md`, on `meta-llama/Llama-3.1-8B-Instruct`, one GPU
(`CUDA_VISIBLE_DEVICES=0`, A100 80GB).

## VERDICT: WORKS-WITH-FINDINGS

The install is clean, the quickstart sweep runs to completion from the release tree
alone and writes a valid artifact, and the canonical scorer prints the full headline
block with a coherent clean arm — but the scorer command **as printed in the README
fails out of the box** (finding 1), and the tree carries author-identifying anonymizer
config plus internal hardcoded-path scripts (findings 2–4).

## What was run

| step | command | result | wall clock |
|---|---|---|---|
| install | `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt` (verbatim from README) | exit 0, no resolver conflicts, no build failures, all pins resolved | 21m17s (inflated by intermittent DNS failures in this environment — pip retried and succeeded; not a release defect) |
| quickstart sweep | README's Llama-3.1-8B command, verbatim, with `CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/<USER>/.cache/huggingface` | exit 0; wrote `runs/llama31-8b/results_add-dim-no-override-achf-2920126.json` + `..._completions.json`, both parse | 1m00.5s (README says ≈10 min; A100 + cached weights) |
| scorer, as printed | `.venv/bin/python tools/controls/score_table.py runs/llama31-8b/results_add-dim-no-override-achf-*_completions.json` | **exit 1 — finding 1** | 6.6s |
| scorer + `--no-adjudicate` | same + the tool's own documented flag | exit 0, full headline block | ~7s |

Model weight **download was not exercised**: the Llama-3.1-8B weights were already in
the HF cache at `HF_HOME` (the meta-llama license gate makes a non-interactive fresh
download untestable). Resolution went through the standard HF cache mechanism the
README documents, and worked.

## Output sanity (n=8 dev slice, noisy by design)

- Clean arm pinned `corr 1.000` (it is the reference) — correct.
- `CLEAN+` (steered-clean) arm: `corr 1.000`, `utilBenign 100.0%/100%` — steering did
  not degrade the clean task at this slice.
- Defended arm not degenerate: acted on 7/8 samples (`noact 0.125`, one `no_action`),
  `goal 0.000`, no attacker-tool calls, `corr 0.750` (one drift, one no-action).
- Caveat the pipeline itself printed: at n=8 the base-XPIA arm's attack rate was 0.000
  (README's full-n dev expectation is ≈0.135, so 0/8 is within binomial noise), and the
  sweep honestly flagged `[INVALID] base-XPIA attack rate is 0.000`. The slice proves
  the pipeline, not the numbers — exactly as the README states.
- Cosmetic: `xunatt` prints `nanx` when the unattacked arm's `noact` is 0 (0/0).

## Findings

### 1. The README's scorer command fails on a fresh install (BREAKS QUICKSTART AS PRINTED)

`tools/controls/score_table.py`, run exactly as printed, exits 1:

```
[judge] XPIA_JUDGE_ENDPOINT is not set. The judge/adjudicator needs an Azure OpenAI endpoint, ...
```

Whenever any parameter drift exists in the artifact (at n=8 one steered-clean sample
drifted, so this is the common case), `score()` calls `adjudicate()` →
`src/judge.py:require_endpoint()` → `SystemExit`. The README's Notes say
`XPIA_JUDGE_ENDPOINT` is needed "for the adaptive-attack optimizer's judge" — it is
also needed by the quickstart's own scoring step. The tool ships an escape hatch,
`--no-adjudicate` (`tools/controls/score_table.py`, arg parsed at line ~810), which
produces the full headline block offline; the README never mentions it.
**Fix: add `--no-adjudicate` to the README quickstart scorer line (or make the scorer
degrade to no-adjudication with a warning when the endpoint is unset).**

### 2. Author-identifying strings shipped inside the anonymizer's own config — REMEDIATED 2026-09-10

The scrub tooling's pattern files (`tools/anon_email.re`, `tools/anon_identity.re`,
`tools/anon_rules.tsv`, and a comment in `tools/anon_scrub.py`) shipped with the export
and contained the author identity they scrub. `tools/anon_check.sh` passed (exit 0)
because these files are its own configuration and are effectively self-excluded.
**Remediation: the anonymizer and all of its pattern files were removed from the export**
(they are build-side tooling, not release content); the identifying literals originally
quoted in this finding were replaced by this note for the same reason.

### 3. Shipped code that hard-depended on internal absolute paths — REMEDIATED 2026-09-10 (all files below removed from the export)

Load-bearing paths in Python (not provenance comments):

- `tools/controls/attribution_positive_control.py:9,12` —
  `sys.path.insert(0, "/datadrive/xpia-steering/tools/controls")`;
  `RUN_DIR = "/datadrive/xpia-steering/runs/gpt-oss-20b-paper"` (a run dir that is not
  shipped at all)
- `tools/controls/framing_format_decomposition.py:13,16` — same pattern
  (`OUT = "/datadrive/xpia-steering/runs/gpt-oss-20b-userabl"`)
- `tools/controls/position_confound_control.py:15` —
  `sys.path.insert(0, "/datadrive/xpia-steering")`

Shell launchers with the same defect (internal one-off job scripts):
`tools/queue_next.sh`, `tools/launch_when_ready.sh`, `tools/run_smoke.sh`,
`tools/run_when_gpu_free.sh`, `tools/controls/autodojo_harvest.sh` (ssh's into fleet
hosts and rsyncs from `/datadrive/xpia-steering/...`).

None of these are on the README's documented paths (quickstart, bring-up, agentic
eval), so the release works without them — but they are shipped code that cannot run
outside the internal box.

### 4. Internal scratch: 12 hidden `tools/.frozen_glm_*.sh` files — REMEDIATED 2026-09-10 (removed)

Hidden dotfiles (`tools/.frozen_glm_T_pass.sh`, `.frozen_glm_disjoint_cont.sh`,
`.frozen_glm_lane.sh`, etc.) are frozen internal launch scripts, several containing
absolute `/datadrive/xpia-steering/...` paths. They read as internal scratch a public
release should not carry (a stranger listing `tools/` will not even see them).

### 5. Dangling cross-references to a `cluster/` directory that is not shipped — REMEDIATED 2026-09-10/13 (references removed or reworded; `ANONYMIZATION.md` dropped from the export)

- `requirements.txt` (header comment) explains itself relative to
  `cluster/requirements.txt`
- `ANONYMIZATION.md` points readers to "the setup section in `cluster/README.md`"

No `cluster/` directory exists in the export. Harmless for running, but the
ANONYMIZATION.md pointer is a reader-facing dead link. (`reference/` checkouts are
documented as fetched separately in the README — that one is fine.)

### 6. `tools/anon_check.sh`'s git-metadata leg is vacuous on the export

Run inside the export (not a git repo) it prints
`fatal: not a git repository` for the file/PDF/git scans yet still reports
`anon check OK (files, rendered PDF text, git metadata)` and exits 0. The gate should
fail loudly or skip explicitly rather than claim scans it could not run.

### 7. Minor / acceptable

- `runs/agentdojo_cells.json` and `runs/paper_disjoint_firm_dataset.dev.json` carry
  `/datadrive/xpia-steering/...` in provenance metadata fields (`screen`, `out`,
  `source_file`). Not load-bearing; acceptable provenance, noted for completeness.
- No internal IPs (RFC-1918 literals), no subscription GUIDs, no real
  Azure endpoints, and no OS username in file contents anywhere in the tree
  (`.venv` excluded). Endpoint use is via `XPIA_JUDGE_ENDPOINT` with no default.
- No stray `*.log`/`nohup` files ship.

## gpt-oss-20b alternative quickstart: CONSISTENT (artifacts only; model not run)

`runs/gpt-oss-20b-userabl/probe_L{12,16,20}.pkl` each contain the direction
`combo_ovr8_pat1` and a positive sigma for the `--match-sigma-to dim_no_override`
anchor (83.4 / 149.6 / 268.3 at L12/16/20). The Llama probes
(`runs/llama31-8b/probe_L{12,16,20}.pkl`) likewise carry `dim_no_override_achf` with
positive sigmas (0.473 / 0.764 / 0.998). Both quickstart lines are internally
consistent with the shipped artifacts.

## Modifications made to the release by this verification

Only: `.venv/` (created per README), the two `runs/llama31-8b/results_add-*` output
files the quickstart wrote, and this `VERIFICATION.md`.

---

# Addendum — 2026-09-13 refresh

The tree was refreshed from the working repository (staged copy scrubbed with the same
placeholder map before merging; the internal tree was not modified). What changed:

- **Code refresh** (12 files): the AgentDojo bridge/runners gained `--steer-mode`
  (projection/ablate operators) and per-cell robustness fixes; the AutoDojo
  target/scorer/plugins, `score_dojo_baselines.py`, `score_ipi_arena.py`, and the fixed
  `general_utility_bench.py` (persists full completions per arm with a stable sha256;
  earlier runs stored plain arms only).
- **New scripts**: `tools/controls/adaptive_sample_level_test.py`,
  `dojo_scrub_regrade.py`, `agri_gate.py`, `build_agri_probe.py`, plus five
  AML-cluster job drivers matching artifacts shipped under `runs/scrub_regrade/`.
- **New artifacts**: `runs/achT_endtoend/` (the framing-held-out refit's preregistered
  end-to-end test pass, four corpora + the AgentDojo 4-arm grid, per-sample completions
  and transcripts), `runs/scrub_regrade/` (typography-normalized offline regrades),
  `runs/general_utility_*.json` (GSM8K/MMLU/IFEval capability checks),
  `runs/agri_probe_*.json` (AGRI rival-gate probe specs), and
  `runs/adaptive_sample_level_tests.json` (paired sample-level inference for the
  adaptive query search).
- **Preregistrations**: new `prereg/` directory (`ach_endtoend_prereg.json`,
  `autodojo_prereg.json`), each written before its test touch.
- **Vendored AGRI port**: `reference/agri/` (probe pipeline of arXiv:2608.02657 with our
  marked patches; upstream `.venv`/`.git` excluded) and `reference/DEVIATIONS.md`, the
  paper-vs-port deviations table.
- **Refreshed directions**: `runs/qwen3-30b-thinking/probe_L*.pkl` now also carry the
  `dim_no_override_achf` / `dim_override_achf` framing-held-out refit keys (new keys;
  previously shipped keys unchanged).
- **Anonymization**: internal cluster platform name replaced with "AML-cluster" /
  `cluster/` tree-wide (including in files staged earlier); host letters kept consistent
  with the existing `<FLEET_HOST_B>` convention; identifying literals that this file
  itself quoted in finding 2 replaced with a remediation note.

Not included, pending: the AutoDojo adaptive-attack arms against the AGRI rival gate
(runs unsigned/still in flight at refresh time). Provenance metadata inside run
artifacts retains internal absolute paths per finding 7's accepted convention.
