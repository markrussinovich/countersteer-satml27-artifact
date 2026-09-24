# CounterSteer

**Neutralizing indirect prompt injection with activation steering.**

CounterSteer defends LLM agents against indirect prompt injection (IPI) by
suppressing, at inference time, the model's internal decision to treat
untrusted retrieved text as instructions. Per model, it fits a residual-stream
direction from behavioral contrasts (paired episodes identical except for
whether an embedded instruction is followed), validates it through
reliability, held-out-generalization and bidirectional causal gates, and
subtracts it from **every tool-result token during prefill** — uniformly,
with no attempt to detect which tokens are malicious. The edit is always on:
there is no detector decision for an attacker to flip, no auxiliary model, no
added inference calls or prompt tokens. The serving stack only needs to know
where tool output *is* (span boundaries it already tracks), never whether it
is poisoned.

This repository contains the complete implementation: corpus builders, the
direction-fitting recipe with its gates, the steering runtime, the evaluation
harnesses (single-turn corpora, AgentDojo/AgentDyn bridges, adaptive attacks),
the canonical scorers, and fitted directions for the evaluated models.

## Layout

| path | contents |
|---|---|
| `xpia_defense.py` | main entry point: probe capture, direction fitting, sweep/confirm evaluation stages |
| `src/` | steering runtime, corpora, chat-format rendering/parsing (harmony, ChatML, GLM-4.5, Gemma-4, Llama-3.1, Phi-3), scoring |
| `tools/controls/` | canonical scorer (`score_table.py`), corpus builders, AgentDojo/AgentDyn bridge + runners, adaptive-attack harnesses, baselines (CachePrune port, detector filters, SecAlign evaluation) |
| `tools/bringup_stage1.sh` | one-command per-model bring-up: capture → factorial → fit → gates |
| `configs/` | one JSON per model: the certified deployed cell (direction, dose, layers, sigma convention, budgets, firing corpora) for all five evaluated models |
| `evaluate.sh` | one-command evaluation runner: `./evaluate.sh configs/<model>.json` (single-turn corpora + canonical scoring; `--agentic` for the AgentDojo battery) |
| `runs/` | fitted probe/direction pickles for **all five models**, all evaluation corpora (webpage dev + held-out test, JSON parameter-abuse with its probe/dev/heldout/test splits and held-out attacker-template sets, LLMail replay set), CachePrune masks, the AgentDojo cell list, and the per-sample eval artifacts behind the reported tables (`achT_endtoend/`, `scrub_regrade/`, `general_utility_*.json`, `agri_probe_*.json`, `adaptive_sample_level_tests.json`, the GCG adaptive-attack runs `gcg_n52/` / `gcg_param/` / `gcg_param_rsn/`, the symmetrized test-split baselines `symtest/`, the AGRI-rival AgentDojo battery `agri_battery_r2/`, and the AutoDojo-vs-AGRI adaptive roots `autodojo/agri/` and `autodojo/agri_qwen/`); the full family-by-family map is the "Shipped run artifacts" table below |
| `prereg/` | preregistrations written before their test touches: the framing-held-out refit's end-to-end certification (`ach_endtoend_prereg.json`) and the AutoDojo adaptive evaluation (`autodojo_prereg.json`) |
| `reference/` | vendored third-party code: `reference/agri/` (the AGRI probe pipeline, arXiv:2608.02657, with our marked patches; `reference/DEVIATIONS.md` is the paper-vs-port deviations table). Other `reference/` checkouts (AgentDyn, AutoDojo, ipi-arena, rc-paper) are fetched separately at the public commits pinned in the module docstrings; AutoDojo additionally needs the shipped patch series `reference/autodojo-patches/` (our plugin seam) applied with `git am` on the pinned upstream commit |

## Install

Python 3.12, one CUDA GPU (a 24 GB card suffices for the Llama-3.1-8B
quickstart; gpt-oss-20b wants 80 GB). Pinned versions are the ones that
produced the paper's numbers.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model weights resolve through the standard Hugging Face cache; set `HF_HOME`
if you keep one elsewhere. `openai/gpt-oss-20b` is ungated;
`meta-llama/Llama-3.1-8B-Instruct` requires accepting the license on the Hub.

## Quickstart: evaluate a shipped defense (≈10 minutes on one GPU)

Every model's certified deployed cell ships as a config; the runner executes
the four-arm evaluation (clean / steered-clean / attacked / defended) and
scores it with the canonical scorer (the severity-ordered headline block):

```bash
./evaluate.sh configs/llama3.1-8b.json --n-eval 8          # smallest model, 24GB GPU
./evaluate.sh configs/gpt-oss-20b.json --n-eval 8          # ungated, 80GB GPU
./evaluate.sh configs/qwen3-30b.json  --corpus paper_param # any firing corpus
./evaluate.sh configs/llama3.1-8b.json --agentic           # AgentDojo 4-arm battery
```

`--n-eval 52 --stage sweep` reproduces a development rung; `--stage confirm`
touches the held-out test split (one preregistered pass — do not rerun).
Small slices prove the pipeline, not the numbers. The scorer's optional LLM
drift adjudicator needs `XPIA_JUDGE_ENDPOINT`; every headline number is
deterministic without it. Configs for all five models (gpt-oss-20b,
Qwen3-30B, Gemma-4-31B, GLM-4.5-Air, Llama-3.1-8B) are in `configs/`, each
carrying its firing corpora and budget conventions; GLM-4.5-Air needs
3–4×80GB GPUs (`device=auto`).

## Fitting a direction for a new model

`bash tools/bringup_stage1.sh` runs capture → behavioral factorial → fit →
gates end-to-end (≈35 min for an 8B model on 3 GPUs; see the script header).
The recipe's gates are pass/fail and preregistered: reliability,
held-out-generalization AUC, and a bidirectional causal test (the direction
must both reduce attack success at +α and increase it at −α). Dose search
then locates a capability-safe deployment window; a dose only counts if the
no-action capability guard stays intact.

**Reproducing the shipped Llama-3.1-8B direction exactly** (the deployed
`dim_no_override_achf` vector uses the FULL factorial design with
action-centring and the `firm` override level excluded, which
`bringup_stage1.sh`'s default `_both`-on-`locked24` path does not):

```bash
M=meta-llama/Llama-3.1-8B-Instruct; RUN=runs/llama31-8b-refit; TAG=llama31-8b-refit-full
python xpia_defense.py --model $M --stage probe    --outdir $RUN --skip-ovr
python xpia_defense.py --model $M --stage validate --outdir $RUN --skip-ovr
# behavioral factorial on the SHIPPED probe split (runs/param_abuse_dataset.probe.json),
# shardable with --shard I --nshard N:
python tools/controls/override_slope_experiment.py --split probe --n 24 \
  --model $M --probe-run $RUN --tag $TAG --batch 8 --max-new 1024
python tools/controls/override_slope_experiment.py --merge   --tag $TAG
python tools/controls/override_slope_experiment.py --analyze --tag $TAG
python tools/controls/build_override_direction.py $RUN runs/override_slope_$TAG.json \
  --key-suffix _achf --exclude-override firm --centre-action
```

Verified twice from fresh installs (2026-09-14 and 2026-09-24): cosine vs
the shipped `runs/llama31-8b` pickles = +1.0000 at L12/L16/L20, sigmas
identical to four decimals (≈44 min on two A100s). Note: `agri_battery_r2`
scores through `tools/controls/score_dojo_soa.py` via symlinks named
`<label>_mn4096.shardN.json` (the scorer's glob); the AutoDojo prereg lives
at `prereg/autodojo_prereg.json` (older docstrings say `runs/autodojo/`).

## Evaluating rival defenses

The same battery machinery evaluates the baselines the paper compares
against, each with its own same-process undefended anchor:

```bash
./evaluate.sh configs/gpt-oss-20b.json --defense pi_detector_piguard   # PIGuard filter
./evaluate.sh configs/gpt-oss-20b.json --defense spotlighting_with_delimiting
./evaluate.sh configs/gpt-oss-20b.json --kv-mask runs/cacheprune_mask.json  # CachePrune
```

Detector checkpoints resolve from the Hub at pinned revisions (PromptGuard-2
is gated; accept its license or set `XPIA_MODEL_STORE` to a directory holding
the checkpoint). The adaptive-attack (AutoDojo) arms for CounterSteer and
every rival are in `tools/controls/autodojo_job.sh` (arms: `defended`,
`cacheprune`, `promptguard`, `piguard`, `deberta`, and `--arm dojodef
--dojo-defense <inbuilt>`); they require the pinned AutoDojo fork under
`reference/autodojo` and an attacker-LLM endpoint (`XPIA_JUDGE_ENDPOINT`).
SecAlign is supported as an evaluation target (a different model through the
same batteries); its DPO training is described in the paper and its
checkpoint is not distributed.

## Agentic evaluation

`tools/controls/agentdojo_smoke.py` (task screening + 3-arm smoke) and
`tools/controls/agentdojo_run.py` (full 180-cell, 4-arm batteries) drive
AgentDojo with the steered model, scored by AgentDojo's own checkers. The
AgentDyn fork and the AutoDojo adaptive-attack optimizer are driven through
the same bridge; see the module docstrings for the vendored-fork pinning
(`reference/` checkouts are fetched separately; pinned commits are named in
the docstrings).

## Results (paper Table I: AgentDojo, defenses × models)

Compromise rate (AgentDojo's own security checker over the 180-case grid) and benign
utility (% of the same model's clean arm, typography-normalized):

| arm | gpt-oss-20b | Qwen3-30B | Gemma-4-31B | GLM-4.5-Air | Llama-3.1-8B |
|---|---|---|---|---|---|
| no defense (cmp / util) | .483 / 100 | .489 / 100 | .239 / 100 | .183 / 100 | .100 / 100 |
| **CounterSteer** (cmp / util) | .091 / 90.6 | .072 / 94.1 | .006 / 92.9 | .056 / 102.2 | .050 / 100.0 |

These are the comparison batteries; the same-configuration certification runs (quoted in
the paper's abstract) read 0.006–0.079 compromise at 91–100% utility, with gpt-oss-20b at
.475 → .079 (94.4%). Under the adaptive attacker (AutoDojo AD@6: success at six
optimization iterations) the undefended/defended rates are .673/.175 (gpt-oss-20b),
.730/.188 (Qwen3-30B), .377/.307 (Llama-3.1-8B). Rival-defense rows and per-model
batteries are in the paper; the artifacts behind them ship under `runs/`.

## Analysis and audit tooling (beyond the batteries)

- `tools/controls/dojo_scrub_regrade.py` — typography-normalized offline regrade of a
  stored AgentDojo battery (Unicode space/dash → ASCII, uniformly across arms) through
  AgentDojo's own unmodified checkers, with a per-episode raw-replay fidelity gate;
  zero GPU, needs the `.transcripts.json` siblings. Artifacts: `runs/scrub_regrade/`.
- `tools/controls/general_utility_bench.py` — capability-benchmark harness
  (GSM8K/MMLU/IFEval), four arms in one process (plain/tool × baseline/steered),
  greedy; persists full completions per arm with a stable sha256. Artifacts:
  `runs/general_utility_*.json`. Companion `tools/controls/gub_failure_probe.py`
  regenerates flip-enriched failure samples for anatomy (batch=1: item-level flips are
  batch-composition-sensitive).
- `tools/controls/adaptive_sample_level_test.py` — sample-level paired inference for
  the adaptive query search (attempts nest within samples, so attempt-level tests are
  anti-conservative): exact sign-flip permutation test on paired per-sample means plus
  a paired bootstrap CI. Artifact: `runs/adaptive_sample_level_tests.json`.
- `tools/controls/adaptive_gcg.py` — GCG adaptive-attack harness against the steering
  defense (white-box / surrogate-transfer / surrogate-defense-transfer arms;
  `--dataset param_abuse` teacher-forces the param-hijack tier-1 event, and
  `--objective reasoned` scores the CE at the reasoned-call offset every real compromise
  uses). Artifacts: `runs/gcg_n52/` (webpage corpus, n=52), `runs/gcg_param/` (param
  class, forced offset), `runs/gcg_param_rsn/` (param class, reasoned offset).
  `runs/gcg_param_rsn/seam_replay/` is a disclosure sidecar: a fresh-process static-arm
  replay (script + 4 outputs) showing the two param-run static arms differ only by
  process-history numerics.
- `runs/symtest/` — test-split baseline symmetrization: CachePrune and the add-combo
  deployed cell re-run on the held-out test split in the same process as their own
  undefended anchors (result + full-completions files per job).
- `runs/agri_battery_r2/` — the AGRI rival arm's AgentDojo battery (gpt-oss-20b and
  Qwen3-30B, 8 shards each, JSON + progress logs; scored by AgentDojo's own checkers).
  Transcript siblings are not shipped for this battery.
- `tools/controls/agri_gate.py` — the AGRI rival arm (arXiv:2608.02657): probe-gated
  anti-injection reasoning prefill, implemented from their released probe pipeline
  (`reference/agri/`) plus the paper's intervention spec; loads the spec JSONs under
  `runs/agri_probe_*.json`. `tools/controls/build_agri_probe.py` converts an AGRI
  probe checkpoint into that deployable spec. Port deviations: `reference/DEVIATIONS.md`.

## Shipped run artifacts (`runs/`): the per-sample completions behind the paper's tables

One line per measurement family; every file is the run's own per-sample output
(result JSON beside its `_completions`/`.transcripts` siblings where the harness
writes them). Paths mirror the provenance comments in the paper sources.

| family (paths under `runs/`) | backs |
|---|---|
| `gpt-oss-20b-userabl/results_confirm_add-combo-ovr8-pat0-combo-ovr8-pat1-38723{77,78,79,80}*` | gpt-oss held-out test pass, 4 corpora (per-model detail + multi-model + flagship tables) |
| `qwen3-30b-thinking/results_confirm_add-dim-no-override-both-{3310513,3310631,809909,810027}*` + dev `...-871922*` and 16σ dev `...-32339{22..25}*` | Qwen3-30B test rung + development counterparts |
| `glm45-air/results_confirm_add-dim-no-override-actioncentred-1436291*` (T), `...-1220531*` (dev), `...-3599721*` (multi-turn boundary), `...-8815*` (screening diagnostics), `glm_dojo_full*` | GLM-4.5-Air certification + its AgentDojo grid |
| `gemma4-31b-it/cert_harvest/` + `results_confirm_add-dim-no-override-both-{3489031,3532482}*`, `results_add-...-639808*` | Gemma-4 held-out certification, T* rung, prose-carrier null |
| `llama31-8b/results_confirm_add-dim-no-override-achf-2373033*` + `llama31_agentdojo_run.shard{0..3}*` | Llama-3.1-8B certification + its AgentDojo grid |
| `dojo_soa_gptoss/`, `dojo_soa_qwen/`, `dojo_soa_llama31/`, `dojo_filters_glm/` | Table I same-process 4-arm AgentDojo batteries + rival-defense arms |
| `judge_utility/`, `soa_fidelity/` | pairwise-judge per-cell artifacts, judge validation controls, fidelity table |
| `dojo_full_mn4096/`, `agentdojo_censor_rerun.json*` | gpt-oss AgentDojo certification grid (0.475 → 0.079) |
| `dose_gptoss/` | gpt-oss AgentDojo dose curve (dose table + figure) |
| `qwen_ad_a1{0,2,4}_def.shard*` / `qwen_ad_a1{0,2,4}_benign.json`, `qwen_agentdojo_run.shard*`, `dojo_baselines_qwen/` | Qwen AgentDojo dose curve + legacy baseline rows |
| `agentdojo_secalign*`, `agentdojo_cacheprune.shard*`, `agentdojo_combo_v2.shard*` | SecAlign / CachePrune AgentDojo shards beside the deployed arm (768-budget) |
| `agentdojo_run.shard{0..3}*`, `dojo_baselines/`, `dojo_baselines_mn4096/`, `dojo_gaps_*.manifest.json` | gpt-oss 768-budget four-arm run and inbuilt-defense baselines (legacy panels; delegated-authority partition table) |
| `cacheprune/`, `gptoss20b-secalign-eval/`, `secalign_pairs.{train,eval}.jsonl` + `.stats.json` | single-turn baseline runs; SecAlign DPO training pairs (the checkpoint itself is not distributed) |
| `agentdyn_harvest/`, `agentdyn_grid_qwen/`, `cluster_presync/xpia-agentdyn-qwen-ipi/`, `agentdyn_rivals/`, `dose_harvest/named-outputs/blob/`, `h100_delta_var/`, `agentdyn_cells*.json`, `agentdyn_screen_llama31.json`, `agentdyn_smoke_llama31.json` | AgentDyn 180-case grids (gpt-oss, Qwen, GLM), rival arms (PromptGuard/PIGuard/SecAlign/CachePrune), dose-frontier arm, fabric-variance replicates |
| `autodojo/full2{,_capped}/` (per-arm `injections.json` + `run_cost.json`, prompt logs, LLM call caches), `autodojo/harvest2/merged/`, `autodojo/harvest4/merged/`, `autodojo/final_merge_score.txt`, fraud-label files | AutoDojo adaptive matrix (AD@6) and its audits; `harvest2/merged` is the signed-off canonical merge, `harvest4/merged` the final full pull |
| `adaptive_framing.shard*`, `qwen_adaptive_framing.shard*`, `adaptive_param_{none,spotlight}.json`, `adaptive_param_cacheprune.shard*`, `secalign_adaptive_{param.shard*,tool}.json` | defense-aware adaptive query attacks (framing search, five-arm parameter-manipulation table) |
| `whitebox_deployed/`, `gcg_harvest/named-outputs/blob/`, `qwen_adaptive_leg2b.*`, `qwen_adaptive_leg3.*` (beside the shipped `gcg_n52/`, `gcg_param/`, `gcg_param_rsn/`) | GCG white-box / surrogate arms and the Qwen adaptive legs |
| `llmail_replay/` (gpt-oss dev+test shards), `llmail_qwen_scale/` | LLMail-Inject replay (0/2052 gpt-oss, 0/1537 Qwen) |
| `relay_multiturn.shard{0..3}.json`, `gpt-oss-20b-userabl/results_add-combo-ovr8-pat1-30533{80..83}*` | relay-attack rung |
| `gpt-oss-20b-userabl/results_add-combo-ovr8-pat1-scopeP4-*-62771{2,3}*` | search-budget appendix's deployed readings |

Not shipped, with cause: the raw LLMail-Inject challenge dumps (public upstream
dataset, >1.9 GB; the replay corpora `llmail_dataset.{dev,test}.json` and every
replay output are shipped); the AutoDojo per-host harvest snapshots superseded
by the deduplicated `merged/` views scored in `final_merge_score.txt`; and the
`agri_battery_r2/` transcript siblings (noted above).

## Notes

- Evaluation conventions (severity hierarchy, holdout doctrine, capability
  guard, both-instruments utility reporting) are documented in
  `tools/controls/score_table.py` and the module docstrings.
- Infrastructure identifiers (hosts, endpoints, storage) are placeholders of
  the form `<...>`; set your own endpoints where a script requires one
  (e.g. `XPIA_JUDGE_ENDPOINT` for the adaptive-attack optimizer's judge).
