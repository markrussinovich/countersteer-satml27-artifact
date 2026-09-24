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
| `runs/` | fitted probe/direction pickles for **all five models**, all evaluation corpora (webpage dev + held-out test, JSON parameter-abuse with its probe/dev/heldout/test splits and held-out attacker-template sets, LLMail replay set), CachePrune masks, the AgentDojo cell list, and the per-sample eval artifacts behind the reported tables (`achT_endtoend/`, `scrub_regrade/`, `general_utility_*.json`, `agri_probe_*.json`, `adaptive_sample_level_tests.json`, the GCG adaptive-attack runs `gcg_n52/` / `gcg_param/` / `gcg_param_rsn/`, the symmetrized test-split baselines `symtest/`, the AGRI-rival AgentDojo battery `agri_battery_r2/`, and the AutoDojo-vs-AGRI adaptive roots `autodojo/agri/` and `autodojo/agri_qwen/`) |
| `prereg/` | preregistrations written before their test touches: the framing-held-out refit's end-to-end certification (`ach_endtoend_prereg.json`) and the AutoDojo adaptive evaluation (`autodojo_prereg.json`) |
| `reference/` | vendored third-party code: `reference/agri/` (the AGRI probe pipeline, arXiv:2608.02657, with our marked patches; `reference/DEVIATIONS.md` is the paper-vs-port deviations table). Other `reference/` checkouts (AgentDojo/AgentDyn/AutoDojo forks) are fetched separately at the commits pinned in the module docstrings |

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

## Notes

- Evaluation conventions (severity hierarchy, holdout doctrine, capability
  guard, both-instruments utility reporting) are documented in
  `tools/controls/score_table.py` and the module docstrings.
- Infrastructure identifiers (hosts, endpoints, storage) are placeholders of
  the form `<...>`; set your own endpoints where a script requires one
  (e.g. `XPIA_JUDGE_ENDPOINT` for the adaptive-attack optimizer's judge).
