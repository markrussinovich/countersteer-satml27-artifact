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
| `runs/` | fitted probe/direction pickles for gpt-oss-20b and Llama-3.1-8B, the webpage evaluation corpora (dev + held-out test splits), CachePrune masks, the AgentDojo cell list |

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

Runs the certified Llama-3.1-8B cell (direction `dim_no_override_achf`,
α=5, layers 12/16/20) on a small slice of the webpage tool-hijack corpus,
producing the four arms (clean / steered-clean / attacked / defended):

```bash
.venv/bin/python xpia_defense.py \
  --model meta-llama/Llama-3.1-8B-Instruct --stage sweep --device cuda:0 \
  --outdir runs/llama31-8b --corpus paper_disjoint --n-eval 8 \
  --directions dim_no_override_achf --alphas 5 --steer-layers 12,16,20 \
  --match-sigma-to dim_no_override_achf --steer-clean
```

Score the artifact it writes with the canonical scorer (the only source of
publishable numbers — it prints the severity-ordered headline block):

```bash
.venv/bin/python tools/controls/score_table.py --no-adjudicate \
  runs/llama31-8b/results_add-dim-no-override-achf-*_completions.json
```

(`--no-adjudicate` skips the optional LLM drift adjudicator, which needs an
Azure OpenAI endpoint in `XPIA_JUDGE_ENDPOINT`; all headline numbers are
computed deterministically without it.)

Expected shape at full n=52 (dev): attacked goal ≈ 0.135 → defended 0.000,
steered-clean correctness 1.000. Small `--n-eval` slices vary; they exist to
prove the pipeline, not to reproduce the numbers.

For gpt-oss-20b (ungated), substitute:
`--model openai/gpt-oss-20b --outdir runs/gpt-oss-20b-userabl
--directions combo_ovr8_pat1 --alphas 8.06 --match-sigma-to dim_no_override`.

## Fitting a direction for a new model

`bash tools/bringup_stage1.sh` runs capture → behavioral factorial → fit →
gates end-to-end (≈35 min for an 8B model on 3 GPUs; see the script header).
The recipe's gates are pass/fail and preregistered: reliability,
held-out-generalization AUC, and a bidirectional causal test (the direction
must both reduce attack success at +α and increase it at −α). Dose search
then locates a capability-safe deployment window; a dose only counts if the
no-action capability guard stays intact.

## Agentic evaluation

`tools/controls/agentdojo_smoke.py` (task screening + 3-arm smoke) and
`tools/controls/agentdojo_run.py` (full 180-cell, 4-arm batteries) drive
AgentDojo with the steered model, scored by AgentDojo's own checkers. The
AgentDyn fork and the AutoDojo adaptive-attack optimizer are driven through
the same bridge; see the module docstrings for the vendored-fork pinning
(`reference/` checkouts are fetched separately; pinned commits are named in
the docstrings).

## Notes

- Evaluation conventions (severity hierarchy, holdout doctrine, capability
  guard, both-instruments utility reporting) are documented in
  `tools/controls/score_table.py` and the module docstrings.
- Infrastructure identifiers (hosts, endpoints, storage) are placeholders of
  the form `<...>`; set your own endpoints where a script requires one
  (e.g. `XPIA_JUDGE_ENDPOINT` for the adaptive-attack optimizer's judge).
