# AGRI port — deviations table (paper/released code vs ours)

AGRI = Action-Guiding Reasoning Intervention, arXiv:2608.02657 ("Your Agentic LLMs
Secretly Encode Latent Signals of Indirect Prompt-Injection Exposure"), code vendored at
`reference/agri` (upstream github.com/jianshuod/IPI-exposure-signal, commit
03908cce41e4dfb3d32ff5ee6be51ad4195f3171, 2026-08-05). The released
repo contains the PROBE pipeline only; the intervention (gate + reasoning prefill) is
"released upon acceptance" and is implemented here from the paper's specification.

| # | theirs (paper / release) | ours | status |
|---|---|---|---|
| 1 | probe trained on full-8 grid: {slack,workspace} × {direct, long_horizon_important_instructions} × {default, safety_reminder_balanced} | full-4 (`expand_suite_attack`): same suites/attacks, `default` system message only — the released `vendor/agentdojo/.../system_messages.yaml` carries ONLY `default`; unknown names silently fall back to default, so the safety-reminder level is NOT reproducible from the release | OPEN (release gap, not closable without inventing their prompt text) |
| 2 | intervention code (gate, prefill mechanics, 3-turn window) released "upon acceptance" — not in the repo | implemented from the paper: threshold t=0.5 on sigmoid(z), prefill for the firing turn + next 3 assistant turns, prefill text VERBATIM from paper Tables 14/15 | OK (paper-specified; parameters auditable in the spec JSON) |
| 3 | probe score computed during the SAME prefill pass as generation (their patched vLLM capture) | one extra scoring forward per assistant turn, stopped at the probe layer (same tokens, same site: decoder-block output, post-assistant token) — compute-only difference, numerically the same read | OK |
| 4 | collection served by their patched vLLM 0.19.0 (patch needed only for the vllm_direct featurize backend) | plain vLLM 0.19.1 serve (a separate serving venv), featurize via their `transformers_hook` backend (bf16, matches both their featurize loader and our deployment dtype) | OK |
| 5 | 3 trajectories per attack instance, temperature 1.0 | same (`--epochs 3 --temperature 1`) | OK |
| 6 | training: linear, single_layer, lr 1e-4, batch 64, 5 epochs, standard z-score, no validation-based checkpoint selection (paper "main probing configuration"; the released example script instead defaults lr 3e-4 / batch 256 / best_val) | paper values: lr 1e-4, batch 64, 5 epochs, `checkpoint_mode: final` | OK (paper config chosen over example-script defaults) |
| 7 | gpt-oss deployed layer: L20/24 (their held-out-strict selection) | layer sweep 0..23 trained; deploy layer 20 (their published selection); our own val AUROC reported beside it | OK |
| 8 | labeling protocol risk_faced (positive = first assistant turn facing each injection round) | same (`--labeling-protocol risk_faced`) | OK |
| 9 | their defense eval: 8 hardest grid points from held-out banking/travel, 3 samples at temp 1.0, AgentDojo checkers | OUR harness smoke: 17 banking/travel `important_instructions` cells (held-out suites), greedy decoding, 4-arm battery (clean/cleanplus/attacked/defended), mn4096, `--system yaml`. YAML is deliberate — the probe trained under AgentDojo's own default system message (= the yaml text); the short SYSTEM constant would be a train/deploy shift. CONSEQUENCE (review 2026-09-11): the prior SoA battery rows ran the SHORT system message, so this row is WITHIN-RUN comparable only (its own clean/attacked anchors); it is not directly comparable to the short-system baseline rows nor to their Table 3 absolute numbers | OK (labelled) |
| 9b | their collection sampling reportedly used top_p 0.95 with temp 1.0 | the released collect CLI cannot set top_p (collect_agentdojo_v2 passes none → server default 1.0) — a release gap; we run temp 1.0 / top_p 1.0 | OPEN (release gap, disclosed) |
| 10 | featurize prefers STORED prompt_token_ids (their patched-vLLM capture); in their system the deployed gate also reads the vLLM serving render, so train and deploy see the SAME prompt bytes | **MEASURED PORT-BOUNDARY LADDER, all four fixed (2026-09-12/13), each caught by the capture-parity gate:** (a) RENDER PROVENANCE — plain vLLM 0.19.1 also returns prompt token ids, so featurize silently trained on the vLLM harmony render (Tools-before-Instructions, 680 tok) while our bridge deploys the HF chat-template render (630 tok); gate-vs-featurized max\|dp\|=0.59. (b) THE RELEASED FLAG IS A NO-OP ON THIS BACKEND — `--rebuild-prompt-token-ids-from-replay` is wired only into the vllm_direct path (cli.py:865); on transformers_hook it is accepted and silently ignored (their §23e-class defect). Fixed by a marked one-line local patch in the vendored copy honoring the flag's documented semantics (strip stored ids so build_probe_prompt takes its chat-template fallback). (c) STACK DRIFT — their venv (transformers 4.57.6/torch 2.10) vs deployment venv (5.14.1/2.7.1) shifts block-20 hidden states cos 0.9994 / rel-L2 3.4%, moving a borderline score 0.545→0.610; fixed by featurizing IN the deployment venv. (d) BATCHED CAPTURE — batch-8 right-padded capture vs the deployment's single-sequence forward: max\|dp\|=0.079 over 16 points (unconfounded, same venv/tokens); fixed by batch-1 featurize. With all four fixed: capture parity EXACTLY 0.0 on 16/16 points (deterministic forwards, byte-identical tokens). Earlier probe artifacts (`features/default`, `features/hfrender`, `probes/esa_risk_faced_L*`, `probes/esa_risk_faced_hfrender`) stay on disk for the record; the deployed probe is the `hfr2` fit. Also RETRACTED: an interim claim that batching alone explained the residual — that comparison was confounded by (b) | FIXED (patch logged; gate enforced before any smoke number) |

Holdout statement for the smoke (per CLAUDE.md axes; CORRECTED per adversarial review
2026-09-11): samples — held out (evaluation cells are banking/travel tasks; the fit is
slack/workspace only); suites — held out (same fact); attacker template — **SHARED
FAMILY, disclosed**: the fit's `long_horizon_important_instructions` is literally the
`important_instructions` jailbreak wording with a benign lead-in prepended
(reference/agri/vendor/agentdojo/src/agentdojo/attacks/long_horizon_attacks.py), so the
probe saw the evaluation attack's template verbatim as a suffix. This matches the
AUTHORS' held-out-group taxonomy but NOT this repo's template-disjoint standard; any AGRI
row is labelled accordingly. Framing level — n/a for a rival baseline (the probe is
theirs).

## AGRI-Qwen program (Qwen/Qwen3-30B-A3B-Thinking-2507) — decisions

- Collect: same full-4 recipe as gpt-oss (their code, vendored AgentDojo, vLLM 0.19.1
  server on <FLEET_HOST_C>, hermes tool parser + qwen3 reasoning parser). 3,972/3,990 traces
  (18 failures = 0.45%, matching their paper's 0.49% discard rate), 15,426 decision
  points, risk_faced positive rate 0.286.
- Model family for featurize: `qwen3.5` (position offset -2), NOT the name-inferred
  `qwen3`: the Thinking-2507 chat template's generation tail is token-identical to
  their Table-19 qwen3.5 row (`<|im_start|>assistant\n<think>\n`, ids
  [151644,77091,198,151667,198]), while their qwen3 row has no <think> suffix.
  Verified by rendering, not assumed.
- Featurize will use `--rebuild-prompt-token-ids-from-replay` from the start (the
  gpt-oss render-provenance lesson) and the probe must pass the same capture-parity
  gate (max|dp| <= 0.05) against the bridge's render before deployment.

## Long-prompt gate scoring (2026-09-13, post-smoke)

An AutoDojo adaptive episode at ~16k tokens OOM'd the AGRI gate's single-pass scoring
forward (gpt-oss eager attention materializes ~14 GiB transients; the SINGLE-pass
reference itself OOMs from ~10k on an empty 80 GB A100). Fix: KV-chunked scoring for
prompts > 8192 tokens (chunk 2048), same capture site. MEASURED: chunk2048-vs-single
max|dp| 0.0001–0.0096 on 5.5–7k prompts (in-budget reference); chunk-size invariance
(2048 vs 1024) |dp| 0.001–0.004 at ~10k; chunk-1024 diverges more (up to 0.11 on one
borderline prompt — MoE chunk-boundary sensitivity, same class as the measured batch-8
capture effect). Prompts <= 8192 keep the byte-identical single-pass path, so every
number recorded before this change (17-cell smoke, all parity gates, completed AutoDojo
groups) is unaffected. Scores above 8192 tokens carry this labeled path.
