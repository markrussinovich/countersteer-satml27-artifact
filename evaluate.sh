#!/bin/bash
# evaluate.sh — run the CounterSteer evaluation for one model from its config.
#
#   ./evaluate.sh configs/<model>.json [options]
#
# Options:
#   --corpus NAME     evaluation corpus (default: the config's default_corpus;
#                     any of the config's firing_corpora is valid)
#   --n-eval N        samples (default 8 for a smoke; 52 reproduces the dev rung;
#                     the held-out test split is touched with --stage confirm ONLY)
#   --stage sweep|confirm   (default sweep = dev split; confirm = held-out test,
#                     one preregistered pass — do not rerun)
#   --gpu K           CUDA device index (default 0); GLM-4.5-Air ignores this and
#                     uses device=auto over all visible GPUs
#   --agentic         run the AgentDojo 4-arm battery instead of a single-turn
#                     corpus (uses the config's agentic budget and runs/agentdojo_cells.json)
#   --undefended-only skip the steered arms (baseline measurement)
#   --defense NAME    evaluate a RIVAL defense instead of CounterSteer (agentic only):
#                     spotlighting_with_delimiting | repeat_user_prompt | reminder |
#                     tool_filter | pi_detector (DeBERTa) | pi_detector_promptguard |
#                     pi_detector_piguard — AgentDojo's own wiring, same 4-arm battery,
#                     same-process undefended anchor. For CachePrune use
#                     --kv-mask runs/cacheprune_mask<model>.json instead.
#   --kv-mask FILE    evaluate the CachePrune port (agentic only)
#
# Output: results artifacts under the config's probe_dir; the canonical scorer
# is run automatically on the single-turn artifact (severity-ordered headline).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3

CFG="${1:?usage: evaluate.sh configs/<model>.json [--corpus C] [--n-eval N] [--stage sweep|confirm] [--gpu K] [--agentic]}"
shift
CORPUS=""; NEVAL=8; GPU=0; AGENTIC=0; STAGE=sweep; UNDEF=0; DEFENSE=""; KVMASK=""; SHARD=0; NSHARD=1
while [[ $# -gt 0 ]]; do case "$1" in
  --corpus) CORPUS="$2"; shift 2;;
  --n-eval) NEVAL="$2"; shift 2;;
  --stage)  STAGE="$2"; shift 2;;
  --gpu)    GPU="$2"; shift 2;;
  --agentic) AGENTIC=1; shift;;
  --undefended-only) UNDEF=1; shift;;
  --defense) DEFENSE="$2"; AGENTIC=1; shift 2;;
  --kv-mask) KVMASK="$2"; AGENTIC=1; shift 2;;
  --shard) SHARD="$2"; shift 2;;
  --nshard) NSHARD="$2"; shift 2;;
  *) echo "unknown option: $1" >&2; exit 2;;
esac; done

cfg() { "$PY" -c "import json,sys; print(json.load(open('$CFG')).get('$1',''))"; }
MODEL=$(cfg model); PD=$(cfg probe_dir); DIR=$(cfg direction); ALPHA=$(cfg alpha)
LAYERS=$(cfg layers); SIGMA=$(cfg match_sigma_to)
MN1=$(cfg max_new_single_turn); MNA=$(cfg max_new_agentic)
[ -n "$CORPUS" ] || CORPUS=$(cfg default_corpus)
DEVICE="cuda:0"
case "$MODEL" in zai-org/GLM-4.5-Air) DEVICE="auto";; esac

if [[ "$AGENTIC" == 1 ]]; then
  RARGS=(--model "$MODEL" --probe-dir "$PD" --max-new "$MNA"
         --cells runs/agentdojo_cells.json --no-adjudicate
         --shard "$SHARD" --nshard "$NSHARD")
  if [[ -n "$KVMASK" ]]; then
    LABEL="cacheprune"; RARGS+=(--kv-mask "$KVMASK")
    echo "== AgentDojo 4-arm battery: $MODEL, CachePrune ($KVMASK, mn$MNA)"
  elif [[ -n "$DEFENSE" ]]; then
    LABEL="$DEFENSE"; RARGS+=(--dojo-defense "$DEFENSE")
    echo "== AgentDojo 4-arm battery: $MODEL, rival defense $DEFENSE (mn$MNA)"
  else
    LABEL="countersteer"
    RARGS+=(--direction "$DIR" --alpha "$ALPHA" --layers "$LAYERS" --match-sigma-to "$SIGMA")
    echo "== AgentDojo 4-arm battery: $MODEL ($DIR @ $ALPHA, L$LAYERS, mn$MNA)"
  fi
  CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/controls/agentdojo_run.py "${RARGS[@]}" \
    --out "$PD/agentdojo_run_${LABEL}.shard${SHARD}.json"
  echo "== artifact: $PD/agentdojo_run_${LABEL}.json (AgentDojo's own checkers inside)"
  exit 0
fi

ARGS=(--model "$MODEL" --stage "$STAGE" --device "$DEVICE" --outdir "$PD"
      --corpus "$CORPUS" --n-eval "$NEVAL" --max-new "$MN1")
if [[ "$UNDEF" == 1 ]]; then
  ARGS+=(--baseline-only)
else
  ARGS+=(--directions "$DIR" --alphas "$ALPHA" --steer-layers "$LAYERS"
         --match-sigma-to "$SIGMA" --steer-clean)
fi
echo "== $STAGE on $CORPUS: $MODEL ($DIR @ $ALPHA, L$LAYERS, sigma='${SIGMA}')"
CUDA_VISIBLE_DEVICES=$GPU "$PY" xpia_defense.py "${ARGS[@]}"

ART=$(ls -t "$PD"/results_*_completions.json 2>/dev/null | head -1)
if [[ -n "$ART" ]]; then
  echo "== canonical scoring: $ART"
  "$PY" tools/controls/score_table.py --no-adjudicate "$ART"
fi
