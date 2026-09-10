#!/usr/bin/env bash
# One AutoDojo adaptive-attack optimization run (arXiv:2606.15057) against ONE arm of
# ONE of our local models: the attacker's optimizer LLM (Azure deployment, az-cli
# bearer) iteratively rewrites each (injection_task x vector) target's injection,
# scoring every candidate by running the agent END-TO-END in AgentDojo's own loop with
# AgentDojo's own security checker. The target agent is OUR SteeredLLM bridge element
# (the same substrate as every recorded AgentDojo number), reached through the
# `plugin:` seam added to the vendored fork (reference/autodojo, branch
# xpia-integration).
#
# Arms:
#   --arm undefended   direction off; the adaptive anchor
#   --arm defended     the deployed CounterSteer cell; reachability filtering runs on
#                      the UNDEFENDED element via AUTODOJO_REACHABILITY_LLM (their own
#                      design: reachability must not be skewed by the defense)
#   --arm cacheprune   the KV-mask element (needs --kv-mask); reachability as above
#   --arm promptguard  PromptGuard-2 as a PIPELINE-level tool-output filter (prereg
#                      v3.3): the SoA batteries' windowed LocalPIDetector, registered
#                      through the fork's plugin seam (autodojo_promptguard_plugin).
#                      The LLM element itself is UNDEFENDED; with defense != None the
#                      evaluator builds its reachability pipeline defense-free from
#                      the same spec automatically. The gated checkpoint resolves via
#                      XPIA_MODEL_STORE when set (boxes without hub credentials).
#
# Usage (gpt-oss deployed cell, banking, 2-target smoke):
#   XPIA_JUDGE_ENDPOINT=https://<resource>.cognitiveservices.azure.com \
#   bash tools/controls/autodojo_job.sh --suite banking --arm defended \
#     --model openai/gpt-oss-20b --name gpt-oss-20b \
#     --probe-dir runs/gpt-oss-20b-userabl --direction combo_ovr8_pat1 --alpha 8.06 \
#     --layers 12,16,20 --match-sigma-to dim_no_override \
#     --gpu 0 --iterations 6 --outdir runs/autodojo/smoke \
#     --injection-tasks injection_task_0,injection_task_5 --vectors injection_bill_text
#
# Produces <outdir>/<suite>/<name>-<arm>/no_defense/injections.json (+ run_cost.json,
# prompt_log.jsonl, llm_cache.jsonl) and prints AUTODOJO-JOB-DONE rc=<rc> at exit.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
FORK="$ROOT/reference/autodojo"

SUITE=""; ARM=""; MODEL=""; NAME=""; PROBEDIR=""; DIRECTION=""; ALPHA=""; LAYERS=""
MATCH_SIGMA="__unset__"; GPU="0"; ITER="6"; OUTDIR=""; INJTASKS=""; VECTORS=""
MAXNEW="4096"; DEPLOYMENT="gpt-5.4"; NVAR="5"; KVMASK=""; EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)          SUITE="$2"; shift 2 ;;
    --arm)            ARM="$2"; shift 2 ;;
    --model)          MODEL="$2"; shift 2 ;;
    --name)           NAME="$2"; shift 2 ;;
    --probe-dir)      PROBEDIR="$2"; shift 2 ;;
    --direction)      DIRECTION="$2"; shift 2 ;;
    --alpha)          ALPHA="$2"; shift 2 ;;
    --layers)         LAYERS="$2"; shift 2 ;;
    --match-sigma-to) MATCH_SIGMA="$2"; shift 2 ;;
    --gpu)            GPU="$2"; shift 2 ;;
    --iterations)     ITER="$2"; shift 2 ;;
    --n-variants)     NVAR="$2"; shift 2 ;;
    --outdir)         OUTDIR="$2"; shift 2 ;;
    --injection-tasks) INJTASKS="$2"; shift 2 ;;
    --vectors)        VECTORS="$2"; shift 2 ;;
    --max-new)        MAXNEW="$2"; shift 2 ;;
    --kv-mask)        KVMASK="$2"; shift 2 ;;
    --deployment)     DEPLOYMENT="$2"; shift 2 ;;
    --dojo-defense)   DOJODEF="$2"; shift 2 ;;
    --extra)          EXTRA+=("$2"); shift 2 ;;     # raw passthrough flag
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$SUITE" && -n "$ARM" && -n "$MODEL" && -n "$NAME" && -n "$OUTDIR" ]] \
  || { echo "FATAL: need --suite --arm --model --name --outdir" >&2; exit 2; }
[[ "$ARM" == "defended" || "$ARM" == "undefended" || "$ARM" == "cacheprune" || "$ARM" == "promptguard" || "$ARM" == "piguard" || "$ARM" == "deberta" || "$ARM" == "dojodef" ]] \
  || { echo "FATAL: --arm defended|undefended|cacheprune|promptguard" >&2; exit 2; }
[[ -n "${XPIA_JUDGE_ENDPOINT:-}" ]] \
  || { echo "FATAL: export XPIA_JUDGE_ENDPOINT (Azure OpenAI resource)" >&2; exit 2; }
[[ "$OUTDIR" = /* ]] || OUTDIR="$ROOT/$OUTDIR"
mkdir -p "$OUTDIR"

# PRE-FLIGHT: one REAL data-plane call through the same auth path the run will use,
# before loading a 20-100GB model for nothing. A locally-mintable token is NOT the
# check -- .9's az identity minted valid-looking tokens with zero RBAC on the
# resource and 401'd every call (voided the first smoke's defended-arm optimizer).
if [[ -n "${AUTODOJO_AZ_TOKEN_CMD:-}" ]]; then
  TOK=$(bash -c "$AUTODOJO_AZ_TOKEN_CMD" | tail -1)
else
  TOK=$(az account get-access-token --resource https://cognitiveservices.azure.com/ \
        --query accessToken -o tsv)
fi
[[ -n "$TOK" ]] || { echo "FATAL: could not mint an Azure token" >&2; exit 1; }
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  "$XPIA_JUDGE_ENDPOINT/openai/deployments/$DEPLOYMENT/chat/completions?api-version=${XPIA_JUDGE_API_VERSION:-2024-10-21}" \
  -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"ping"}],"max_completion_tokens":16}')
[[ "$code" == "200" ]] \
  || { echo "FATAL: data-plane pre-flight to $DEPLOYMENT returned HTTP $code (identity lacks RBAC on the resource? set AUTODOJO_AZ_TOKEN_CMD)" >&2; exit 1; }
echo "[autodojo-job] azure pre-flight OK ($DEPLOYMENT via $([[ -n "${AUTODOJO_AZ_TOKEN_CMD:-}" ]] && echo token-relay || echo local-az))"

# our plugin element wraps a stateful single-GPU HF model with module-level hooks;
# concurrent query() would race. Their PARALLEL_EVAL_SAFE_DEFENSES contains None, so
# the optimizer would ACCEPT --parallel-eval -- refuse it here instead.
for x in "${EXTRA[@]:-}"; do
  [[ "$x" == "--parallel-eval" ]] \
    && { echo "FATAL: --parallel-eval is unsafe for plugin targets (stateful GPU element)" >&2; exit 2; }
done

# ── target plugin specs ───────────────────────────────────────────────────────
base="plugin:autodojo_target?model=$MODEL&max_new=$MAXNEW"
undef_spec="$base&name=${NAME}-undefended"
if [[ "$ARM" == "defended" ]]; then
  [[ -n "$DIRECTION" && -n "$PROBEDIR" && -n "$ALPHA" && -n "$LAYERS" ]] \
    || { echo "FATAL: defended arm needs --direction --probe-dir --alpha --layers" >&2; exit 2; }
  spec="$base&probe_dir=$PROBEDIR&direction=$DIRECTION&alpha=$ALPHA&layers=$LAYERS"
  [[ "$MATCH_SIGMA" != "__unset__" ]] && spec="$spec&match_sigma_to=$MATCH_SIGMA"
  spec="$spec&name=${NAME}-countersteer"
  export AUTODOJO_REACHABILITY_LLM="$undef_spec"
elif [[ "$ARM" == "cacheprune" ]]; then
  # CachePrune (arXiv:2504.21228) as the best-alternate-defense arm: the KV-mask
  # element the SoA batteries ran, inside the same plugin seam. Reachability runs
  # on the undefended element, exactly as for the steering arm.
  [[ -n "$KVMASK" ]] || { echo "FATAL: cacheprune arm needs --kv-mask" >&2; exit 2; }
  spec="$base&kv_mask=$KVMASK&name=${NAME}-cacheprune"
  export AUTODOJO_REACHABILITY_LLM="$undef_spec"
elif [[ "$ARM" == "piguard" ]]; then
  # PIGuard (owner-approved 2026-09-10): same pipeline-level filter seam as promptguard,
  # strongest static detector — the decisive adaptive-robustness cell.
  spec="$undef_spec"
  export AGENTDOJO_DEFENSE_PLUGINS="autodojo_piguard_plugin"
  export XPIA_MODEL_STORE="${XPIA_MODEL_STORE:-$ROOT/models}"
  [[ -d "$XPIA_MODEL_STORE/PIGuard" ]] \
    || { echo "FATAL: PIGuard checkpoint missing at $XPIA_MODEL_STORE/PIGuard (sync it first)" >&2; exit 1; }
  EXTRA+=(--defense piguard_soa --run-defense)
elif [[ "$ARM" == "deberta" ]]; then
  # DeBERTa/ProtectAI filter via the same SoA LocalPIDetector wiring (figure-completeness
  # program, owner 2026-09-10: adaptive attacks for every defense).
  spec="$undef_spec"
  export AGENTDOJO_DEFENSE_PLUGINS="autodojo_deberta_plugin"
  export XPIA_MODEL_STORE="${XPIA_MODEL_STORE:-$ROOT/models}"
  EXTRA+=(--defense deberta_soa --run-defense)
elif [[ "$ARM" == "dojodef" ]]; then
  # Generic inbuilt-defense arm (spotlighting / repeat_user_prompt / reminder /
  # tool_filter — the fork's own registry names). Requires --dojo-defense.
  [[ -n "${DOJODEF:-}" ]] || { echo "FATAL: --arm dojodef needs --dojo-defense NAME" >&2; exit 2; }
  case "$DOJODEF" in spotlighting|repeat_user_prompt|reminder|tool_filter) ;; *)
    echo "FATAL: --dojo-defense must be spotlighting|repeat_user_prompt|reminder|tool_filter" >&2; exit 2 ;; esac
  spec="$undef_spec"
  EXTRA+=(--defense "$DOJODEF" --run-defense)
  ARM="$DOJODEF"   # output dirs/caches are per-defense, not one shared "dojodef"
  spec="plugin:autodojo_target?model=$MODEL&max_new=$MAXNEW&name=${NAME}-undefended"
elif [[ "$ARM" == "promptguard" ]]; then
  # PromptGuard-2 (prereg v3.3): pipeline-level filter, UNDEFENDED LLM element.
  # defense != None makes the evaluator build a separate defense-free reachability
  # pipeline from this same spec -- no AUTODOJO_REACHABILITY_LLM override.
  spec="$undef_spec"
  export AGENTDOJO_DEFENSE_PLUGINS="autodojo_promptguard_plugin"
  # validated setting as DEFAULT (review fix 2): resolve the gated checkpoint from the
  # repo model store; the hub fallback 401s on boxes without an HF token.
  export XPIA_MODEL_STORE="${XPIA_MODEL_STORE:-$ROOT/models}"
  [[ -d "$XPIA_MODEL_STORE/Llama-Prompt-Guard-2-86M" ]] \
    || { echo "FATAL: PromptGuard checkpoint missing at $XPIA_MODEL_STORE/Llama-Prompt-Guard-2-86M (sync it first)" >&2; exit 1; }
  EXTRA+=(--defense promptguard_soa --run-defense)
else
  spec="$undef_spec"
fi

export AUTODOJO_OUTPUT_DIR="$OUTDIR"
export AUTODOJO_LLM_CACHE="$OUTDIR/llm_cache_${SUITE}_${NAME}-${ARM}.jsonl"
export CUDA_VISIBLE_DEVICES="$GPU"
# fragmentation headroom: the defended slack channel-group backfill OOM'd at a
# 11.68 GiB eager-attention allocation with 10.36 GiB free BUT 5.22 GiB reserved-
# unallocated (2026-09-07). The SoA batteries always exported this; do the same.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$FORK/agentdojo/src:$FORK/agentdojo/variant_generation:$ROOT/tools/controls:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ -d /datadrive/hf_cache ]] && export HF_HOME="${HF_HOME:-/datadrive/hf_cache}"

args=(agentdojo/variant_generation/optimize_variants.py
      --suite "$SUITE" --eval-asr
      --target-model "$spec" --target-model-id "${NAME}-${ARM}"
      --model "$DEPLOYMENT" --provider azure
      --iterations "$ITER" --n-variants "$NVAR"
      --analyzer-prompt "analyzer_${SUITE}"
      --injection-prompt "injection_task_iterative_${SUITE}"
      --seed-styles rlhammer topicattack
      --store-traces --resume
      # per-GROUP prompt log: optimize_variants UNLINKS an existing log at setup,
      # so a shared per-suite name would wipe earlier groups' refusal telemetry
      # when the lane driver invokes one group per (suite, vector)
      --prompt-log "$OUTDIR/prompt_log_${SUITE}_${NAME}-${ARM}$([[ -n "$VECTORS" ]] && echo "_${VECTORS//[^A-Za-z0-9_]/-}").jsonl")
if [[ -n "$INJTASKS" ]]; then
  IFS=',' read -r -a ts <<< "$INJTASKS"
  for t in "${ts[@]}"; do args+=(--injection-tasks "$t"); done
fi
if [[ -n "$VECTORS" ]]; then
  IFS=',' read -r -a vs <<< "$VECTORS"
  for v in "${vs[@]}"; do args+=(--vectors "$v"); done
fi
[[ ${#EXTRA[@]} -gt 0 ]] && args+=("${EXTRA[@]}")

echo "[autodojo-job] suite=$SUITE arm=$ARM model=$MODEL gpu=$GPU iter=$ITER outdir=$OUTDIR"
echo "[autodojo-job] target spec: $spec"
[[ "$ARM" == "defended" ]] && echo "[autodojo-job] reachability: $AUTODOJO_REACHABILITY_LLM"

cd "$FORK"
"$PY" -u "${args[@]}"
rc=$?
echo "AUTODOJO-JOB-DONE rc=$rc suite=$SUITE arm=$ARM name=$NAME"
exit "$rc"
