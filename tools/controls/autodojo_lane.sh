#!/usr/bin/env bash
# One LANE of the AutoDojo full run: every (suite, vector, tasks) group of a plan TSV,
# sequentially, for ONE (model, arm) on ONE GPU. Each group is one autodojo_job.sh
# invocation (their optimizer runs tasks x vectors as a cross product; the
# pre-registered pair list in runs/autodojo/prereg.json decomposes exactly into these
# groups). --resume accumulates all groups of a suite into one injections.json per
# (suite, model-arm) under --outdir.
#
# Usage:
#   XPIA_JUDGE_ENDPOINT=... bash tools/controls/autodojo_lane.sh \
#     --plan runs/autodojo/full_plan.tsv --outdir runs/autodojo/full \
#     --gpu 0 --iterations 6 --arm defended \
#     --model openai/gpt-oss-20b --name gpt-oss-20b \
#     --probe-dir runs/gpt-oss-20b-userabl --direction combo_ovr8_pat1 \
#     --alpha 8.06 --layers 12,16,20 --match-sigma-to dim_no_override
#
# Prints AUTODOJO-LANE-DONE rc=<n_failed_groups> at the end.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PLAN=""; PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) PLAN="$2"; shift 2 ;;
    *) PASS+=("$1" "$2"); shift 2 ;;
  esac
done
[[ -n "$PLAN" ]] || { echo "FATAL: need --plan" >&2; exit 2; }
[[ "$PLAN" = /* ]] || PLAN="$ROOT/$PLAN"
[[ -f "$PLAN" ]] || { echo "FATAL: missing plan $PLAN" >&2; exit 1; }

fails=0; n=0; total=$(grep -cv '^\s*$' "$PLAN")
while IFS=$'\t' read -r suite vector tasks; do
  [[ -z "$suite" || "$suite" == \#* ]] && continue
  n=$((n+1))
  echo "[lane] group $n/$total: suite=$suite vector=$vector tasks=$tasks $(date -u '+%F %T')"
  bash "$ROOT/tools/controls/autodojo_job.sh" \
    --suite "$suite" --vectors "$vector" --injection-tasks "$tasks" "${PASS[@]}"
  rc=$?
  [[ $rc -ne 0 ]] && { echo "[lane] group $n FAILED rc=$rc"; fails=$((fails+1)); }
done < "$PLAN"
echo "AUTODOJO-LANE-DONE rc=$fails groups=$n"
exit "$fails"
