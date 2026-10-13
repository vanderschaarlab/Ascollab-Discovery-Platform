#!/usr/bin/env bash
# Experiment 05: network scaling. Full ASCollab with N agents, compared with N=16.
# Dataset: kirc. Smaller networks keep the 6/5/5 GPT-4o/GPT-5/DeepSeek proportions (largest
# remainder); reviewers per paper K = min(2, N-1); N=1 uses the single-agent settings of 01.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=05
SIZES="${SIZES:-1 2 3 4 5 10 16}"

# N -> "gpt4o gpt5 deepseek"
mix_for() {
  case "$1" in
    1) echo "0 1 0" ;;
    2) echo "1 1 0" ;;
    3) echo "1 1 1" ;;
    4) echo "2 1 1" ;;
    5) echo "2 2 1" ;;
    10) echo "4 3 3" ;;
    16) echo "6 5 5" ;;
    *) echo "ERROR: no model mix defined for N=$1" >&2; return 1 ;;
  esac
}

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for n in ${SIZES}; do
      profile_main
      read -r V_GPT4O V_GPT5 V_DEEPSEEK <<<"$(mix_for "${n}")"
      EXTRA=("network.num_agents=${n}")
      if (( n == 1 )); then
        V_COLLAB_TOOLS=false V_CODE_COLLAB=false
        EXTRA+=("review.num_reviewers_per_paper=1" "review.allow_self_review=true")
      elif (( n == 2 )); then
        EXTRA+=("review.num_reviewers_per_paper=1")
      fi
      varying_overrides
      run_condition "${EXP}" "n${n}" "${dataset}" "${seed}" "${EXTRA[@]}"
    done
  done
done
