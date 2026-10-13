#!/usr/bin/env bash
# Experiment 01: network ablations.
# Conditions: single_agent, indep_homo, collab_homo, ascollab. Datasets: kirc luad paad nhanes.
#
#   DATASETS=kirc SEEDS=1 bash scripts/experiments/01_network_ablations/run.sh
#   DRY_RUN=1 bash scripts/experiments/01_network_ablations/run.sh     # print commands only
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=01
CONDITIONS="${CONDITIONS:-single_agent indep_homo collab_homo ascollab}"

condition_overrides() {
  profile_main
  EXTRA=()
  case "$1" in
    single_agent)
      # One GPT-5 agent; reviews its own papers (no peers).
      V_COLLAB_TOOLS=false V_MEMORY_TOOLS=true V_CODE_COLLAB=false
      V_GPT4O=0 V_GPT5=1 V_DEEPSEEK=0 V_LLAMA=0
      EXTRA=("network.num_agents=1" "review.num_reviewers_per_paper=1" "review.allow_self_review=true")
      ;;
    indep_homo)
      # 16 independent GPT-4o agents: no collaboration, no social memory.
      V_SETTING=indep_homo V_COLLAB_TOOLS=false V_MEMORY_TOOLS=false V_CODE_COLLAB=false
      V_GPT4O=16 V_GPT5=0 V_DEEPSEEK=0 V_LLAMA=0
      ;;
    collab_homo)
      # 16 homogeneous GPT-4o agents with collaboration and social memory.
      V_SETTING=network_homo
      V_GPT4O=16 V_GPT5=0 V_DEEPSEEK=0 V_LLAMA=0
      ;;
    ascollab)
      # Full system: heterogeneous model mix, collaboration and social memory.
      ;;
    *) echo "ERROR: unknown condition $1" >&2; return 1 ;;
  esac
  varying_overrides
}

for dataset in $(datasets_or_default "kirc luad paad nhanes"); do
  for seed in ${SEEDS}; do
    for condition in ${CONDITIONS}; do
      condition_overrides "${condition}"
      run_condition "${EXP}" "${condition}" "${dataset}" "${seed}" "${EXTRA[@]}"
    done
  done
done
