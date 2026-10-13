#!/usr/bin/env bash
# Experiment 06: closed vs open-ended task, on TCGA KIRC data. The closed arm's goal asks
# agents to test three prespecified hypotheses (ABL1, CDK4/6 and JAK inhibition).
# There is no ranker step: the paper compares the closed arm's findings with a published study.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=06
CONDITIONS="${CONDITIONS:-open_ended closed}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for condition in ${CONDITIONS}; do
      profile_extended
      V_GPT4O=6 V_GPT5=7 V_DEEPSEEK=3
      if [[ "${condition}" == "closed" ]]; then
        V_SESSION_TIMEOUT=3600
        V_TARGET=kirc_closed_hypotheses
      else
        V_SESSION_TIMEOUT=720
        V_TARGET=tcga
      fi
      varying_overrides
      run_condition "${EXP}" "${condition}" "${dataset}" "${seed}"
    done
  done
done
