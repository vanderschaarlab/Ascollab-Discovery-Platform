#!/usr/bin/env bash
# Experiment 02 ranking: fixed_workflow vs each experiment-01 condition (same dataset and seed),
# x {novelty, significance, soundness, overall}, top 25, all pairs. Needs the 01 runs.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=02
RUBRICS="${RUBRICS:-novelty significance soundness default}"
TOP_N="${TOP_N:-25}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for other in single_agent indep_homo collab_homo ascollab; do
      for rubric in ${RUBRICS}; do
        rank_pair "${EXP}" "${dataset}" "${seed}" fixed_workflow 02 "${other}" 01 "${rubric}" "${TOP_N}" all
      done
    done
  done
done
