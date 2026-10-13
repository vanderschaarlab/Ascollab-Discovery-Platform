#!/usr/bin/env bash
# Experiment 01 ranking: all 6 condition pairs x {novelty, significance, soundness, overall},
# top 25 papers per side, full Cartesian product of pairs.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=01
RUBRICS="${RUBRICS:-novelty significance soundness default}"
TOP_N="${TOP_N:-25}"
CONDITIONS=(single_agent indep_homo collab_homo ascollab)

for dataset in $(datasets_or_default "kirc luad paad nhanes"); do
  for seed in ${SEEDS}; do
    for ((i = 0; i < ${#CONDITIONS[@]}; i++)); do
      for ((j = i + 1; j < ${#CONDITIONS[@]}; j++)); do
        for rubric in ${RUBRICS}; do
          rank_pair "${EXP}" "${dataset}" "${seed}" "${CONDITIONS[$i]}" "${EXP}" "${CONDITIONS[$j]}" "${EXP}" \
            "${rubric}" "${TOP_N}" all
        done
      done
    done
  done
done
