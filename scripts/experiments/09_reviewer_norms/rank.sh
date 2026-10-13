#!/usr/bin/env bash
# Experiment 09 ranking: norm_a vs norm_b x {novelty, significance, soundness}, top 15, all pairs.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=09
RUBRICS="${RUBRICS:-novelty significance soundness}"
TOP_N="${TOP_N:-15}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for rubric in ${RUBRICS}; do
      rank_pair "${EXP}" "${dataset}" "${seed}" norm_a_novelty "${EXP}" norm_b_soundness "${EXP}" \
        "${rubric}" "${TOP_N}" all
    done
  done
done
