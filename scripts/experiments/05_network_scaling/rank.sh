#!/usr/bin/env bash
# Experiment 05 ranking: each network size vs N=16 x {novelty, significance, soundness, overall},
# top 25 per side, all pairs.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=05
RUBRICS="${RUBRICS:-novelty significance soundness default}"
TOP_N="${TOP_N:-25}"
SIZES="${SIZES:-1 2 3 4 5 10}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for n in ${SIZES}; do
      for rubric in ${RUBRICS}; do
        rank_pair "${EXP}" "${dataset}" "${seed}" "n${n}" "${EXP}" n16 "${EXP}" "${rubric}" "${TOP_N}" all
      done
    done
  done
done
