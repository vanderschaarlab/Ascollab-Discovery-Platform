#!/usr/bin/env bash
# Experiment 10 ranking: n=1, n=2 (shared baseline) and n=3 each vs n=4,
# x {novelty, significance, soundness}, top 15 per side, all pairs.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=10
RUBRICS="${RUBRICS:-novelty significance soundness}"
TOP_N="${TOP_N:-15}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for rubric in ${RUBRICS}; do
      rank_pair "${EXP}" "${dataset}" "${seed}" n1 "${EXP}" n4 "${EXP}" "${rubric}" "${TOP_N}" all
      rank_pair "${EXP}" "${dataset}" "${seed}" baseline rb n4 "${EXP}" "${rubric}" "${TOP_N}" all
      rank_pair "${EXP}" "${dataset}" "${seed}" n3 "${EXP}" n4 "${EXP}" "${rubric}" "${TOP_N}" all
    done
  done
done
