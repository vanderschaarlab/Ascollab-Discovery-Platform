#!/usr/bin/env bash
# Experiment 07 ranking: baseline vs no_reviewers, overall rubric, top 15 per side, all pairs.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=07
RUBRICS="${RUBRICS:-default}"
TOP_N="${TOP_N:-15}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for rubric in ${RUBRICS}; do
      rank_pair "${EXP}" "${dataset}" "${seed}" baseline rb no_reviewers "${EXP}" "${rubric}" "${TOP_N}" all
    done
  done
done
