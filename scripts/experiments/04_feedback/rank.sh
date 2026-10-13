#!/usr/bin/env bash
# Experiment 04 ranking: each feedback variant vs no_feedback x {novelty, significance, soundness,
# overall}, top 25 per side, 500 randomly sampled pairs.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=04
RUBRICS="${RUBRICS:-novelty significance soundness default}"
TOP_N="${TOP_N:-25}"
N_PAIRS="${N_PAIRS:-500}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for variant in meta_review_only meta_review_with_metrics metrics_only; do
      for rubric in ${RUBRICS}; do
        rank_pair "${EXP}" "${dataset}" "${seed}" "${variant}" "${EXP}" no_feedback "${EXP}" \
          "${rubric}" "${TOP_N}" random "${N_PAIRS}"
      done
    done
  done
done
