#!/usr/bin/env bash
# Experiment 09: reviewer norms. Every reviewer prompt carries a norm: novelty-first
# (norm_a) or soundness-first (norm_b). Dataset: kirc.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=09
NORMS="${NORMS:-norm_a_novelty norm_b_soundness}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for norm in ${NORMS}; do
      profile_extended
      varying_overrides
      run_condition "${EXP}" "${norm}" "${dataset}" "${seed}" "${SHARED_BASELINE_K2[@]}" \
        "+review_norm@review.norm_prompt=${norm}"
    done
  done
done
