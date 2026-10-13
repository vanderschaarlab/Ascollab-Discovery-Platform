#!/usr/bin/env bash
# Experiment 10: number of reviewers per paper, n in {1, 2, 3, 4}. The n=2 arm is the
# shared baseline (default K=2), so it is reused rather than rerun. Dataset: kirc.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=10
REVIEWER_COUNTS="${REVIEWER_COUNTS:-1 2 3 4}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for n in ${REVIEWER_COUNTS}; do
      if (( n == 2 )); then
        run_shared_baseline "${dataset}" "${seed}"
        continue
      fi
      profile_extended
      varying_overrides
      run_condition "${EXP}" "n${n}" "${dataset}" "${seed}" \
        "review.num_reviewers_per_paper=${n}" \
        "review.reviewer_assignment_condition_label='n_reviewers=${n}'"
    done
  done
done
