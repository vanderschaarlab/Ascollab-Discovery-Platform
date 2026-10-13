#!/usr/bin/env bash
# Experiment 03: reviewer matching. Re-runs the review step on fixed cohorts of
# papers from an experiment-01 `ascollab` run, once with expertise-matched reviewers and once
# with randomly assigned reviewers. Dataset: nhanes. Needs the 01 run for that dataset.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=03
ROOT_OUT="${ROOT_OUT:-outputs/reviewer_matching}"
TARGET_ITERATIONS="${TARGET_ITERATIONS:-5 10 20}"

for dataset in $(datasets_or_default "nhanes"); do
  for seed in ${SEEDS}; do
    source_dir="$(snapshot_dir 01 ascollab "${dataset}" "${seed}")"
    # shellcheck disable=SC2086
    emit "${EXP}" review_replay "${dataset}" "${seed}" other -- \
      "${PYTHON}" -m scripts.analysis.reviewer_matching.run_grid \
      --datasets "${dataset}" \
      --snapshot-dirs "${dataset}=${source_dir}" \
      --target-iterations ${TARGET_ITERATIONS} \
      --cohorts-per-iteration 5 \
      --cohort-size 20 \
      --seed 123 \
      --root-output-dir "${ROOT_OUT}/${dataset}_seed_${seed}" \
      --skip-autoranker
  done
done
