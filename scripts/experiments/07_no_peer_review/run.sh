#!/usr/bin/env bash
# Experiment 07: remove the peer-review stage. The meta-reviewer ranks submissions
# directly, without reviews. Compared with the shared baseline. Dataset: kirc.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=07

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    run_shared_baseline "${dataset}" "${seed}"
    profile_extended
    varying_overrides
    # No reviewers, so reviewers per paper and the assignment label do not apply.
    run_condition "${EXP}" no_reviewers "${dataset}" "${seed}" \
      "metareview.reviewerless.enabled=true"
  done
done
