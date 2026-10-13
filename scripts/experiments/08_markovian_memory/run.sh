#!/usr/bin/env bash
# Experiment 08: Markovian social memory. The shared archive only shows papers from
# the previous round. Compared with the shared baseline. Dataset: kirc.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=08

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    run_shared_baseline "${dataset}" "${seed}"
    profile_extended
    varying_overrides
    run_condition "${EXP}" markovian_memory "${dataset}" "${seed}" "${SHARED_BASELINE_K2[@]}" \
      "network.social_memory.mode=markovian"
  done
done
