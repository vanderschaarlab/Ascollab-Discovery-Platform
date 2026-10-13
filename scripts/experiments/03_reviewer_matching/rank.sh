#!/usr/bin/env bash
# Experiment 03 ranking: AutoRanker scores for each fixed cohort, then the comparison of
# expertise-matched vs random reviewer assignment. Run after run.sh: the cohorts come from the 01
# snapshot that run.sh recorded in the output folder's manifests, not from a new lookup.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=03
ROOT_OUT="${ROOT_OUT:-outputs/reviewer_matching}"
TARGET_ITERATIONS="${TARGET_ITERATIONS:-5 10 20}"

# JUDGE and MAX_WORKERS only when set: the AutoRanker's defaults are gpt5-high and 4 workers.
RANKER_ARGS=()
if [[ -n "${JUDGE:-}" ]]; then RANKER_ARGS+=(--judge "${JUDGE}"); fi
if [[ -n "${MAX_WORKERS:-}" ]]; then RANKER_ARGS+=(--max-workers "${MAX_WORKERS}"); fi

for dataset in $(datasets_or_default "nhanes"); do
  for seed in ${SEEDS}; do
    root="${ROOT_OUT}/${dataset}_seed_${seed}"
    # shellcheck disable=SC2086
    emit "${EXP}" autoranker "${dataset}" "${seed}" rank -- \
      "${PYTHON}" -m scripts.analysis.reviewer_matching.run_grid \
      --datasets "${dataset}" \
      --target-iterations ${TARGET_ITERATIONS} \
      --cohorts-per-iteration 5 \
      --cohort-size 20 \
      --seed 123 \
      --root-output-dir "${root}" \
      --skip-reviews \
      ${RANKER_ARGS[@]+"${RANKER_ARGS[@]}"}
    emit "${EXP}" analysis "${dataset}" "${seed}" other -- \
      "${PYTHON}" -m scripts.analysis.reviewer_matching.analyze_results \
      --root "${root}" \
      --output-dir "${root}/analysis"
  done
done
