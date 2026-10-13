#!/usr/bin/env bash
# Experiment 04: feedback. Homogeneous GPT-4o network; agents receive feedback on
# their previous submission in one of three forms, compared with no feedback. Dataset: kirc.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=04
VARIANTS="${VARIANTS:-no_feedback meta_review_only meta_review_with_metrics metrics_only}"

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    for variant in ${VARIANTS}; do
      profile_main
      V_T=20
      V_SETTING=network_homo
      V_GPT4O=16 V_GPT5=0 V_DEEPSEEK=0 V_LLAMA=0
      V_PRIOR_PAPERS=true V_FULL_CITING=true
      V_SELF_REFLECTION=true V_NATURE_STYLE=true V_GRAPH_LOGIC=true
      if [[ "${variant}" == "no_feedback" ]]; then
        V_METACOGNITION=false mode=none
      else
        V_METACOGNITION=true mode="${variant}"
      fi
      varying_overrides
      run_condition "${EXP}" "${variant}" "${dataset}" "${seed}" \
        "research.metacognition_feedback.mode=${mode}" \
        "research.metacognition_feedback.source_iteration=previous" \
        "research.metacognition_feedback.apply_on_resume_first_iteration=true" \
        "research.metacognition_feedback.strict_source_required=false" \
        "research.metacognition_feedback.best_effort=true" \
        "research.metacognition_feedback.write_artifacts=true" \
        "graph_logic.require_plot_citations_in_research=true" \
        "graph_logic.plot_only_rewrite_cap=2" \
        "graph_logic.allow_plot_validation_waiver_after_cap=true"
    done
  done
done
