#!/usr/bin/env bash
# Experiment 02: fixed-workflow baseline. Each agent gets a required collaborator instead of
# choosing one; otherwise identical to the 01 `ascollab` condition. Dataset: kirc.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

EXP=02

profile_main
varying_overrides
FIXED_WORKFLOW=(
  "research.fixed_workflow.enabled=true"
  "research.fixed_workflow.assignment_policy=weakest_agent"
  "research.fixed_workflow.tie_break=seeded_random"
  "research.fixed_workflow.self_fallback=second_weakest"
  "research.fixed_workflow.require_establish_before_first_code=true"
  "research.fixed_workflow.require_collab_message=true"
  "research.fixed_workflow.min_collab_messages=1"
  "research.fixed_workflow.force_tool_target=true"
  "research.fixed_workflow.disable_agent_registry_tool=true"
  "research.fixed_workflow.write_artifacts=true"
)

for dataset in $(datasets_or_default "kirc"); do
  for seed in ${SEEDS}; do
    run_condition "${EXP}" fixed_workflow "${dataset}" "${seed}" "${FIXED_WORKFLOW[@]}"
  done
done
