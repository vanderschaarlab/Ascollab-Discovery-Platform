#!/usr/bin/env bash
# Shared helpers for scripts/experiments/*/run.sh and rank.sh. Sourced, not executed.
#
# Environment knobs (all optional):
#   DATASETS        space/comma list of dataset bundles (default: per experiment)
#   SEEDS           space/comma list of seeds (default: 1)
#   BASE_NAME       run-name prefix (default: ascollab)
#   PYTHON          python executable (default: python)
#   CONFIG_NAME     Hydra config name (default: ascollab)
#   ITERATIONS      override the number of rounds T (smoke tests only); rank.sh and the shared
#                   baseline reuse then take runs that finished this many rounds
#   NUM_AGENTS      override the network size N (smoke tests only): scales the condition's model
#                   mix to N and sets reviewers per paper K = N-1 (1 to 2); skipped by conditions
#                   that set N. Conditions that set K keep it (shared baseline, 08, 09: 2; 10: 1-4),
#                   so K can exceed N-1; the reviewer matcher then warns and assigns fewer
#   DRY_RUN=1       print the commands instead of running them
#   DRY_RUN_FORMAT  text (default) | json (one JSON object per command, used by tests)
#   JUDGE           ranker judge backend: gpt5-high (default), gpt5, gpt4o, deepseek, llama or
#                   primary; its endpoint file must exist (docs/models.md)
#   MAX_WORKERS     ranker parallelism (default: 8; experiment 03's AutoRanker: 4)
#   SNAPSHOT_DIR_<EXPERIMENT>_<DATASET>_<CONDITION>_SEED_<SEED>  the snapshot dir of one run (for
#                   rank.sh, 03, and the shared baseline: experiment RB, condition BASELINE), e.g.
#                   SNAPSHOT_DIR_01_KIRC_ASCOLLAB_SEED_1=snapshots/ascollab_01_kirc_ascollab_seed_1_260101_1200
#   FORCE_BASELINE=1  run the shared baseline even if a complete one exists

set -euo pipefail

EXP_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${EXP_LIB_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
CONFIG_NAME="${CONFIG_NAME:-ascollab}"
BASE_NAME="${BASE_NAME:-ascollab}"
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_FORMAT="${DRY_RUN_FORMAT:-text}"
SEEDS="${SEEDS:-1}"
SEEDS="${SEEDS//,/ }"

# ---------------------------------------------------------------------------
# Dataset bundles (public data only; see docs/data.md)
# ---------------------------------------------------------------------------
check_bundle() {
  if [[ ! -f "sys_config/dataset/$1.yaml" ]]; then
    echo "ERROR: unknown dataset bundle '$1' (see sys_config/dataset/: kirc luad paad nhanes)" >&2
    return 1
  fi
}

# datasets_or_default "<default list>": the DATASETS env var if set, else the default.
datasets_or_default() {
  local ds="${DATASETS:-$1}"
  echo "${ds//,/ }"
}

# ---------------------------------------------------------------------------
# Settings that vary between experiments. Every run sets all of them
# explicitly, so no script depends on a default that differs between experiments.
# Callers set the V_* variables (or call a profile_* function) before varying_overrides.
# ---------------------------------------------------------------------------
profile_main() {
  # Settings of experiments 01-05.
  V_T=40
  V_SETTING=network_hetero
  V_COLLAB_TOOLS=true
  V_MEMORY_TOOLS=true
  V_CODE_COLLAB=true
  V_GPT4O=6 V_GPT5=5 V_DEEPSEEK=5 V_LLAMA=0
  V_TARGET=""  # discovery_dataset.target (goal and package set); empty: the bundle's
  V_PRIOR_PAPERS=false
  V_FULL_CITING=false
  V_CODE_TIMEOUT=480
  V_SESSION_TIMEOUT=720
  V_MAX_CONCURRENT=null
  V_SELF_REFLECTION=false
  V_NATURE_STYLE=false
  V_GRAPH_LOGIC=false
  V_METACOGNITION=false
}

profile_extended() {
  # Settings of experiments 06-10.
  V_T=20
  V_SETTING=network_hetero
  V_COLLAB_TOOLS=true
  V_MEMORY_TOOLS=true
  V_CODE_COLLAB=true
  V_GPT4O=6 V_GPT5=6 V_DEEPSEEK=4 V_LLAMA=0
  V_TARGET=""  # 06 sets one per arm
  V_PRIOR_PAPERS=true
  V_FULL_CITING=true
  V_CODE_TIMEOUT=900
  V_SESSION_TIMEOUT=3600
  V_MAX_CONCURRENT=10
  V_SELF_REFLECTION=false
  V_NATURE_STYLE=false
  V_GRAPH_LOGIC=false
  V_METACOGNITION=false
}

varying_overrides() {
  VARYING=(
    "network.iterations=${ITERATIONS:-${V_T}}"
    "network.setting=${V_SETTING}"
    "network.enable_collaboration_tools=${V_COLLAB_TOOLS}"
    "network.enable_social_memory_tools=${V_MEMORY_TOOLS}"
    "research.code_collaboration=${V_CODE_COLLAB}"
    "llm_backends.counts.gpt4o=${V_GPT4O}"
    "llm_backends.counts.gpt5=${V_GPT5}"
    "llm_backends.counts.deepseek=${V_DEEPSEEK}"
    "llm_backends.counts.llama=${V_LLAMA}"
    "research.prior_papers.enabled=${V_PRIOR_PAPERS}"
    "research.full_citing=${V_FULL_CITING}"
    "research.code_timeout_seconds=${V_CODE_TIMEOUT}"
    "research.total_session_timeout_seconds=${V_SESSION_TIMEOUT}"
    "research.max_concurrent_research_agents=${V_MAX_CONCURRENT}"
    "research.self_reflection_prompt.enabled=${V_SELF_REFLECTION}"
    "research.nature_style_papers_prompt.enabled=${V_NATURE_STYLE}"
    "graph_logic.enabled=${V_GRAPH_LOGIC}"
    "research.metacognition_feedback.enabled=${V_METACOGNITION}"
    "bootstrap.enabled=false"
  )
  if [[ "${V_PRIOR_PAPERS}" == "true" ]]; then
    VARYING+=("research.prior_papers.mode=full")
  fi
}

# ---------------------------------------------------------------------------
# Command emission
# ---------------------------------------------------------------------------
# emit <experiment> <condition> <dataset> <seed> <kind> -- <argv...>
# kind: run | rank | other. In DRY_RUN mode prints instead of executing.
emit() {
  local experiment="$1" condition="$2" dataset="$3" seed="$4" kind="$5"
  shift 5
  [[ "$1" == "--" ]] && shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${DRY_RUN_FORMAT}" == "json" ]]; then
      python3 -c 'import json,sys; e,c,d,s,k=sys.argv[1:6]; print(json.dumps({"experiment":e,"condition":c,"dataset":d,"seed":s,"kind":k,"argv":sys.argv[6:]}))' \
        "${experiment}" "${condition}" "${dataset}" "${seed}" "${kind}" "$@"
    else
      printf '%q ' "$@"
      printf '\n'
    fi
  else
    echo "==> [${experiment}] ${kind} condition=${condition} dataset=${dataset} seed=${seed}" >&2
    "$@"
  fi
}

run_name() { echo "${BASE_NAME}_$1_$3_$2_seed_$4"; }  # <experiment> <condition> <dataset> <seed>

# smoke_size_overrides <extra overrides...>: with NUM_AGENTS set and no network.num_agents among the
# extra overrides, rewrite VARYING's model mix for NUM_AGENTS agents (largest remainder) and append
# the network size and, unless set, a reviewers-per-paper cap of N-1.
smoke_size_overrides() {
  SMOKE=()
  [[ -n "${NUM_AGENTS:-}" ]] || return 0
  local arg
  for arg in "$@"; do [[ "${arg}" == network.num_agents=* ]] && return 0; done
  local counts
  counts="$(awk -v n="${NUM_AGENTS}" -v c="${V_GPT4O} ${V_GPT5} ${V_DEEPSEEK} ${V_LLAMA}" 'BEGIN {
    k = split(c, w, " "); t = 0; for (i = 1; i <= k; i++) t += w[i]
    a = 0; for (i = 1; i <= k; i++) { q[i] = n * w[i] / t; f[i] = int(q[i]); r[i] = q[i] - f[i]; a += f[i] }
    while (a < n) { b = 1; for (i = 2; i <= k; i++) if (r[i] > r[b]) b = i; f[b]++; r[b] = -1; a++ }
    printf "%d %d %d %d", f[1], f[2], f[3], f[4] }')"
  local g4 g5 ds ll i
  read -r g4 g5 ds ll <<<"${counts}"
  for i in "${!VARYING[@]}"; do
    case "${VARYING[$i]}" in
      llm_backends.counts.gpt4o=*) VARYING[i]="llm_backends.counts.gpt4o=${g4}" ;;
      llm_backends.counts.gpt5=*) VARYING[i]="llm_backends.counts.gpt5=${g5}" ;;
      llm_backends.counts.deepseek=*) VARYING[i]="llm_backends.counts.deepseek=${ds}" ;;
      llm_backends.counts.llama=*) VARYING[i]="llm_backends.counts.llama=${ll}" ;;
    esac
  done
  SMOKE=("network.num_agents=${NUM_AGENTS}")
  for arg in "$@"; do [[ "${arg}" == review.num_reviewers_per_paper=* ]] && return 0; done
  local k=$((NUM_AGENTS - 1)); (( k > 2 )) && k=2; (( k < 1 )) && k=1
  SMOKE+=("review.num_reviewers_per_paper=${k}")
}

# run_condition <experiment> <condition> <dataset> <seed> [extra overrides...]
# Uses the VARYING array built by varying_overrides.
run_condition() {
  local experiment="$1" condition="$2" dataset="$3" seed="$4"
  shift 4
  check_bundle "${dataset}"
  smoke_size_overrides "$@"
  # The target (goal + package set) defaults to the bundle's; experiment 06 sets V_TARGET per arm.
  local target_override=()
  if [[ -n "${V_TARGET:-}" ]]; then target_override=("discovery_dataset.target=${V_TARGET}"); fi
  emit "${experiment}" "${condition}" "${dataset}" "${seed}" run -- \
    "${PYTHON}" -m system.main --config-path ../sys_config --config-name "${CONFIG_NAME}" \
    "exp_name=$(run_name "${experiment}" "${condition}" "${dataset}" "${seed}")" \
    "SEED=${seed}" \
    "dataset=${dataset}" \
    ${target_override[@]+"${target_override[@]}"} \
    "${VARYING[@]}" \
    ${SMOKE[@]+"${SMOKE[@]}"} \
    "$@"
}

# ---------------------------------------------------------------------------
# Shared baseline (experiment id "rb"): the reference arm for 07 (no peer review),
# 08 (Markovian memory) and the n=2 arm of 10 (number of reviewers). Reviewers per paper are the
# paper's K=2; the condition label salts reviewer assignment like the 10 arms do.
# ---------------------------------------------------------------------------
# The label contains '=', so Hydra needs it quoted.
SHARED_BASELINE_K2=(
  "review.num_reviewers_per_paper=2"
  "review.reviewer_assignment_condition_label='n_reviewers=2'"
)

run_shared_baseline() {
  local dataset="$1" seed="$2" var dir
  profile_extended
  varying_overrides
  var="$(snapshot_var rb baseline "${dataset}" "${seed}")"
  if [[ "${DRY_RUN}" != "1" && "${FORCE_BASELINE:-0}" != "1" ]]; then
    if [[ -n "${!var:-}" ]]; then
      if ! snapshot_complete "${!var}" "$(rounds_for rb)"; then
        echo "ERROR: ${var}=${!var} is not a complete baseline run (no iteration_$(rounds_for rb)/snapshot_metadata.json); unset it to run the baseline" >&2
        return 1
      fi
      echo "==> [rb] baseline for ${dataset} seed ${seed}: ${var}=${!var}; reusing it" >&2
      return 0
    fi
    if dir="$(latest_complete_snapshot rb baseline "${dataset}" "${seed}")"; then
      echo "==> [rb] baseline for ${dataset} seed ${seed} exists (${dir}); reusing it (FORCE_BASELINE=1 to rerun)" >&2
      return 0
    fi
  fi
  run_condition rb baseline "${dataset}" "${seed}" "${SHARED_BASELINE_K2[@]}"
}

# ---------------------------------------------------------------------------
# Snapshot lookup and ranking
# ---------------------------------------------------------------------------
# rounds_for <experiment>: the number of rounds T of its runs (ITERATIONS when set). Must match the
# network.iterations its run.sh sets (tests/scripts/experiments/test_experiment_lib.py checks it).
rounds_for() {
  if [[ -n "${ITERATIONS:-}" ]]; then
    echo "${ITERATIONS}"
    return 0
  fi
  case "$1" in
    01 | 02 | 05) echo 40 ;;
    04 | 06 | 07 | 08 | 09 | 10 | rb) echo 20 ;;
    *) echo "ERROR: no round count for experiment $1" >&2; return 1 ;;
  esac
}

# snapshot_complete <dir> <T>: the run in <dir> finished round T. The pipeline writes
# iteration_<N>/snapshot_metadata.json last, after the round's archives (docs/outputs.md), so a run
# that stopped at start-up or before round T, or a shorter smoke run, does not count.
snapshot_complete() { [[ -f "$1/iteration_$2/snapshot_metadata.json" ]]; }

# latest_complete_snapshot <experiment> <condition> <dataset> <seed>: the newest matching run that
# finished its last round; newer unfinished matches are skipped with a message.
latest_complete_snapshot() {
  local rounds dir matches=()
  rounds="$(rounds_for "$1")" || return 1
  mapfile -t matches < <(compgen -G "snapshots/$(run_name "$@")_*" || true)
  (( ${#matches[@]} > 0 )) || return 1
  while IFS= read -r dir; do
    if snapshot_complete "${dir}" "${rounds}"; then
      echo "${dir}"
      return 0
    fi
    echo "==> skipping ${dir}: not a complete run (no iteration_${rounds}/snapshot_metadata.json)" >&2
  done < <(ls -1dt "${matches[@]}")
  return 1
}

# snapshot_var <experiment> <condition> <dataset> <seed>: the variable that sets one run's snapshot,
# SNAPSHOT_DIR_<EXPERIMENT>_<DATASET>_<CONDITION>_SEED_<SEED> (the run name without the prefix).
snapshot_var() {
  echo "SNAPSHOT_DIR_$(echo "$1_$3_$2_seed_$4" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9\n' '_')"
}

# snapshot_dir <experiment> <condition> <dataset> <seed>: explicit override or latest complete match.
# An override must be a folder; one that did not finish its last round is used with a warning.
snapshot_dir() {
  local experiment="$1" condition="$2" dataset="$3" seed="$4" var rounds
  var="$(snapshot_var "$@")"
  if [[ -n "${!var:-}" ]]; then
    if [[ ! -d "${!var}" ]]; then
      echo "ERROR: ${var}=${!var} is not a folder" >&2
      return 1
    fi
    rounds="$(rounds_for "${experiment}")" || return 1
    if ! snapshot_complete "${!var}" "${rounds}"; then
      echo "WARNING: ${var}=${!var} has no iteration_${rounds}/snapshot_metadata.json (not a complete run); ranking it as set" >&2
    fi
    echo "${!var}"
    return 0
  fi
  local pattern="snapshots/$(run_name "${experiment}" "${condition}" "${dataset}" "${seed}")_*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "<latest:${pattern}>"
    return 0
  fi
  if ! latest_complete_snapshot "${experiment}" "${condition}" "${dataset}" "${seed}"; then
    echo "ERROR: no complete run for ${pattern} (set ${var} to point at one)" >&2
    return 1
  fi
}

# rank_pair <experiment> <dataset> <seed> <cond_a> <exp_a> <cond_b> <exp_b> <rubric> <top_n> <sampling> [n_pairs]
# exp_a/exp_b: the experiment id whose runs hold each condition (lets 02 compare against 01).
# sampling: all | random (random uses n_pairs, or every possible pair if there are fewer).
rank_pair() {
  local experiment="$1" dataset="$2" seed="$3" cond_a="$4" exp_a="$5" cond_b="$6" exp_b="$7"
  local rubric="$8" top_n="$9" sampling="${10}" n_pairs="${11:-0}"
  local dir_a dir_b
  dir_a="$(snapshot_dir "${exp_a}" "${cond_a}" "${dataset}" "${seed}")"
  dir_b="$(snapshot_dir "${exp_b}" "${cond_b}" "${dataset}" "${seed}")"
  local sampling_args=("sampling.mode=${sampling}" "sampling.seed=${seed}")
  if [[ "${sampling}" == "all" ]]; then
    sampling_args+=("sampling.n_repeats=1")
  else
    sampling_args+=("sampling.n_pairs=${n_pairs}" "sampling.cap_n_pairs=true")
  fi
  emit "${experiment}" "${cond_a}_vs_${cond_b}__${rubric}" "${dataset}" "${seed}" rank -- \
    "${PYTHON}" -m ranker.run_sample_and_rank_paired \
    "hydra.run.dir=outputs/ranker/${experiment}/${dataset}/${cond_a}_vs_${cond_b}/${rubric}/seed${seed}" \
    "experiments.experiment_a.snapshot_dirs=[\"${dir_a}\"]" \
    "experiments.experiment_b.snapshot_dirs=[\"${dir_b}\"]" \
    "experiments.experiment_a.label=${dataset}_${cond_a}" \
    "experiments.experiment_b.label=${dataset}_${cond_b}" \
    "exp_name=${BASE_NAME}_${experiment}_${dataset}_${cond_a}_vs_${cond_b}__${rubric}__top${top_n}_seed${seed}" \
    "ranker_cfg@ranker=${rubric}" \
    "ranker.llm_backend=${JUDGE:-gpt5-high}" \
    "parallelism.max_workers=${MAX_WORKERS:-8}" \
    "${sampling_args[@]}" \
    "paper_selection.iterations=last" \
    "paper_selection.cumulative=true" \
    "paper_selection.top_n.enabled=true" \
    "paper_selection.top_n.n=${top_n}" \
    "paper_selection.balance.enabled=true" \
    "paper_selection.balance.method=top_metareview" \
    "paper_selection.balance.seed=${seed}" \
    "SEED=${seed}"
}
