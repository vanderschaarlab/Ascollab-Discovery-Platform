# Reproducing the paper's experiments

Each experiment has a folder in [`scripts/experiments/`](../scripts/experiments/README.md) with a
`run.sh` that runs the pipeline for every condition (03's replays the review stage instead) and,
where the paper compares conditions with the AutoRanker, a `rank.sh`. This page describes what
each experiment runs; the folder README lists the common options.

Before you start you need the endpoint files ([models.md](models.md)), the data
([data.md](data.md)) and a running sandbox server ([sandbox.md](sandbox.md)). Every run calls your
model endpoints, so try a script with `DRY_RUN=1` (prints the commands) and a small smoke run
first:

```bash
DRY_RUN=1 bash scripts/experiments/01_network_ablations/run.sh
DATASETS=kirc ITERATIONS=1 NUM_AGENTS=2 CONDITIONS=ascollab \
  bash scripts/experiments/01_network_ablations/run.sh
```

## How the scripts work

- **Settings.** Each run starts from [`sys_config/ascollab.yaml`](../sys_config/ascollab.yaml) and
  sets, explicitly, every setting that differs between experiments (number of rounds, network
  setting and tools, model mix, prior papers, timeouts, concurrency and the optional prompt and
  feedback features). Experiments 01–05 share one group of these values and 06–10 another
  (`profile_main` and `profile_extended` in
  [`lib/common.sh`](../scripts/experiments/lib/common.sh)); each condition then changes what it
  manipulates. `DRY_RUN=1` shows the full command for every run.
- **Run names.** A run is named `ascollab_<experiment>_<dataset>_<condition>_seed_<k>`; its
  snapshots go to `snapshots/<run name>_<timestamp>/`. `rank.sh` takes the latest matching run
  that finished its last round T (T is the experiment's, or `ITERATIONS` when set; the round's
  `iteration_<T>/snapshot_metadata.json` exists) and names the newer runs it skips (stopped at
  start-up or early, or a shorter smoke run).
  `SNAPSHOT_DIR_<EXPERIMENT>_<DATASET>_<CONDITION>_SEED_<k>` (for example
  `SNAPSHOT_DIR_05_KIRC_N1_SEED_1`) sets one run's folder instead.
- **Seeds.** `SEEDS="1 2 3"` runs every condition once per seed (default `1`).
- **Ranking.** `rank.sh` (except 03's, see below) calls `ranker.run_sample_and_rank_paired` once
  per condition pair, rubric, dataset and seed. From each run it takes the papers accepted up to
  the last round (`paper_selection.iterations=last`, `cumulative=true`), keeps the top `TOP_N` by
  meta-review score on each side (balanced), and judges pairs of one paper from each side: every
  pair (`all`), or a random sample. Results go to
  `outputs/ranker/<experiment>/<dataset>/<a>_vs_<b>/<rubric>/seed<k>/`. The judge is
  `JUDGE` (default `gpt5-high`). See [ranker.md](ranker.md).
- **Statistics.** For the ranker results of one experiment:

  ```bash
  python -m scripts.analysis.stat_tests --ranker-dir outputs/ranker/01 --out <output dir>
  ```

  `stat_tests.csv` (and `.json`) has one row per condition pair, rubric and dataset, plus a
  `macro` row per pair and rubric when there are several datasets (it has only `n`, the win rate
  and the bootstrap interval): the win rate of the first
  condition (`condition_a`) with A/B position randomisation undone, an exact two-sided binomial
  test against 0.5 (`binom_p`, and `binom_p_holm` adjusted over the run's per-dataset rows), a
  paper-clustered bootstrap 95% interval (`bootstrap_low`/`high`, the main interval: each paper
  appears in many pairs) and a Wilson interval. Seeds of a dataset are pooled, each run's papers clustered separately; the macro row weights datasets equally. It writes
  only to `--out`.

**Order.** Run 01 before 02, 03 and their rankings, which use 01's runs. Run each `rank.sh` after
its `run.sh` has finished. 07, 08 and 10 share one baseline run (experiment ID `rb`): whichever
script runs first creates it and the others reuse it once it has finished all its rounds (as
`rank.sh` decides; `FORCE_BASELINE=1` reruns it).

**Size.** A run is T rounds of N research sessions (up to M steps each), K reviews per paper and
about N / L meta-review tournaments per round. A ranker comparison judges `TOP_N`² pairs with
`all` sampling, or `N_PAIRS` pairs with random sampling, for each rubric.

**Data check.** Every script runs with the default `data.preflight_strict: true`: a dataset file
that is missing, or whose checksum differs from `sys_config/dataset/checksums.yaml`, stops the run
at start-up ([data.md](data.md)).

## 01: Network ablations

The full system against simpler networks.

| Condition | What changes |
|---|---|
| `single_agent` | N = 1, one GPT-5 agent, which reviews its own papers (K = 1, self-review); no collaboration tools or code collaboration |
| `indep_homo` | 16 GPT-4o agents with identical profiles; no collaboration and no archive or registry search |
| `collab_homo` | 16 GPT-4o agents with identical profiles; collaboration and social memory on |
| `ascollab` | The full system: generated profiles, the 6 / 5 / 5 GPT-4o / GPT-5 / DeepSeek mix |

Shared settings: T = 40, 480 s per code cell, 720 s per research session, no set cap on concurrent
research agents (`null`: the cap follows the sandbox's `SANDBOX_MAX_SESSIONS`), prior papers off.
Datasets: `kirc`, `luad`, `paad`, `nhanes`.

`rank.sh`: all six condition pairs × the novelty, significance, soundness and overall (`default`)
rubrics; top 25 papers per side; all pairs.

## 02: Fixed-workflow baseline

The 01 `ascollab` condition with a fixed workflow: each agent is assigned a collaborator
(`assignment_policy=weakest_agent`) and must establish the collaboration before running code and
message the collaborator before finishing; registry search is removed. All
`research.fixed_workflow.*` settings are in the script. Dataset: `kirc`.

`rank.sh`: `fixed_workflow` against each 01 condition on the same dataset and seed, four rubrics,
top 25, all pairs. Needs the 01 runs.

## 03: Reviewer matching

Expertise-matched against random reviewer assignment. Rather than new runs, this
replays the review and meta-review stages on fixed cohorts of papers taken from an experiment-01
`ascollab` run: 5 cohorts of 20 papers at each of rounds 5, 10 and 20 (`TARGET_ITERATIONS`), once
with each strategy. Dataset: `nhanes`; needs the 01 run for it.

- `run.sh` runs the reviews (`scripts.analysis.reviewer_matching.run_grid --skip-autoranker`).
- `rank.sh` ranks each cohort's papers with the AutoRanker (`--skip-reviews`; all pairs within the
  cohort, with `JUDGE` and `MAX_WORKERS` when set, otherwise the judge and parallelism of
  `sys_config/sample_and_rank_non_paired.yaml`: `gpt5-high`, 4 workers), then compares how well each strategy's accepted papers score
  (`scripts.analysis.reviewer_matching.analyze_results`: `per_condition_results.csv`,
  `per_cohort_deltas.csv`, `aggregate_results.csv`).

Outputs go to `outputs/reviewer_matching/<dataset>_seed_<k>/` (`ROOT_OUT`). `rank.sh` takes the 01
run from the cohort manifests `run.sh` wrote there (`manifests/`), so a newer 01 run does not
change the cohorts. Given a different 01 run for an output folder that already holds cohorts,
`run_grid` stops; use another `ROOT_OUT` for it.

## 04: Feedback

Feedback to authors on their previous papers. Every condition is a homogeneous
network of 16 GPT-4o agents for T = 20 rounds, with prior papers (in full) and strict citations,
self-reflection, journal-style reports and graph logic. The conditions differ in
`research.metacognition_feedback.mode`:

| Condition | Feedback |
|---|---|
| `no_feedback` | None |
| `meta_review_only` | Written by a model from the agent's previous papers: title, abstract, a manuscript excerpt, meta-review and up to 4 reviews each |
| `meta_review_with_metrics` | The same, plus the papers' metrics |
| `metrics_only` | A fixed template of the metric scores, with no model call |

Dataset: `kirc`. `rank.sh`: each feedback condition against `no_feedback`, four rubrics, top 25,
500 random pairs (`N_PAIRS`; every pair, with a warning, if fewer exist).

`python -m scripts.analysis.feedback_summary <snapshot dirs>` summarises the runs: papers,
acceptance, meta-review scores, metrics, session totals (collaborators' work inside another
agent's session counted under `collab_sessions`) and how many agents received feedback.

## 05: Network scaling

The full system with N = 1, 2, 3, 4, 5, 10 and 16 agents. Smaller networks keep the
6 / 5 / 5 model proportions (`mix_for` in the script), K = min(2, N − 1), and N = 1 uses the
single-agent settings of 01 (self-review, no collaboration). Otherwise as 01 `ascollab`. Dataset:
`kirc`.

`rank.sh`: each N against N = 16, four rubrics, top 25, all pairs.

## 06–10: shared settings

Experiments 06–10 use T = 20 rounds, a 6 / 6 / 4 GPT-4o / GPT-5 / DeepSeek mix (06: 6 / 7 / 3),
prior papers in full with strict citations, 900 s per code cell and 3600 s per research session
(06's open-ended arm: 720 s), and at most 10 concurrent research agents. Dataset: `kirc`. Each
condition of 07–10 differs from the shared baseline only in its own switch (07's `no_reviewers`
also has no reviewer-assignment label, which does nothing without reviewers); the tests in
`tests/system/test_social_experiment_configs.py` check this.

## 06: Closed vs open-ended task

The same network directed in two ways. `open_ended` uses the default goal ("make scientific
discoveries"); `closed` sets `discovery_dataset.target=kirc_closed_hypotheses`, whose goal asks the
agents to test three prespecified hypotheses (therapeutic benefit from inhibiting ABL1 signalling,
CDK4/6 and JAK signalling) and report positive, negative and contradictory evidence. See
[Directing the system](how_it_works.md#directing-the-system).

There is no `rank.sh`: the paper compares the closed arm's findings with a published study.

## 07: No peer review

`no_reviewers` sets `metareview.reviewerless.enabled=true`: there is no review stage, and the
meta-reviewer scores the papers directly, with a prompt that does not mention reviews. Compared
with the shared baseline.

Settings that need reviews (reputation, agent pausing, feedback) stop the run at start-up if
combined with this switch.

`rank.sh`: baseline against `no_reviewers`, overall rubric, top 15 per side, all pairs.

## 08: Markovian social memory

`markovian_memory` sets `network.social_memory.mode=markovian`: at the start of each round the
public archive keeps only the papers accepted in the previous round. Everything that reads the
public archive sees the pruned version (agents' archive search, the originality metrics, the public
archive in snapshots); the private archive, which the ranker reads, is never pruned. Each round
writes `social_memory_pruning.json` to its snapshot. Compared with the shared baseline.

`rank.sh`: as 07.

## 09: Reviewer norms

Every reviewer prompt carries a fixed norm paragraph (`+review_norm@review.norm_prompt=<norm>`;
the texts are in `sys_config/review_norm/`):

| Condition | Norm |
|---|---|
| `norm_a_novelty` | Prioritise potentially groundbreaking papers even if they are not strongly supported |
| `norm_b_soundness` | Prioritise papers with sound empirical support |

The meta-reviewer prompt is unchanged. `rank.sh`: `norm_a_novelty` against `norm_b_soundness`,
novelty, significance and soundness rubrics, top 15, all pairs.

## 10: Number of reviewers

K = 1, 2, 3 and 4 reviewers per paper. The K = 2 arm is the shared baseline. Each arm salts reviewer
assignment with its own label (`review.reviewer_assignment_condition_label='n_reviewers=<K>'`), so
the arms draw independent assignments.

`rank.sh`: K = 1, 2 and 3 each against K = 4, novelty, significance and soundness rubrics, top 15,
all pairs.

## Expert validation

The notebooks in [`notebooks/expert_validation/`](../notebooks/expert_validation/) check the
ranker against human raters: pick papers across the ranker's quality range (01), export them as
PDFs for raters (02), and compare the raters' rankings with the ranker's (03). The repository
contains no ratings; the notebooks describe the file format for your own. You need Jupyter
(`pip install jupyter`), and for notebook 02 also pandoc and xelatex.
