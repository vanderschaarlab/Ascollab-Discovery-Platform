# Experiment scripts

One folder per experiment. `run.sh` runs the pipeline for each condition (03's replays the review
stage on an 01 run); `rank.sh` compares the conditions with the AutoRanker (06 has none). Run
both from any directory; they `cd` to the repository root. What each experiment runs, compares and
writes is described in
[docs/reproducing_paper.md](../../docs/reproducing_paper.md).

| Folder | Experiment | Conditions | Default datasets |
|---|---|---|---|
| `01_network_ablations` | Network ablations | `single_agent`, `indep_homo`, `collab_homo`, `ascollab` | kirc luad paad nhanes |
| `02_fixed_workflow` | Fixed-workflow baseline | `fixed_workflow` (ranked against 01) | kirc |
| `03_reviewer_matching` | Reviewer matching | expertise-matched vs random reviewers, replayed on a 01 `ascollab` run | nhanes |
| `04_feedback` | Feedback | `no_feedback`, `meta_review_only`, `meta_review_with_metrics`, `metrics_only` | kirc |
| `05_network_scaling` | Network scaling | N = 1, 2, 3, 4, 5, 10, 16 agents | kirc |
| `06_closed_vs_open_task` | Closed vs open-ended task | `open_ended`, `closed` | kirc |
| `07_no_peer_review` | No peer review | shared baseline, `no_reviewers` | kirc |
| `08_markovian_memory` | Markovian social memory | shared baseline, `markovian_memory` | kirc |
| `09_reviewer_norms` | Reviewer norms | `norm_a_novelty`, `norm_b_soundness` | kirc |
| `10_reviewer_count` | Number of reviewers | n = 1, 2 (shared baseline), 3, 4 | kirc |

## Common options

Set as environment variables:

- `DATASETS="kirc luad"`: which dataset bundles to run (`sys_config/dataset/<bundle>.yaml`, passed as `dataset=<bundle>`).
- `SEEDS="1 2 3"`: which seeds to run.
- `DRY_RUN=1`: print the commands instead of running them.
- `SNAPSHOT_DIR_<EXPERIMENT>_<DATASET>_<CONDITION>_SEED_<SEED>=…`: use this snapshot folder for
  one run (its run name without the `ascollab_` prefix, upper-case), for example
  `SNAPSHOT_DIR_01_KIRC_ASCOLLAB_SEED_1=snapshots/…`; it applies wherever that run is used (01's
  runs also in 02 and 03). The shared baseline is `SNAPSHOT_DIR_RB_KIRC_BASELINE_SEED_<k>`, which
  07, 08 and 10 then reuse instead of running it; it must be a complete run. A folder that has not
  finished its last round is ranked with a warning. Without it, `rank.sh` takes the latest run of
  the condition that finished its last round (the experiment's T, or `ITERATIONS` when set) and
  names the newer, unfinished runs it skips.

- `ITERATIONS=1 NUM_AGENTS=2`: a small smoke run (one round, two agents; the model mix is scaled
  down and reviewers per paper set to N-1, at most 2). Conditions that set their own network size
  keep it. Conditions that set their own reviewers per paper (the shared baseline, 08, 09, 10) keep
  that too, so with two agents their papers can get fewer reviewers than asked for (with a warning).

The rest, including `FORCE_BASELINE=1` (rerun the shared baseline), are listed at the top of
`lib/common.sh`.

Each script sets every setting that differs between experiments explicitly. Settings shared by all
experiments come from `sys_config/ascollab.yaml`.

## Order

1. Run 01 first: 02, 03 and the rankings that compare against 01 need its runs.
2. 07, 08 and 10 share one baseline run. Whichever script runs first creates it, and the others
   reuse it once it has finished all its rounds.
3. Run `rank.sh` after the corresponding `run.sh` has finished.

## Statistics

`rank.sh` writes each comparison to `outputs/ranker/<NN>/<dataset>/<a>_vs_<b>/<rubric>/seed<k>/`.
Compute win rates, exact binomial tests (Holm-adjusted), Wilson and paper-clustered bootstrap
intervals into a directory of your choice:

```bash
python -m scripts.analysis.stat_tests --ranker-dir outputs/ranker/01 --out <output dir>
```

Experiment 03 is analysed by its own `rank.sh` (`scripts.analysis.reviewer_matching.analyze_results`),
and `python -m scripts.analysis.feedback_summary` summarises experiment 04's runs. Experiment 06 has
no ranking step.
