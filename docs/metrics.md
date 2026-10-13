# Review metrics

Before reviewers see a round's papers, the pipeline computes 25 automatic metrics for every paper
submitted for review in that round. They come from three cheap sources:

- the paper's research session: tool calls, code executions, lines of code;
- the manuscript text: keyword and regular-expression heuristics;
- the public archive: embedding similarity to earlier accepted papers.

The entry point is `build_review_metrics_for_papers` in
[`system/review_metrics.py`](../system/review_metrics.py). No chat model is called. The
embedding-based metrics use the public archive's embedder, so they do make embedding API calls.

## Where the metrics are used

| Consumer | What it gets |
|---|---|
| Reviewers | The paper's metrics as a JSON block in the review request. The reviewer prompt describes each enabled metric and tells reviewers to base their numeric ratings mainly on the metrics. Reviewers may move a rating up or down by one point when their reading of the paper strongly supports it, and are asked to cite the metric values behind their overall recommendation. |
| Feedback | With `research.metacognition_feedback.enabled=true` [false], the modes `meta_review_with_metrics` and `metrics_only` use the previous round's metrics in each agent's feedback (experiment 04 in [reproducing_paper.md](reproducing_paper.md)). |
| Snapshots | `iteration_<N>/metrics.jsonl`, plus a `review_metrics` field on each paper in the private archive ([outputs.md](outputs.md)). |
| Run log | One `[signal]` line per metric per round with mean, std, min and max, and a `collapsed metric` warning when a metric has the same value for every paper. With reputation or agent pausing on and review metrics computed (not with `review.metrics_pipeline=none`), one `[signal] reputation` line per round (after the reputation update) gives the spread of the agents' reputation scores and warns when its std is below 0.03. |
| Ranker | The sample-and-rank CLIs copy each pool paper's metrics into their `papers/paper_data.json` ([ranker.md](ranker.md)). |

The meta-reviewer does not see the metrics.

## Switching metrics on and off

`review.metrics_pipeline` [default] selects the mode:

| Value | Behaviour |
|---|---|
| `default` | Computes all 25 metrics and sends them all to reviewers. `review.metrics_allowlist` is ignored. |
| `experimental` | Keeps only the metrics in `review.metrics_allowlist`. The reviewer prompt describes only those. |
| `none` (or `off`, `disabled`) | Computes nothing. No metrics block is sent to reviewers, no `metrics.jsonl` is written, and each paper's `review_metrics` is empty. |

An unknown value falls back to `default` with a warning. In `none` mode the reviewer system prompt
also leaves out its description of the metrics and does not ask reviewers to ground their ratings
in metric values (other modes, and the review replay of experiment 03, keep them).

`review.metrics_allowlist` (used only in `experimental` mode) is a list of entries in any of these
forms:

| Entry | Selects |
|---|---|
| `support.lines_of_code` | One metric |
| `support`, `support.*` or `support.all` | Every metric in a group |

A `null` or empty list selects every metric. Unknown entries are logged, mentioned in the reviewer
prompt, and otherwise ignored. A group with no selected metric is left out of the output entirely.
The default list in [`sys_config/ascollab.yaml`](../sys_config/ascollab.yaml) names all 25 metrics.

```bash
python -m system.main "review.metrics_pipeline=experimental" \
  "review.metrics_allowlist=[support,originality.embedding_novelty_percentile]"
```

Two related settings:

- The feedback feature filters metrics separately, with `research.metacognition_feedback.axes`
  [all four groups] and `research.metacognition_feedback.metrics_allowlist` [null, meaning all].
  It reads the metrics computed in the previous round, so it gets no metric values when
  `review.metrics_pipeline=none`.
- `review.metric_weights` is not about these metrics. It weights the four review rating axes per
  agent in the reviewer prompt, and is empty by default.

## The metrics

Metrics are grouped like the review ratings. "Archive" means the public archive of accepted
papers at the time of review.

| Group | Metric | Meaning |
|---|---|---|
| support | `external_papers_read` | Number of `arxiv` and `search_internal_archive` tool calls in the research session, collaborators' calls included (calls, not papers) |
| support | `successful_code_executions` | Number of successful `execute_code` calls |
| support | `lines_of_code` | Non-blank, non-comment lines sent to `execute_code`, failed calls included |
| support | `evidence_types_present` | Kinds of evidence named in the text: `numerical_data`, `statistical_test`, `causal_graph_dag`, `survival_plot_hazard_ratio`, `correlation_regression_clustering` |
| support | `num_statistical_code_cells` | `execute_code` calls whose code mentions a statistics library or method (scipy, statsmodels, lifelines, Cox, AUC, p-value, ...); equals `successful_code_executions` when none match |
| support | `evidence_breadth_index` | `external_papers_read` plus the number of evidence types |
| support | `analytical_depth` | Same value as `num_statistical_code_cells` |
| support | `claim_evidence_alignment_max_cosine` | Highest cosine similarity in [0, 1] between a claim-like and an evidence-like sentence; 0.5 for texts under two sentences |
| soundness | `internal_consistency_score` | Starts at 0.45, rises for rigour phrases (confidence intervals, sensitivity analysis, cross-validation, multiple-testing correction, ...), falls for weakness phrases (post hoc, underpowered, unadjusted, ...); clamped to [0, 1] |
| soundness | `num_statistical_validity_checks_total` | How many of 13 kinds of model diagnostic the text mentions (VIF, residuals, normality, proportional hazards, calibration, ...) |
| soundness | `num_statistical_validity_checks_passed` | Those whose first mention has no failure word ("failed", "violated", "not met", "issue", ...) |
| soundness | `diagnostic_validity_pass_rate` | passed / total; 0 when total is 0 |
| significance | `clinical_biological_relevance` | Number of 16 clinical keywords found (patient, cohort, therapy, biomarker, ...) divided by 8, capped at 1 |
| significance | `clinical_relevance_score` | Same value as `clinical_biological_relevance` |
| significance | `effect_size_strength` | Mean strength score of effect sizes reported in the text (hazard, odds and risk ratios, AUC, Cohen's d, correlations, coefficients, fold changes); 0 when none is found |
| significance | `distance_to_prior_mean`, `_min`, `_max` | Mean, minimum and maximum of 1 − similarity to the five archive papers a search for the paper's abstract returns (fewer if `memory.max_k` [5] is lower). With the citation boost on [default] these are the top five by boosted score among the ten most similar. Null with fewer than two neighbours |
| originality | `hypothesis_overlap_jaccard` | Mean word overlap (Jaccard, [0, 1]) between this paper's hypothesis sentences and those of the archive papers it cites and the five archive papers its search returns; lower is more novel; null without prior papers |
| originality | `concept_pair_rarity` | Mean −log co-occurrence frequency, across archive texts, of pairs of the paper's content words; higher means rarer combinations |
| originality | `concept_entropy` | Shannon entropy (natural log) of the paper's content-word frequencies; higher means a broader vocabulary |
| originality | `semantic_novelty_knn_distance` | Mean 1 − similarity to up to 10 archive papers from the same search, capped by `memory.max_k` [5] |
| originality | `embedding_novelty_percentile` | Percentile (0–100) of `semantic_novelty_knn_distance` among this round's papers; null when the paper has no distance |
| originality | `methodological_novelty_flag` | True when one sentence has a method word (method, model, pipeline, framework, ...) and a novelty word (novel, new, first, state-of-the-art, ...) |
| originality | `mechanistic_novelty_flag` | True when the text says "novel mechanism/pathway/interaction", "previously unreported/undescribed" or "new mechanistic/pathway" |

## Output

Each line of `snapshots/<run_name>/iteration_<N>/metrics.jsonl` is one paper:

```json
{
  "iteration": 3,
  "paper_id": "paper_t03_agent_05",
  "primary_agent_id": "agent_05",
  "paper_title": "...",
  "metrics": {
    "support": {"external_papers_read": 4, "successful_code_executions": 9, "...": "..."},
    "soundness": {"internal_consistency_score": 0.58, "...": "..."},
    "significance": {"distance_to_prior_mean": 0.21, "...": "..."},
    "originality": {"embedding_novelty_percentile": 62.5, "...": "..."}
  }
}
```

The values above are only illustrations. To plot per-agent metric trajectories over a run as
interactive HTML (written to `plots/` by default):

```bash
python -m scripts.tools.plot_metrics --run-name <run_name>
```
