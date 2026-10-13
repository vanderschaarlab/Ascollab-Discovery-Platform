# Outputs

A pipeline run writes two trees: a Hydra run directory under `outputs/` with the logs and the
composed config, and a snapshot directory under `snapshots/` with the full state of the system
after every round. This page lists each file, when it is written and what it holds, and how to
resume a run. How to start and monitor runs is in [running.md](running.md).

## Where things go

```text
outputs/<YYYY-MM-DD>/<HH-MM-SS>/          one per invocation (Hydra run dir)
├── .hydra/config.yaml                    composed config, saved as written (not resolved)
├── .hydra/hydra.yaml, .hydra/overrides.yaml
├── main.log                              Hydra's job log
└── run.log                               pipeline log, including printed output (10 MB rotation)

snapshots/<run_name>/                     one per run; a resumed run continues in it
├── data_manifest.json                    dataset preflight report
├── data_card.json                        compact dataset summary
├── notepads/<agent_id>/notepad.jsonl     experimental notepad only
├── iteration_1/ ... iteration_<T>/       one snapshot per round
└── iteration_<N>__incomplete_<ts>/       an unfinished round set aside by resume
```

Both trees are relative to the directory you launch from (the repository root). The Hydra folder
name is Hydra's default; `hydra.run.dir` changes it ([running.md](running.md#run-name-and-outputs)).
The ranker CLIs write their own Hydra run directories under `outputs/_ranker/`,
`outputs/_sample_and_rank_paired/` and `outputs/_sample_and_rank_non_paired/`
([ranker.md](ranker.md)).

**Run name.** `run_name` defaults to `${exp_name}_${now:%y%m%d_%H%M}`, for example
`ascollab_261008_1332`. The experiment scripts set `exp_name` to
`<base>_<experiment>_<dataset>_<condition>_seed_<seed>`, which gives run names such as
`ascollab_01_kirc_ascollab_seed_1_260101_1200`.

**Iterations.** `iteration_<N>` holds the state after round N (1-based). Plots are written into it
while the round runs; the other files are written at the end of the round, and
`snapshot_metadata.json` is written last.

**Scope.** Each file below is marked with one of these scopes:

- *Current*: covers round N only.
- *Cumulative*: the state after round N, including everything from earlier rounds. The last
  iteration directory alone is enough for most analysis.
- *Run*: the same in every round, except that rounds after a resume also record the resume
  settings.

**Formats.** JSON files are UTF-8 with `indent=2`; JSONL files hold one JSON object per line. The
archive `.pkl` files hold the same plain data as their `.json` twins (resume reads the `.pkl`
first).
`agent_registry_data.pkl` and `session_stats_data.pkl` contain objects from
[`system/schema.py`](../system/schema.py) and [`system/trackers.py`](../system/trackers.py), so
unpickle them with the repository on the Python path.

## Run-level files

| File | Written when | Contents |
|---|---|---|
| `data_manifest.json` | At the start of every invocation, unless the preflight is off (`data.preflight` [true], or `DISABLE_PREFLIGHT=1`); written even when the check fails | Top level includes `ok`, `errors`, `cwd` and `dataset_root`. Per dataset key: `ok`, `paths`, `tried_paths` and per file `path`, `exists`, `size_bytes`, `schema` (`n_rows`, `n_cols`, `columns`, `dtypes`), `sample_head` (first rows), `sha256` |
| `data_card.json` | Same, when the preflight passes, or fails with `data.preflight_strict=false` | `ok`; per dataset key, the files found, with `path`, `exists`, `n_rows`, `n_cols` and the first 10 `columns` |

These are a record of the data the run saw; the pipeline does not read them back. Both hold
absolute paths, and the manifest holds sample rows of your data. See [data.md](data.md) for the
dataset registry the preflight checks.

## Snapshot files

### Metadata and config

| File | Written | Scope | Contents and use |
|---|---|---|---|
| `snapshot_metadata.json` | Every round, last | Current | `iteration`, `run_name`, `timestamp` (`YYYYMMDD_HHMMSS`), `snapshot_created_at` (ISO 8601), `num_agents`, `num_papers_public`, `files_saved`, `citation_tags`; `meta_researcher` and `metacognition_feedback` summaries when those features are on. Marks the snapshot as finished; resume takes the run name from it |
| `exp_config.yaml` | Every round | Run | The full config the run used, saved with `OmegaConf.save`. Every key named `api_key` is replaced by `REDACTED`; interpolations stay unresolved (for example `run_name: ${exp_name}_${now:%y%m%d_%H%M}`; after a resume, `run_name` holds the literal name). Resume loads it as the config baseline |

`files_saved` lists the snapshot's files, with `plots/` as a folder and without `timelines/`. `citation_tags` is the citation-tag check for this round: `research.entries` has one
entry per paper written this round (statuses, `citation_compliant`, tag `counts`, `errors`,
`invalid_archive_refs`, `code_evidence`, the parsed `tags`), `review.entries` one per review, and
`research.citation_integrity` counts accepted papers that failed the check and the archive
citations pointing to them, with `alerts`. `schema` and `trust_policy` restate the tag formats and
the rule that every accepted paper can be cited as an `[Archive: ...]` source.

### Archives and papers

| File | Written | Scope | Contents and use |
|---|---|---|---|
| `archive_data.json`, `archive_data.pkl` | Every round | Cumulative | Public archive: `papers` (flat list of accepted papers) and its retrieval settings `with_citation_boost`, `with_recency_boost`, `w_citation`, `w_recency`, `max_k`. With `network.social_memory.mode=markovian` it holds only the papers of rounds N−1 and N |
| `private_archive_data.json`, `private_archive_data.pkl` | Every round | Cumulative | Private archive: `papers` (a list of lists, one inner list per author), `reviews_by_paper`, `review_assignments` (empty in pipeline runs) and the same retrieval settings. Holds every paper that reached a meta-review decision, accepted or rejected, and is never pruned. The ranker and the analysis scripts read it |

Each paper record (the same in both archives and in the ranker pools) has these fields:

| Field | Meaning |
|---|---|
| `paper_id` | `paper_t<NN>_<agent_id>`, for example `paper_t02_agent_12`. Suffix `_meta` for meta-research papers; `_incomplete` or `_timeout` when the session did not finish (a timed-out paper's `manuscript` is `TIMEOUT`) |
| `primary_agent_id`, `collab_agent_ids` | Author and collaborators |
| `title`, `abstract`, `manuscript` | The Markdown final report, and the title and abstract taken from it |
| `code_script` | The code the session ran |
| `publication_t` | The round the paper was written in |
| `cited_paper_ids` | Outgoing archive citations, as a list of `{paper_id, agent_id}` |
| `citation_count` | Incoming citations: how many accepted papers cite this one |
| `metareview` | `paper_id`, `meta_review_text`, `overall_score` (0 to 1), `rank` (for example `"1/4"`), `justification` (null), `decision` (`accept` or `reject`) |
| `review_metrics` | The paper's metrics, as in `metrics.jsonl` |
| `status`, `publication_status` | `accepted` or `rejected` once decided |
| `research_validation_status` | Outcome of the research-stage report checks: `under_review` if the report passed, otherwise a reason such as `invalid_citation_report` or `invalid_action_report`. Sessions that did not finish also keep `under_review`; their `paper_id` ends in `_incomplete` or `_timeout` |
| `citation_compliant`, `citation_errors`, `citation_error_count`, `invalid_archive_refs` | Citation-tag check of the manuscript; `invalid_archive_refs` lists cited archive ids that are not accepted papers |
| `is_citable_archive_source`, `citable_with_warnings` | True for accepted papers; the second is true when an accepted paper failed the citation check |
| `plot_artifacts`, `plot_filenames`, `plot_handles`, `plot_handle_to_filename` | Plots available to the paper under graph logic ([plots_and_graph_logic.md](plots_and_graph_logic.md)) |
| `paper_kind`, `research_phase`, `research_session_id` | `research`, `primary` and `<agent_id>_t<NN>` for normal papers |
| `meta_research_*`, `meta_quality_*`, `meta_revision_attempts`, `meta_*_count`, `meta_claim_checks`, `citation_lint` | Meta-research fields (experimental); defaults otherwise |

### Reviews and meta-reviews

| Location | Written | Scope | Contents |
|---|---|---|---|
| `private_archive_data.*` → `reviews_by_paper[paper_id]` | After each review stage | Cumulative | List of `{iteration, reviewer_id, review_text, scores}`. `scores` has `Support`, `Soundness`, `Significance`, `Originality` (1 to 4) and `Overall Recommendation` (1 to 5) |
| Paper field `metareview` | After meta-review | Cumulative | Score, rank in its tournament and decision (see the table above) |
| `snapshot_metadata.json` → `citation_tags.review` | Every round | Current | Citation tags found in this round's reviews |
| `timelines/<reviewer_id>_<paper_id>_timeline.jsonl` | Every round | Current | Events of one review session |

With `metareview.reviewerless.enabled=true` there are no reviews: `reviews_by_paper` gains no
entries and only the meta-review is recorded. How reviewers are assigned and how acceptance works
is in [how_it_works.md](how_it_works.md#review).

### Agents and the registry

| File | Written | Scope | Contents and use |
|---|---|---|---|
| `agent_registry_data.pkl` | Every round | Cumulative | `agents` (a list of `AgentProfile`), `max_citations_seen` and the registry's retrieval settings `with_citation_boost`, `w_sim`, `w_cit`, `max_k`. Resume restores the agents and these settings from it |

`AgentProfile` fields:

| Field | Meaning |
|---|---|
| `agent_id` | `agent_01`, `agent_02`, ... |
| `behavior`, `expertise`, `expertise_topics` | The persona and expertise generated at the start of the run |
| `citation_count`, `num_accepted_papers` | Totals over the run so far |
| `metric_weights` | Per-agent metric weights from `review.metric_weights` (policy learning adjusts them); null when none are set |
| `learning_state` | Written by metacognition feedback and by policy learning; null otherwise |
| `reputation`, `pause_state` | Experimental reputation and pause features; null when they are off |

The model each agent uses is not part of its profile. It is recorded per session in
`session_stats_data.pkl` (`compute_requirements.llm`) and in the `[llm-assign]` lines of `run.log`.

### Session statistics and timelines

| File | Written | Scope | Contents and use |
|---|---|---|---|
| `session_stats_data.pkl` | Every round | Cumulative | `{"registry": {iteration: {phase: {session_id: stats}}}}` for the phases `research`, `review` and `metareview`. Resume restores it; the metrics read code-execution counts from it |
| `timelines/` | Every round; a failure here does not fail the snapshot | Current | One JSONL file per session that recorded events |

Session ids and timeline file names:

| Phase | Session id | Timeline file |
|---|---|---|
| research | `<agent_id>_t<NN>` (`<agent_id>_t<NN>_meta` for meta-research); a collaborator's log inside another agent's session: `<collab_id>_t<NN>_collab_for_<agent_id>` | `research_<session_id>_timeline.jsonl` |
| review | `<reviewer_id>_<paper_id>` | `<session_id>_timeline.jsonl` |
| metareview | `metareview_meta_reviewer_t<N>_<k>_t<NN>_tour<KKK>` for tournament k, e.g. `metareview_meta_reviewer_t3_0_t03_tour000` | `metareview_<session_id>_timeline.jsonl` |

Each `stats` dict has `started_at`, `ended_at`, `duration` (seconds), `turns`, `invalid_actions`,
`input_tokens`, `output_tokens`, `total_tokens`, `tool_calls` (count per tool), `tool_inputs`
(raw inputs per tool), `tool_events`, `timeline`, `log_type` and `compute_requirements`
(`session_type`, `llm.backend_key`, `llm.deployment`, `llm.reasoning_effort`, `runtime_limits` and
the sandbox `limits`). Research sessions add `code_executions`, `code_executions_successful`,
`code_executions_failed`, `lines_of_code`, `images_created`, `collab_established`,
`collab_interactions` and `collab_logs`.

Each timeline line is an event with `ts` (Unix time) and `kind`, plus fields that depend on the
kind; most carry `agent_id`, `iteration` and `step`. Common kinds are `think` (the model's
`response`), `action_parsed`, `tool_call` (`tool`, `args`), `tool_result_*` (for example
`tool_result_execute_code`), `retry` and `terminal_route`.

### Metrics

| File | Written | Scope | Contents and use |
|---|---|---|---|
| `metrics.jsonl` | Every round that produced papers, unless `review.metrics_pipeline=none` | Current | One line per paper: `iteration`, `paper_id`, `primary_agent_id`, `paper_title`, `metrics` with the groups `support`, `soundness`, `significance` and `originality`. Reviewers see these numbers; `ranker.snapshot_import.collect_metrics` reads them |

The fields of each group are defined in [metrics.md](metrics.md).

### Plots

| File | Written | Scope | Contents and use |
|---|---|---|---|
| `plots/*.png` | During the round, for each image returned by a successful `execute_code` call | Current | Named `<agent>_sid-<session>_job-<job>_cell-<cell>_step-<step>_img-<NN>.png` |
| `plots/manifest.jsonl` | With the PNGs | Current | One line per image: `ts`, `run_name`, `iteration`, `step`, `agent_id`, `session_id`, `job_id`, `cell_no`, `image_index`, `filename`, `path`, `bytes`, `sha256`; `plot_filename` under graph logic |
| `plots/summary_manifest.jsonl`, `plots/summaries/<png name>.json` | With the PNGs, when `graph_logic.enabled=true` | Current | One summary per plot (statistics extracted for citation); manifest lines add `plot_handle`, `plot_citation`, `plot_path`, `summary_path`, `generation_backend`, `stats_completeness` |

Failed or timed-out code runs write no plots. Plot citations and summaries are described in
[plots_and_graph_logic.md](plots_and_graph_logic.md).

### Ranker pools

| File | Written | Scope | Contents and use |
|---|---|---|---|
| `ranker_pools/papers_all.jsonl` | Every round | Cumulative | Accepted papers from the private archive, timed-out papers removed, sorted by `publication_t` then `paper_id`; one paper record per line |
| `ranker_pools/papers_standard.jsonl`, `ranker_pools/papers_meta.jsonl` | Every round | Cumulative | The same split by `paper_kind` (`research`, `meta_research`) |
| `ranker_pools/manifest.json` | Every round | Current | `run_name`, `iteration`, `generated_at_utc`, `counts` (`all`, `standard`, `meta`), `filters` (`include_rejected: false`, `exclude_timeout: true`), `source_files`, `schema_version` |

`collect_papers` reads a pool when its own filters match the manifest's, and the private archive
otherwise (for example with `include_rejected=True`). With the meta-researcher on,
`research.meta_researcher.telemetry.write_ranker_pools` [true] can turn the pools off.

### Files from the paper's optional features

These features are off by default and switched on by the experiment scripts
([reproducing_paper.md](reproducing_paper.md)).

| File | Written | Scope | Contents |
|---|---|---|---|
| `social_memory_pruning.json` | Every round when `network.social_memory.mode=markovian` | Current | The public-archive pruning done at the start of the round: `mode`, `iteration`, `retained_publication_t` (N−1), `retained_count`, `dropped_paper_ids`, `dropped_count` |
| `fixed_workflow_assignments.json` | Every round when `research.fixed_workflow.enabled=true` (and `write_artifacts` [true]) | Current | `assignment_policy`, `tie_break`, `self_fallback`, `primary_target_id`, `weakest_agent_id`, `assignments` (agent to collaborator), `agent_scores`, `summary` (`passed`, `failed`, `total`) |
| `fixed_workflow_compliance.jsonl` | Same | Current | One line per paper: `paper_id`, `primary_agent_id`, `assigned_collaborator_id`, `passed`, `fail_reasons`. Papers that fail are left out of review and the archives |
| `metacognition_feedback.json` | Every round when `research.metacognition_feedback.enabled=true`, `mode` is not `none` and `write_artifacts` [true] | Current | `iteration`, `source_iteration`, `mode`, `feedback_by_agent`, `source_paper_counts_by_agent`, `warnings_by_agent`, and `llm_usage_by_agent` when a model wrote the feedback. Feedback is built from the source round's papers and its `metrics.jsonl` |

## Experimental feature files

These features are experimental and off by default; see
[how_it_works.md](how_it_works.md#experimental-features-off-by-default).

| Feature | Files | Written |
|---|---|---|
| `research.notepad` | `notepad_data.json`, `notepad_data.pkl`: `entries_by_agent`, `notepad_file_paths`, `config` (cumulative, capped per agent) | Every round, always; empty when the feature is off |
| `research.notepad` | `notepads/<agent_id>/notepad.jsonl` at run level, one idea per line | When an agent records an idea |
| `research.meta_researcher` | `meta_researcher.json` (trigger decision, selected agents and source papers, generated and promoted meta-paper ids, `skipped_reason`); `meta_quality_report.json`, `evidence_map.json`, `meta_claim_checks.json`, `citation_lint.json`; `_meta` papers in the archives and `papers_meta.jsonl` | `meta_researcher.json` every round when on; the others in rounds where the meta pass runs, each switchable under `research.meta_researcher.telemetry` |
| `bootstrap` | A snapshot `iteration_<bootstrap.iteration>/` of the seeded state, with no `metrics.jsonl` and an empty `timelines/`. `bootstrap.pool_path` takes earlier snapshot iteration directories (needing `agent_registry_data.pkl`, and reading the archives and notepad data when present) or a JSON pool file | Before the first round, when `bootstrap.write_snapshot` [true] |
| `reputation`, `reputation_pause` | `reputation`, `pause_state`, `metric_weights` and `learning_state` in the agent profiles | Every round when on |
| `review.require_assertion_justification` | `citation_tags.review.strict_required: true` in `snapshot_metadata.json` | Every round when on |

## Resuming from a snapshot

Run with `resume.enabled=true resume.snapshot_path=<run dir or iteration dir>`. Command-line
overrides other than `resume.*` and `network.iterations` are ignored (`run.log` lists them); pass
`network.iterations=<T>` to run to a different last round. The pipeline restores the highest *complete* iteration, at or below N if you pass `iteration_<N>/`,
and continues with the next round. An iteration directory is complete when:

1. its name is `iteration_<N>`;
2. `snapshot_metadata.json` exists, parses, and its `iteration` is N;
3. `agent_registry_data.pkl` exists;
4. `archive_data.pkl` or `archive_data.json` exists;
5. `private_archive_data.pkl` or `private_archive_data.json` exists.

Resume also needs `exp_config.yaml` in that directory, because it is the config baseline. It
restores, when present, `session_stats_data.pkl` (so stats and code-execution counts carry on) and
`notepad_data.pkl` or `.json`. Because the snapshot's `api_key` values are redacted, the endpoints
always come from the current invocation's config.

With `resume.archive_incomplete_tail` [true], incomplete directories above the restored iteration
(a round that crashed or was stopped, often holding only `plots/`) are renamed to
`iteration_<N>__incomplete_<YYYYMMDD_HHMMSS>` (UTC) so the round can be rewritten cleanly.

New rounds are written to `snapshots/<run_name>/`, with the run name from the restored
`snapshot_metadata.json`, so resume a run from its place under `snapshots/`, not from a copy
elsewhere. To resume a longer run from an earlier iteration and keep its later iterations, copy the
run under a new name first.

## Reading snapshots from Python

Run from the repository root. Reading the files directly:

```python
import json
import pickle
from pathlib import Path

from system.resume_snapshot import find_last_complete_iteration

run = Path("snapshots/<run_name>")
last, it = find_last_complete_iteration(run)  # same completeness rule as resume

private = json.loads((it / "private_archive_data.json").read_text())
papers = [p for group in private["papers"] for p in group]
for p in papers:
    reviews = private["reviews_by_paper"].get(p["paper_id"], [])
    score = (p.get("metareview") or {}).get("overall_score")
    print(p["paper_id"], p["status"], score, len(reviews))

with open(it / "agent_registry_data.pkl", "rb") as f:
    agents = pickle.load(f)["agents"]
print([(a.agent_id, a.num_accepted_papers, a.citation_count) for a in agents])
```

With the ranker's import helpers, which add provenance fields (`source_snapshot`,
`source_snapshot_dir`, `source_iteration`), a `code` field and each paper's `reviews`:

```python
from ranker.snapshot_import import collect_metrics, collect_papers, paper_uid

papers = collect_papers("snapshots/<run_name>", iterations="last", include_rejected=True)
metrics = collect_metrics(papers)  # paper_uid(p) -> metrics.jsonl record
for p in papers:
    has_metrics = paper_uid(p) in metrics
    print(p["paper_id"], p["status"], p["source_iteration"], len(p["reviews"]), has_metrics)
```

`collect_papers` returns accepted papers only by default and skips timed-out papers; its options
(`iterations`, `cumulative`, `paper_kind_filter`) are documented in
[ranker/snapshot_import.py](../ranker/snapshot_import.py) and [ranker.md](ranker.md).

## Version control and sharing

`outputs/`, `snapshots/` and `artifacts/` are listed in `.gitignore` and are never committed.

Hydra saves the composed config to `.hydra/config.yaml` in the run directory as written, endpoint
values included: a key pasted into an endpoint file is saved there in plain text. Write
`api_key: ${oc.env:AZURE_OPENAI_API_KEY}` in the endpoint files so only the reference is saved
([models.md](models.md#keys)), and do not share `outputs/`. Snapshots redact `api_key` only:
`exp_config.yaml` keeps endpoint URLs and deployment names, and `data_manifest.json` holds absolute
paths and sample rows of the data. Check both before sharing a snapshot. See also
[running.md](running.md#keys-in-outputs).
