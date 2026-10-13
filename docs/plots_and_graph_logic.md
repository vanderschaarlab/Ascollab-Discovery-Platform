# Plots and graph logic

Agents run their analysis code in the [sandbox](sandbox.md). After each code cell, the sandbox
returns every matplotlib figure that is still open as a PNG, and the pipeline saves these images
in the run's snapshot. This always happens.

**Graph logic** is an optional layer on top, off by default (`graph_logic.enabled` [false]). It
writes a structured summary for each saved plot, gives each plot a short citation handle, asks
agents to cite their plots in their papers and checks those citations, and shows reviewers the
plots' summaries. The paper's feedback experiment (04 in
[reproducing_paper.md](reproducing_paper.md)) runs with it on.

The research prompt asks agents to make at least one plot whenever they compute statistics,
whether or not graph logic is on.

## Saved plots

Plots from successful `execute_code` calls by research agents are saved to (collaborators cannot
run code, so they make no plots):

```text
snapshots/<run_name>/iteration_<N>/plots/<filename>.png
snapshots/<run_name>/iteration_<N>/plots/manifest.jsonl
```

`snapshots/` is under the directory you launch the pipeline from (outside a Hydra run, under
`PROJECT_DIR` if it is set). File names are built from the run state:

```text
<agent>_sid-<session>_job-<job>_cell-<cell>_step-<step>_img-<NN>.png
```

If the name already exists, the first 10 hex characters of the image's SHA-256 are appended.

An image is saved only if the call's status is `success`, the run name and iteration are known,
and the payload is a valid base64 PNG. Two environment variables, read by the pipeline process (not
the sandbox server) at start-up, set limits:

| Variable | Default | Meaning |
|---|---|---|
| `SANDBOX_MAX_PLOTS_PER_EXECUTION` | 8 | Images kept per `execute_code` call; the rest are dropped |
| `SANDBOX_MAX_PLOT_BYTES` | 8000000 | Largest PNG kept, in bytes |

Skipped images are recorded as warnings and never stop the agent. The agent's observation lists
the saved files as `PLOTS = [...]`, with paths relative to `iteration_<N>/`.

`plots/manifest.jsonl` has one row per saved image:

| Field | Meaning |
|---|---|
| `ts` | UTC time saved |
| `run_name`, `iteration`, `step` | Run, round and agent step |
| `agent_id`, `session_id`, `job_id`, `cell_no` | Who ran the code, and where in the sandbox |
| `image_index` | Position in the call's image list (from 1) |
| `filename`, `path` | PNG name, and path relative to `iteration_<N>/` |
| `bytes`, `sha256` | Size and hash |
| `plot_filename` | Same as `filename`; only with graph logic on |

## Graph logic

Turn it on with:

```bash
python -m system.main --config-path ../sys_config --config-name ascollab \
  "graph_logic.enabled=true"
```

With graph logic on, for each saved plot the pipeline:

1. writes a summary JSON and a row in a summary manifest (see [Plot summaries](#plot-summaries));
2. gives the plot a handle, `plot_` plus the first 8 hex characters of the SHA-1 of its file name,
   and a citation token, `[Plot: plot_ab12cd34]`;
3. adds `PLOT_ARTEFACTS` (one record per plot) and `PLOT_CITATIONS` (the tokens) to the agent's
   observation. The prompt tells the agent to copy these tokens and cite figures in-line;
4. checks the plot citations in the agent's final report ([Plot citations](#plot-citations));
5. stores the plot records on the paper and shows reviewers up to 12 of them (citation token,
   handle, file name, paths, a summary excerpt, the summary backend and a statistics digest).

Settings, under `graph_logic` in [`sys_config/ascollab.yaml`](../sys_config/ascollab.yaml):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | false | Turns graph logic on |
| `require_plot_citations_in_research` | true | If the agent saved at least one plot this round, its final report needs at least one valid `[Plot: ...]` tag |
| `plot_only_rewrite_cap` | 2 | Rewrite requests for plot-only citation errors before the waiver is tried |
| `allow_plot_validation_waiver_after_cap` | true | After the cap, accept the report if every non-plot citation check passes |
| `summary.backend` | `code_stats_llm` | Added at the front of `fallback_order` if it is not already in it |
| `summary.fallback_order` | `[code_stats_llm, agent_sidecar, heuristic_fallback_only]` | Backends tried in order; `heuristic_fallback_only` is always appended |
| `summary.include_descriptive_stats` | true | Per-series statistics in summaries |
| `summary.include_comparative_stats` | true | A comparison of the first two series |
| `summary.max_series_per_plot` | 12 | Most series per summary |

To change which backend runs first, set `summary.fallback_order`.

### Summary backends

| Backend | What it does |
|---|---|
| `code_stats_llm` | Reads the plotting code for the title, axis labels and limits, plot type and series labels, and fills the series statistics from numbers printed in the cell's output. |
| `agent_sidecar` | Copies summary fields from `plot_summary_data`, `plot_summaries` or `plot_summary` keys in the sandbox response. |
| `heuristic_fallback_only` | A placeholder summary with no statistics and confidence 0.2. |

The series statistics from `code_stats_llm` are computed over all numbers printed by the cell,
split evenly across the series labels found in the code.

A summary failure falls back to the placeholder summary and never stops the agent.

## Plot summaries

With graph logic on, each plot also gets:

```text
snapshots/<run_name>/iteration_<N>/plots/summaries/<plot_filename>.json
snapshots/<run_name>/iteration_<N>/plots/summary_manifest.jsonl
```

Summary JSON fields:

| Field | Meaning |
|---|---|
| `plot_filename`, `plot_path`, `summary_version`, `generation_backend` | Identity and the backend that produced the summary |
| `plot_type` | `kaplan_meier`, `heatmap`, `scatter`, `histogram`, `box`, `bar`, `line` or `other`, guessed from function names in the code |
| `title`, `axes` | Title; `x_label`, `y_label`, `x_range`, `y_range` from `plt`/`ax` calls |
| `statistics.series` | Per series: `name`, `n`, `min`, `max`, `mean`, `median`, `std`, `q1`, `q3`, `iqr` |
| `statistics.comparisons` | First vs second series: `delta_mean`, `effect_size` (pooled-SD standardised), `p_value` (first `p <`, `p =` or `p >` in the output) |
| `statistics.annotations_detected` | `error_bars`, `confidence_intervals`, `p_value_text`: keywords found in the output |
| `statistics.stats_completeness` | Share of nine items present: title, plot type, two axis labels, two axis ranges, series, `point_of_plot`, `interpretation_guidance` |
| `point_of_plot`, `statistical_takeaway`, `interpretation_guidance` | Short generated sentences |
| `confidence` | 0.75 with `include_descriptive_stats` on; 0.45 with it off; 0.2 for the placeholder |
| `warnings` | For example `title_not_detected`, `descriptive_stats_disabled_by_config`, `agent_sidecar_missing` |

Each `summary_manifest.jsonl` row has the manifest's run, agent and sandbox fields plus
`plot_filename`, `plot_handle`, `plot_citation`, `plot_path`, `summary_path`,
`generation_backend` and `stats_completeness`.

## Plot citations

A paper cites a plot with `[Plot: <token>]`, where the token is a plot handle (preferred) or an
exact plot file name. It is one of the four tags of the strict citation format, with
`[Archive: ...]`, `[Arxiv: ...]` and `[Code: ...]`.

When the agent submits its final report with graph logic on, the report is checked
(`system/citation_tags.py`):

- Each `[Plot: ...]` tag holds one token: no spaces or `|`, only `A-Za-z0-9_.:-`. Surrounding
  quotes, a leading path and trailing punctuation are removed first.
- The token must match a plot saved in this research session: by handle, then by file name, then
  either case-insensitively.
- With `require_plot_citations_in_research=true` and at least one saved plot, the report needs at
  least one valid plot tag.

If the check fails:

1. **Auto-repair.** Plot tags that resolve are rewritten to the canonical `[Plot: <handle>]`. If
   the report is then valid, it is accepted.
2. **Rewrite request.** Otherwise the agent is asked to rewrite the report, with the list of
   valid tokens and the errors. A request where every error concerns plots counts towards
   `plot_only_rewrite_cap`.
3. **Waiver.** After `plot_only_rewrite_cap` plot-only requests, with
   `allow_plot_validation_waiver_after_cap=true`, a report that passes every non-plot check is
   accepted. The paper then counts as citation-compliant, so its archived `citable_with_warnings`
   is false; the research timeline records the waiver as a `plot_validation_waived` event.

An agent that runs out of steps first stops with an error, and its paper is marked incomplete
(`research.research_max_iters`, see [running.md](running.md#research)). Plot-tag errors found
when the paper is recorded are stored on it (`citation_compliant`, `citation_errors`).
They change the paper's status only with `research.full_citing=true` (which takes effect only with
`research.prior_papers.enabled=true` and `mode=full`); it turns any invalid citation into the status
`invalid_citation_report`.

With graph logic off, plot tags are not checked against saved plots.

## Related snapshot files

| File | Plot-related content |
|---|---|
| `iteration_<N>/plots/` | PNGs, `manifest.jsonl`, and with graph logic `summary_manifest.jsonl` and `summaries/` |
| `iteration_<N>/timelines/research_<session_id>_timeline.jsonl` | `tool_result_execute_code` events with `image_count`, `plot_paths`, `plots_skipped`, `plot_manifest`, `plot_warnings`, and with graph logic `plot_filenames`, `plot_handles`, `plot_citations`, `plot_summary_paths`, `plot_artifacts`, `plot_summary_manifest`, `plot_summary_backend`, `plot_summary_stats_completeness`; `tool_warning_execute_code_images` events for skipped images; `plot_citation_auto_repaired`, `plot_citation_auto_repair_partial`, `plot_only_rewrite_attempt`, `plot_validation_waived` and `plot_validation_waiver_rejected` events |
| `iteration_<N>/snapshot_metadata.json` | `files_saved` lists `plots/` and the plot manifests and `summaries/` folder present (not each PNG); `citation_tags` counts include `plot` |
| `iteration_<N>/private_archive_data.json` | Per paper: `plot_artifacts`, `plot_filenames`, `plot_handles`, `plot_handle_to_filename` |

Session statistics also count `images_created`. All snapshot files are described in
[outputs.md](outputs.md).

## Troubleshooting

If a run has no `plots/` directory, look at the research timelines:

- `status` on `tool_result_execute_code` must be `success` and `image_count` above zero;
- `plot_warnings` and `tool_warning_execute_code_images` give the reason an image was skipped
  (`not_png`, `invalid_base64`, `image_too_large_decoded`, `max_images_limit:...`,
  `missing_run_name`, ...).

Common causes: the cell failed or timed out; the code closed its figures (`plt.close()`), so no
figure was open when the cell ended; or the plot was not made with matplotlib.
