# The AutoRanker

The AutoRanker is an LLM judge that compares two papers. It reads both, decides which one is
better under a rubric (or, in one rubric, whether they test the same hypothesis), and writes a
justification. The paper's experiments use it to compare the papers produced by two conditions.

Each comparison is a short agent loop: the judge reasons, may call a tool, and ends with a verdict
block. If the reply has no parseable verdict, the judge is asked again.

## What the judge sees

Each paper is shown in up to four sections, in this order:

| Section | Content | Setting (`ranker.*`) |
|---|---|---|
| Dataset Description | Each dataset key the run used, with its description from [`env_setup/datasets.py`](../env_setup/datasets.py) (for example `Dataset 1: tcga_kirc_neurips_hiseqv2`) | `include_dataset_description` |
| Manuscript | Title, abstract and full text | always shown |
| Code | The code the research session submitted | `include_code` |
| Cited Papers | Every paper this paper builds on through citations to the archive, followed transitively (citations of citations), each with title, abstract, manuscript and, when `include_code` is on, its code | `include_cited_papers`, `cited_papers_order` |

All three settings default to `true`. A section that is on but empty says so ("No code was
provided for this paper.", "This paper does not cite any other papers."). The system prompt lists
the sections that are on, using the texts in `ranker.prompt_section_notes`.

The batch tools fill these sections from the snapshots:

- The dataset keys come from the run's `exp_config.yaml` (`discovery_dataset.dataset_keys`), and
  their descriptions from `env_setup/datasets.py`.
- Cited papers are resolved within each paper's own run, from all its accepted papers up to the
  iteration read, so ancestors that the selection settings left out of the comparison still
  appear. Lookups use `<snapshot dir>::<paper_id>`, because paper IDs repeat across runs.

**Tools.** Two tools exist, both off by default (`rank.sh` leaves them off):

| Tool | Enable with | What it does |
|---|---|---|
| `arxiv` | `ranker.tools.arxiv.enabled=true` | Searches arXiv and returns summaries of the top `top_k_results` (5) hits |
| `sandbox` | `ranker.tools.sandbox.enabled=true` | Runs Python in a sandbox session kept for the whole comparison; needs a running [sandbox server](sandbox.md) at `ranker.tools.sandbox.base_url` or, if that is `null`, at `SANDBOX_BASE_URL` (default `http://127.0.0.1:9000`) |

When a tool is on, the system prompt describes it and the judge decides whether to use it.

**Position.** The batch tools randomise which paper is shown as "Paper A". For the comparison with
index *k* (0-based, in generation order), `random.Random(sampling.seed + k)` decides the swap, so
the same seed gives the same positions. In paired runs each record stores which side each slot
came from. The single-pair CLI shows the papers in the order you give them.

## Rubrics

A rubric is a config file in [`sys_config/ranker_cfg/`](../sys_config/ranker_cfg/), selected with
`ranker_cfg@ranker=<name>`. Each sets `ranker.criteria_prompt`; the variants inherit everything
else from `default.yaml`.

| `ranker_cfg@ranker=` | Mode | Judge is asked to |
|---|---|---|
| `default` | `comparison` | Weigh novelty, significance and soundness "as a researcher would" |
| `novelty` | `comparison` | Compare only novelty (conceptual uniqueness against the field) |
| `significance` | `comparison` | Compare only significance (scientific value and broader importance) |
| `soundness` | `comparison` | Compare only soundness (are the claims supported by coherent evidence) |
| `hypothesis_similarity` | `hypothesis_similarity` | Decide whether the two papers test the same hypothesis (same core question and causal or relational claim, whatever the methods) |

In `comparison` mode the verdict is `Winner: A|B` and the score is 1 when Paper A wins, 0 when
Paper B wins. In `hypothesis_similarity` mode the verdict is `Decision: Same|Different` and the
score is 0 for same, 1 for different. Use the `hypothesis_similarity` rubric rather than setting
`ranker.mode` alone: the mode does not change the criteria text.

For your own rubric, override the text (`"ranker.criteria_prompt=Compare only the statistical
methods."`) or add a YAML file next to the others with `defaults: [default, _self_]` and a new
`criteria_prompt`.

## The judge model

`ranker.llm_backend` names the endpoint the judge uses. The default `gpt5-high` reads
`sys_config/llm_api/gpt-5-high.yaml`, which you create from
[`sys_config/llm_api/default_to_complete.yaml`](../sys_config/llm_api/default_to_complete.yaml) as
described in [models.md](models.md): a GPT-5 endpoint with `reasoning_effort: high`. The ranker
configs also load the pipeline's endpoint files when they exist (`primary` from `default.yaml`,
`gpt4o`, `gpt5`, `deepseek`, `llama`), so any of them can be the judge, for example
`ranker.llm_backend=llama`. For another endpoint file, add it and select it:
`+llm_api@endpoints.<key>=<file name> ranker.llm_backend=<key>`. The judge's tool calls are plain
text (`Action: {...}`), so any chat model works.

## Command-line tools

All three are Hydra apps, so every setting below is a command-line override. To check settings
without calling a model: `python -m ranker.run_ranker --cfg job --package ranker <overrides>`.

### One pair: `ranker.run_ranker`

```bash
python -m ranker.run_ranker input_file=pair.json ranker_cfg@ranker=soundness
```

`pair.json` holds the two papers; `code` is optional:

```json
{
  "paper_a": {"title": "...", "abstract": "...", "manuscript": "...", "code": "..."},
  "paper_b": {"title": "...", "abstract": "...", "manuscript": "..."}
}
```

Short papers can also be given inline with `+paper_a.title=... +paper_a.abstract=...
+paper_a.manuscript=...` and the same for `paper_b`. The tool prints the outcome, the score, turns,
token counts and the justification; `score=None` means the judge gave no parseable verdict. It
writes `run.log` and `ranker_trace.json` to `outputs/_ranker/<date>/<time>/`.

This tool uses the rubric, mode, tools, `max_iterations` and retry settings. To give the judge
dataset descriptions or cited papers, or to set the section settings (`include_*`,
`cited_papers_order`, `paper_separator`, `prompt_section_notes`), use the Python API.

### Condition A against condition B: `ranker.run_sample_and_rank_paired`

```bash
python -m ranker.run_sample_and_rank_paired \
    experiments.experiment_a.label=kirc_ascollab \
    experiments.experiment_a.snapshot_dirs='["snapshots/<run_a>"]' \
    experiments.experiment_b.label=kirc_single_agent \
    experiments.experiment_b.snapshot_dirs='["snapshots/<run_b>"]' \
    ranker_cfg@ranker=novelty
```

Each side is a list of snapshot run folders (one per seed, for example); their papers are pooled.
Every comparison pairs a paper from A with a paper from B.

### One pool: `ranker.run_sample_and_rank_non_paired`

```bash
python -m ranker.run_sample_and_rank_non_paired \
    paper_pool.label=kirc_ascollab \
    paper_pool.snapshot_dirs='["snapshots/<run_seed_1>","snapshots/<run_seed_2>"]' \
    sampling.mode=random sampling.n_pairs=50
```

All papers go into one pool and pairs are drawn within it (`paper_*_experiment` is `null`). Both
batch tools write to `outputs/_sample_and_rank_{paired,non_paired}/<date>/<time>/`; set
`hydra.run.dir=<dir>` to choose the folder.

## Batch settings

Defaults are in [`sys_config/sample_and_rank_base.yaml`](../sys_config/sample_and_rank_base.yaml)
and the paired and non-paired files next to it.

### Paper selection

Papers are filtered by status, timeout and kind, then by iteration, then top-N, then balance.

| Key | Default | Meaning |
|---|---|---|
| `paper_selection.include_rejected` | `false` | Also take papers the meta-review rejected |
| `paper_selection.exclude_timeout` | `true` | Drop papers whose research session timed out (`paper_id` ends in `_timeout` or manuscript `TIMEOUT`) |
| `paper_selection.iterations` | `last` | `last`, `all`, or a list of iteration numbers such as `[10]` or `[2,4]` |
| `paper_selection.cumulative` | `true` | `true`: every paper published up to the last (or the given) iteration. `false`: only papers published in the given iterations; with `last`, the latest iteration that has papers. A list of several iterations needs `false` |
| `paper_selection.paper_kind_filter` | `all` | `all`, `standard_only` or `meta_only`; matters only for runs with the experimental meta-researcher (see [how_it_works.md](how_it_works.md#experimental-features-off-by-default)) |
| `paper_selection.paper_kind_filter_a` / `_b` | `null` | Paired only: a per-side value that replaces `paper_kind_filter` |
| `paper_selection.top_n.enabled` | `false` | Keep the `n` papers with the highest meta-review `overall_score` (per side in paired runs); papers without a score rank last |
| `paper_selection.top_n.n` | `10` | How many papers top-N keeps |
| `paper_selection.top_n.per_iteration` | `false` | Apply top-N within each publication iteration and keep the union (`n` per iteration) |
| `paper_selection.balance.enabled` | `false` | Paired only: cut the larger side down to the size of the smaller one |
| `paper_selection.balance.method` | `random` | `random` (seeded sample) or `top_metareview` (keep the best-scored) |
| `paper_selection.balance.seed` | `${SEED}` (42) | Seed for `random` balancing |

### Sampling and parallelism

| Key | Default | Meaning |
|---|---|---|
| `sampling.mode` | `all` | `all`: every pair (paired: each A paper with each B paper; non-paired: each unordered pair once). `random`: draw `n_pairs` pairs uniformly |
| `sampling.n_repeats` | `1` | `all` only: judge each pair this many times (each repeat gets its own position draw) |
| `sampling.n_pairs` | `100` | `random` only: number of comparisons |
| `sampling.with_replacement` | `false` | `random` only: allow the same pair twice. Without replacement, `n_pairs` cannot exceed the number of possible pairs |
| `sampling.cap_n_pairs` | `false` | Paired, `random` without replacement: if fewer than `n_pairs` pairs are possible, judge each once (with a warning) instead of stopping |
| `sampling.allow_self_comparison` | `false` | Non-paired only: allow a paper to be compared with itself |
| `sampling.seed` | `${SEED}` (42) | Seeds the pair draw and the A/B positions |
| `parallelism.max_workers` | `4` | Comparisons judged at the same time (threads) |

### Judge settings

These come from [`sys_config/ranker_cfg/default.yaml`](../sys_config/ranker_cfg/default.yaml) and
apply to all three tools, except where the single-pair section above says otherwise.

| Key | Default | Meaning |
|---|---|---|
| `ranker.llm_backend` | `gpt5-high` | Endpoint key of the judge |
| `ranker.mode` | `comparison` | `comparison` or `hypothesis_similarity` (set by the rubric) |
| `ranker.criteria_prompt` | rubric text | What the judge compares |
| `ranker.max_iterations` | `10` | Limit on the judge's steps (tool calls and failed verdict checks). Once reached, no more tools run: the next reply is checked for a verdict, with up to `max_parse_retries` re-asks, and without one the comparison ends with no decision |
| `ranker.output_parsing.max_parse_retries` | `2` | Times the judge is re-asked for a verdict in the required format after a reply without one |
| `ranker.llm_retry.max_retries` | `3` | Retries on rate-limit and API errors, with exponential backoff from `base_delay_seconds` (2.0) up to `max_delay_seconds` (30.0) |
| `ranker.tools.arxiv.*` | off | `enabled`, `top_k_results` (5), `load_max_docs` (5) |
| `ranker.tools.sandbox.*` | off | `enabled`, `base_url` (`null`), `code_timeout_seconds` (120) |
| `ranker.include_code` | `true` | Show the Code section |
| `ranker.include_dataset_description` | `true` | Show the Dataset Description section |
| `ranker.include_cited_papers` | `true` | Show the Cited Papers section |
| `ranker.cited_papers_order` | `newest_first` | Order of cited papers: `newest_first` or `oldest_first` |
| `ranker.paper_separator` | three `---` lines | Text between Paper A and Paper B |
| `ranker.prompt_section_notes.*` | see YAML | System-prompt line for each section (`preamble`, `manuscript`, `include_dataset_description`, `include_code`, `include_cited_papers`) |

## Outputs of a batch run

```
<run dir>/
├── run.log
├── comparisons.json       # one record per comparison
├── run_summary.json       # counts, settings, selection report
├── ranker_config.json     # the ranker.* settings used
├── experiment_configs/    # <run_name>.yaml: each snapshot's exp_config.yaml
├── ranker_traces/         # cmp_NNNN.json: full trace of each judged comparison
└── papers/paper_data.json # one record per selected paper
```

**`comparisons.json`** is a JSON list that `pandas.read_json` loads directly. Fields of a record:

| Field | Meaning |
|---|---|
| `comparison_id` | `cmp_0001`, `cmp_0002`, ... in generation order |
| `run_id`, `run_datetime` | The batch run (`run_name`) and its UTC start time |
| `status` | `completed`; `unparsed_verdict` (the judge answered but gave no parseable verdict after every re-ask); `failed` (an error, see `error`) |
| `score` | `comparison`: 1 = Paper A better, 0 = Paper B better. `hypothesis_similarity`: 0 = same, 1 = different. `null` unless completed |
| `outcome` | `A` / `B` or `same` / `different`; `null` unless completed |
| `justification` | The judge's reasoning; empty when unparsed, `null` when failed |
| `paper_a_*`, `paper_b_*` | For each slot: `id`, `title`, `experiment` (the side's label, `null` in non-paired runs), `source_snapshot` (run name), `source_snapshot_dir`, `source_iteration`, `primary_agent_id` |
| `iteration_selection`, `cumulative` | The iteration settings used |
| `repeat_index` | Repeat number in `all` mode with `n_repeats > 1`, else 0 |
| `turns`, `input_tokens`, `output_tokens`, `tool_calls`, `duration_seconds` | Judge cost and tool use for this comparison |
| `ranker_config_ref`, `experiment_configs_ref`, `ranker_trace_ref` | Relative paths of the matching files (failed comparisons have no trace file) |
| `error` | Error message for `failed`, else `null` |

To count wins of side A, map the slots back to the sides and leave out records that are not
completed:

```python
import pandas as pd

df = pd.read_json("comparisons.json")
done = df[df.status == "completed"]
a = "kirc_ascollab"  # experiments.experiment_a.label
a_won = (done.paper_a_experiment == a) == (done.score == 1)  # side a won from either slot
print(a_won.sum(), len(done), df.status.value_counts().to_dict())
```

**`run_summary.json`** holds `run_id`, timing, the sides (`experiments`: label, snapshot folders,
papers collected per snapshot; non-paired runs have one `paper_pool` entry), the iteration and
sampling settings with `total_unique_pairs` and `total_comparisons`, `results` (`completed`,
`failed`, `unparsed_verdict`, and token and time totals over completed comparisons), `paper_data`
(papers with and without review metrics), `parallelism`, and `paper_selection` (filters, counts
before and after top-N and balance, and per-iteration counts when `per_iteration` is on).

**`papers/paper_data.json`** has one record per selected paper: every field of the snapshot paper
record (see [outputs.md](outputs.md)), plus `source_snapshot`, `source_snapshot_dir`,
`source_iteration`, `code`, `reviews` (each with `iteration`, `reviewer_id`, `review_text`,
`scores`), `experiment` (the side or pool label), `experiment_condition` (`A`, `B` or `null`) and
`metrics` (the paper's review metrics from `metrics.jsonl`, see [metrics.md](metrics.md), or
`null`).

**`ranker_traces/cmp_NNNN.json`** records every model call (with its full input messages), tool
call and verdict-parse attempt, plus a summary of counts and tokens. These files can be large.

Hydra also writes `.hydra/` with the composed config, including the endpoint settings. If an
endpoint file holds the API key itself rather than `${oc.env:...}`, so does that folder.

## Python API

The CLIs are thin wrappers around three functions. Build the judge from the same configs:

```python
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from env_setup.build_llms import build_llm_pool
from ranker import run_ranker

with initialize_config_dir(config_dir=os.path.abspath("sys_config"), version_base=None):
    cfg = compose(config_name="sample_and_rank_paired", overrides=["ranker_cfg@ranker=novelty"])
key = cfg.ranker.llm_backend
judge = build_llm_pool(cfg.endpoints, keys=[key], primary_alias=key == "primary")[key]
ranker_cfg = OmegaConf.to_container(cfg.ranker, resolve=True)

# One pair, with your own dataset description and no cited papers
result = run_ranker(
    paper_a={"title": "...", "abstract": "...", "manuscript": "...", "code": "..."},
    paper_b={"title": "...", "abstract": "...", "manuscript": "..."},
    llm=judge,
    criteria_prompt=ranker_cfg["criteria_prompt"],
    dataset_descriptions_a={"tcga_kirc_neurips_survival": "Survival table ..."},
    dataset_descriptions_b={"tcga_kirc_neurips_survival": "Survival table ..."},
    include_cited_papers=False,
)
if result.valid_result:
    print(result.score, result.justification)
```

Pass `criteria_prompt` explicitly: without it, `run_ranker` uses a built-in five-dimension
fallback, not the `default` rubric. The result also carries `raw_response`, `tool_calls`, `turns`,
token counts and `trace`.

The batch functions take the config sections as plain dicts, write the files above to
`output_dir` and return `{"comparisons", "summary", "output_dir"}`:

```python
from ranker.sample_and_rank_non_paired import run_sample_and_rank_non_paired
from ranker.sample_and_rank_paired import run_sample_and_rank_paired

common = dict(
    ranker_cfg=ranker_cfg,
    parallelism_cfg={"max_workers": 4},
    llm=judge,
    paper_selection_cfg={"iterations": "last", "cumulative": True},
)
paired = run_sample_and_rank_paired(
    experiments_cfg={
        "experiment_a": {"label": "kirc_ascollab", "snapshot_dirs": ["snapshots/<run_a>"]},
        "experiment_b": {"label": "kirc_single_agent", "snapshot_dirs": ["snapshots/<run_b>"]},
    },
    sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
    output_dir="outputs/my_paired_run",
    **common,
)
pool = run_sample_and_rank_non_paired(
    paper_pool_cfg={"label": "kirc_ascollab", "snapshot_dirs": ["snapshots/<run_a>"]},
    sampling_cfg={"mode": "random", "n_pairs": 50, "seed": 42},
    output_dir="outputs/my_pool_run",
    **common,
)
```

To build your own pairs from snapshots, `ranker.snapshot_import` provides `collect_papers`,
`to_ranker_format`, `collect_dataset_descriptions` and `build_citation_chains`; see their
docstrings in [ranker/snapshot_import.py](../ranker/snapshot_import.py).

## Use in the paper's experiments

Each experiment folder in `scripts/experiments/` except 03 and 06 has a `rank.sh` that runs
`ranker.run_sample_and_rank_paired` for every condition pair, rubric, dataset and seed. It ranks
the top-N papers of each side by meta-review score (`iterations=last`, `cumulative=true`,
`balance.method=top_metareview`), judges every pair once (`sampling.mode=all`; 04 instead draws
500 random pairs without replacement, `sampling.mode=random`, seeded by the run's seed), labels
the sides `<dataset>_<condition>`, and writes each run to
`outputs/ranker/<NN>/<dataset>/<a>_vs_<b>/<rubric>/seed<k>/`. The environment variables `JUDGE`
(default `gpt5-high`) and `MAX_WORKERS` (default 8) set the judge and the parallelism. 03's
`rank.sh` instead scores fixed cohorts of papers with `scripts.analysis.reviewer_matching.run_grid`
and `analyze_results` (outputs under `outputs/reviewer_matching/`); with `JUDGE` and `MAX_WORKERS`
unset, its judge and parallelism are those of `sys_config/sample_and_rank_non_paired.yaml`
(`gpt5-high`, 4 workers). See [reproducing_paper.md](reproducing_paper.md)
for each experiment's conditions and rubrics.

The statistics read those folders:

```bash
python -m scripts.analysis.stat_tests --ranker-dir outputs/ranker/01 --out results/01
```

It writes `stat_tests.csv` and `stat_tests.json` to `--out`. For each condition pair and rubric
there is a row per dataset (seeds pooled): the number of comparisons and condition A's win rate,
an exact two-sided binomial test against 0.5 (with Holm-adjusted p-values), a Wilson interval and
a paper-clustered bootstrap interval (`--n-boot`, default 2000; `--seed`, default 12345). With
several datasets a macro-average row adds the number of comparisons, the win rate and the
bootstrap interval only. Only completed comparisons count.
