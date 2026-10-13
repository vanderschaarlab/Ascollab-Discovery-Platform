# Running the pipeline

`python -m system.main` runs one experiment: a network of research agents works for a number of
iterations (rounds), and each iteration runs research, review and meta-review, updates the paper
archives and writes a snapshot. This page covers how to configure and start a run and read its logs;
resuming a run is in [outputs.md](outputs.md#resuming-from-a-snapshot). What happens inside an iteration is in [how_it_works.md](how_it_works.md).

## Before you run

Install the requirements as described in the [README](../README.md#install). Create the endpoint
files for the models the run uses ([models.md](models.md)): always `sys_config/llm_api/default.yaml`
(the primary model) and `sys_config/embedder_api/default.yaml`, plus one file per backend that has
a share in the model mix. Get the data for the dataset bundle you run ([data.md](data.md)); the
default bundle is TCGA KIRC (`python -m scripts.data.download_tcga --cohorts kirc`). Start the
sandbox server ([sandbox.md](sandbox.md)) before the run: the pipeline finds it at
`SANDBOX_BASE_URL` (default `http://127.0.0.1:9000`), and you can check it with
`curl http://127.0.0.1:9000/base/health`.

## A first run

A small run with two agents, one iteration and one reviewer per paper:

```bash
python -m system.main \
  exp_name=smoke SEED=1 \
  network.num_agents=2 network.iterations=1 \
  review.num_reviewers_per_paper=1 \
  llm_backends.counts=null
```

`llm_backends.counts=null` gives every agent the primary model, so only the two required endpoint
files are needed. The run checks the KIRC files, generates two agent personas, runs one iteration
and writes `snapshots/smoke_<yymmdd_HHMM>/iteration_1/`. Its log is
`outputs/<YYYY-MM-DD>/<HH-MM-SS>/run.log`.

## How a run is configured

### Config file and overrides

The pipeline is a [Hydra](https://hydra.cc) application whose config is
[`sys_config/ascollab.yaml`](../sys_config/ascollab.yaml). `python -m system.main` loads it by
default; the experiment scripts pass `--config-path ../sys_config --config-name ascollab`, which
is the same. To keep your own defaults, copy the file to `sys_config/<name>.yaml` and pass
`--config-name <name>`.

Change any setting on the command line with `key=value`:

| Form | Example |
|---|---|
| Set a value | `network.num_agents=8` |
| Set to null | `research.max_concurrent_research_agents=null` |
| Set a list | `"discovery_dataset.dataset_keys=[nhanes,mi_gwas_genes]"` |
| Add a key the file doesn't have | `+llm_backends.fractions.gpt5=0.5` |
| Choose a config group option | `dataset=luad` |
| A value containing `=` | `"review.reviewer_assignment_condition_label='n_reviewers=2'"` |

Quote overrides that contain brackets, spaces or `$`. To see the composed config without running,
print one section:

```bash
python -m system.main --cfg job -p network network.num_agents=8
```

Without `-p` this prints the whole config, including your endpoint files' contents.

### Dataset bundle

`dataset=<bundle>` selects a file in [`sys_config/dataset/`](../sys_config/dataset/): `kirc`
(default), `luad`, `paad` or `nhanes`. Each bundle sets `discovery_dataset.target` (the research
goal and package list given to the agents, from `env_setup/goals.py` and `env_setup/packages.py`)
and `discovery_dataset.dataset_keys` (the dataset descriptions in `env_setup/datasets.py`). Override
either directly to use other keys (see [Your own data](#your-own-data)).

### Run name and outputs

| Key | Default | Meaning |
|---|---|---|
| `exp_name` | `ascollab` | Prefix of the run name |
| `SEED` | 1 | Seeds Python and NumPy, the model assignment, reviewer assignment and other random choices; the endpoint template also sends it to the model as `seed`. |
| `run_name` | `${exp_name}_${now:%y%m%d_%H%M}` | Name of the snapshot folder |

A run writes to two places:

| Path | Contents |
|---|---|
| `snapshots/<run_name>/iteration_<N>/` | The snapshot written at the end of each iteration ([outputs.md](outputs.md)) |
| `snapshots/<run_name>/data_manifest.json`, `data_card.json` | The dataset check's report |
| `outputs/<YYYY-MM-DD>/<HH-MM-SS>/` | Hydra's run folder: `run.log`, `main.log` and `.hydra/` (see [Logs](#logs)) |

Keep run names unique: two runs with the same `run_name` write into the same folder. To give the
Hydra folder the same name as the snapshot folder, add `'hydra.run.dir=outputs/${run_name}'`
(single quotes, so the shell leaves `${...}` alone).

## Main settings

Defaults are those in [`sys_config/ascollab.yaml`](../sys_config/ascollab.yaml). The letters N, T,
M and K are the paper's symbols. The experiment scripts set their own values for many of these
keys.

### Network

| Key | Default | Meaning |
|---|---|---|
| `network.num_agents` | 16 | N, the number of research agents. Change the model mix with it (see [Model mix](#model-mix)). |
| `network.iterations` | 40 | T, the last iteration to run (iterations are numbered from 1) |
| `network.setting` | `network_hetero` | `network_hetero`: each agent gets a persona and expertise generated by the primary model. `network_homo`, `indep_homo`: all agents share one default persona. The model mix is set separately. |
| `network.enable_collaboration_tools` | true | Tools to start a collaboration and message the collaborator. `null`: on for `network_*`, off for `indep_homo`. |
| `network.enable_social_memory_tools` | true | Search of the public paper archive and the agent registry. `null`: as above. |
| `network.social_memory.mode` | `cumulative` | `cumulative`: the public archive keeps every accepted paper. `markovian`: it keeps only the papers accepted in the previous iteration (the private archive keeps everything). |
| `memory.max_k` | 5 | Most results one archive or registry search returns (the agents' search tools ask for 3 unless they pass `k`). The other `memory.*` keys weight the ranking of results by citations and recency. |
| `tools.arxiv.enabled` | true | The arXiv search tool (`tools.arxiv.top_k_results` [3]; `tools.arxiv_http_timeout_seconds` [30]) |

### Research

| Key | Default | Meaning |
|---|---|---|
| `research.research_max_iters` | 40 | M, the step limit of one research session. A step is one action or one attempt at the final report. A session with no valid report after M − 1 steps stops with an error after step M, and its paper is marked incomplete. A collaborator's reply has the same limit; reaching it stops the author's session the same way. |
| `research.code_timeout_seconds` | 900 | Time limit for one code execution in the sandbox |
| `research.total_session_timeout_seconds` | 720 | Wall-clock limit of one research session; a session that reaches it ends as a timeout |
| `research.session_create_hard_timeout_seconds` | 1800 | How long an agent waits for a free sandbox session before giving up |
| `research.max_concurrent_research_agents` | 10 | Most research sessions running at once. `null`: derived from the sandbox's `SANDBOX_MAX_SESSIONS`. |
| `research.execute_code_retries`, `research.execute_code_retry_delay_seconds` | 120, 5 | Retries, and the delay between them, when the sandbox session is busy or a request fails |
| `research.code_collaboration` | true | With collaboration tools on: collaborators edit one shared Python script. `false`: they only exchange messages. |
| `research.prior_papers.enabled` | false | Put the agent's own earlier papers in its prompt. `research.prior_papers.mode` [`full`]: all in full; `summary`: the latest `keep_full` [1] in full, older ones summarised by the model in `summary_cfg_path`. |
| `research.full_citing` | false | Strict citation tags in papers. Takes effect only with `prior_papers.enabled=true` and `mode=full`; otherwise the run logs a warning and turns it off. |
| `research.self_reflection_prompt.enabled` | false | From iteration 2, agents first write a self-reflection on their weakest review scores |
| `research.nature_style_papers_prompt.enabled` | false | Ask for a fixed paper structure (title, summary, research question, hypothesis and findings, ...) |
| `research.metacognition_feedback.enabled`, `.mode` | false, `none` | Give each agent feedback on its earlier papers: `meta_review_only`, `meta_review_with_metrics` or `metrics_only` (experiment 04) |
| `research.fixed_workflow.enabled` | false | Assign each agent a collaborator and drop papers that don't comply before review (experiment 02) |
| `graph_logic.enabled` | false | Plot citations and plot summaries ([plots_and_graph_logic.md](plots_and_graph_logic.md)) |

### Review

Reviewers are research agents; each reviews with its own model.

| Key | Default | Meaning |
|---|---|---|
| `review.num_reviewers_per_paper` | 2 | K, reviewers per paper |
| `review.allow_self_review` | false | Let agents review their own papers (needed when N=1) |
| `review.reviewer_assignment_strategy` | `expertise_matching` | `expertise_matching`: reviewers found by searching the agent registry for the paper's topic, balancing load. `random`: seeded random assignment. |
| `review.reviewer_assignment_seed` | `${SEED}` | Seed of the reviewer assignment |
| `review.reviewer_assignment_condition_label` | null | Extra salt for the assignment, so conditions that share a seed get different assignments |
| `review.norm_prompt` | off | A norm sentence added to every reviewer prompt. Choose one with `+review_norm@review.norm_prompt=<name>`: `norm_a_novelty`, `norm_b_soundness` or `none` ([`sys_config/review_norm/`](../sys_config/review_norm/)). |
| `review.metrics_pipeline` | `default` | `default`: review metrics are computed and shown to reviewers. `experimental`: only those in `review.metrics_allowlist`. `none`: no metrics. See [metrics.md](metrics.md). |
| `review.include_code_in_reviews` | true | Give reviewers the paper's code |
| `review.review_max_iterations` | 6 | Step limit of one review session, counted as for `research.research_max_iters`; a reviewer that runs out gives no review |
| `review.total_session_timeout_seconds` | 1200 | Wall-clock limit of one review session |

### Meta-review

| Key | Default | Meaning |
|---|---|---|
| `metareview.tournament_size` | 4 | L, papers per meta-review tournament. The meta-reviewer scores each paper in its tournament; across tournaments, the top-scoring 1/L of the papers are accepted |
| `metareview.shuffle_before_tournament` | true | Shuffle papers before grouping them |
| `metareview.total_session_timeout_seconds` | 1200 | Wall-clock limit of one meta-review session |
| `metareview.reviewerless.enabled` | false | Skip review; the meta-reviewer scores papers directly. The run refuses to start if `reputation.enabled=true`, `reputation_pause.mode` is not `none`, or `research.metacognition_feedback.enabled=true`. |

### Model mix

| Key | Default | Meaning |
|---|---|---|
| `llm_backends.counts` | `gpt4o: 6`, `gpt5: 5`, `deepseek: 5`, `llama: 0` | Agents per backend |
| `llm_backends.fractions` | not set | Shares per backend (add with `+llm_backends.fractions.<key>=<share>`); they fill the agents the counts leave over |
| `llm_backends.shuffle` | true | Shuffle which agent gets which backend (seeded by `SEED`) |

Agents the counts and fractions don't cover use the primary model. Counts that add up to more
than N are scaled down to N, keeping their proportions, with a warning. Endpoint files and backends
are in [models.md](models.md).

### Data check

| Key | Default | Meaning |
|---|---|---|
| `data.preflight` | true | Before the run, read each file the dataset descriptions point to and write `data_manifest.json` and `data_card.json` to `snapshots/<run_name>/` |
| `data.preflight_strict` | true | Stop the run when a check fails (no file found for a dataset key, unknown key, checksum mismatch). `false`: log the report and continue. |

## Common recipes

**Another dataset bundle.**

```bash
python -m system.main exp_name=luad_run dataset=luad
```

**Another model mix.** Eight agents, four on GPT-5 and four on DeepSeek (needs
`sys_config/llm_api/gpt-5.yaml` and `deepseek.yaml`):

```bash
python -m system.main exp_name=mix_run network.num_agents=8 \
  llm_backends.counts.gpt4o=0 llm_backends.counts.gpt5=4 \
  llm_backends.counts.deepseek=4 llm_backends.counts.llama=0
```

**One agent.** With no other agent to review its papers, it reviews its own:

```bash
python -m system.main exp_name=single network.num_agents=1 \
  review.allow_self_review=true review.num_reviewers_per_paper=1 \
  network.enable_collaboration_tools=false research.code_collaboration=false \
  llm_backends.counts=null
```

### Your own data

Describe your files in `env_setup/datasets.py` and register their keys as explained in
[data.md](data.md#adding-or-describing-a-dataset). Then pass the keys directly:

```bash
python -m system.main exp_name=my_data "discovery_dataset.dataset_keys=[my_table,my_labels]"
```

or make a bundle, `sys_config/dataset/my_data.yaml`:

```yaml
# @package discovery_dataset
target: tcga # goal and package set (env_setup/goals.py, env_setup/packages.py)
dataset_keys:
  - my_table
  - my_labels
```

and run it with `dataset=my_data`. The `tcga` goal asks the agents to make scientific discoveries
from the data they are given; to set a different goal, add an entry to `GOALS` in
`env_setup/goals.py` and to `PYTHON_PACKAGE_SETS` in `env_setup/packages.py` under the same key,
and use that key as `target`. Check the files before a run with
`python -m env_setup.preflight_data --config sys_config/dataset/my_data.yaml`.

## The paper's experiments

Each experiment in the paper has a script in [`scripts/experiments/`](../scripts/experiments/)
that calls `python -m system.main` once per condition, dataset and seed (03's instead replays the
review stage on an 01 run), setting every setting that differs between experiments explicitly.
[reproducing_paper.md](reproducing_paper.md) describes each experiment and its settings, and
[`scripts/experiments/README.md`](../scripts/experiments/README.md) lists the scripts' options;
`DRY_RUN=1` prints the commands a script would run. The social-system switches used by
experiments 07–10 (`metareview.reviewerless.enabled`, `network.social_memory.mode`,
`review.norm_prompt`, `review.num_reviewers_per_paper`) are in the tables above.

## Logs

Each process writes to its Hydra run folder, `outputs/<YYYY-MM-DD>/<HH-MM-SS>/`:

| File | Contents |
|---|---|
| `run.log` | Log records at INFO and above from every module (WARNING and above from the `openai` and `httpx` clients), plus everything printed to stdout and stderr. Its first line gives its path. Rotates at 10 MB, keeping `run.log.1` to `run.log.3`. |
| `main.log` | Hydra's job log: the same records in Hydra's format, and debug records when you turn them on |
| `.hydra/config.yaml`, `hydra.yaml`, `overrides.yaml` | The composed config, Hydra's settings and the command-line overrides |

The console shows the same records. Output written to stderr, such as progress bars, is logged
under the name `STDERR`. Records from `system/main.py` carry the logger name `__main__`.

Lines are tagged by stage, so you can follow one part of a run, for example:

```bash
grep -E "\[(capacity|llm-mix|resume|review)\]" outputs/<date>/<time>/run.log
```

The pipeline's modules log through Python's `logging`, and debug records are off by default. To
turn them on for the project's own modules:

```bash
python -m system.main ... "hydra.verbose=[__main__,system,agents,env_setup]"
```

Debug records go to the console and `main.log`; `run.log` stays at INFO. `hydra.verbose=true`
turns on debug output for every library as well, which is very noisy.

## Keys in outputs

- Hydra saves the composed config in `.hydra/config.yaml` in the run folder under `outputs/`,
  including your endpoint files' values as written: a key pasted into an endpoint file is saved
  there in plain text. Command-line overrides are saved in `.hydra/overrides.yaml`, so don't pass
  keys on the command line.
- Snapshots save the config in each iteration's `exp_config.yaml` with every `api_key` replaced by
  `REDACTED`. Other endpoint values (endpoint URL, deployment or model name) are kept.
- Read keys from the environment, `api_key: ${oc.env:AZURE_OPENAI_API_KEY}`: `.hydra/config.yaml`
  then holds only the `${oc.env:...}` expression, and the snapshot copy `REDACTED`
  ([models.md](models.md#keys)).
- Don't share `outputs/`. Check snapshots for endpoint URLs before sharing them. Both folders are
  ignored by git.

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| `Missing endpoint file(s): ...` | Create the named file from `default_to_complete.yaml` in the same folder ([models.md](models.md)) |
| `... is not filled in (azure_deployment is empty)` (`model` for `provider: openai`) | Fill in that endpoint file, or give its backend no share in `llm_backends` |
| `[llm_backends] requested backends [...] have no endpoint file` | A backend in the mix has no file, so its agents are filled from `llm_backends.fractions` if set, otherwise from the primary model. Create the file or change the counts. |
| `Environment variable '...' not found` | An endpoint file reads its key with `${oc.env:...}` and the variable is not set |
| `Pre-flight failed` | `run.log` holds the report. Check the files ([data.md](data.md)), or run `python -m env_setup.preflight_data --config sys_config/dataset/<bundle>.yaml`. `No description found` means a dataset key that is not in `env_setup/datasets.py`. |
| Connection error to `127.0.0.1:9000` | The sandbox server is not running, or `SANDBOX_BASE_URL` points elsewhere ([sandbox.md](sandbox.md)) |
| `Gave up after ...s waiting for capacity` | Lower `research.max_concurrent_research_agents`, or raise `SANDBOX_MAX_SESSIONS` on the server |
| Code executions or sessions time out | Raise `research.code_timeout_seconds` or `research.total_session_timeout_seconds`; container CPU and memory are set on the sandbox server |
| Sandbox sessions left over after a crash | `python -c "from env_setup.sandbox_utils import SandboxClient; SandboxClient().shutdown_all_sessions()"` |
| `candidate papers received ZERO reviews and will skip meta-review` | No eligible reviewer (for example N=1 without `review.allow_self_review=true`), or the review sessions failed; see `run.log` |
| `Got unknown network.setting` | Use `network_hetero`, `network_homo` or `indep_homo` |
| `Resumed snapshot has ... agents but network.num_agents=...` | Set `resume.strict_agent_count_match=false` |
| `No complete iteration found in run directory` | The run never finished an iteration; start a new run |

## Experimental features

Some features are experimental and off by default; they are listed in
[how_it_works.md](how_it_works.md#experimental-features-off-by-default).
