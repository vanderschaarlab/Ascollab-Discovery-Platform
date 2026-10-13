# ASCollab

ASCollab is a multi-agent system for open-ended scientific discovery. A network of LLM research
agents analyses a dataset over many rounds: each agent writes short papers backed by code it ran
in a sandbox, collaborates with other agents, and builds on earlier published work; other agents
review the papers, meta-reviewers decide which are published, and published papers become citable
work that shapes what comes next. An LLM judge, the AutoRanker, compares the papers different
setups produce.

This repository holds the code for the paper **Open-Ended Scientific Discovery and the Social
Dynamics of Evolving Agent Networks**. It contains no data and no results: the documentation
explains how to get the public data and how to run every experiment in the paper.

## Contents

| Path | What it holds |
|---|---|
| `agents/` | Research, collaborator, reviewer and meta-reviewer agents (LangGraph) |
| `system/` | The round loop (`system/main.py`), review assignment, metrics, archive updates, snapshots, resume |
| `stores/` | Public and private paper archives, the agent registry (FAISS) |
| `env_setup/` | Model clients, dataset descriptions, discovery goals, agent tools, sandbox client |
| `sandbox/` | The Docker sandbox server where agents run code |
| `ranker/` | The AutoRanker: pairwise LLM judge and batch comparisons |
| `scripts/experiments/` | One `run.sh` per experiment in the paper, and a `rank.sh` for each except 06 |
| `scripts/analysis/` | Statistics and analysis of experiment outputs |
| `scripts/data/` | Data download |
| `notebooks/expert_validation/` | Ranker vs human raters |
| `sys_config/` | Hydra configs: the pipeline, datasets, endpoints, ranker |
| `docs/` | Documentation |
| `tests/` | Unit, end-to-end and golden tests (no model calls, except those marked `integration`, which call live LLM and arXiv APIs) |

## Documentation

| Read | To |
|---|---|
| [docs/how_it_works.md](docs/how_it_works.md) | Understand the system: agents, sessions, review, meta-review, social memory |
| [docs/models.md](docs/models.md) | Set up model endpoints, choose a model mix, add a backend |
| [docs/data.md](docs/data.md) | Get the public data, or describe your own |
| [docs/sandbox.md](docs/sandbox.md) | Run the sandbox server |
| [docs/running.md](docs/running.md) | Run and configure the pipeline |
| [docs/reproducing_paper.md](docs/reproducing_paper.md) | Run the paper's experiments, rank them, compute the statistics |
| [docs/ranker.md](docs/ranker.md) | Use the AutoRanker |
| [docs/outputs.md](docs/outputs.md) | Read what a run writes; resume a run |
| [docs/metrics.md](docs/metrics.md), [docs/plots_and_graph_logic.md](docs/plots_and_graph_logic.md) | Review metrics; plots and graph logic |
| [docs/dev/testing.md](docs/dev/testing.md) | Develop: tests, linters and hooks |

## Install

Python 3.11 and Docker are needed.

```bash
conda create -n ascollab python=3.11
conda activate ascollab
pip install -r requirements.txt        # pipeline, ranker, analysis, sandbox server
pip install -r requirements-dev.txt    # tests and linters (optional)
```

The sandbox image installs its own packages when you build it (see below).

## Set up model endpoints

Create the endpoint files from the templates and fill them in. They hold API keys and are ignored
by git: never commit them.

```bash
cp sys_config/llm_api/default_to_complete.yaml sys_config/llm_api/default.yaml
cp sys_config/embedder_api/default_to_complete.yaml sys_config/embedder_api/default.yaml
```

`default.yaml` is the primary model and the embedder file the embeddings model; both are always
needed. The paper's default mix of research-agent models (GPT-4o, GPT-5, DeepSeek) needs one more
file per model, and the ranker one for its judge. Azure OpenAI and any OpenAI-compatible server
are supported. Keys can come from the environment (`api_key: ${oc.env:AZURE_OPENAI_API_KEY}`).
Details: [docs/models.md](docs/models.md).

## Get the data

```bash
python -m scripts.data.download_tcga      # TCGA KIRC, LUAD and PAAD from UCSC Xena
```

The files go to `sandbox/datasets/` (or to `$SANDBOX_DATASETS_DIR` if set, or `--dest`) and are
checked against recorded checksums. NHANES and the myocardial-infarction GWAS data have to be
obtained and prepared separately; see [docs/data.md](docs/data.md), which also explains how to use
your own data.

## Start the sandbox

```bash
docker build -t agent-sandbox:py311 \
  --build-arg SANDBOX_UID=$(id -u) --build-arg SANDBOX_GID=$(id -g) sandbox/

export SANDBOX_IMAGE=agent-sandbox:py311
export SANDBOX_DATASETS_DIR=$PWD/sandbox/datasets
export SANDBOX_USER_UID=$(id -u) SANDBOX_USER_GID=$(id -g)
gunicorn sandbox.server_docker:app -k gthread -w 1 --threads 8 \
  -b 127.0.0.1:9000 --timeout 180 --graceful-timeout 30 --keep-alive 30
```

The server has no authentication: keep it on `127.0.0.1`. Settings and limits:
[docs/sandbox.md](docs/sandbox.md).

## Quickstart

A small run: two agents, one round, the KIRC data, all agents on the primary model.

```bash
python -m system.main exp_name=quickstart \
  network.num_agents=2 network.iterations=1 review.num_reviewers_per_paper=1 \
  llm_backends.counts.gpt4o=0 llm_backends.counts.gpt5=0 llm_backends.counts.deepseek=0
```

**Cost.** Every run calls your model endpoints. Even this run makes many calls (two research
sessions of up to 40 steps each, reviews and a meta-review). The paper's configurations (16
agents, 20 to 40 rounds, several conditions, datasets and seeds, plus the ranker) are far larger.
Check your provider's pricing and quotas first, and use `DRY_RUN=1` and the smoke-run options in
[docs/reproducing_paper.md](docs/reproducing_paper.md) before launching an experiment.

The run's snapshots are under `snapshots/quickstart_<timestamp>/`
([docs/outputs.md](docs/outputs.md)); configuration and further recipes are in
[docs/running.md](docs/running.md).

## Experiments in the paper

| Script | Experiment | Conditions |
|---|---|---|
| `01_network_ablations` | Network ablations | single agent, independent agents, homogeneous network, full ASCollab |
| `02_fixed_workflow` | Fixed-workflow baseline | assigned collaborators |
| `03_reviewer_matching` | Reviewer matching | expertise-matched vs random reviewers |
| `04_feedback` | Feedback to authors | none, meta-reviews, meta-reviews with metrics, metrics |
| `05_network_scaling` | Network scaling | 1 to 16 agents |
| `06_closed_vs_open_task` | Closed vs open-ended task | open-ended goal, prespecified hypotheses |
| `07_no_peer_review` | No peer review | with and without reviewers |
| `08_markovian_memory` | Markovian social memory | cumulative vs last-round archive |
| `09_reviewer_norms` | Reviewer norms | novelty-first vs soundness-first norm |
| `10_reviewer_count` | Number of reviewers | 1 to 4 reviewers per paper |

```bash
bash scripts/experiments/01_network_ablations/run.sh     # runs every condition
bash scripts/experiments/01_network_ablations/rank.sh    # ranks them with the AutoRanker
python -m scripts.analysis.stat_tests --ranker-dir outputs/ranker/01 --out <output dir>
```

What each experiment runs and compares: [docs/reproducing_paper.md](docs/reproducing_paper.md).

## Responsible use

- Agents write and run code. It runs in Docker containers without network access, with CPU,
  memory and process limits ([docs/sandbox.md](docs/sandbox.md)). Treat generated code and outputs
  as untrusted.
- The papers the system writes are machine-generated hypotheses about the data, not validated
  findings. They are not medical advice and must not inform clinical decisions without independent
  verification.
- Follow the data providers' terms (TCGA / GDC, NHANES, FinnGen; see
  [docs/data.md](docs/data.md)), in particular never attempt to identify participants.
- Keep API keys out of version control, and don't share `outputs/` if your endpoint files contain
  keys ([docs/models.md](docs/models.md#keys)).

## Citing

If you use this code, please cite the paper *Open-Ended Scientific Discovery and the Social
Dynamics of Evolving Agent Networks*.
[CITATION.cff](CITATION.cff) has the citation for the software.

## Licence and contact

Apache License 2.0; see [LICENSE](LICENSE). Questions and bug reports: please open a GitHub issue.
