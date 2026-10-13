# Testing and checks

## Running the tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q -m "not integration and not sandbox"
```

This is what CI runs. It makes no model calls and needs no sandbox server. Tests run without
`LANGCHAIN_*` and `LANGSMITH_*` variables, in-process and in the subprocesses they start
(`tests/isolation.py`), so LangSmith tracing set in your shell sends nothing. The pipeline stops
without the primary-model and embedder files, and the fake models replace only backends that have
a file, so CI first copies the templates into place; do the same if you have none:

```bash
for f in default gpt-4o gpt-5 gpt-5-high deepseek llama; do
  cp sys_config/llm_api/default_to_complete.yaml sys_config/llm_api/$f.yaml
done
cp sys_config/embedder_api/default_to_complete.yaml sys_config/embedder_api/default.yaml
```

Markers (`pytest.ini`):

| Marker | Needs | Run with |
|---|---|---|
| `integration` | Live services: a filled-in model endpoint (it costs money) or the arXiv API | `-m integration` |
| `sandbox` | A running sandbox server ([sandbox.md](../sandbox.md)) | `-m sandbox` |
| `slow` | Nothing external; long-running | included by default |

## Layout

Test folders mirror the source folders (`tests/agents/`, `tests/system/`, ...), and no test module
lives outside `tests/` (`tests/test_repo_test_layout_contract.py`). Two folders test the pipeline as
a whole:

- **`tests/golden/`** composes the config of every `system.main` run of the experiment scripts
  (from their `DRY_RUN` output; not experiment 03's review replay) and records the resolved
  config (without endpoints), the agents' tool lists, the research, reviewer and meta-reviewer
  prompts, and a reviewer assignment for a
  fixed set of papers. Any change to what an experiment runs shows up as a fixture diff.
- **`tests/e2e/`** runs every condition through `system.main` for two rounds with at most five
  agents, using a scripted LLM, embedder and sandbox (`tests/e2e/fakes.py`). Fixtures record the
  exit code, LLM calls per kind, representative prompts, the number of papers in the public
  archive after each round and the structure of every snapshot file. Other e2e tests cover
  resume, the ranker reading pipeline snapshots, a `rank.sh` command with a scripted judge
  (`tests/e2e/fake_ranker.py`) and the statistics on its output.

When a change is intended, regenerate the fixtures, review the diff and commit it:

```bash
GOLDEN_UPDATE=1 python -m pytest tests/golden
E2E_UPDATE=1 python -m pytest tests/e2e
```

Repository checks in `tests/`: relative links in the Markdown docs resolve
(`test_repo_docs_links.py`), and no function rebinds a module's `log` logger
(`test_repo_logging_contract.py`). The checks that list tracked files (links, test layout) are
skipped outside a git checkout, e.g. in a `git archive` export. `tests/notebooks/` runs the expert-validation notebooks on
synthetic ranker outputs.

## Linters and hooks

```bash
python -m pre_commit install           # once
python -m pre_commit run --all-files
```

The hooks run ruff (lint and format, including notebooks), mypy, interrogate (docstring coverage),
pydoclint (with `.pydoclint-baseline`), nbstripout,
standard file checks, gitleaks, and guards that refuse:

- endpoint files other than the two templates (they hold API keys);
- data, archive and image files outside `tests/**/fixtures/` and `docs/assets/`;
- absolute home-directory paths and personal email addresses.

CI ([`.github/workflows/tests.yml`](../../.github/workflows/tests.yml)) runs the hooks, gitleaks over
all commits, the test suite, every experiment script with `DRY_RUN=1`, and checks that only the
endpoint templates are tracked.

## Logging

Library modules log through `log = logging.getLogger(__name__)`; command-line entry points print.
Debug output (tool-call traces, sandbox history dumps, per-paper metrics) is off unless you raise the
log level (see [running.md](../running.md)).
