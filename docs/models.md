# Models

ASCollab talks to chat models and one embeddings model through endpoint files: small YAML files
that you create from a template. They hold API keys, so git ignores them; never commit them.

## Endpoint files

| File | Config key | Used for | Needed when |
|---|---|---|---|
| `sys_config/llm_api/default.yaml` | `endpoints.llm_primary` | The primary model | Always |
| `sys_config/embedder_api/default.yaml` | `endpoints.embedder_primary` | Embeddings for archive and registry search | Always |
| `sys_config/llm_api/gpt-4o.yaml` | `endpoints.gpt4o` | Research agents on GPT-4o | `llm_backends` gives `gpt4o` a share |
| `sys_config/llm_api/gpt-5.yaml` | `endpoints.gpt5` | Research agents on GPT-5 | `gpt5` has a share |
| `sys_config/llm_api/deepseek.yaml` | `endpoints.deepseek` | Research agents on DeepSeek | `deepseek` has a share |
| `sys_config/llm_api/llama.yaml` | `endpoints.llama` | Research agents on Llama | `llama` has a share |
| `sys_config/llm_api/gpt-5-high.yaml` | `endpoints.gpt5-high` | The ranker's judge | Running the ranker |

The default mix (`llm_backends.counts`: 6 GPT-4o, 5 GPT-5, 5 DeepSeek, 0 Llama) needs the first
five files. Create each one from the template and fill it in:

```bash
cp sys_config/llm_api/default_to_complete.yaml sys_config/llm_api/default.yaml
cp sys_config/embedder_api/default_to_complete.yaml sys_config/embedder_api/default.yaml
cp sys_config/llm_api/default_to_complete.yaml sys_config/llm_api/gpt-5.yaml   # one per backend
```

The file names are only conventions: the `defaults` list at the end of
[`sys_config/ascollab.yaml`](../sys_config/ascollab.yaml) maps each config key to a file (for
example `optional llm_api@endpoints.gpt5: gpt-5`).

### Providers

`provider` selects the client:

- **Azure OpenAI** (`provider: azure`, the default): `azure_deployment`, `azure_endpoint`,
  `api_version`, `api_key`. For a DeepSeek or Llama deployment the client also sends the
  deployment name as the request's model name, which some of these endpoints require.
- **OpenAI or any OpenAI-compatible server** (`provider: openai`): `model`, `api_key`, and
  `base_url` for a server other than OpenAI's (for example vLLM serving Llama or DeepSeek). For
  embeddings from such a server, also set `check_embedding_ctx_length: false`.

A run can mix providers: each file is independent.

### Keys

Put the key in the file, or better, read it from the environment so it is never written anywhere:

```yaml
api_key: ${oc.env:AZURE_OPENAI_API_KEY}
```

Hydra saves the composed config of every run under `outputs/` (`.hydra/config.yaml`), with
endpoint values as written in your files. With the environment form only the `${oc.env:...}`
expression is saved. Snapshots replace `api_key` values with `REDACTED`. Don't share `outputs/`
if your files contain keys.

### Checks at start-up

A missing primary or embeddings file, or one still holding the empty template, stops the run with
a message naming the file. So does the file of `research.metacognition_feedback.llm_backend` when
feedback is on in a mode that uses a model (`meta_review_only`, `meta_review_with_metrics`). A
backend that has a share in the mix but no file gets a warning, and its agents are filled from
`llm_backends.fractions` if set (none by default), otherwise from the primary model. The ranker checks its judge's file the same way.

## Model mixes

`llm_backends.counts` gives the number of agents per backend. `llm_backends.fractions` gives
shares instead; if both are set, counts win. If the counts add up to less than
`network.num_agents`, the remaining agents are filled from the fractions if set, otherwise they use
the primary model. If they add up to more, they are scaled down to `network.num_agents`, keeping
their proportions (largest remainder), with a warning: the default 6 / 5 / 5 mix becomes 1 / 1 / 0
for two agents. `llm_backends.shuffle` [true] shuffles which agent gets which backend (seeded by
`SEED`). For example, 16 GPT-5 agents:

```bash
python -m system.main "llm_backends.counts.gpt4o=0" "llm_backends.counts.gpt5=16" \
  "llm_backends.counts.deepseek=0" "llm_backends.counts.llama=0"
```

Only backends with a non-zero share are built, so you need files only for those.

## Adding a backend

1. Create `sys_config/llm_api/<name>.yaml` from the template.
2. Add `- optional llm_api@endpoints.<key>: <name>` to the `defaults` list in
   `sys_config/ascollab.yaml`.
3. Give it a share: `+llm_backends.counts.<key>=<n>` (the `+` adds a key the config does not
   have), or add the key under `llm_backends.counts` in `sys_config/ascollab.yaml`.

Sampling settings in the file (`temperature`, `top_p`, `max_tokens`) are sent to GPT-4o-family
models only, and `reasoning_effort` to GPT-5-family models only; the family is read from
`azure_deployment` or `model`. Research agents whose deployment or model name contains "llama"
get a tool-calling shim (`env_setup/llama_emulation.py`) that adds explicit tool-call instructions
to their system prompt.

## The ranker's judge

The ranker uses one judge model for every comparison, `ranker.llm_backend` [`gpt5-high`], which
maps to `sys_config/llm_api/gpt-5-high.yaml`: a GPT-5 deployment with `reasoning_effort: high`.
The ranker configs (`sys_config/ranker.yaml`, `sample_and_rank_paired.yaml`,
`sample_and_rank_non_paired.yaml`) load `gpt5-high` and the pipeline's endpoints (`llm_primary`,
as `primary`, `gpt4o`, `gpt5`, `deepseek`, `llama`), each when its file exists, so any of them can
be the judge: `ranker.llm_backend=<key>`. For a new endpoint, add it to that list and set
`ranker.llm_backend=<key>`.
See [ranker.md](ranker.md).
