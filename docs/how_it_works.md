# How it works

ASCollab simulates a research community. A network of LLM research agents works on a shared
dataset for a number of rounds. In each round every agent writes a short paper backed by code it
ran, other agents review the papers, meta-reviewers pick which ones are published, and published
papers become citable work that later papers can find and build on.

This page describes the system as implemented. Settings are keys in
[`sys_config/ascollab.yaml`](../sys_config/ascollab.yaml); defaults are shown in brackets.

## One round

```
for round t = 1 .. T:
    research     every active agent runs one research session and submits a paper
    metrics      optional per-paper metrics from the session logs and the archive
    review       K reviewers per paper write reviews with scores
    meta-review  papers are grouped into tournaments of L; meta-reviewers score them
    update       the top 1/L are accepted; archives, citations and the registry are updated
    snapshot     everything is written to snapshots/<run>/iteration_<t>/
```

`network.num_agents` (N) [16], `network.iterations` (T) [40],
`review.num_reviewers_per_paper` (K) [2] and `metareview.tournament_size` (L) [4] are the main
sizes. Snapshots, and resuming from them, are described in [outputs.md](outputs.md).

## Agents and the registry

Each agent has an ID (`agent_01`, ...), a profile and a model:

- **Profile.** In the heterogeneous setting the primary model writes, for each agent, a
  behaviour persona (how it chooses ideas, collaborates, scopes work, evaluates, uses the
  literature and resources) and an expertise description grounded in the dataset descriptions.
  In the homogeneous settings all agents share one fixed profile.
- **Model.** `llm_backends.counts` assigns models to agents [6 GPT-4o, 5 GPT-5, 5 DeepSeek];
  agents review with their own model too. See [models.md](models.md).

The **agent registry** stores the profiles with each agent's citation count and number of accepted
papers. Agents search it by describing the expertise they need; results are ranked by embedding
similarity plus a citation term (`memory.registry`: `w_sim` [0.8], `w_cit` [0.2]), so
well-cited agents are easier to find.

### Network settings

| `network.setting` | Profiles | Collaboration tools | Archive and registry search |
|---|---|---|---|
| `network_hetero` [default] | generated, one per agent | yes | yes |
| `network_homo` | identical | yes | yes |
| `indep_homo` | identical | no | no |

The tool columns apply only when `network.enable_collaboration_tools` and
`network.enable_social_memory_tools` are `null`. Both are true by default and override the
setting's tool choice, so `network.setting=indep_homo` alone keeps all tools; experiment 01's
`indep_homo` condition also sets both to false. A single-agent run is N = 1 with self-review (see
[reproducing_paper.md](reproducing_paper.md)).

## A research session

Each round, every active agent runs one session: a LangGraph loop in which the model thinks,
calls a tool, reads the observation, and repeats until it writes its final report or runs out of
steps (`research.research_max_iters` (M) [40]) or time (`research.total_session_timeout_seconds`
[720]). Up to `research.max_concurrent_research_agents` [10] sessions run at once.

**What the agent sees.** The session starts from the discovery goal (see
[Directing the system](#directing-the-system)) and a system prompt with: its role, its persona and
expertise, the size of its research community, a "Your Review Reputation" section from round 2
(with reputation off, the default, it says no scores are available), the tools, the dataset
descriptions and the Python packages available, the research process and the report format.

**Tools.**

| Tool | What it does |
|---|---|
| `execute_code` | Runs Python in the agent's sandbox session (state persists between calls; see [sandbox.md](sandbox.md)) |
| `arxiv` | Searches arXiv |
| `search_internal_archive` | Searches the published papers of this run |
| `search_agent_registry` | Searches other agents by expertise |
| `establish_collaboration` | Starts a collaboration with another agent |
| `communicate_with_collaborator` | Sends a message to the collaborator |

**Collaboration.** `establish_collaboration` builds a collaborator agent from the chosen agent's
profile (its papers are not passed on). An agent cannot collaborate with itself. The collaborator
can search arXiv and the archive but cannot run code; only the agent that started the
collaboration runs code. With `research.code_collaboration` [true] the two agents
pair-program: each message carries the current shared script, and the collaborator replies with a
revised one. With it off, messages are plain text.

**The paper.** The final report is a short paper with fixed sections (title, research question,
hypothesis and key findings, rationale, empirical evidence, literature evidence, references) and
the script that produced the results. Papers cite earlier papers from this run with
`[Archive: <paper_id>]` tags; the tags are checked against the archive. A session that times out,
runs out of steps or fails still produces a paper record, with an ID ending in `_timeout` or
`_incomplete`.

## Social memory

| Store | What it holds | Who sees it |
|---|---|---|
| Public archive | Accepted papers | Agents, through `search_internal_archive` |
| Agent registry | Profiles, citation counts, accepted-paper counts | Agents, through `search_agent_registry`; reviewer matching |
| Private archive | Every paper with a meta-review decision, accepted or rejected, with its reviews and meta-review | Not searchable by agents; the ranker and the analysis read it |

Archive search ranks papers by similarity to the query plus a citation boost
(`memory.archive.citation_boost` [true], `w_citation` [0.1]) and an optional recency boost
[off]. Citations therefore compound: a cited paper is easier to find, so it is more likely to be
cited again.

`network.social_memory.mode` [cumulative] controls what the public archive keeps. With
`markovian`, at the start of each round it keeps only the papers accepted in the previous round.
The private archive is never pruned.

With `research.prior_papers.enabled` [false], each agent's prompt also includes its own earlier
papers from the private archive (from round 2; with `research.prior_papers.mode` [full], all of
them in full); `research.full_citing` [false] then requires strict citation tags for every source.

## Review

**Assignment.** `review.reviewer_assignment_strategy` [expertise_matching] searches the registry
for the agents whose expertise best matches each paper's title (at most `memory.max_k` [5]),
excludes the paper's author, and takes the matches that have reviewed least in the round first; if
the search returns too few, other agents are added in a seeded random order. `random` draws
reviewers at random, seeded by `review.reviewer_assignment_seed` [`${SEED}`]. Assignments are
deterministic for a seed.

**Reviewing.** Reviewers are research agents reviewing with their own model. Each review session
sees the paper and its code (`review.include_code_in_reviews` [true]) and, unless
`review.metrics_pipeline` is `none`, the paper's metrics (see [metrics.md](metrics.md)).
Reviewers can search arXiv and the archive but cannot run code. A review has a structured text and
five scores: support, soundness, significance and originality (1–4) and an overall recommendation
(1–5).

`review.norm_prompt` adds a fixed "community reviewing norms" paragraph to every reviewer prompt
(the paper's reviewer-norm experiment; see `sys_config/review_norm/`).

## Meta-review and acceptance

Papers with at least one review are shuffled (`metareview.shuffle_before_tournament` [true]) into
tournaments of L papers. For each tournament a meta-reviewer, using the primary model, reads the
papers and their reviews and gives each paper a score from 0 to 1 and a rank. Across all
tournaments of the round, the `max(1, n // L)` papers with the highest scores are accepted, where n
is the number of papers meta-reviewed: about 1 in L.

With `metareview.reviewerless.enabled` [false] the review stage is skipped and the meta-reviewer
scores the papers directly (the paper's no-peer-review experiment).

## Update

After meta-review:

1. Accepted papers go into the public archive; accepted and rejected papers go into the private
   archive with their reviews and meta-reviews. A paper that received no review has no meta-review
   and stays undecided.
2. Each accepted paper's `[Archive: ...]` citations raise the cited paper's citation count (in
   both archives) and the cited author's count in the registry.
3. Authors' accepted-paper counts are updated.

Then the round's snapshot is written.

## Directing the system

The discovery goal is the first message of every research session. It comes from
[`env_setup/goals.py`](../env_setup/goals.py), selected by `discovery_dataset.target`, which also
selects the Python package list shown to agents ([`env_setup/packages.py`](../env_setup/packages.py)).
The dataset bundles set `target: tcga`, whose goal is open-ended ("make scientific discoveries").

To give the agents a specific task, add a goal and a package set under a new key, then run with
`discovery_dataset.target=<key>`. Experiment 06 does this with `kirc_closed_hypotheses`, which asks
the agents to test three prespecified hypotheses on the KIRC data.

## Optional features used in the paper's experiments

These are supported and off by default; the experiment scripts switch them on
([reproducing_paper.md](reproducing_paper.md)).

| Setting | What it does |
|---|---|
| `research.fixed_workflow.enabled` | Assigns each agent a collaborator and requires it to collaborate before running code and before finishing; registry search is removed |
| `research.metacognition_feedback.enabled` | Before a round, gives each agent feedback on its previous papers (`mode`): in `meta_review_only` and `meta_review_with_metrics` a model writes it from each paper's title, abstract, a manuscript excerpt, meta-review and up to `max_reviews_per_paper` [4] reviews, plus its metrics in `meta_review_with_metrics`; `metrics_only` is a fixed template of the metric scores, with no model call |
| `research.self_reflection_prompt.enabled` | From round 2, agents start with a self-reflection on their lowest review scores and a plan to improve them |
| `research.nature_style_papers_prompt.enabled` | Asks for a longer, journal-style final report |
| `graph_logic.enabled` | Requires plots to be cited in papers and adds plot summaries ([plots_and_graph_logic.md](plots_and_graph_logic.md)) |
| `research.prior_papers.enabled`, `research.full_citing` | Own earlier papers in the prompt; strict citation tags for archive, arXiv and code evidence, including the agent's own papers |
| `metareview.reviewerless.enabled` | No reviewers |
| `network.social_memory.mode: markovian` | Public archive keeps only the last round |
| `review.norm_prompt` | Reviewer norm paragraph |

## Experimental features (off by default)

These features are in the code and tested; no experiment in the paper uses them.

| Setting | What it does |
|---|---|
| `review.require_assertion_justification` | Reviewers must back every assertion with archive, arXiv or code evidence; reviews that fail the citation format get one corrective retry |
| `reputation.enabled` (+ `reputation.policy_learning`) | Computes each agent's reputation from the review scores of its papers (recency-weighted) and shows it in the "Your Review Reputation" prompt section; policy learning adjusts per-agent metric weights from it |
| `reputation_pause.mode` (+ `replacement`) | Pauses low-reputation agents from research, review or collaboration, and can replace them with new agents |
| `research.meta_researcher.enabled` | Every few rounds, a second research pass writes "meta-research" papers from the public archive |
| `bootstrap.enabled` | Seeds round 1's archives and registry from earlier runs' snapshots (`bootstrap.pool_path`) |
| `research.notepad.enabled` | Gives each agent a private notepad kept across rounds |

Their files in snapshots are described in [outputs.md](outputs.md).
