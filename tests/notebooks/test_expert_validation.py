"""The expert-validation notebooks run end to end on synthetic ranker outputs.

Each notebook's code cells run in order in one namespace, as in Jupyter, from a scratch copy of
the repo layout (``<root>/notebooks/expert_validation``). Making PDFs needs pandoc and LaTeX, so
notebook 02's ``generate_pdf`` is swapped for a stub that writes the Markdown instead. No model is
called and nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import itertools
import json
import random
import re
from pathlib import Path

import matplotlib
import nbformat
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO_ROOT / "notebooks" / "expert_validation"
DIMS = ("novelty", "significance", "soundness", "default")
N_PAPERS = 30  # SELECTION_SET_INDEX = 2 needs two disjoint sets of 12, plus an incomplete paper


def _snapshot(seed: int) -> str:
    return f"snapshots/social_exp_kirc_baseline_seed_{seed}_260101_120{seed}"


def _papers() -> list[dict]:
    papers = []
    for i in range(N_PAPERS):
        papers.append(
            {
                "paper_id": f"paper_{i}" + ("_incomplete" if i == 0 else ""),
                "source_snapshot_dir": _snapshot(1 + i % 2),
                "source_snapshot": Path(_snapshot(1 + i % 2)).name,
                "title": f"Paper {i}",
                "abstract": f"Abstract {i}.",
                "manuscript": f"# Paper {i}\n\n## Methods\n\nText {i}.",
                "code": "x = 1",
                "primary_agent_id": f"agent_{i % 4}",
                "publication_t": i % 5,
                "publication_status": "published",
                "citation_count": i % 3,
                # Same parity, so the cited paper is in the same snapshot.
                "cited_paper_ids": [f"paper_{i - 2}"] if i >= 3 else [],
                "metrics": {
                    "support": {
                        "external_papers_read": i,
                        "successful_code_executions": i,
                        "lines_of_code": 10 * i,
                        "analytical_depth": 0.5,
                    },
                    "soundness": {
                        "internal_consistency_score": 0.9,
                        "diagnostic_validity_pass_rate": 1.0,
                    },
                    "significance": {"clinical_biological_relevance": 0.4},
                    "originality": {
                        "semantic_novelty_knn_distance": 0.25,
                        "embedding_novelty_percentile": 50,
                    },
                },
                "reviews": [
                    {
                        "reviewer_id": "agent_1",
                        "iteration": 1,
                        "scores": {"overall": 3},
                        "review_text": "## Summary\n\nFine.",
                    }
                ],
                "metareview": {
                    "decision": "accept",
                    "overall_score": 3.5,
                    "rank": 1,
                    "meta_review_text": "Accept.",
                },
            }
        )
    return papers


def _write_ranker_runs(out_dir: Path, papers: list[dict]) -> None:
    """One all-pairs non-paired run per rubric; a paper's quality is its index plus noise."""
    rng = random.Random(0)
    for dim in DIMS:
        run = out_dir / "ranker" / dim
        (run / "papers").mkdir(parents=True)
        records = []
        for a, b in itertools.combinations(papers, 2):
            if rng.random() < 0.5:
                a, b = b, a
            qa = int(a["title"].split()[1]) + rng.gauss(0, 4)
            qb = int(b["title"].split()[1]) + rng.gauss(0, 4)
            records.append(
                {
                    "status": "completed",
                    "score": int(qa > qb),
                    "paper_a_id": a["paper_id"],
                    "paper_b_id": b["paper_id"],
                    "paper_a_source_snapshot_dir": a["source_snapshot_dir"],
                    "paper_b_source_snapshot_dir": b["source_snapshot_dir"],
                }
            )
        (run / "comparisons.json").write_text(json.dumps(records))
        (run / "papers" / "paper_data.json").write_text(json.dumps(papers))


def _run_notebook(name: str, stubs: dict | None = None) -> dict:
    """Execute a notebook's code cells; replace any function named in ``stubs`` once defined."""
    stubs = stubs or {}
    ns: dict = {"display": lambda *args, **kwargs: None, "__name__": "__main__"}
    nb = nbformat.read(NOTEBOOKS / name, as_version=4)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        exec(compile(cell.source, f"{name}[{i}]", "exec"), ns)
        ns.update({k: v for k, v in stubs.items() if k in ns})
    plt.close("all")
    return ns


def _fake_pdf(markdown_content: str, output_path: Path, timeout: int = 300) -> bool:
    Path(output_path).write_text(markdown_content, encoding="utf-8")
    return True


def test_notebooks_have_no_outputs_and_no_absolute_paths():
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert cell.outputs == [] and cell.execution_count is None, path.name
            # Windows drive paths: C:\, C:\\ (escaped in a string) and C:/.
            assert not re.search(r"/home/|/mnt/|[A-Z]:[\\/]", cell.source), path.name


def test_expert_validation_notebooks_run_on_synthetic_ranker_outputs(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs" / "expert_validation"
    workdir = tmp_path / "notebooks" / "expert_validation"
    workdir.mkdir(parents=True)
    papers = _papers()
    _write_ranker_runs(out_dir, papers)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    # 01: win rates and the selection.
    ns = _run_notebook("01_pick_papers.ipynb")
    selection_list = json.loads((out_dir / "selection_list.json").read_text())
    (selected,) = selection_list["selection_sets"].values()
    assert selection_list["selection_set_index"] == 2
    assert (
        selection_list["paper_data_path"]
        == "outputs/expert_validation/ranker/default/papers/paper_data.json"
    )
    assert len(selected) == 12 and not any("_incomplete" in uid for uid in selected)
    assert len(json.loads((out_dir / "win_rates.json").read_text())) == N_PAPERS
    # Set 2 excludes set 1, which holds the overall best papers.
    assert ns["df_full"].loc[selected, "wr_default"].max() < ns["df_full"]["wr_default"].max()

    # 02: two documents per selected paper, named by the paper key raters use.
    _run_notebook("02_export_papers.ipynb", stubs={"generate_pdf": _fake_pdf})
    pdf_dir = out_dir / "paper_pdfs" / "set_02"
    pdfs = sorted(p.name for p in pdf_dir.glob("*.pdf"))
    assert len(pdfs) == 24
    assert all(re.fullmatch(r"seed_\d_260101_120\d__paper_\d+(_reviews)?\.pdf", n) for n in pdfs)
    paper_md = next(p for p in pdf_dir.glob("*.pdf") if not p.stem.endswith("_reviews")).read_text()
    assert "\\newpage" in paper_md and "\newpage" not in paper_md
    assert "## Cited Papers" in paper_md and "## Code" in paper_md
    # Citation chains reach the export: paper_k cites paper_(k-2) of the same run.
    k, cited_md = next(
        (int(p.stem.rsplit("_", 1)[1]), p.read_text())
        for p in sorted(pdf_dir.glob("*.pdf"))
        if not p.stem.endswith("_reviews") and int(p.stem.rsplit("_", 1)[1]) >= 3
    )
    assert f"**Title:** Paper {k - 2}" in cited_md

    # 03: Rater A agrees with the ranker; Rater B reverses it.
    ranker = pd.read_json(out_dir / "selection.json")
    keys = [p.removesuffix(".pdf") for p in pdfs if not p.endswith("_reviews.pdf")]
    key_of = {k.split("__", 1)[1]: k for k in keys}
    ranker["paper_id_short"] = ranker["paper_id"].map(key_of)
    order = ranker.sort_values(["wr_default", "paper_id_short"], ascending=[False, True])
    ratings = out_dir / "ratings"
    ratings.mkdir()
    for name, ids in (
        ("first.csv", order["paper_id_short"]),
        ("second.csv", order["paper_id_short"][::-1]),
    ):
        ranks = range(1, len(ids) + 1)
        cols = ("novelty_rank", "significance_rank", "soundness_rank", "overall_rank")
        pd.DataFrame({"paper_id": list(ids), **{c: list(ranks) for c in cols}}).to_csv(
            ratings / name, index=False
        )

    ns = _run_notebook("03_ranker_vs_expert.ipynb")
    summary = ns["summary_expert_vs_ranker"].set_index(["expert", "dimension"])
    assert summary.loc[("Rater A", "overall_default"), "spearman_rho"] == pytest.approx(1.0)
    assert summary.loc[("Rater B", "overall_default"), "spearman_rho"] == pytest.approx(-1.0)
    assert set(summary["n"]) == {12}
    pairs = ns["summary_expert_vs_expert"].set_index("dimension")
    assert pairs.loc["overall", "spearman_rho"] == pytest.approx(-1.0)

    # Everything the notebooks write lands under outputs/.
    assert list(workdir.iterdir()) == []
