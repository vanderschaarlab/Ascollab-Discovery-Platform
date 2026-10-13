"""Shared fixtures for ranker tests."""

import os

import pytest
from hydra import compose, initialize_config_dir

# Absolute path to the sys_config directory
_SYS_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sys_config"))


@pytest.fixture(scope="session")
def ranker_cfg():
    """Load ranker.yaml via Hydra compose API."""
    with initialize_config_dir(config_dir=_SYS_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="ranker")
    return cfg


@pytest.fixture(scope="session")
def llm(ranker_cfg):
    """Build the real judge LLM from the ranker config (only its endpoint file is needed)."""
    from env_setup.build_llms import build_llm_pool

    backend = ranker_cfg.ranker.llm_backend
    pool = build_llm_pool(ranker_cfg.endpoints, keys=[backend], primary_alias=backend == "primary")
    if backend not in pool:
        pytest.skip(f"LLM backend '{backend}' not available in pool: {list(pool.keys())}")
    return pool[backend]


@pytest.fixture(autouse=True)
def _real_llm_tests_are_marked_integration(request):
    """A test that uses the real `llm` fixture calls a live API, so it must be marked integration.

    Otherwise a selection such as ``-m sandbox`` would make paid model calls. The check runs
    before the test body, so an unmarked test fails without calling the model.
    """
    if "llm" in request.fixturenames and request.node.get_closest_marker("integration") is None:
        pytest.fail("tests that use the real `llm` fixture must be marked integration")


@pytest.fixture
def sample_papers():
    """Return a (paper_a, paper_b) tuple with obvious quality difference."""
    paper_a = {
        "title": "Statistical analysis of gene expression in TCGA-KIRC",
        "abstract": (
            "We performed differential expression analysis on 530 TCGA-KIRC "
            "samples, identifying 47 significantly dysregulated genes "
            "(FDR < 0.01). Survival analysis confirms prognostic value with "
            "a concordance index of 0.78."
        ),
        "manuscript": (
            "Introduction: Kidney renal clear cell carcinoma (KIRC) is the most "
            "common subtype of kidney cancer, accounting for approximately 70% of "
            "all renal cell carcinomas. Understanding the molecular landscape of "
            "KIRC is essential for developing targeted therapies.\n\n"
            "Methods: We downloaded RNA-seq data for 530 KIRC samples from TCGA. "
            "Differential expression analysis was performed using DESeq2 with "
            "Benjamini-Hochberg correction (FDR < 0.01). Gene set enrichment "
            "analysis was conducted using GSEA v4.2. Survival analysis used Cox "
            "proportional hazards models with 5-fold cross-validation.\n\n"
            "Results: We identified 47 significantly dysregulated genes. Pathway "
            "analysis revealed enrichment of angiogenesis, hypoxia response, and "
            "metabolic reprogramming pathways. Our 10-gene prognostic signature "
            "achieved a concordance index of 0.78 (95% CI: 0.74-0.82) in "
            "cross-validation. Three genes (VHL, PBRM1, SETD2) showed strong "
            "association with overall survival (p < 0.001).\n\n"
            "Discussion: Our comprehensive analysis provides a robust gene "
            "signature for KIRC prognosis. The identified pathways are consistent "
            "with known KIRC biology, validating our approach."
        ),
    }
    paper_b = {
        "title": "Thoughts on cancer",
        "abstract": "Cancer is a very important disease. We think about it.",
        "manuscript": (
            "Cancer is bad. More research is needed. We believe our approach "
            "is novel. We did some analysis and found interesting things. "
            "Our method is better than previous ones because we say so."
        ),
    }
    return paper_a, paper_b


@pytest.fixture
def sandbox_client():
    """Build a SandboxClient, or skip if unavailable."""
    url = os.environ.get("SANDBOX_BASE_URL")
    if not url:
        pytest.skip("SANDBOX_BASE_URL not set")
    from env_setup.sandbox_utils import SandboxClient

    return SandboxClient(base_url=url)
