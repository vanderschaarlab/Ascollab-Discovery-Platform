# system/paper_schema.py
from typing import Any

from pydantic import BaseModel, Field, field_validator

FORBIDDEN_PATTERNS = [
    "dataset access issue",
    "unable to access",
    "once accessible",
    "since I cannot access",
    "without access to the datasets",
    "file not found",
    "issue with accessing the dataset",
]


def _text_ok(s: str, *, min_len: int) -> None:
    """Validate text length and ensure required phrases are present/absent."""
    if not s or len(s.strip()) < min_len:
        raise ValueError(f"text too short (<{min_len} chars)")
    low = s.lower()
    for pat in FORBIDDEN_PATTERNS:
        if pat in low:
            raise ValueError(f"forbidden phrase: {pat}")
    # Require Methods marker in the manuscript
    if min_len >= 800 and ("methods" not in low):
        raise ValueError("manuscript must include a 'Methods' section")


def make_paper_model(*, allowed_dataset_keys: list[str]):
    """Build a Pydantic PaperModel enforcing title/abstract/manuscript constraints.

    Args:
        allowed_dataset_keys: List of dataset keys permitted in `datasets_used`.

    Returns:
        Tuple of (PaperModel class, JSON schema dict).
    """

    class PaperModel(BaseModel):
        title: str = Field(..., description="Concise, specific title.")
        abstract: str = Field(..., description="~150–300 words; concrete summary.")
        manuscript: str = Field(..., description="Full paper text with Methods & Results sections.")
        datasets_used: list[str] = Field(
            ...,
            description=f"Subset of dataset keys actually used. Valid: {allowed_dataset_keys}",
            min_items=1,
        )
        citations: list[dict[str, Any]] | None = None
        methods_name: str | None = Field(
            default=None, description="Optional short name for the method/approach."
        )

        @field_validator("title")
        @classmethod
        def _v_title(cls, v: str):
            _text_ok(v, min_len=15)
            return v

        @field_validator("abstract")
        @classmethod
        def _v_abstract(cls, v: str):
            _text_ok(v, min_len=120)
            return v

        @field_validator("manuscript")
        @classmethod
        def _v_manuscript(cls, v: str):
            _text_ok(v, min_len=800)
            # also require Results/Findings/Analysis cue
            low = v.lower()
            if not any(k in low for k in ["results", "findings", "analysis"]):
                raise ValueError(
                    "manuscript must include a 'Results'/'Findings'/'Analysis' section"
                )
            return v

        @field_validator("datasets_used")
        @classmethod
        def _v_datasets(cls, v: list[str]):
            if not v:
                raise ValueError("datasets_used must not be empty")
            if allowed_dataset_keys:
                bad = [x for x in v if x not in allowed_dataset_keys]
                if bad:
                    raise ValueError(
                        f"unknown dataset keys: {bad}; allowed: {allowed_dataset_keys}"
                    )
            # de-dup while preserving order
            seen, out = set(), []
            for x in v:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

    # JSON schema (optional; useful if you want to show the spec to the model)
    return PaperModel, PaperModel.model_json_schema()
