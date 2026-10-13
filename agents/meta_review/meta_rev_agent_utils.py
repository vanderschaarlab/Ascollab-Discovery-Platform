import re
from typing import Any

from system.schema import PaperMetadataPrivate, PaperReview


def make_metareviewer_system_prompt(review_free: bool = False) -> str:
    """
    make_metareviewer_system_prompt

    Args:
        review_free: When True (the reviewerless ablation, experiment 07), the prompt must not
            mention reviews; the meta-reviewer scores submissions directly from their
            content. The Decision/score/rank/justification output contract is identical
            in both modes.

    Returns:
        str: The prompt
    """

    if review_free:
        prompt = """
1. Role and Objectives
You are serving as an expert meta-reviewer. Your responsibility is to evaluate multiple research reports based on their content. For each submission, you must:
- Write a comprehensive meta-review that synthesizes its main strengths and weaknesses.
- Assign a numerical score reflecting its overall merit.
- Provide an overall ranking of all submissions relative to one another.

2. Evaluation Criteria
Each report should be assessed on the following dimensions:
1. Support: Are the claims well supported by evidence, reasoning, and/or consistency with prior knowledge?
2. Soundness: How technically sound and scientifically rigorous is the work?
3. Significance: To what extent does the work advance knowledge or practice in the field?
4. Originality: How novel are the ideas, methods, or results?
"""
    else:
        prompt = """
1. Role and Objectives
You are serving as an expert meta-reviewer. Your responsibility is to evaluate multiple research reports based on their content and the expert reviews they have received. For each submission, you must:
- Write a comprehensive meta-review that synthesizes its main strengths and weaknesses.
- Assign a numerical score reflecting its overall merit.
- Provide an overall ranking of all submissions relative to one another.

2. Evaluation Criteria
Each report should be assessed on the following dimensions:
1. Support: Are the claims well supported by evidence, reasoning, and/or consistency with prior knowledge?
2. Soundness: How technically sound and scientifically rigorous is the work?
3. Significance: To what extent does the work advance knowledge or practice in the field?
4. Originality: How novel are the ideas, methods, or results?
"""

    prompt += """
3. Instructions for Meta-Review
For each report, provide a meta-review following this exact structure:

Paper ID: <id of the report>

1. Brief Summary
- A 1–2 sentence bullet-point summary of its main contributions.

2. Comparative Analysis
- 2–3 bullet points assessing the submission against the criteria.

3. Decision
- score: <float between 0 and 1> (assign each report a score on a 0–1 scale, where 1 = best overall quality)
- rank: <integer rank, 1 is best> (assign each submission a rank from 1 to N, where 1 = best. No ties allowed)
- justification: <brief justification>
"""

    return prompt.strip()


def make_metareviewer_user_message(
    candidates: list[PaperMetadataPrivate],
    reviews: dict[str, list[PaperReview]],
    review_free: bool = False,
) -> str:
    """
    make metareviewer user message

    Args:
        candidates (List[PaperMetadataPrivate]): A list of candidate papers
        reviews (Dict[str, List[PaperReview]]): A mapping of paper IDs to their reviews
        review_free: When True (the reviewerless ablation, experiment 07), omit any mention of
            reviews and do not render review sections, even if `reviews` is non-empty.

    Returns:
        str: The user message for the metareviewer agent
    """
    if review_free:
        prompt = "You have received the following research reports:\n\n"
    else:
        prompt = "You have received the following research reports and their reviews:\n\n"
    for paper in candidates:
        prompt += f"Paper ID: {paper.paper_id}\n"
        prompt += f"Title: {paper.title}\n"
        prompt += f"Manuscript: {paper.manuscript}\n"
        prompt += "-----------------------------\n"

        if not review_free:
            paper_reviews = reviews.get(paper.paper_id, [])
            for review_idx, review in enumerate(paper_reviews):
                prompt += f"Review {review_idx + 1}:\n{review.review_text}"
                prompt += "----------------------------\n"
        prompt += "\n"

    return prompt.strip()


def parse_metareview(text: str, num_papers: int) -> tuple[dict[str, dict[str, Any]], bool]:
    """
    parse_metareview

    Args:
        text (str): The metareview text
        num_papers (int): The number of papers expected in the metareview

    Returns:
        Tuple[Dict[str, Dict[str, Any]], bool]: A tuple containing the parsed metareview data and a boolean validity flag
            return
                {
                    "text": section.strip(),
                    "score": score,
                    "rank": rank,
                    "justification": justification
                },
                boolean_validity
    """
    parsed_dict = {}
    paper_sections = re.split(r"(?:Meta-Review for\s+)?Paper ID:\s*", text)

    if paper_sections and not paper_sections[0].strip():
        paper_sections = paper_sections[1:]

    for section in paper_sections:
        if not section.strip():
            continue

        lines = section.strip().split("\n")
        if not lines:
            continue

        paper_id = lines[0].strip()
        if not paper_id:
            continue

        decision_match = re.search(
            r"(?:3\.\s*Decision|#### 3\. Decision)\s*\n(.*?)(?=(?:Meta-Review for\s+)?Paper ID:|$)",
            section,
            re.DOTALL,
        )
        if not decision_match:
            continue

        decision_text = decision_match.group(1)

        score_match = re.search(
            r"(?:^|\n)\s*-?\s*score:\s*([0-9]*\.?[0-9]+)", decision_text, re.IGNORECASE
        )
        if not score_match:
            continue

        score = float(score_match.group(1))

        rank_match = re.search(r"(?:^|\n)\s*-?\s*rank:\s*([0-9]+)", decision_text, re.IGNORECASE)
        if not rank_match:
            continue
        rank = int(rank_match.group(1))

        justification_match = re.search(
            r"(?:^|\n)\s*-?\s*justification:\s*(.*?)(?=\n|$)", decision_text, re.IGNORECASE
        )
        justification = ""
        if justification_match:
            justification = justification_match.group(1).strip()

        parsed_dict[paper_id] = {
            "text": section.strip(),
            "score": score,
            "rank": rank,
            "justification": justification,
        }

    valid = True

    if len(parsed_dict) != num_papers:
        valid = False

    scores = [data["score"] for data in parsed_dict.values()]
    if not all(0 <= score <= 1 for score in scores):
        valid = False

    ranks = [data["rank"] for data in parsed_dict.values()]
    if sorted(ranks) != list(range(1, num_papers + 1)):
        valid = False

    return parsed_dict, valid
