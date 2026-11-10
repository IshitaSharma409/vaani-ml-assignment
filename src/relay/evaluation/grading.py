"""Manual grading rubric and validation helpers for the evaluation set."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AnswerGrade = Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]

RUBRIC = {
    "correct": "Substantively supported by the referenced paper and correctly cited to its source location.",
    "wrong": "Contains an unsupported claim, incorrect fact, incorrect paper/version/page, or fabricated citation.",
    "refused_correctly": "Refuses because the fixed three-paper corpus does not answer the question.",
    "refused_wrongly": "Refuses even though the required evidence is present and retrievable in the fixed corpus.",
}


class ManualGrade(BaseModel):
    question_id: str
    request_id: str
    answer_grade: AnswerGrade
    reason: str = Field(min_length=8, max_length=500)
    evidence_pages: list[int] = Field(default_factory=list)
    grader: str = "manual-review"


def validate_grade_coverage(expected_request_ids: set[str], grades: list[ManualGrade]) -> None:
    """Require exactly one human rubric result per answer-evaluation request."""
    actual = [grade.request_id for grade in grades]
    if len(actual) != len(set(actual)):
        raise ValueError("A request has more than one manual grade")
    if set(actual) != expected_request_ids:
        missing = expected_request_ids - set(actual)
        extra = set(actual) - expected_request_ids
        raise ValueError(f"Manual grade coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
