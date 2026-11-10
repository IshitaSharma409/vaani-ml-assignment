"""Validated endpoint and required per-request result schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AnswerGrade = Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]
ResultStatus = Literal["ok", "rate_limited", "timeout", "error"]


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    scenario: str = "interactive"
    retrieval_setting: str = "medium_reranked"


class QuestionResponse(BaseModel):
    request_id: str
    status: ResultStatus
    user_outcome: Literal["success", "refused", "rejected", "timed_out", "abandoned", "error"]
    answer: str
    model_name: str


class ResultRecord(BaseModel):
    """One CSV row per user request, retaining required columns unchanged."""

    request_id: str
    scenario: str
    source: Literal["real", "stand-in"]
    start_time: str
    first_token_ms: float | None = None
    total_ms: float
    status: ResultStatus
    retries: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    retrieval_setting: str
    answer_grade: AnswerGrade | None = None
    user_outcome: str
    queue_wait_ms: float = 0.0
    active_work_ms: float = 0.0
    retry_wait_ms: float = 0.0
    retrieval_ms: float = 0.0
    verification_ms: float = 0.0
    first_model_token_ms: float | None = None
    first_user_token_ms: float | None = None
    llm_calls: int = 0
    queue_depth_at_arrival: int = 0
    peak_queue_length: int = 0
    model_name: str
    prompt_version: str = "relay-rag-v2"
    tokens_source: Literal["provider", "simulated", "estimated", "unavailable", "mixed"] = "unavailable"
    estimated_cost_usd: float | None = None
    cost_basis: str = "not calculated; pricing snapshot not configured"
    error_class: str | None = None
    http_status: int | None = None
    retry_after_s: float | None = None
