"""Typed shared request state for the answer workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class RelayState(TypedDict, total=False):
    request_id: str
    user_query: str
    normalized_query: str
    retrieval_setting: str
    retrieved_passages: list[dict[str, Any]]
    retrieval_attempts: int
    evidence_sufficient: bool
    tool_calls: list[dict[str, Any]]
    draft_claims: list[dict[str, Any]]
    draft_answer: str
    verification_passed: bool
    verification_issues: str
    revision_attempts: int
    refusal_reason: str
    final_response: str
    final_status: str
    call_usage: list[dict[str, Any]]
    stage_timings: dict[str, float]

