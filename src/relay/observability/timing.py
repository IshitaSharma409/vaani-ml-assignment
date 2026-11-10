"""UTC and monotonic request timing context shared with model-call wrappers."""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RequestTrace:
    """Mutable trace for one end-user request; durations use a monotonic clock."""

    question: str
    scenario: str = "interactive"
    source: str = "real"
    retrieval_setting: str = "medium_reranked"
    model_name: str = "unknown"
    deadline_seconds: float = 60.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_monotonic: float = field(default_factory=time.monotonic)
    enqueued_monotonic: float | None = None
    dequeued_monotonic: float | None = None
    queue_depth_at_arrival: int = 0
    queue_wait_ms: float = 0.0
    retry_wait_ms: float = 0.0
    first_model_token_ms: float | None = None
    first_user_token_ms: float | None = None
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cancel_reason: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    error_class: str | None = None
    error_detail: str | None = None
    http_status: int | None = None
    retry_after_s: float | None = None
    final_status: str | None = None
    final_response: str = ""
    answer_grade: str | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    completed_monotonic: float | None = None

    @property
    def deadline_monotonic(self) -> float:
        return self.started_monotonic + self.deadline_seconds

    def mark_enqueued(self, queue_depth: int) -> None:
        self.enqueued_monotonic = time.monotonic()
        self.queue_depth_at_arrival = queue_depth

    def mark_dequeued(self) -> None:
        self.dequeued_monotonic = time.monotonic()
        if self.enqueued_monotonic is not None:
            self.queue_wait_ms = max(0.0, (self.dequeued_monotonic - self.enqueued_monotonic) * 1000)

    def note_model_token(self, call_started: float) -> float:
        now = time.monotonic()
        call_ms = max(0.0, (now - call_started) * 1000)
        if self.cancel_event.is_set():
            return call_ms
        if self.first_model_token_ms is None:
            self.first_model_token_ms = max(0.0, (now - self.started_monotonic) * 1000)
        return call_ms

    def cancel(self, reason: str) -> None:
        self.cancel_reason = reason
        self.cancel_event.set()

    def finish(self) -> None:
        if self.completed_monotonic is None:
            self.completed_monotonic = time.monotonic()

    @property
    def total_ms(self) -> float:
        end = self.completed_monotonic or time.monotonic()
        return max(0.0, (end - self.started_monotonic) * 1000)

    @property
    def active_work_ms(self) -> float:
        return max(0.0, self.total_ms - self.queue_wait_ms - self.retry_wait_ms)

    def record_call(self, call: dict[str, Any]) -> None:
        self.model_calls.append(call)
        self.retry_wait_ms += float(call.get("retry_wait_seconds", 0.0)) * 1000


_current_trace: contextvars.ContextVar[RequestTrace | None] = contextvars.ContextVar("relay_request_trace", default=None)


def current_trace() -> RequestTrace | None:
    return _current_trace.get()


def set_current_trace(trace: RequestTrace):
    return _current_trace.set(trace)


def reset_current_trace(token: contextvars.Token) -> None:
    _current_trace.reset(token)
