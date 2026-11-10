from __future__ import annotations

import json
import threading
import time

import pytest
from langchain_core.messages import AIMessageChunk

from relay.observability.timing import RequestTrace, reset_current_trace, set_current_trace
from relay.reliability.limiter import BudgetLimitExceeded, LocalBudgetLimiter
from relay.reliability.retry import RetryingModel, RetryPolicy, classify_error


class StatusError(Exception):
    def __init__(self, status_code: int, message: str, headers=None):
        self.status_code = status_code
        self.response = type("Response", (), {"status_code": status_code, "headers": headers or {}})()
        super().__init__(message)


def test_error_classifier_separates_retryable_minute_and_terminal_daily_limits():
    minute = classify_error(StatusError(429, "tokens per minute limit", {"retry-after": "8"}))
    simulated_minute = classify_error(StatusError(429, "Simulated tokens-per-minute limit reached."))
    concurrency = classify_error(StatusError(429, "Simulated in-flight request limit reached."))
    daily = classify_error(StatusError(429, "requests per day quota exceeded"))
    auth = classify_error(StatusError(401, "unauthorized"))
    assert (minute.category, minute.retryable, minute.retry_after_seconds) == ("minute_rate_limit", True, 8.0)
    assert (simulated_minute.category, simulated_minute.retryable) == ("minute_rate_limit", True)
    assert (concurrency.category, concurrency.retryable) == ("concurrency_limit", False)
    assert (daily.category, daily.retryable) == ("daily_budget", False)
    assert (auth.category, auth.retryable) == ("authentication_or_permission", False)


def test_budget_reservations_reconcile_and_survive_restart(tmp_path):
    state_path = tmp_path / "usage.json"
    budget = LocalBudgetLimiter(state_path, requests_per_minute=2, tokens_per_minute=100,
                                requests_per_day=3, tokens_per_day=150)
    reservation = budget.reserve_call(40)
    budget.reconcile(reservation, 25)
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["tokens_today"] == 25
    reloaded = LocalBudgetLimiter(state_path, requests_per_minute=2, tokens_per_minute=100,
                                  requests_per_day=3, tokens_per_day=150)
    reloaded.reserve_call(40)
    with pytest.raises(BudgetLimitExceeded, match="requests_per_minute"):
        reloaded.reserve_call(1)


def test_retry_wrapper_streams_buffers_and_records_one_bounded_retry():
    class FirstCallFails:
        def __init__(self):
            self.calls = 0

        def stream(self, _prompt, config=None, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("temporary connection failure")
            yield AIMessageChunk(content="buffered response")

    model = FirstCallFails()
    trace = RequestTrace("test", deadline_seconds=5)
    token = set_current_trace(trace)
    try:
        response = RetryingModel(model, RetryPolicy(2, 0.001, 0.002, 5)).invoke("short prompt")
    finally:
        reset_current_trace(token)
    assert response.content == "buffered response"
    assert model.calls == 2
    assert trace.model_calls[0]["retry_count"] == 1
    assert trace.model_calls[0]["attempt_trace"][0]["category"] == "transient_network"
    assert trace.model_calls[0]["token_source"] == "estimated"
    assert trace.retry_wait_ms > 0


def test_retry_after_is_never_shortened_by_backoff_cap():
    assert RetryPolicy(2, 0.1, 1.0, 1).delay(0, retry_after=8.2) == 8.2


@pytest.mark.parametrize(
    ("limits", "reservations", "dimension"),
    [
        ({"requests_per_minute": 10, "tokens_per_minute": 50, "requests_per_day": 10, "tokens_per_day": 500}, [40, 20], "tokens_per_minute"),
        ({"requests_per_minute": 10, "tokens_per_minute": 500, "requests_per_day": 1, "tokens_per_day": 500}, [40, 20], "requests_per_day"),
        ({"requests_per_minute": 10, "tokens_per_minute": 500, "requests_per_day": 10, "tokens_per_day": 50}, [40, 20], "tokens_per_day"),
    ],
)
def test_budget_enforces_token_minute_and_daily_dimensions(tmp_path, limits, reservations, dimension):
    budget = LocalBudgetLimiter(tmp_path / "limits.json", **limits)
    budget.reserve_call(reservations[0])
    with pytest.raises(BudgetLimitExceeded) as caught:
        budget.reserve_call(reservations[1])
    assert caught.value.dimension == dimension


def test_daily_provider_limit_is_terminal_and_does_not_retry():
    class DailyLimitModel:
        calls = 0

        def stream(self, _prompt, config=None, **_kwargs):
            self.calls += 1
            raise StatusError(429, "requests per day quota exceeded")
            yield  # Make this a generator, as required by the chat-model stream contract.

    model = DailyLimitModel()
    trace = RequestTrace("daily exhaustion", deadline_seconds=5)
    token = set_current_trace(trace)
    try:
        with pytest.raises(StatusError):
            RetryingModel(model, RetryPolicy(3, 0, 0, 1)).invoke("prompt")
    finally:
        reset_current_trace(token)
    assert model.calls == 1
    assert trace.model_calls[0]["outcome"] == "daily_budget"
    assert trace.model_calls[0]["retry_count"] == 0


def test_model_call_semaphore_caps_underlying_concurrency():
    class TrackingModel:
        def __init__(self):
            self.active = 0
            self.maximum = 0
            self.lock = threading.Lock()

        def stream(self, _prompt, config=None, **_kwargs):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            yield AIMessageChunk(content="done")

    underlying = TrackingModel()
    wrapped = RetryingModel(underlying, RetryPolicy(max_attempts=1), max_concurrent_calls=1)
    threads = [threading.Thread(target=wrapped.invoke, args=(f"prompt {index}",)) for index in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in threads)
    assert underlying.maximum == 1
