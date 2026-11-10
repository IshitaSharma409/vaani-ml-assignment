"""Explicit, bounded model-call retries and error classification."""

from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessageChunk

from relay.observability.timing import current_trace
from relay.reliability.deadlines import check_request_live
from relay.reliability.limiter import BudgetLimitExceeded, LocalBudgetLimiter, estimate_tokens, flatten_prompt


@dataclass(frozen=True)
class Failure:
    category: str
    status_code: int | None
    retry_after_seconds: float | None
    retryable: bool


def classify_error(error: Exception) -> Failure:
    """Classify common provider/transport failures without retrying terminal cases."""
    if isinstance(error, BudgetLimitExceeded):
        return Failure(error.dimension, None, error.reset_after_seconds, False)
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    message = str(error).lower().replace("-", " ")
    headers = getattr(response, "headers", {}) or {}
    retry_after = None
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if value:
        try:
            retry_after = float(value)
        except (TypeError, ValueError):
            pass
    if retry_after is None:
        match = re.search(r"(?:try again|retry)(?: in| after)\s*([0-9.]+)\s*s", message)
        if match:
            retry_after = float(match.group(1))
    if status == 429 or "ratelimiterror" in type(error).__name__.lower():
        if any(term in message for term in ("concurrent", "in flight", "concurrency")):
            return Failure("concurrency_limit", status or 429, retry_after, False)
        if any(term in message for term in ("per day", "daily", "tokens per day", "requests per day")):
            return Failure("daily_budget", status or 429, retry_after, False)
        if any(term in message for term in ("per minute", "tokens per minute", "requests per minute", "tpm", "rpm")):
            return Failure("minute_rate_limit", status or 429, retry_after, True)
        return Failure("rate_limit_unknown", status or 429, retry_after, False)
    if status in (401, 403):
        return Failure("authentication_or_permission", status, None, False)
    if status is not None and 500 <= int(status) < 600:
        return Failure("server_error", int(status), retry_after, True)
    if status in (400, 404, 422):
        return Failure("invalid_request", int(status), None, False)
    name = type(error).__name__.lower()
    if "timeout" in name or isinstance(error, TimeoutError):
        return Failure("timeout", status, None, False)
    if "connect" in name or "connection" in name or "network" in name or isinstance(error, (ConnectionError, OSError)):
        return Failure("transient_network", status, retry_after, True)
    return Failure("server_error" if status is None else "error", status, retry_after, False)


class RetryPolicy:
    def __init__(self, max_attempts: int = 2, base_seconds: float = 0.5,
                 max_seconds: float = 8.0, jitter_seed: int = 29):
        if max_attempts < 1 or base_seconds < 0 or max_seconds < base_seconds:
            raise ValueError("Invalid retry policy bounds")
        self.max_attempts, self.base_seconds, self.max_seconds = max_attempts, base_seconds, max_seconds
        self._rng, self._lock = random.Random(jitter_seed), threading.Lock()

    def delay(self, retry_index: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return max(0.0, retry_after)
        ceiling = min(self.max_seconds, self.base_seconds * 2 ** max(0, retry_index))
        with self._lock:
            return self._rng.uniform(ceiling / 2, ceiling) if ceiling else 0.0


class RetryingModel:
    """Wrap a LangChain chat model; own retry counts and stream buffering."""

    def __init__(self, model: Any, retry_policy: RetryPolicy, budget: LocalBudgetLimiter | None = None,
                 output_token_reservation: int = 512, max_concurrent_calls: int = 2):
        self.model, self.retry_policy, self.budget = model, retry_policy, budget
        self.output_token_reservation = output_token_reservation
        self._semaphore = threading.BoundedSemaphore(max_concurrent_calls)

    def invoke(self, prompt: Any, config: Any = None, **kwargs: Any):
        trace = current_trace()
        call_started = time.monotonic()
        prompt_text = flatten_prompt(prompt)
        input_estimate = estimate_tokens(prompt_text)
        attempts, retry_wait = 0, 0.0
        attempt_trace = []
        last_error: Exception | None = None
        while attempts < self.retry_policy.max_attempts:
            if trace and trace.cancel_event.is_set():
                raise TimeoutError(f"Request cancelled: {trace.cancel_reason}")
            check_request_live()
            remaining = (trace.deadline_monotonic - time.monotonic()) if trace else 60.0
            acquired = remaining > 0 and self._semaphore.acquire(timeout=max(0.0, remaining))
            if not acquired:
                if trace:
                    trace.cancel("deadline")
                    trace.error_class = "ModelConcurrencyDeadlineExceeded"
                raise TimeoutError("Timed out waiting for model-call concurrency slot")
            reservation = None
            try:
                if self.budget is not None:
                    reservation = self.budget.reserve_call(input_estimate + self.output_token_reservation)
                attempts += 1
                response, first_token_ms = self._stream_buffered(prompt, config, kwargs, trace, call_started)
                usage = getattr(response, "usage_metadata", None) or {}
                has_provider_usage = "input_tokens" in usage and "output_tokens" in usage
                actual = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                if reservation and has_provider_usage:
                    self.budget.reconcile(reservation, actual)
                metadata = dict(getattr(response, "response_metadata", {}) or {})
                token_source = "simulated" if trace and trace.source == "stand-in" else (
                    "provider" if has_provider_usage else "estimated"
                )
                metadata["relay_retry"] = {"attempts": attempts, "retry_count": attempts - 1,
                                           "retry_wait_seconds": retry_wait, "attempt_trace": attempt_trace}
                response.response_metadata = metadata
                if trace:
                    trace.record_call({"attempts": attempts, "retry_count": attempts - 1,
                                       "retry_wait_seconds": retry_wait, "elapsed_seconds": time.monotonic() - call_started,
                                       "tokens_in": usage.get("input_tokens", input_estimate),
                                       "tokens_out": usage.get("output_tokens", self.output_token_reservation),
                                       "token_source": token_source,
                                       "first_token_ms": first_token_ms, "attempt_trace": attempt_trace})
                return response
            except Exception as exc:
                last_error = exc
                failure = classify_error(exc)
                attempt_trace.append({"attempt": max(attempts, 1), "category": failure.category,
                                      "status_code": failure.status_code, "error_class": type(exc).__name__})
                if not failure.retryable or attempts >= self.retry_policy.max_attempts:
                    if trace:
                        trace.record_call({"attempts": attempts, "retry_count": max(0, attempts - 1),
                                           "retry_wait_seconds": retry_wait, "elapsed_seconds": time.monotonic() - call_started,
                                           "tokens_in": None, "tokens_out": None, "token_source": "unavailable",
                                           "outcome": failure.category, "error_class": type(exc).__name__,
                                           "http_status": failure.status_code, "retry_after_s": failure.retry_after_seconds,
                                           "attempt_trace": attempt_trace})
                        trace.error_class = type(exc).__name__
                        trace.http_status = failure.status_code
                        trace.retry_after_s = failure.retry_after_seconds
                    try:
                        exc.relay_failure = failure
                        exc.relay_attempts = attempts
                        exc.relay_retry_wait_seconds = retry_wait
                    except Exception:
                        pass
                    raise
                delay = self.retry_policy.delay(attempts - 1, failure.retry_after_seconds)
                if trace and time.monotonic() + delay >= trace.deadline_monotonic:
                    trace.error_class = type(exc).__name__
                    trace.record_call({"attempts": attempts, "retry_count": max(0, attempts - 1),
                                       "retry_wait_seconds": retry_wait,
                                       "elapsed_seconds": time.monotonic() - call_started,
                                       "tokens_in": None, "tokens_out": None, "token_source": "unavailable",
                                       "outcome": failure.category, "error_class": type(exc).__name__,
                                       "http_status": failure.status_code,
                                       "retry_after_s": failure.retry_after_seconds,
                                       "attempt_trace": attempt_trace})
                    try:
                        exc.relay_failure = failure
                        exc.relay_attempts = attempts
                        exc.relay_retry_wait_seconds = retry_wait
                    except Exception:
                        pass
                    raise
                if trace and trace.cancel_event.is_set():
                    raise
                attempt_trace[-1]["backoff_seconds"] = delay
                self._semaphore.release()
                acquired = False
                if delay:
                    time.sleep(delay)
                    retry_wait += delay
                continue
            finally:
                if acquired:
                    self._semaphore.release()
        assert last_error is not None
        raise last_error

    def _stream_buffered(self, prompt: Any, config: Any, kwargs: dict[str, Any], trace: Any, call_started: float):
        chunks = []
        first_token_ms = None
        for chunk in self.model.stream(prompt, config=config, **kwargs):
            chunks.append(chunk)
            content = getattr(chunk, "content", None)
            if content and trace and first_token_ms is None:
                first_token_ms = trace.note_model_token(call_started)
        if not chunks:
            raise RuntimeError("Model stream returned no chunks")
        aggregate = chunks[0]
        for chunk in chunks[1:]:
            aggregate = aggregate + chunk
        if hasattr(aggregate, "to_message"):
            return aggregate.to_message(), first_token_ms
        return aggregate, first_token_ms
