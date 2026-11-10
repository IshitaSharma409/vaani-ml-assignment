"""Request-scoped cancellation and deadline context helpers."""

from __future__ import annotations

from relay.observability.timing import RequestTrace, current_trace


class RequestDeadlineExceeded(TimeoutError):
    """The request deadline elapsed before another model call could begin."""


class RequestAbandoned(RuntimeError):
    """The client disconnected and no further model calls should be started."""


def check_request_live() -> RequestTrace | None:
    trace = current_trace()
    if trace is None:
        return None
    if trace.cancel_reason == "abandoned":
        raise RequestAbandoned("Client abandoned the request")
    if trace.cancel_event.is_set():
        raise RequestDeadlineExceeded(trace.cancel_reason or "Request cancelled")
    if trace.deadline_monotonic <= __import__("time").monotonic():
        trace.cancel("deadline")
        raise RequestDeadlineExceeded("Request deadline exceeded")
    return trace
