"""Deterministic OpenAI-compatible chat simulation with controllable limits."""

from __future__ import annotations

import asyncio
import random
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from relay.reliability.limiter import estimate_tokens
from relay.standin.profiles import StandInProfile, load_profile


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = None


@dataclass
class SimulationCall:
    completion_id: str
    output: str
    input_tokens: int
    output_tokens: int
    first_token_ms: float
    target_total_ms: float
    active_load_delay_ms: float
    profile_id: str
    started: float
    partial_stream_after_chunks: int | None = None


class SimulatorRejection(Exception):
    def __init__(self, status_code: int, body: dict[str, Any], headers: dict[str, str] | None = None):
        self.status_code, self.body, self.headers = status_code, body, headers or {}
        super().__init__(body.get("error", {}).get("message", "Simulated request rejected"))


class StandInSimulator:
    """A local-only deterministic simulator; all limits and failures are synthetic."""

    def __init__(self, profile: StandInProfile | None = None):
        self.profile = profile or load_profile()
        self._rng = random.Random(self.profile.seed)
        self._lock = threading.RLock()
        self._active = 0
        self._request_count = 0
        self._tokens_today = 0
        self._utc_day = datetime.now(timezone.utc).date().isoformat()
        self._request_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()
        self._forced = list(self.profile.failures.forced_failure_statuses)

    @property
    def active_requests(self) -> int:
        with self._lock:
            return self._active

    def reset(self) -> dict[str, int]:
        """Reset all synthetic counters and restore the configured forced-failure list."""
        with self._lock:
            self._active = 0
            self._request_count = 0
            self._tokens_today = 0
            self._utc_day = datetime.now(timezone.utc).date().isoformat()
            self._request_times.clear()
            self._token_events.clear()
            self._forced = list(self.profile.failures.forced_failure_statuses)
            return {"request_count": self._request_count, "tokens_today": self._tokens_today}

    def _content(self, request: ChatCompletionRequest) -> str:
        messages = request.messages
        system = " ".join(str(message.content) for message in messages if message.role == "system")
        human = " ".join(str(message.content) for message in messages if message.role == "human" or message.role == "user")
        if "claims" in human.lower() and ("draft" in system.lower() or "claim" in system.lower()):
            return '{"supported": true, "issues": ""}'
        match = re.search(r'"chunk_id"\s*:\s*"([a-f0-9]+)"', human)
        claims = []
        if match:
            claims = [{"text": "SIMULATED stand-in output; no paper fact is asserted.", "citation_ids": [match.group(1)]}]
        import json
        return json.dumps({"claims": claims, "refusal": not bool(claims)}, ensure_ascii=False)

    @staticmethod
    def _error(status: int, message: str, code: str, kind: str = "simulated_error") -> dict[str, Any]:
        return {"error": {"message": message, "type": kind, "code": code}}

    def _limit_headers(self, requests_remaining: int, tokens_remaining: int, retry_after: str | None = None) -> dict[str, str]:
        headers = {
            "X-RateLimit-Limit-Requests": str(self.profile.limits.requests_per_day),
            "X-RateLimit-Remaining-Requests": str(max(0, requests_remaining)),
            "X-RateLimit-Limit-Tokens": str(self.profile.limits.tokens_per_minute),
            "X-RateLimit-Remaining-Tokens": str(max(0, tokens_remaining)),
            "X-RateLimit-Reset-Requests": "24h",
            "X-RateLimit-Reset-Tokens": "60s",
        }
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        return headers

    def prepare(self, request: ChatCompletionRequest) -> SimulationCall:
        now = time.monotonic()
        text = "\n".join(str(message.content) for message in request.messages)
        input_tokens = estimate_tokens(text)
        output = self._content(request)
        output_tokens = estimate_tokens(output)
        with self._lock:
            current_day = datetime.now(timezone.utc).date().isoformat()
            if current_day != self._utc_day:
                self._request_count = 0
                self._tokens_today = 0
                self._utc_day = current_day
            self._request_count += 1
            count = self._request_count
            while self._request_times and now - self._request_times[0] >= 60:
                self._request_times.popleft()
            while self._token_events and now - self._token_events[0][0] >= 60:
                self._token_events.popleft()
            used_tpm = sum(tokens for _, tokens in self._token_events)
            used_rpm = len(self._request_times)
            if self._active >= self.profile.limits.concurrent_requests:
                raise SimulatorRejection(429, self._error(429, "Simulated in-flight request limit reached.", "simulated_concurrency"), self._limit_headers(self.profile.limits.requests_per_day - self._request_count, self.profile.limits.tokens_per_minute - used_tpm, "1"))
            if used_rpm + 1 > self.profile.limits.requests_per_minute:
                raise SimulatorRejection(429, self._error(429, "Simulated requests-per-minute limit reached.", "requests_per_minute"), self._limit_headers(self.profile.limits.requests_per_day - self._request_count, self.profile.limits.tokens_per_minute - used_tpm, "60"))
            if used_tpm + input_tokens + output_tokens > self.profile.limits.tokens_per_minute:
                raise SimulatorRejection(429, self._error(429, "Simulated tokens-per-minute limit reached.", "tokens_per_minute"), self._limit_headers(self.profile.limits.requests_per_day - self._request_count, self.profile.limits.tokens_per_minute - used_tpm, "60"))
            if count > self.profile.limits.requests_per_day:
                raise SimulatorRejection(429, self._error(429, "Simulated daily request budget exhausted.", "requests_per_day"), self._limit_headers(0, self.profile.limits.tokens_per_minute - used_tpm, "86400"))
            if self._tokens_today + input_tokens + output_tokens > self.profile.limits.tokens_per_day:
                raise SimulatorRejection(429, self._error(429, "Simulated daily token budget exhausted.", "tokens_per_day"), self._limit_headers(self.profile.limits.requests_per_day - self._request_count, self.profile.limits.tokens_per_minute - used_tpm, "86400"))
            forced = self._forced.pop(0) if self._forced else None
            if forced is None and self.profile.failures.fail_every_n_requests and count % self.profile.failures.fail_every_n_requests == 0:
                forced = 500
            if forced is None and self._rng.random() < self.profile.failures.fail_probability:
                forced = 500
            partial_stream = self.profile.failures.partial_stream_after_chunks if forced == "partial_stream" else None
            if forced == "partial_stream":
                forced = None if request.stream else 503
            if forced:
                if forced == "connection_error":
                    raise SimulatorRejection(503, self._error(503, "Simulated connection-style failure.", "connection_error"), {"X-Simulated-Failure": "connection_error"})
                if forced == "timeout":
                    raise SimulatorRejection(504, self._error(504, "Simulated timeout.", "timeout"), {"X-Simulated-Failure": "timeout"})
                status = int(forced)
                raise SimulatorRejection(status, self._error(status, f"Forced simulated HTTP {status}.", "forced_failure"),
                                         {"Retry-After": "1"} if status == 429 else {})
            self._active += 1
            self._request_times.append(now)
            token_total = input_tokens + output_tokens
            self._tokens_today += token_total
            self._token_events.append((now, token_total))
            active = self._active
            first_sample, total_sample = self._rng.choice(list(zip(
                self.profile.latency.first_token_ms_samples,
                self.profile.latency.total_ms_samples,
            )))
        load_delay = max(0, active - 1) * self.profile.latency.load_slowdown_ms_per_extra_active_request
        return SimulationCall(str(uuid.uuid4()), output, input_tokens, output_tokens,
                              first_sample + load_delay, total_sample + load_delay,
                              load_delay, self.profile.profile_id, time.monotonic(), partial_stream)

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    async def stream_chunks(self, call: SimulationCall):
        """Yield piecewise chunks with empirical first/total delay and assumed pacing."""
        await asyncio.sleep(call.first_token_ms / 1000)
        chunks = [call.output[index:index + self.profile.latency.chunk_chars]
                  for index in range(0, len(call.output), self.profile.latency.chunk_chars)] or [""]
        yield chunks[0]
        for piece in chunks[1:]:
            await asyncio.sleep(self.profile.latency.inter_chunk_ms / 1000)
            yield piece
        elapsed_ms = (time.monotonic() - call.started) * 1000
        remaining_ms = call.target_total_ms - elapsed_ms
        if remaining_ms > 0:
            await asyncio.sleep(remaining_ms / 1000)

    async def complete(self, call: SimulationCall) -> None:
        remaining_ms = call.target_total_ms - (time.monotonic() - call.started) * 1000
        if remaining_ms > 0:
            await asyncio.sleep(remaining_ms / 1000)
