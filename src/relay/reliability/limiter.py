"""Persistent local request/token budgets for real-provider calls."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BudgetLimitExceeded(RuntimeError):
    def __init__(self, dimension: str, reset_after_seconds: float | None = None):
        self.dimension = dimension
        self.reset_after_seconds = reset_after_seconds
        super().__init__(f"Local {dimension} safety budget is exhausted")


def estimate_tokens(text: str) -> int:
    """Named conservative input estimate: UTF-8 character count divided by four."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def flatten_prompt(prompt: Any) -> str:
    """Extract model prompt text for a transparent token estimate."""
    if isinstance(prompt, str):
        return prompt
    messages = getattr(prompt, "messages", None)
    if messages is None and isinstance(prompt, (list, tuple)):
        messages = prompt
    if messages is None:
        return str(prompt)
    return "\n".join(str(getattr(message, "content", message)) for message in messages)


class LocalBudgetLimiter:
    """A conservative per-process limiter whose daily counters survive restarts."""

    def __init__(self, state_path: Path, *, requests_per_minute: int, tokens_per_minute: int,
                 requests_per_day: int, tokens_per_day: int):
        self.state_path = state_path
        self.limits = {"rpm": requests_per_minute, "tpm": tokens_per_minute,
                       "rpd": requests_per_day, "tpd": tokens_per_day}
        self._lock = threading.RLock()
        self._state = self._load()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"utc_date": datetime.now(timezone.utc).date().isoformat(), "requests_today": 0,
                "tokens_today": 0, "minute_events": [], "reservations": {}}

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._empty_state()
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            required = {"utc_date", "requests_today", "tokens_today", "minute_events", "reservations"}
            if not required.issubset(state):
                raise ValueError("required fields are missing")
            return state
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Local budget state is invalid; refusing real calls: {exc}") from exc

    def _roll_day(self) -> None:
        current_date = datetime.now(timezone.utc).date().isoformat()
        if self._state["utc_date"] != current_date:
            self._state = self._empty_state()

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def reserve_call(self, estimated_tokens: int) -> str:
        now = time.time()
        with self._lock:
            self._roll_day()
            recent = [event for event in self._state["minute_events"] if now - event["at"] < 60]
            minute_requests = len(recent)
            minute_tokens = sum(event["tokens"] for event in recent)
            if minute_requests + 1 > self.limits["rpm"]:
                raise BudgetLimitExceeded("requests_per_minute", max(0.0, 60 - (now - recent[0]["at"])))
            if minute_tokens + estimated_tokens > self.limits["tpm"]:
                raise BudgetLimitExceeded("tokens_per_minute", max(0.0, 60 - (now - recent[0]["at"])) if recent else 60.0)
            if self._state["requests_today"] + 1 > self.limits["rpd"]:
                raise BudgetLimitExceeded("requests_per_day")
            if self._state["tokens_today"] + estimated_tokens > self.limits["tpd"]:
                raise BudgetLimitExceeded("tokens_per_day")
            reservation = str(uuid.uuid4())
            self._state["requests_today"] += 1
            self._state["tokens_today"] += estimated_tokens
            self._state["minute_events"] = recent + [{"id": reservation, "at": now, "tokens": estimated_tokens}]
            self._state["reservations"][reservation] = estimated_tokens
            self._persist()
            return reservation

    def reconcile(self, reservation: str, actual_tokens: int) -> None:
        """Replace a local estimate with provider usage when it is available."""
        with self._lock:
            self._roll_day()
            estimate = self._state["reservations"].pop(reservation, None)
            if estimate is None:
                return
            delta = int(actual_tokens) - estimate
            self._state["tokens_today"] = max(0, self._state["tokens_today"] + delta)
            for event in self._state["minute_events"]:
                if event["id"] == reservation:
                    event["tokens"] = max(0, event["tokens"] + delta)
                    break
            self._persist()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._roll_day()
            return {"utc_date": self._state["utc_date"], "requests_today": self._state["requests_today"],
                    "tokens_today": self._state["tokens_today"], "limits": dict(self.limits)}
