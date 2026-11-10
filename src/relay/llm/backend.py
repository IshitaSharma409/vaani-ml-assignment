"""Select the real Groq or local stand-in model without changing graph logic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from relay.llm.groq_backend import GroqBackend
from relay.llm.standin_backend import StandInBackend
from relay.reliability.limiter import LocalBudgetLimiter
from relay.reliability.retry import RetryingModel, RetryPolicy


ROOT = Path(__file__).resolve().parents[3]


def create_model_backend(name: str | None = None, *, budget: LocalBudgetLimiter | None = None,
                         base_url: str | None = None) -> tuple[Any, str]:
    """Create a wrapped LangChain model and its truthful source label."""
    app = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf-8"))
    settings = app["llm"]
    name = (name or os.getenv(settings.get("backend_env_var", "RELAY_BACKEND"), settings["provider"])).lower()
    if name == "groq":
        model = GroqBackend(timeout_seconds=settings["timeout_seconds"], max_tokens=settings["max_output_tokens"]).create_model()
        source = "real"
        if budget is None and app["local_budgets"].get("enabled_for_groq", True):
            limits = app["local_budgets"]
            budget = LocalBudgetLimiter(ROOT / limits["daily_state_path"],
                requests_per_minute=limits["requests_per_minute"], tokens_per_minute=limits["tokens_per_minute"],
                requests_per_day=limits["requests_per_day"], tokens_per_day=limits["tokens_per_day"])
        applied_budget = budget
    elif name in {"standin", "stand-in", "local"}:
        model = StandInBackend(base_url=base_url, max_tokens=settings["max_output_tokens"]).create_model()
        source, applied_budget = "stand-in", None
    else:
        raise ValueError(f"Unknown model backend {name!r}; choose 'groq' or 'standin'")
    retry = app["local_budgets"]
    policy = RetryPolicy(retry["retry_max_attempts"], retry["retry_base_seconds"],
                         retry["retry_max_seconds"], retry["retry_jitter_seed"])
    limiter = app["request"]["max_concurrent_model_calls"]
    wrapped = RetryingModel(model, policy, applied_budget, settings["max_output_tokens"], limiter)
    return wrapped, source
