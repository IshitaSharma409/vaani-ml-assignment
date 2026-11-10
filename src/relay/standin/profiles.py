"""Validated stand-in profile loading and deterministic defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LatencyProfile(BaseModel):
    first_token_ms_samples: list[float] = Field(min_length=1)
    total_ms_samples: list[float] = Field(min_length=1)
    chunk_chars: int = Field(ge=1)
    inter_chunk_ms: float = Field(ge=0)
    load_slowdown_ms_per_extra_active_request: float = Field(ge=0)


class LimitProfile(BaseModel):
    requests_per_minute: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)
    requests_per_day: int = Field(gt=0)
    tokens_per_day: int = Field(gt=0)
    concurrent_requests: int = Field(gt=0)


class FailureProfile(BaseModel):
    forced_failure_statuses: list[int | str] = Field(default_factory=list)
    fail_probability: float = Field(ge=0, le=1)
    fail_every_n_requests: int = Field(ge=0)
    partial_stream_after_chunks: int | None = Field(default=None, ge=1)


class StandInProfile(BaseModel):
    profile_id: str
    model_id: str
    provider_label: str
    seed: int
    latency: LatencyProfile
    limits: LimitProfile
    failures: FailureProfile
    calibration: dict[str, Any]


def load_profile(path: Path | None = None) -> StandInProfile:
    """Load the repository's profile or an explicitly supplied test profile."""
    profile_path = path or Path(__file__).resolve().parents[3] / "config/standin_profile.yaml"
    return StandInProfile.model_validate(yaml.safe_load(profile_path.read_text(encoding="utf-8")))
