"""Run a deliberately modest streamed baseline against the configured Groq model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import HumanMessage

from relay.config import GroqSettings


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "raw" / "groq_probe.jsonl"
RUN_OUTPUT = ROOT / "experiments" / "raw" / "groq_probe_runs.jsonl"
PROMPT = "Reply with the single word READY."
MAX_REQUESTS = 5


def _safe_error_fields(error: Exception) -> dict[str, Any]:
    """Extract status and rate-limit metadata without retaining credentials."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    safe_headers = {
        key.lower(): value
        for key, value in headers.items()
        if key.lower().startswith("x-ratelimit-") or key.lower() == "retry-after"
    }
    response_body = getattr(error, "body", None)
    body_text = json.dumps(response_body, default=str) if response_body else None
    if body_text:
        body_text = body_text.replace("GROQ_API_KEY", "[REDACTED]")
        body_text = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", body_text)[:1000]
    return {
        "error_class": type(error).__name__,
        "http_status": getattr(error, "status_code", None)
        or getattr(response, "status_code", None),
        "rate_limit_headers": safe_headers,
        "error_body_excerpt": body_text,
    }


def run_probe(request_count: int, pause_seconds: float) -> list[dict[str, Any]]:
    """Execute short streaming prompts; do not retry or load test the free tier."""
    if not 1 <= request_count <= MAX_REQUESTS:
        raise ValueError(f"--requests must be between 1 and {MAX_REQUESTS}")
    if pause_seconds < 0:
        raise ValueError("--pause-seconds cannot be negative")

    settings = GroqSettings.from_environment()
    run_id = str(uuid.uuid4())
    run_started_at = datetime.now(timezone.utc)
    api_key = settings.api_key.get_secret_value()
    with httpx.Client(timeout=settings.timeout_seconds) as account_client:
        models_response = account_client.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if models_response.status_code in (401, 403):
            raise RuntimeError(
                f"Groq rejected the configured credentials (HTTP {models_response.status_code}); "
                "check the local key and model permissions."
            )
        models_response.raise_for_status()
        model_ids = {
            item.get("id")
            for item in models_response.json().get("data", [])
            if isinstance(item, dict)
        }
        if settings.model not in model_ids:
            raise RuntimeError(
                "GROQ_MODEL is not present in the model list available to this account. "
                "Update it from the account's current model catalog before probing."
            )
        print(
            json.dumps(
                {
                    "event": "model_access_verified",
                    "run_id": run_id,
                    "provider": "groq",
                    "configured_model": settings.model,
                    "accessible_model_count": len(model_ids),
                    "source": "real",
                }
            )
        )

    latest_rate_headers: dict[str, str] = {}

    def capture_rate_headers(response: httpx.Response) -> None:
        latest_rate_headers.clear()
        latest_rate_headers.update(
            {
                name.lower(): value
                for name, value in response.headers.items()
                if name.lower().startswith("x-ratelimit-") or name.lower() == "retry-after"
            }
        )

    http_client = httpx.Client(
        timeout=settings.timeout_seconds,
        event_hooks={"response": [capture_rate_headers]},
    )
    model = settings.create_chat_model(http_client=http_client)
    records = []
    try:
        for index in range(request_count):
            latest_rate_headers.clear()
            started_at = datetime.now(timezone.utc)
            started = time.perf_counter()
            first_token_ms = None
            tokens_in = tokens_out = None
            response_parts: list[str] = []
            record: dict[str, Any] = {
                "request_id": str(uuid.uuid4()),
                "run_id": run_id,
                "scenario": "phase1_real_baseline",
                "source": "real",
                "provider": "groq",
                "model_name": settings.model,
                "start_time": started_at.isoformat(),
                "prompt": PROMPT,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "request_index": index + 1,
                "retries": 0,
            }
            try:
                for chunk in model.stream([HumanMessage(content=PROMPT)]):
                    content = chunk.content
                    text = content if isinstance(content, str) else ""
                    if text and first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    if text:
                        response_parts.append(text)
                    usage = chunk.usage_metadata
                    if usage:
                        tokens_in = usage.get("input_tokens", tokens_in)
                        tokens_out = usage.get("output_tokens", tokens_out)
                record.update(
                    {
                        "status": "ok",
                        "first_model_token_ms": first_token_ms,
                        "total_ms": (time.perf_counter() - started) * 1000,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "tokens_source": "provider"
                        if tokens_in is not None or tokens_out is not None
                        else "unavailable",
                        "provider_rate_limit_headers": dict(latest_rate_headers),
                        "response_excerpt": "".join(response_parts)[:300],
                    }
                )
            except Exception as error:  # Record each failed probe; never retry it here.
                error_fields = _safe_error_fields(error)
                http_status = error_fields.get("http_status")
                record.update(
                    {
                        "status": "rate_limited" if http_status == 429 else "error",
                        "first_model_token_ms": first_token_ms,
                        "total_ms": (time.perf_counter() - started) * 1000,
                        "provider_rate_limit_headers": dict(latest_rate_headers),
                        **error_fields,
                    }
                )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))
            if index + 1 < request_count:
                time.sleep(pause_seconds)
    finally:
        http_client.close()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    run_summary = {
        "run_id": run_id,
        "scenario": "phase1_real_baseline",
        "source": "real",
        "provider": "groq",
        "model_name": settings.model,
        "account_model_access_verified": True,
        "started_at": run_started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "requested_calls": request_count,
        "pause_seconds": pause_seconds,
        "retry_policy": "no retry",
        "status_counts": {
            status: sum(record["status"] == status for record in records)
            for status in sorted({record["status"] for record in records})
        },
        "request_ids": [record["request_id"] for record in records],
    }
    with RUN_OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(run_summary, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    args = parser.parse_args()
    run_probe(args.requests, args.pause_seconds)


if __name__ == "__main__":
    main()
