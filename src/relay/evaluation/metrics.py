"""Recomputable Phase 4 latency/outcome summaries derived from raw CSV rows."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def summarize_scenario(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one scenario with explicit successful-population percentiles."""
    statuses = Counter(str(row.get("status", "")) for row in rows)
    outcomes = Counter(str(row.get("user_outcome", "")) for row in rows)
    successful = [row for row in rows if row.get("status") == "ok"]
    first_tokens = [_number(row.get("first_token_ms")) for row in successful]
    first_tokens = [value for value in first_tokens if value is not None]
    total_times = [_number(row.get("total_ms")) for row in successful]
    total_times = [value for value in total_times if value is not None]
    percentiles = {}
    for name, values in (("first_token_ms", first_tokens), ("total_ms", total_times)):
        percentiles[name] = {
            "population": "status=ok" + (" and first_token_ms is present" if name == "first_token_ms" else ""),
            "n": len(values), "p50": float(np.quantile(values, 0.50, method="linear")) if values else None,
            "p95": float(np.quantile(values, 0.95, method="linear")) if values else None,
            "p99": float(np.quantile(values, 0.99, method="linear")) if values else None,
        }
    tokens = [int(row["tokens_in"]) + int(row["tokens_out"]) for row in rows
              if row.get("tokens_in") not in (None, "") and row.get("tokens_out") not in (None, "")]
    return {
        "request_count": len(rows), "successful_count": len(successful),
        "status_counts": dict(statuses), "user_outcome_counts": dict(outcomes),
        "latency_percentiles": percentiles,
        "mean_tokens_per_request": float(np.mean(tokens)) if tokens else None,
        "total_retries": sum(int(row.get("retries") or 0) for row in rows),
        "peak_queue_length": max((int(row.get("peak_queue_length") or 0) for row in rows), default=0),
        "mean_queue_wait_ms": float(np.mean([float(row.get("queue_wait_ms") or 0) for row in rows])) if rows else None,
        "mean_active_work_ms": float(np.mean([float(row.get("active_work_ms") or 0) for row in rows])) if rows else None,
        "mean_retry_wait_ms": float(np.mean([float(row.get("retry_wait_ms") or 0) for row in rows])) if rows else None,
    }


def summarize_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact page/version retrieval hit and rank metrics by setting."""
    result = {}
    for setting in sorted({str(row["setting"]) for row in records}):
        rows = [row for row in records if row["setting"] == setting and row.get("answerable")]
        ranks = [row.get("relevant_rank") for row in rows]
        result[setting] = {
            "n_answerable": len(rows), "hit_count": sum(rank is not None for rank in ranks),
            "hit_at_3": sum(rank is not None for rank in ranks) / len(rows) if rows else None,
            "mrr_at_3": sum(1 / rank if rank else 0 for rank in ranks) / len(rows) if rows else None,
        }
    return result
