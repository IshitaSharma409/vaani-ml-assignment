"""Validate the assignment's one-row-per-request CSV against its raw traces."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
    "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
]
STATUSES = {"ok", "rate_limited", "timeout", "error"}
SOURCES = {"real", "stand-in"}
GRADES = {"", "correct", "wrong", "refused_correctly", "refused_wrongly"}


def _sum_known(calls: list[dict], key: str) -> int | None:
    values = [int(call[key]) for call in calls if call.get(key) is not None]
    return sum(values) if values else None


def validate() -> tuple[int, int]:
    csv_path = ROOT / "experiments/raw/results.csv"
    trace_path = ROOT / "experiments/raw/request_traces.jsonl"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    if fields[:len(REQUIRED)] != REQUIRED:
        raise ValueError(f"Mandatory column names/order differ: {fields[:len(REQUIRED)]}")
    ids = [row["request_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("results.csv contains duplicate request_id values")
    for line_number, row in enumerate(rows, start=2):
        if row["status"] not in STATUSES or row["source"] not in SOURCES:
            raise ValueError(f"Invalid status/source at CSV line {line_number}")
        if row["answer_grade"] not in GRADES:
            raise ValueError(f"Invalid answer_grade at CSV line {line_number}")
        if not row["scenario"] or not row["start_time"] or not row["retrieval_setting"]:
            raise ValueError(f"Missing required per-request value at CSV line {line_number}")

    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trace_by_id = {trace["request_id"]: trace for trace in traces}
    if len(trace_by_id) != len(traces):
        raise ValueError("request_traces.jsonl contains duplicate request_id values")
    if set(trace_by_id) != set(ids):
        raise ValueError("CSV and trace request ID sets differ")
    missing = set(ids) - set(trace_by_id)
    if missing:
        raise ValueError(f"CSV rows have no matching request trace: {sorted(missing)[:5]}")

    for row in rows:
        trace = trace_by_id[row["request_id"]]
        calls = trace.get("model_calls", [])
        for csv_field, trace_field in (("tokens_in", "tokens_in"), ("tokens_out", "tokens_out")):
            expected = _sum_known(calls, trace_field)
            actual = int(row[csv_field]) if row[csv_field] else None
            if actual != expected:
                raise ValueError(f"{csv_field} does not aggregate model calls for {row['request_id']}")
        expected_retries = sum(int(call.get("retry_count", 0)) for call in calls)
        if int(row["retries"] or 0) != expected_retries:
            raise ValueError(f"Retry count does not aggregate model calls for {row['request_id']}")
    return len(rows), len(ids)


def main() -> None:
    row_count, unique_count = validate()
    print(f"Validated {row_count} end-user CSV rows; {unique_count} unique request IDs; token/retry aggregates match traces.")


if __name__ == "__main__":
    main()
