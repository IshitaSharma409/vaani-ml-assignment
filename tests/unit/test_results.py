from __future__ import annotations

import csv

import pytest
from pydantic import ValidationError

from relay.observability.csv_writer import CSV_COLUMNS, REQUIRED_COLUMNS, ResultWriter
from relay.schemas import ResultRecord


def record(request_id="r-1"):
    return ResultRecord(request_id=request_id, scenario="unit", source="real", start_time="2026-11-10T00:00:00Z",
        total_ms=10.0, status="ok", retrieval_setting="medium", model_name="test", user_outcome="success")


def test_results_schema_preserves_assignment_columns_and_allowed_values(tmp_path):
    assert CSV_COLUMNS[:len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS
    assert REQUIRED_COLUMNS == ("request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
        "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade")
    with pytest.raises(ValidationError):
        ResultRecord(**{**record().model_dump(), "status": "refused_correctly"})
    with pytest.raises(ValidationError):
        ResultRecord(**{**record().model_dump(), "answer_grade": "maybe"})
    writer = ResultWriter(tmp_path / "results.csv", tmp_path / "trace.jsonl")
    writer.write(record(), {"request_id": "r-1", "model_calls": []})
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert tuple(rows[0]) == CSV_COLUMNS
    with pytest.raises(ValueError, match="Duplicate"):
        writer.write(record(), {"request_id": "r-1"})
