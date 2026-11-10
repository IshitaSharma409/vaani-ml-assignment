"""Thread-safe CSV and JSONL persistence with one CSV row per request ID."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

from relay.schemas import ResultRecord


REQUIRED_COLUMNS = (
    "request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status",
    "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
)
EXTRA_COLUMNS = tuple(field for field in ResultRecord.model_fields if field not in REQUIRED_COLUMNS)
CSV_COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS


class ResultWriter:
    """Append a validated result row and its companion detailed trace exactly once."""

    def __init__(self, csv_path: Path, jsonl_path: Path):
        self.csv_path, self.jsonl_path = csv_path, jsonl_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._written: set[str] = set()
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                    raise ValueError("Existing results.csv header differs from the fixed schema")
                self._written = {row["request_id"] for row in reader}
        else:
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=CSV_COLUMNS).writeheader()

    def write(self, record: ResultRecord, detailed_trace: dict) -> None:
        data = record.model_dump(mode="json")
        with self._lock:
            if data["request_id"] in self._written:
                raise ValueError(f"Duplicate result for request {data['request_id']}")
            with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=CSV_COLUMNS).writerow(data)
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(detailed_trace, ensure_ascii=False) + "\n")
            self._written.add(data["request_id"])
