"""Derive answer quality metrics from preserved Phase 4 answer and result files."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _number(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def main() -> None:
    answers = [json.loads(line) for line in (ROOT / "experiments/raw/evaluation_answers.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()]
    with (ROOT / "experiments/raw/results.csv").open(newline="", encoding="utf-8") as stream:
        result_by_id = {row["request_id"]: row for row in csv.DictReader(stream)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for answer in answers:
        grouped[answer["retrieval_setting"]].append(answer)
    question_counts = Counter(answer["question_id"] for answer in answers)
    question_setting_counts = Counter((answer["question_id"], answer["retrieval_setting"]) for answer in answers)
    unique_question_count = len(question_counts)
    repeats = sum(count - 1 for count in question_setting_counts.values() if count > 1)
    summary = {}
    for setting, rows in sorted(grouped.items()):
        grades = Counter(row["answer_grade"] for row in rows if row.get("answer_grade"))
        successful = [row for row in rows if row["status"] == "ok"]
        latencies = [_number(result_by_id[row["request_id"]].get("total_ms")) for row in successful
                     if row["request_id"] in result_by_id]
        tokens_by_source: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            result = result_by_id.get(row["request_id"])
            if result:
                input_tokens, output_tokens = _number(result.get("tokens_in")), _number(result.get("tokens_out"))
                if input_tokens is not None or output_tokens is not None:
                    tokens_by_source[result.get("tokens_source") or "unavailable"].append(
                        int((input_tokens or 0) + (output_tokens or 0))
                    )
        summary[setting] = {
            "offered": len(rows),
            "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
            "usable_answer_count": len(successful),
            "grade_counts_on_usable_answers": dict(sorted(grades.items())),
            "usable_answer_latency_ms": {
                "population": "status=ok; application total_ms in results.csv for real-source requests",
                "n": len(latencies),
                "p50": float(np.quantile(latencies, 0.50, method="linear")) if latencies else None,
                "p95": float(np.quantile(latencies, 0.95, method="linear")) if latencies else None,
                "p99": float(np.quantile(latencies, 0.99, method="linear")) if latencies else None,
            },
            "token_counts_by_provenance": {
                source: {"requests_with_counts": len(values), "total_tokens": sum(values)}
                for source, values in sorted(tokens_by_source.items())
            },
        }
    payload = {
        "source": "real Groq low-rate application requests; derived from evaluation_answers.jsonl and results.csv",
        "request_count": len(answers),
        "sample_design": (f"{unique_question_count} unique answerable questions x {len(grouped)} retrieval settings; "
                          f"{repeats} repeated request(s) are retained in addition to the unique-question comparison"),
        "warning": "Small exploratory sample; operational errors and local safety limiter outcomes are retained.",
        "summary": summary,
    }
    destination = ROOT / "experiments/summaries/phase4_answer_metrics.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
