"""Validate evidence-review grades and propagate them to Phase 4 raw outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from relay.evaluation.grading import ManualGrade, validate_grade_coverage
from relay.schemas import ResultRecord


ROOT = Path(__file__).resolve().parents[1]


def _replace_lines(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    answer_path = ROOT / "experiments/raw/evaluation_answers.jsonl"
    trace_path = ROOT / "experiments/raw/request_traces.jsonl"
    results_path = ROOT / "experiments/raw/results.csv"
    grade_path = ROOT / "experiments/raw/evaluation_grades.yaml"
    answers = [json.loads(line) for line in answer_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    answer_by_id = {item["request_id"]: item for item in answers}
    usable_ids = {item["request_id"] for item in answers if item["status"] == "ok"}
    grade_doc = yaml.safe_load(grade_path.read_text(encoding="utf-8"))
    grades = [ManualGrade.model_validate(item) for item in grade_doc.get("grades", [])]
    validate_grade_coverage(usable_ids, grades)
    question_by_id = {item["id"]: item for item in yaml.safe_load(
        (ROOT / "src/relay/evaluation/questions.yaml").read_text(encoding="utf-8")).get("questions", [])}
    for grade in grades:
        if answer_by_id[grade.request_id]["question_id"] != grade.question_id:
            raise ValueError(f"Grade question ID does not match request {grade.request_id}")
        if grade.question_id not in question_by_id:
            raise ValueError(f"Unknown evaluation question: {grade.question_id}")
        answer_by_id[grade.request_id]["answer_grade"] = grade.answer_grade
        answer_by_id[grade.request_id]["grade_reason"] = grade.reason
        answer_by_id[grade.request_id]["grade_evidence_pages"] = grade.evidence_pages
    _replace_lines(answer_path, answers)

    trace_records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grade_by_id = {grade.request_id: grade for grade in grades}
    for trace in trace_records:
        if trace.get("request_id") in grade_by_id:
            trace["answer_grade"] = grade_by_id[trace["request_id"]].answer_grade
            trace["grade_reason"] = grade_by_id[trace["request_id"]].reason
    _replace_lines(trace_path, trace_records)

    with results_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("results.csv is missing its header")
    for row in rows:
        if row["request_id"] in grade_by_id:
            row["answer_grade"] = grade_by_id[row["request_id"]].answer_grade
        # CSV encodes nullable fields as empty strings. Normalize those values
        # before validating the row against its typed application schema.
        normalized = {key: (None if value == "" else value) for key, value in row.items()}
        ResultRecord.model_validate(normalized)
    temporary = results_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results_path)
    print(f"Applied {len(grades)} evidence-based grades; {len(answers) - len(grades)} operational failures remain ungraded.")


if __name__ == "__main__":
    main()
