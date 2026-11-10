from __future__ import annotations

import pytest

from relay.evaluation.grading import ManualGrade, validate_grade_coverage
from relay.evaluation.metrics import summarize_scenario, summarize_retrieval
from relay.evaluation.run_eval import balanced_answer_sample, existing_question_ids, load_questions


def test_question_set_has_corpus_versions_evidence_and_negative_controls():
    questions = load_questions()
    assert len(questions) == 18
    assert sum(question["answerable"] for question in questions) == 15
    assert sum(not question["answerable"] for question in questions) == 3
    sample = balanced_answer_sample(questions, 3)
    assert {question["expected_paper_id"] for question in sample} == {"1706.03762", "2410.00037", "2507.07955"}


def test_expanded_balanced_sample_is_unique_and_spans_all_papers():
    questions = load_questions()
    selected = balanced_answer_sample(questions, 15)
    assert len(selected) == 15
    assert len({item["id"] for item in selected}) == 15
    assert {item["expected_paper_id"] for item in selected} == {"1706.03762", "2410.00037", "2507.07955"}


def test_resume_skips_questions_by_recorded_backend_source(tmp_path):
    from pathlib import Path
    answer_path = Path(tmp_path) / "answers.jsonl"
    answer_path.write_text(
        '{"question_id":"q-real","source":"real"}\n'
        '{"question_id":"q-sim","source":"stand-in"}\n', encoding="utf-8")
    assert existing_question_ids(answer_path, "groq") == {"q-real"}
    assert existing_question_ids(answer_path, "standin") == {"q-sim"}


def test_manual_grade_schema_and_exact_coverage():
    grades = [
        ManualGrade(question_id="q1", request_id="r1", answer_grade="correct",
                    reason="Supported and correctly cited.", evidence_pages=[4]),
        ManualGrade(question_id="q2", request_id="r2", answer_grade="refused_correctly",
                    reason="This fact is absent from all three papers."),
    ]
    validate_grade_coverage({"r1", "r2"}, grades)
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_grade_coverage({"r1", "r3"}, grades)
    with pytest.raises(ValueError):
        ManualGrade(question_id="q1", request_id="r1", answer_grade="correct", reason="short")


def test_metrics_are_derived_with_documented_populations_and_percentiles():
    rows = [
        {"status": "ok", "user_outcome": "success", "first_token_ms": "10", "total_ms": "20",
         "tokens_in": "4", "tokens_out": "3", "retries": "0", "peak_queue_length": "2"},
        {"status": "ok", "user_outcome": "success", "first_token_ms": "30", "total_ms": "40",
         "tokens_in": "6", "tokens_out": "5", "retries": "1", "peak_queue_length": "3"},
        {"status": "timeout", "user_outcome": "timed_out", "first_token_ms": "", "total_ms": "50",
         "tokens_in": "", "tokens_out": "", "retries": "0", "peak_queue_length": "3"},
    ]
    summary = summarize_scenario(rows)
    assert summary["request_count"] == 3
    assert summary["latency_percentiles"]["first_token_ms"]["n"] == 2
    assert summary["latency_percentiles"]["first_token_ms"]["p50"] == 20
    assert summary["latency_percentiles"]["total_ms"]["n"] == 2
    assert summary["status_counts"] == {"ok": 2, "timeout": 1}
    retrieval = summarize_retrieval([
        {"setting": "s", "answerable": True, "relevant_rank": 1},
        {"setting": "s", "answerable": True, "relevant_rank": None},
    ])
    assert retrieval["s"]["hit_count"] == 1
    assert retrieval["s"]["mrr_at_3"] == 0.5
