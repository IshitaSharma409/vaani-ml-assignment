"""Repeatable retrieval comparison and balanced, controlled answer evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from relay.graph.workflow import compile_workflow, corpus_page_counts
from relay.llm.backend import create_model_backend
from relay.observability.csv_writer import ResultWriter
from relay.reliability.admission import AdmissionController
from relay.retrieval.index import RetrievalService


ROOT = Path(__file__).resolve().parents[3]
QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
SETTINGS = ("small_fast", "medium", "medium_reranked")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = document.get("questions", [])
    ids = [item.get("id") for item in questions]
    if len(questions) < 15 or len(ids) != len(set(ids)):
        raise ValueError("Evaluation dataset requires at least 15 uniquely identified questions")
    for item in questions:
        if not item.get("question") or not item.get("reference_answer") or not item.get("grading_note"):
            raise ValueError(f"Incomplete evaluation question: {item.get('id')}")
        if item["answerable"] and (not item["expected_paper_id"] or not item["expected_pdf_pages"]):
            raise ValueError(f"Answerable question lacks paper/page evidence: {item['id']}")
    return questions


def retrieval_comparison(service: RetrievalService, questions: list[dict[str, Any]],
                         output_path: Path, summary_path: Path) -> dict[str, Any]:
    """Run every evaluation item through each retrieval setting and preserve top-k evidence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with output_path.open("w", encoding="utf-8") as output:
        for question in questions:
            for setting in SETTINGS:
                passages, timings = service.search(question["question"], setting)
                first_relevant_rank = None
                if question["answerable"]:
                    for rank, passage in enumerate(passages, start=1):
                        if (passage["paper_id"] == question["expected_paper_id"]
                                and passage["arxiv_version"] == question["expected_version"]
                                and passage["pdf_page_number"] in question["expected_pdf_pages"]):
                            first_relevant_rank = rank
                            break
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "scenario": "phase4_retrieval_comparison", "source": "local_retrieval",
                    "question_id": question["id"], "question": question["question"],
                    "answerable": question["answerable"], "setting": setting,
                    "retrieval_parameters": service.config["settings"][setting],
                    "timings": timings, "relevant_rank": first_relevant_rank,
                    "hit_at_k": first_relevant_rank is not None if question["answerable"] else None,
                    "results": [{key: passage.get(key) for key in (
                        "chunk_id", "paper_id", "title", "arxiv_version", "pdf_page_number",
                        "retrieval_score", "rerank_score", "chunk_text"
                    )} for passage in passages],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                by_setting[setting].append(record)
    for setting, rows in by_setting.items():
        answerable = [row for row in rows if row["answerable"]]
        hits = [row for row in answerable if row["hit_at_k"]]
        reciprocal_ranks = [1 / row["relevant_rank"] if row["relevant_rank"] else 0.0 for row in answerable]
        summary[setting] = {
            "questions_total": len(rows), "answerable_questions": len(answerable),
            "answerable_hit_at_3_count": len(hits),
            "answerable_hit_at_3": len(hits) / len(answerable) if answerable else None,
            "mean_reciprocal_rank_at_3": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
            "unanswerable_questions": sum(not row["answerable"] for row in rows),
            "mean_vector_search_ms": 1000 * sum(row["timings"]["vector_search_seconds"] for row in rows) / max(len(rows), 1),
            "mean_rerank_ms": 1000 * sum(row["timings"]["rerank_seconds"] for row in rows) / max(len(rows), 1),
        }
    summary_path.write_text(json.dumps({
        "source": "local_retrieval", "method": "page-and-version exact hit within configured top_k",
        "dataset_path": str(QUESTIONS_PATH.relative_to(ROOT)), "summary": summary,
    }, indent=2), encoding="utf-8")
    return summary


def balanced_answer_sample(questions: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    """Choose a round-robin paper-balanced set of answerable questions."""
    answerable = [item for item in questions if item["answerable"]]
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in answerable:
        by_paper[item["expected_paper_id"]].append(item)
    papers = sorted(by_paper)
    selected = []
    while len(selected) < min(sample_size, len(answerable)):
        made_progress = False
        for paper in papers:
            if by_paper[paper] and len(selected) < sample_size:
                selected.append(by_paper[paper].pop(0))
                made_progress = True
        if not made_progress:
            break
    return selected


def existing_question_ids(answer_path: Path, backend: str) -> set[str]:
    """Return question IDs already sampled under the source label used in JSONL."""
    if not answer_path.exists():
        return set()
    expected_source = {"groq": "real", "standin": "stand-in"}[backend]
    records = [json.loads(line) for line in answer_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {item["question_id"] for item in records if item.get("source") == expected_source}


async def run_answer_sample(questions: list[dict[str, Any]], sample_size: int,
                           delay_seconds: float, backend: str,
                           exclude_question_ids: set[str] | None = None) -> list[dict[str, Any]]:
    app_config = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf-8"))
    model, source = create_model_backend(backend)
    underlying = getattr(model, "model", None)
    model_name = getattr(underlying, "model_name", getattr(underlying, "model", "unknown"))
    workflow = compile_workflow(RetrievalService(), model, corpus_page_counts())
    writer = ResultWriter(ROOT / "experiments/raw/results.csv", ROOT / "experiments/raw/request_traces.jsonl")
    limits = app_config["request"]
    admission = AdmissionController(workflow, capacity=limits["queue_capacity"], workers=1,
        deadline_seconds=limits["deadline_seconds"], source=source, model_name=str(model_name), result_writer=writer)
    sample = [item for item in balanced_answer_sample(questions, sample_size)
              if item["id"] not in (exclude_question_ids or set())]
    records = []
    await admission.start()
    try:
        for question in sample:
            for setting in SETTINGS:
                response = await admission.submit(question["question"], "phase4_retrieval_quality", setting)
                records.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(), "source": source,
                    "scenario": "phase4_retrieval_quality", "question_id": question["id"],
                    "question": question["question"], "expected_reference": question["reference_answer"],
                    "expected_paper_id": question["expected_paper_id"],
                    "expected_version": question["expected_version"],
                    "expected_pdf_pages": question["expected_pdf_pages"],
                    "retrieval_setting": setting, "request_id": response.request_id,
                    "status": response.status, "user_outcome": response.user_outcome,
                    "answer": response.answer, "answer_grade": None,
                    "grading_note": "Manual review against the cited PDF page is required.",
                })
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
    finally:
        await admission.close()
    output = ROOT / "experiments/raw/evaluation_answers.jsonl"
    with output.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("retrieval", "answers"), default="retrieval")
    parser.add_argument("--backend", choices=("groq", "standin"), default="standin",
                        help="Answer-run backend; local stand-in is the safe default.")
    parser.add_argument("--sample-size", type=int, default=3,
                        help="Balanced supported-question count for each retrieval setting.")
    parser.add_argument("--delay-seconds", type=float, default=10.0,
                        help="Pause between end-user requests in answer mode.")
    parser.add_argument("--exclude-existing-questions", action="store_true",
                        help="Skip question IDs already present in evaluation_answers.jsonl; useful for a safe resume.")
    args = parser.parse_args()
    questions = load_questions()
    if args.mode == "retrieval":
        result = retrieval_comparison(RetrievalService(), questions,
            ROOT / "experiments/raw/retrieval_evaluation.jsonl",
            ROOT / "experiments/summaries/retrieval_evaluation.json")
        print(json.dumps(result, indent=2))
    else:
        excluded: set[str] = set()
        if args.exclude_existing_questions:
            excluded = existing_question_ids(ROOT / "experiments/raw/evaluation_answers.jsonl", args.backend)
        records = asyncio.run(run_answer_sample(questions, args.sample_size, args.delay_seconds,
                                                args.backend, excluded))
        print(f"Wrote {len(records)} {args.backend} answer records for manual grading.")


if __name__ == "__main__":
    main()
