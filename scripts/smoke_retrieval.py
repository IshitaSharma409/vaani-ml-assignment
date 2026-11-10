"""Inspect representative local retrieval results from all three papers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from relay.retrieval.index import RetrievalService


QUESTIONS = [
    "Why are dot products scaled by the square root of key dimension d_k in attention?",
    "What is Mimi and which techniques does it combine to tokenize audio?",
    "How does H-Net learn content-aware chunk boundaries through end-to-end training?",
]


def main() -> None:
    service = RetrievalService()
    output_path = Path(__file__).resolve().parents[1] / "experiments/raw/phase2_retrieval_smoke.jsonl"
    for question in QUESTIONS:
        for setting in ("small_fast", "medium", "medium_reranked"):
            passages, timing = service.search(question, setting)
            record = {
                "source": "local_retrieval",
                "scenario": "phase2_retrieval_smoke",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "retrieval_setting": setting,
                "retrieval_parameters": service.config["settings"][setting],
                "timings": timing,
                "results": [{key: passage.get(key) for key in ("chunk_id", "paper_id", "title", "arxiv_version", "pdf_page_number", "retrieval_score", "rerank_score")} for passage in passages],
            }
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"\nSETTING: {setting} | QUESTION: {question}\nTIMING: {timing}")
            for passage in passages:
                excerpt = " ".join(passage["chunk_text"].split())[:180].encode("ascii", "backslashreplace").decode("ascii")
                print(f"{passage['paper_id']}v{passage['arxiv_version']} PDF p.{passage['pdf_page_number']} vector={passage['retrieval_score']:.3f} rerank={passage['rerank_score']}: {excerpt}")


if __name__ == "__main__":
    main()
