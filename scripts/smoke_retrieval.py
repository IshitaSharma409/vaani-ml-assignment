"""Inspect representative local retrieval results from all three papers."""

from __future__ import annotations

from relay.retrieval.index import RetrievalService


QUESTIONS = [
    "Why are dot products scaled by the square root of key dimension d_k in attention?",
    "What is Mimi and which techniques does it combine to tokenize audio?",
    "How does H-Net learn content-aware chunk boundaries through end-to-end training?",
]


def main() -> None:
    service = RetrievalService()
    for question in QUESTIONS:
        passages, timing = service.search(question, "medium")
        print(f"\nQUESTION: {question}\nTIMING: {timing}")
        for passage in passages:
            excerpt = " ".join(passage["chunk_text"].split())[:260].encode("ascii", "backslashreplace").decode("ascii")
            print(f"{passage['paper_id']}v{passage['arxiv_version']} PDF p.{passage['pdf_page_number']} score={passage['score']:.3f}: {excerpt}")


if __name__ == "__main__":
    main()
