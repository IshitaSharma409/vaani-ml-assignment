"""Run the small, controlled Phase 2 correctness sample against Groq."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from relay.config import GroqSettings
from relay.graph.workflow import compile_workflow
from relay.retrieval.index import RetrievalService


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/raw/phase2_groq_smoke.jsonl"
CASES = [
    ("attention", "Why are dot products scaled by the square root of key dimension d_k in attention?"),
    ("moshi", "What is Mimi and how does it tokenize audio using semantic and acoustic information?"),
    ("dynamic_chunking", "How does H-Net learn content-aware chunk boundaries through end-to-end training?"),
    ("outside_library", "What is the boiling point of water at sea level in Celsius?"),
]


def main() -> None:
    model_settings = GroqSettings.from_environment()
    model = model_settings.create_chat_model()
    workflow = compile_workflow(RetrievalService(), model)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for case_id, question in CASES:
        request_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        try:
            state = workflow.invoke({"request_id": request_id, "user_query": question, "retrieval_setting": "medium"})
            record = {
                "request_id": request_id,
                "case_id": case_id,
                "source": "real_groq",
                "model": model_settings.model,
                "prompt_version": "relay-rag-v1",
                "scenario": "phase2_correctness_smoke",
                "start_time": started_at,
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
                "question": question,
                "status": state.get("final_status", "error"),
                "answer": state.get("final_response", ""),
                "retrieval_setting": "medium",
                "retrieval_attempts": state.get("retrieval_attempts"),
                "tool_calls": state.get("tool_calls", []),
                "citations": [{key: passage.get(key) for key in ("chunk_id", "paper_id", "arxiv_version", "pdf_page_number", "score")} for passage in state.get("retrieved_passages", [])],
                "model_calls": state.get("call_usage", []),
                "verification_passed": state.get("verification_passed"),
                "revision_attempts": state.get("revision_attempts"),
            }
        except Exception as exc:
            record = {"request_id": request_id, "case_id": case_id, "source": "real_groq", "model": model_settings.model,
                      "scenario": "phase2_correctness_smoke", "start_time": started_at,
                      "total_ms": round((time.perf_counter() - started) * 1000, 1), "question": question,
                      "status": "error", "error_class": type(exc).__name__, "error": str(exc)}
        with OUTPUT.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{case_id}: {record['status']} ({record['total_ms']} ms)")
        print(record.get("answer", record.get("error", "")))


if __name__ == "__main__":
    main()
