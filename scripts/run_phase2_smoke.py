"""Run the small, controlled Phase 2 correctness sample against Groq."""

from __future__ import annotations

import json
import time
import uuid
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from relay.config import GroqSettings
from relay.graph.nodes import PROMPT_HASH, PROMPT_VERSION
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case_id for case_id, _ in CASES], help="Run one smoke case")
    parser.add_argument("--skip", action="append", default=[], choices=[case_id for case_id, _ in CASES], help="Skip a case already run")
    args = parser.parse_args()
    model_settings = GroqSettings.from_environment()
    model = model_settings.create_chat_model()
    retrieval = RetrievalService()
    workflow = compile_workflow(retrieval, model)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    selected_cases = [item for item in CASES if (args.case is None or item[0] == args.case) and item[0] not in args.skip]
    for case_id, question in selected_cases:
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
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_HASH,
                "scenario": "phase2_correctness_smoke",
                "start_time": started_at,
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
                "question": question,
                "status": state.get("final_status", "error"),
                "answer": state.get("final_response", ""),
                "retrieval_setting": "medium",
                "retrieval_parameters": retrieval.config["settings"]["medium"],
                "retrieval_attempts": state.get("retrieval_attempts"),
                "tool_calls": state.get("tool_calls", []),
                "stage_timings": state.get("stage_timings", {}),
                "citations": [{key: passage.get(key) for key in ("chunk_id", "paper_id", "arxiv_version", "pdf_page_number", "score")} for passage in state.get("retrieved_passages", [])],
                "model_calls": state.get("call_usage", []),
                "verification_passed": state.get("verification_passed"),
                "revision_attempts": state.get("revision_attempts"),
                "failure_stage": state.get("failure_stage"),
                "failure_detail": state.get("failure_detail"),
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
