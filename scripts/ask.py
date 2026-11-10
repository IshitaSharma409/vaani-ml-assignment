"""Ask one question using the real Groq backend and local paper index."""

from __future__ import annotations

import argparse
import json
import logging
import uuid

from relay.graph.workflow import compile_workflow
from relay.llm.backend import create_model_backend
from relay.retrieval.index import RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--setting", default="medium_reranked")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    model, _source = create_model_backend()
    workflow = compile_workflow(RetrievalService(), model)
    response = workflow.invoke({"request_id": str(uuid.uuid4()), "user_query": args.question, "retrieval_setting": args.setting})
    print(response["final_response"])
    print(json.dumps({"request_id": response["request_id"], "status": response["final_status"],
                      "retrieval_setting": args.setting, "tool_calls": response.get("tool_calls", []),
                      "model_calls": response.get("call_usage", [])}, indent=2))


if __name__ == "__main__":
    main()
