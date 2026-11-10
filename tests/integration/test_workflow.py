from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from relay.graph.workflow import compile_workflow


class FakeRetrieval:
    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    def search(self, query, setting, top_k=None):
        self.calls.append(query)
        passages = self.passages(query) if callable(self.passages) else self.passages
        return passages, {"vector_search_seconds": 0.001, "rerank_seconds": 0.0}


PASSAGE = {
    "chunk_id": "c1", "chunk_text": "The Transformer uses self-attention and feed-forward layers in its encoder and decoder.",
    "paper_id": "1706.03762", "title": "Attention Is All You Need", "arxiv_version": "7",
    "pdf_page_number": 3, "source_filename": "1706.03762v7.pdf", "score": 0.7,
}


def response(claims=None, refusal=False):
    return json.dumps({"claims": claims or [], "refusal": refusal})


def test_supported_question_is_retrieved_drafted_verified_and_cited():
    fake_model = FakeListChatModel(responses=[
        response([{"text": "The Transformer uses self-attention and feed-forward layers.", "citation_ids": ["c1"]}]),
        json.dumps({"supported": True, "issues": ""}),
    ])
    retrieval = FakeRetrieval([PASSAGE])
    graph = compile_workflow(retrieval, fake_model, {"1706.03762": 15})
    state = graph.invoke({"request_id": "test", "user_query": "What does the Transformer use for attention?"})
    assert state["final_status"] == "ok"
    assert "PDF p. 3" in state["final_response"]
    assert len(state["tool_calls"]) == 1
    assert len(state["call_usage"]) == 2


def test_unsupported_question_refines_once_then_refuses_without_model_call():
    fake_model = FakeListChatModel(responses=[])
    retrieval = FakeRetrieval([])
    graph = compile_workflow(retrieval, fake_model, {"1706.03762": 15})
    state = graph.invoke({"request_id": "test", "user_query": "What is the weather on Mars today?"})
    assert state["final_status"] == "refused_correctly"
    assert len(retrieval.calls) == 2
    assert len(state["tool_calls"]) == 2


def test_failed_verification_allows_one_revision_then_rechecks():
    fake_model = FakeListChatModel(responses=[
        response([{"text": "Possibly unsupported initial claim.", "citation_ids": ["c1"]}]),
        json.dumps({"supported": False, "issues": "Remove the unsupported detail."}),
        response([{"text": "The Transformer uses self-attention.", "citation_ids": ["c1"]}]),
        json.dumps({"supported": True, "issues": ""}),
    ])
    graph = compile_workflow(FakeRetrieval([PASSAGE]), fake_model, {"1706.03762": 15})
    state = graph.invoke({"request_id": "test", "user_query": "What does the Transformer use for attention?"})
    assert state["final_status"] == "ok"
    assert state["revision_attempts"] == 1
    assert len(state["call_usage"]) == 4
