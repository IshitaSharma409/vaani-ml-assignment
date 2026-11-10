"""Compile the bounded evidence retrieval and answer-verification graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from relay.graph.nodes import create_nodes
from relay.graph.state import RelayState
from relay.retrieval.tools import make_search_tool


def corpus_page_counts() -> dict[str, int]:
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "data/papers/manifest.json").read_text(encoding="utf-8"))
    return {paper["arxiv_id"]: int(paper["page_count"]) for paper in manifest["papers"]}


def compile_workflow(retrieval_service: Any, model: Any, page_counts: dict[str, int] | None = None):
    """Compile a graph with explicit tool invocation and hard-bounded recovery loops."""
    search_tool = make_search_tool(retrieval_service)
    nodes = create_nodes(search_tool, model, page_counts or corpus_page_counts())
    graph = StateGraph(RelayState)
    for name in ("prepare_request", "retrieve_passages", "check_evidence", "refine_query", "draft_answer", "verify_answer", "revise_answer", "refuse_safely", "finalize_response"):
        graph.add_node(name, nodes[name])
    graph.add_edge(START, "prepare_request")
    graph.add_edge("prepare_request", "retrieve_passages")
    graph.add_edge("retrieve_passages", "check_evidence")

    def evidence_route(state: RelayState) -> str:
        if state.get("evidence_sufficient"):
            return "draft_answer"
        if state.get("retrieval_attempts", 0) < 2:
            return "refine_query"
        return "refuse_safely"

    graph.add_conditional_edges("check_evidence", evidence_route,
                                {"draft_answer": "draft_answer", "refine_query": "refine_query", "refuse_safely": "refuse_safely"})
    graph.add_edge("refine_query", "retrieve_passages")

    def draft_route(state: RelayState) -> str:
        if state.get("final_status") == "error":
            return "end"
        if state.get("refusal_reason"):
            return "refuse_safely"
        return "verify_answer"

    graph.add_conditional_edges("draft_answer", draft_route,
                                {"verify_answer": "verify_answer", "refuse_safely": "refuse_safely", "end": END})

    def verification_route(state: RelayState) -> str:
        if state.get("final_status") == "error":
            return "end"
        if state.get("verification_passed"):
            return "finalize_response"
        if state.get("revision_attempts", 0) < 1:
            return "revise_answer"
        return "refuse_safely"

    graph.add_conditional_edges("verify_answer", verification_route,
                                {"finalize_response": "finalize_response", "revise_answer": "revise_answer", "refuse_safely": "refuse_safely", "end": END})
    def revision_route(state: RelayState) -> str:
        if state.get("final_status") == "error":
            return "end"
        if state.get("refusal_reason"):
            return "refuse_safely"
        return "verify_answer"

    graph.add_conditional_edges("revise_answer", revision_route,
                                {"end": END, "refuse_safely": "refuse_safely", "verify_answer": "verify_answer"})
    graph.add_edge("finalize_response", END)
    graph.add_edge("refuse_safely", END)
    return graph.compile()
