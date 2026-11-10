"""Single-purpose LangGraph nodes and output contracts."""

from __future__ import annotations

import json
import hashlib
import re
import time
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from relay.graph.state import RelayState
from relay.retrieval.citations import CitationError, render_claims


ANSWER_HUMAN_TEMPLATE = "Question: {question}\nRetrieved excerpts (JSON): {passages}"
REVIEW_HUMAN_TEMPLATE = "Question: {question}\nClaims: {claims}\nExcerpts: {passages}"
PROMPT_SPECS = {
    "relay-rag-v1": {
        "answer_system": "You answer solely from the supplied excerpts in a fixed three-paper research library. Do not use outside knowledge. If the excerpts do not support an answer, return refusal=true and no claims. Otherwise make concise independently verifiable claims. Every claim must cite one or more exact retrieved chunk_id values. Never invent page numbers, sources, or identifiers. Return only the required JSON.\n{format_instructions}",
        "review_system": "Check only whether each draft claim is entailed by its cited supplied excerpts and citation IDs exist. Do not add outside facts. Return JSON.\n{format_instructions}",
        "answer_human": ANSWER_HUMAN_TEMPLATE,
        "review_human": REVIEW_HUMAN_TEMPLATE,
    },
    "relay-rag-v2": {
        "answer_system": "You answer solely from the supplied excerpts in a fixed three-paper research library. Do not use outside knowledge. If the excerpts do not support an answer, return refusal=true and no claims. Otherwise provide at most two concise, independently verifiable claims (no more than 35 words each). Every claim must cite one or more exact retrieved chunk_id values. Never invent page numbers, sources, or identifiers. Return only the required JSON.\n{format_instructions}",
        "review_system": "Check whether each draft claim is entailed by its cited supplied excerpts and citation IDs exist. Do not add facts. Keep any issue summary under 20 words. Return only JSON.\n{format_instructions}",
        "answer_human": ANSWER_HUMAN_TEMPLATE,
        "review_human": REVIEW_HUMAN_TEMPLATE,
    },
}
PROMPT_VERSION = "relay-rag-v2"
PROMPT_HASHES = {version: hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
                 for version, spec in PROMPT_SPECS.items()}
PROMPT_HASH = PROMPT_HASHES[PROMPT_VERSION]
STOPWORDS = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "did", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "with", "by", "from", "about", "paper", "papers", "this", "that", "it", "they", "their", "there", "can", "could", "explain", "describe", "tell", "me"}


class Claim(BaseModel):
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class Draft(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    refusal: bool = False


class Review(BaseModel):
    supported: bool
    issues: str = ""


def _track_usage(state: RelayState, response: Any, started: float, label: str) -> list[dict[str, Any]]:
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = getattr(response, "usage_metadata", None) or metadata.get("token_usage")
    if usage and "input_tokens" in usage:
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    else:
        input_tokens = (usage or {}).get("prompt_tokens")
        output_tokens = (usage or {}).get("completion_tokens")
    return state.get("call_usage", []) + [{
        "stage": label,
        "elapsed_seconds": time.perf_counter() - started,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usage_source": "provider" if usage else "unavailable",
    }]


def _safe_error(exc: Exception) -> str:
    """Keep useful exception context while redacting common credential patterns."""
    detail = str(exc).replace("\n", " ")[:500]
    detail = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", detail)
    detail = re.sub(r"org_[A-Za-z0-9_-]+", "[ORG_ID]", detail)
    return f"{type(exc).__name__}: {detail}"


def create_nodes(search_tool: Any, model: Any, page_counts: dict[str, int]) -> dict[str, Any]:
    """Create node callables with explicit, testable service dependencies."""
    answer_parser = PydanticOutputParser(pydantic_object=Draft)
    review_parser = PydanticOutputParser(pydantic_object=Review)
    prompt_spec = PROMPT_SPECS[PROMPT_VERSION]
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_spec["answer_system"]),
        ("human", prompt_spec["answer_human"]),
    ]).partial(format_instructions=answer_parser.get_format_instructions())
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_spec["review_system"]),
        ("human", prompt_spec["review_human"]),
    ]).partial(format_instructions=review_parser.get_format_instructions())

    def prepare_request(state: RelayState) -> RelayState:
        query = " ".join(state["user_query"].split())
        if not query:
            raise ValueError("Question cannot be empty")
        return {"normalized_query": query, "retrieval_attempts": 0, "revision_attempts": 0,
                "tool_calls": [], "call_usage": [], "stage_timings": {}}

    def retrieve_passages(state: RelayState) -> RelayState:
        query = state.get("normalized_query", state["user_query"])
        started = time.perf_counter()
        result = search_tool.invoke({"query": query, "retrieval_setting": state.get("retrieval_setting", "medium_reranked")})
        timings = dict(state.get("stage_timings", {}))
        timings["retrieval_seconds"] = timings.get("retrieval_seconds", 0.0) + result["tool_seconds"]
        timings["vector_search_seconds"] = timings.get("vector_search_seconds", 0.0) + result["timings"]["vector_search_seconds"]
        timings["rerank_seconds"] = timings.get("rerank_seconds", 0.0) + result["timings"]["rerank_seconds"]
        tool_calls = state.get("tool_calls", []) + [{"name": "search_research_library", "query": query, "elapsed_seconds": time.perf_counter() - started, "passage_count": len(result["passages"])}]
        return {"retrieved_passages": result["passages"], "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
                "tool_calls": tool_calls, "stage_timings": timings}

    def check_evidence(state: RelayState) -> RelayState:
        terms = {term for term in re.findall(r"[a-z0-9]+", state["normalized_query"].lower()) if term not in STOPWORDS and len(term) > 2}
        text = " ".join(p["chunk_text"] for p in state.get("retrieved_passages", [])).lower()
        matched = sum(term in text for term in terms)
        # Conservative lexical gate prevents obvious out-of-library prompts from reaching generation.
        sufficient = bool(state.get("retrieved_passages")) and bool(terms) and matched / len(terms) >= 0.2
        return {"evidence_sufficient": sufficient}

    def refine_query(state: RelayState) -> RelayState:
        # Deterministic, bounded normalization only: no extra model call or fabricated subject matter.
        source = state["normalized_query"]
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", source)
        focused = " ".join(token for token in tokens if token.lower() not in STOPWORDS)
        return {"normalized_query": focused or source}

    def draft_answer(state: RelayState) -> RelayState:
        started = time.perf_counter()
        messages = answer_prompt.invoke({"question": state["user_query"], "passages": json.dumps(state["retrieved_passages"], ensure_ascii=False)})
        try:
            response = model.invoke(messages)
        except Exception as exc:
            return {"final_response": "The assistant could not complete a verified answer. Please try again later.",
                    "final_status": "error", "failure_stage": "answer_model_call", "failure_detail": _safe_error(exc),
                    "refusal_reason": f"Answer generation failed ({type(exc).__name__}).",
                    "call_usage": state.get("call_usage", []) + [{"stage": "answer", "elapsed_seconds": time.perf_counter() - started, "input_tokens": None, "output_tokens": None, "usage_source": "unavailable", "outcome": "error"}]}
        usage = _track_usage(state, response, started, "answer")
        try:
            draft = answer_parser.parse(response.content if isinstance(response.content, str) else json.dumps(response.content))
        except Exception as exc:
            return {"final_response": "The assistant could not complete a verified answer. Please try again later.",
                    "final_status": "error", "failure_stage": "answer_parse", "failure_detail": _safe_error(exc),
                    "refusal_reason": f"Answer output could not be parsed ({type(exc).__name__}).", "call_usage": usage}
        if draft.refusal:
            return {"draft_claims": [], "draft_answer": "", "refusal_reason": "The retrieved papers do not support an answer.", "call_usage": usage}
        if not draft.claims:
            return {"final_response": "The assistant could not complete a verified answer. Please try again later.",
                    "final_status": "error", "refusal_reason": "The model returned no claims."}
        claims = [claim.model_dump() for claim in draft.claims]
        return {"draft_claims": claims, "draft_answer": render_claims(claims, state["retrieved_passages"], page_counts), "call_usage": usage}

    def verify_answer(state: RelayState) -> RelayState:
        try:
            # Citation validity is deterministic and precedes the model's support review.
            for claim in state.get("draft_claims", []):
                from relay.retrieval.citations import validate_citation_ids
                validate_citation_ids(claim["citation_ids"], state["retrieved_passages"], page_counts)
        except (CitationError, KeyError) as exc:
            return {"verification_passed": False, "verification_issues": str(exc)}
        started = time.perf_counter()
        messages = review_prompt.invoke({
            "question": state["user_query"], "claims": json.dumps(state["draft_claims"], ensure_ascii=False),
            "passages": json.dumps(state["retrieved_passages"], ensure_ascii=False),
        })
        try:
            response = model.invoke(messages)
        except Exception as exc:
            return {"verification_passed": False, "verification_issues": f"Verification unavailable ({type(exc).__name__}).",
                    "final_response": "The assistant could not verify this answer because the model service is temporarily unavailable. Please try again later.",
                    "final_status": "error", "failure_stage": "verification", "failure_detail": _safe_error(exc),
                    "call_usage": state.get("call_usage", []) + [{"stage": "verification", "elapsed_seconds": time.perf_counter() - started,
                                                                    "input_tokens": None, "output_tokens": None,
                                                                    "usage_source": "unavailable", "outcome": "error"}]}
        usage = _track_usage(state, response, started, "verification")
        try:
            review = review_parser.parse(response.content if isinstance(response.content, str) else json.dumps(response.content))
        except Exception as exc:
            return {"verification_passed": False, "verification_issues": f"Verification output invalid ({type(exc).__name__}).",
                    "final_response": "The assistant could not verify this answer because the verification response was invalid. Please try again later.",
                    "final_status": "error", "failure_stage": "verification_parse", "failure_detail": _safe_error(exc),
                    "call_usage": usage}
        return {"verification_passed": review.supported, "verification_issues": review.issues,
                "call_usage": usage}

    def revise_answer(state: RelayState) -> RelayState:
        started = time.perf_counter()
        correction_prompt = answer_prompt.invoke({
            "question": state["user_query"] + "\nCorrect the draft to address this verification issue: " + state.get("verification_issues", "unsupported statement"),
            "passages": json.dumps(state["retrieved_passages"], ensure_ascii=False),
        })
        try:
            response = model.invoke(correction_prompt)
        except Exception as exc:
            return {"revision_attempts": 1,
                    "final_response": "The assistant could not complete a verified correction because the model service is temporarily unavailable. Please try again later.",
                    "final_status": "error", "refusal_reason": f"The correction could not be validated ({type(exc).__name__}).",
                    "failure_stage": "revision", "failure_detail": _safe_error(exc),
                    "call_usage": state.get("call_usage", []) + [{"stage": "revision", "elapsed_seconds": time.perf_counter() - started,
                                                                    "input_tokens": None, "output_tokens": None,
                                                                    "usage_source": "unavailable", "outcome": "error"}]}
        usage = _track_usage(state, response, started, "revision")
        try:
            draft = answer_parser.parse(response.content if isinstance(response.content, str) else json.dumps(response.content))
            if draft.refusal or not draft.claims:
                return {"revision_attempts": 1, "refusal_reason": "The correction was not supportable from retrieved evidence.",
                        "call_usage": usage}
            claims = [claim.model_dump() for claim in draft.claims]
            answer = render_claims(claims, state["retrieved_passages"], page_counts)
            return {"revision_attempts": 1, "draft_claims": claims, "draft_answer": answer, "call_usage": usage}
        except Exception as exc:
            return {"revision_attempts": 1,
                    "final_response": "The assistant could not validate the correction returned by the model. Please try again later.",
                    "final_status": "error", "failure_stage": "revision_parse", "failure_detail": _safe_error(exc),
                    "call_usage": usage}

    def refuse_safely(state: RelayState) -> RelayState:
        reason = state.get("refusal_reason", "The fixed paper library does not contain sufficient evidence to answer this question.")
        return {"final_response": f"I can’t answer this from the three-paper library. {reason}", "final_status": "refused_correctly"}

    def finalize_response(state: RelayState) -> RelayState:
        return {"final_response": state["draft_answer"], "final_status": "ok"}

    return locals()
