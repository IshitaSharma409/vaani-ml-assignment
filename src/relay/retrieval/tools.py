"""Explicit LangChain tool interface for the fixed paper library."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from relay.retrieval.index import RetrievalService


class SearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    retrieval_setting: str = Field(default="medium")
    top_k: int | None = Field(default=None, ge=1, le=10)


def make_search_tool(service: RetrievalService):
    """Bind an explicit search tool to a validated, local-only retrieval service."""
    @tool("search_research_library", args_schema=SearchInput)
    def search_research_library(query: str, retrieval_setting: str = "medium", top_k: int | None = None) -> dict[str, Any]:
        """Search only the three assignment papers and return page-grounded passages."""
        started = time.perf_counter()
        passages, timings = service.search(query, retrieval_setting, top_k)
        return {"passages": passages, "timings": timings, "tool_seconds": time.perf_counter() - started}

    return search_research_library
