"""LangChain ChatOpenAI adapter configured only for the local simulator URL."""

from __future__ import annotations

from langchain_openai import ChatOpenAI


class StandInBackend:
    def __init__(self, base_url: str | None = None, max_tokens: int = 512):
        self.base_url = base_url or "http://127.0.0.1:8001/v1"
        self.max_tokens = max_tokens

    def create_model(self) -> ChatOpenAI:
        """Return a LangChain client that targets loopback only."""
        return ChatOpenAI(
            model="relay-standin-v1",
            base_url=self.base_url,
            api_key="local-simulator-no-secret",
            temperature=0,
            max_tokens=self.max_tokens,
            max_retries=0,
            timeout=30,
            streaming=True,
        )
