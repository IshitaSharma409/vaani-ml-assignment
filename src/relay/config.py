"""Environment-backed settings for real Groq inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr


@dataclass(frozen=True)
class GroqSettings:
    """Validated settings for one Groq model; never expose the API key in repr."""

    api_key: SecretStr = field(repr=False)
    model: str
    timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "GroqSettings":
        """Load the project .env if present and require key/model for a real run."""
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model = os.getenv("GROQ_MODEL", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Set it in the local .env or environment "
                "before selecting the real Groq backend."
            )
        if not model:
            raise RuntimeError(
                "GROQ_MODEL is missing. Set the exact model ID available to your "
                "Groq account before selecting the real Groq backend."
            )
        return cls(api_key=SecretStr(api_key), model=model)

    def create_chat_model(self, http_client: Any | None = None, *, max_tokens: int = 512,
                          timeout_seconds: float | None = None) -> ChatGroq:
        """Create a ChatGroq client with hidden SDK retries disabled.

        Application-level retry logic will own and count retry attempts.
        """
        client_options = {"http_client": http_client} if http_client is not None else {}
        return ChatGroq(
            model=self.model,
            api_key=self.api_key.get_secret_value(),
            temperature=0,
            timeout=timeout_seconds or self.timeout_seconds,
            max_tokens=max_tokens,
            max_retries=0,
            streaming=True,
            # The installed Groq SDK accepts undocumented API body fields via
            # extra_body; passing stream_options as a direct kwarg is rejected.
            model_kwargs={
                "extra_body": {"stream_options": {"include_usage": True}}
            },
            **client_options,
        )
