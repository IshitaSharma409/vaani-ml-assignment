"""Configured ChatGroq construction with SDK retries disabled."""

from __future__ import annotations

from typing import Any

from relay.config import GroqSettings


class GroqBackend:
    def __init__(self, timeout_seconds: float = 30, max_tokens: int = 512):
        self.timeout_seconds, self.max_tokens = timeout_seconds, max_tokens

    def create_model(self, http_client: Any | None = None):
        settings = GroqSettings.from_environment()
        return settings.create_chat_model(http_client=http_client, max_tokens=self.max_tokens,
                                          timeout_seconds=self.timeout_seconds)
