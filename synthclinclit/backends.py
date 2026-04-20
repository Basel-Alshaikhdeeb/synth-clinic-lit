"""Pluggable LLM backends for artefact extraction.

Two backends are bundled:
  - AnthropicBackend: hosted Claude models via the Anthropic API.
  - OllamaBackend: any locally-hosted model served by Ollama (free, runs on your machine).

Both implement the same `LLMBackend` protocol so the extractor is backend-agnostic.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


class LLMBackend(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, json_schema: dict[str, Any]) -> str:
        """Return the raw model output text for a single prompt."""
        ...


@dataclass
class AnthropicBackend:
    """Hosted Claude via the Anthropic API."""
    model: str = "claude-opus-4-7"
    max_tokens: int = 4096
    api_key: str | None = None
    name: str = "anthropic"

    def __post_init__(self) -> None:
        from anthropic import Anthropic  # lazy import
        self._client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, json_schema: dict[str, Any]) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()


@dataclass
class OllamaBackend:
    """Local models via Ollama (https://ollama.com).

    Ollama exposes a REST API at http://localhost:11434. Recent versions accept
    a JSON schema in the `format` parameter, which forces the model to produce
    output that matches the schema — this is much more reliable than asking a
    small model to emit JSON via the prompt alone.
    """
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    num_ctx: int | None = 8192       # context window; raise for long articles
    temperature: float = 0.0
    timeout: float = 600.0
    name: str = "ollama"

    def __post_init__(self) -> None:
        import httpx  # lazy import
        self._client = httpx.Client(base_url=self.host, timeout=self.timeout)

    def complete(self, system: str, user: str, json_schema: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Ollama accepts a JSON schema here to constrain decoding.
            # Older servers (<0.5) only support the literal string "json";
            # we fall back to that if the schema call fails.
            "format": json_schema,
            "options": {"temperature": self.temperature},
        }
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx

        r = self._client.post("/api/chat", json=payload)
        if r.status_code == 400 and "format" in r.text.lower():
            payload["format"] = "json"
            r = self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()

    def close(self) -> None:
        self._client.close()


def build_backend(
    backend: str = "anthropic",
    model: str | None = None,
    **kwargs: Any,
) -> LLMBackend:
    """Construct a backend by name. Unknown kwargs are forwarded to the backend."""
    backend = backend.lower()
    if backend == "anthropic":
        return AnthropicBackend(model=model or "claude-opus-4-7", **kwargs)
    if backend == "ollama":
        return OllamaBackend(model=model or "llama3.1:8b", **kwargs)
    raise ValueError(f"Unknown backend: {backend!r}. Use 'anthropic' or 'ollama'.")
