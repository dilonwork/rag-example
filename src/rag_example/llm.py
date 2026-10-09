"""Stage 7 — LLM integration: generate the final answer."""
from __future__ import annotations

from typing import Protocol

import requests


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OllamaClient:
    """Chat against a local Ollama server (default http://localhost:11434)."""

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=120,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"cannot reach Ollama at {self.host} — is `ollama serve` running? "
                "Set RAG_LLM_PROVIDER=openai with OPENAI_API_KEY to use OpenAI instead."
            ) from exc
        return resp.json()["message"]["content"].strip()


class OpenAICompatibleClient:
    """Chat against any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai provider")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def build_llm_client(
    provider: str,
    *,
    model: str,
    ollama_host: str,
    openai_api_key: str,
    openai_base_url: str,
) -> LLMClient:
    provider = provider.lower()
    if provider == "ollama":
        return OllamaClient(model=model or "llama3.1", host=ollama_host)
    if provider == "openai":
        return OpenAICompatibleClient(
            model=model or "gpt-4o-mini", api_key=openai_api_key, base_url=openai_base_url
        )
    raise ValueError(f"unknown RAG_LLM_PROVIDER: {provider!r} (expected 'ollama' or 'openai')")
