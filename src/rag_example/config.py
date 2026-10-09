"""Runtime configuration, driven by environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    docs_dir: str = os.environ.get("RAG_DOCS_DIR", "data/docs")
    chunk_size: int = int(os.environ.get("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.environ.get("RAG_CHUNK_OVERLAP", "120"))
    embedding_model: str = os.environ.get("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    top_k: int = int(os.environ.get("RAG_TOP_K", "4"))
    # llm provider: "ollama" (local, default) or "openai" (any OpenAI-compatible API)
    llm_provider: str = os.environ.get("RAG_LLM_PROVIDER", "ollama")
    llm_model: str = os.environ.get("RAG_LLM_MODEL", "llama3.1")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
