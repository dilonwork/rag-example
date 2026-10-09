"""Stage 5 — Retrieval: find the chunks most relevant to a question."""
from __future__ import annotations

from .embeddings import Embedder
from .vector_store import InMemoryVectorStore, ScoredChunk


class Retriever:
    def __init__(
        self, embedder: Embedder, store: InMemoryVectorStore, top_k: int = 4
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[ScoredChunk]:
        if not question.strip():
            raise ValueError("question must not be empty")
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, self.top_k)
