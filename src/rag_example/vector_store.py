"""Stage 4 — Vector storage: persist vectors and search them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    """Minimal in-memory store with cosine-similarity search.

    Vectors are L2-normalized on the way in, so cosine similarity reduces to
    a dot product. Good enough for an example — swap this class for
    Chroma / pgvector / Qdrant in a real deployment; the interface stays the same.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        self._chunks.extend(chunks)
        self._vectors = (
            vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if self._vectors is None or not self._chunks:
            raise ValueError("vector store is empty — ingest documents first")
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._vectors @ q
        top = np.argsort(scores)[::-1][:top_k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._chunks)
