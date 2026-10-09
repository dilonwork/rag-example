"""Stage 3 — Embeddings: turn text into dense vectors."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (no API key needed).

    The model downloads on first use (~90 MB for all-MiniLM-L6-v2)
    and is cached under ~/.cache afterwards.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)
