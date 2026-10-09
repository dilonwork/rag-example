"""Stage 2 — Chunking: split documents into overlapping text chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .ingestion import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    index: int


class Chunker(Protocol):
    """Anything with a .chunk(documents) method can be plugged into the pipeline."""

    def chunk(self, documents: list[Document]) -> list[Chunk]: ...


class FixedSizeChunker:
    """Fixed-size chunker with character overlap.

    Splits on character count (not tokens) to keep the example dependency-free,
    and prefers whitespace boundaries so words are not cut in half.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for index, text in enumerate(self._split(doc.text)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{index}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=text,
                        source=doc.source,
                        index=index,
                    )
                )
        return chunks

    def _split(self, text: str) -> list[str]:
        parts: list[str] = []
        start = 0
        size, overlap = self.chunk_size, self.chunk_overlap
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + size // 2:
                    end = boundary
            parts.append(text[start:end].strip())
            if end >= len(text):
                break
            start = end - overlap
        return [p for p in parts if p]
