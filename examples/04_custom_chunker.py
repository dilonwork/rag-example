"""Example 4: swap the chunking strategy without touching the pipeline.

Implements a naive sentence-based chunker (N sentences per chunk) and
injects it into RAGPipeline via the `chunker=` argument. Anything with a
`.chunk(documents)` method works — see the Chunker protocol in
src/rag_example/chunking.py.

Run from the repo root:
    python examples/04_custom_chunker.py
"""
from __future__ import annotations

import re

from src.rag_example.chunking import Chunk
from src.rag_example.config import Settings
from src.rag_example.ingestion import Document
from src.rag_example.pipeline import RAGPipeline


class SentenceChunker:
    """One chunk per N sentences. Dumb but transparent."""

    def __init__(self, sentences_per_chunk: int = 2) -> None:
        self.sentences_per_chunk = sentences_per_chunk

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", doc.text)
                if s.strip()
            ]
            groups = [
                sentences[i : i + self.sentences_per_chunk]
                for i in range(0, len(sentences), self.sentences_per_chunk)
            ]
            for index, group in enumerate(groups):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#s{index}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=" ".join(group),
                        source=doc.source,
                        index=index,
                    )
                )
        return chunks


def main() -> None:
    pipeline = RAGPipeline(Settings(), chunker=SentenceChunker(sentences_per_chunk=2))
    n_chunks = len(pipeline.ingest())
    print(f"indexed {n_chunks} sentence-chunks\n")

    question = "What is a vector database?"
    print(f"Q: {question}\n")
    for hit in pipeline.retrieve(question):
        print(f"[{hit.score:.3f}] {hit.chunk.chunk_id}: {hit.chunk.text[:220]}")
        print()


if __name__ == "__main__":
    main()
