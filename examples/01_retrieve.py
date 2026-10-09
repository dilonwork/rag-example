"""Example 1: pull content OUT of the RAG pipeline.

Ingests the sample docs, then shows exactly what retrieval returns
for a question: ranked chunks, similarity scores, and where each
chunk came from. No LLM needed.

Run from the repo root:
    python examples/01_retrieve.py
"""
from __future__ import annotations

from src.rag_example.config import Settings
from src.rag_example.pipeline import RAGPipeline

QUESTION = "How should I choose a chunk size?"


def main() -> None:
    pipeline = RAGPipeline(Settings())
    n_chunks = len(pipeline.ingest())
    print(f"indexed {n_chunks} chunks\n")

    hits = pipeline.retrieve(QUESTION)
    print(f"Q: {QUESTION}\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"--- rank {rank} | similarity {hit.score:.3f} ---")
        print(f"from: {hit.chunk.title} ({hit.chunk.source}, chunk {hit.chunk.index})")
        print(hit.chunk.text)
        print()


if __name__ == "__main__":
    main()
