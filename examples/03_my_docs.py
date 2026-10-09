"""Example 3: index YOUR OWN documents instead of the samples.

Point RAG_DOCS_DIR at any folder of .md/.txt files, then ask away.
Retrieval only — no LLM needed.

Run from the repo root:
    RAG_DOCS_DIR=~/my-notes python examples/03_my_docs.py --ask "What did I write about caching?"
"""
from __future__ import annotations

import argparse

from src.rag_example.config import Settings
from src.rag_example.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over your own documents.")
    parser.add_argument("--ask", required=True, help="question to answer")
    parser.add_argument("--top-k", type=int, default=4, help="chunks to retrieve")
    args = parser.parse_args()

    settings = Settings()
    settings.top_k = args.top_k
    pipeline = RAGPipeline(settings)

    n_chunks = len(pipeline.ingest())
    print(f"indexed {n_chunks} chunks from {settings.docs_dir}\n")

    print(f"Q: {args.ask}\n")
    for hit in pipeline.retrieve(args.ask):
        print(f"[{hit.score:.3f}] {hit.chunk.title} ({hit.chunk.source})")
        print(hit.chunk.text[:350])
        print()


if __name__ == "__main__":
    main()
