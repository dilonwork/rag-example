"""Example 2: full question answering with cited sources.

Runs the whole pipeline — retrieve, build prompt, generate — and prints
the answer together with the sources it was built from.

Requires an LLM: local Ollama by default
(`ollama serve` + `ollama pull llama3.1`), or set
RAG_LLM_PROVIDER=openai with OPENAI_API_KEY.

Run from the repo root:
    python examples/02_answer.py
"""
from __future__ import annotations

from src.rag_example.config import Settings
from src.rag_example.pipeline import RAGPipeline

QUESTION = "What are the failure modes of RAG and how do I fix them?"


def main() -> None:
    pipeline = RAGPipeline(Settings())
    print(f"indexed {len(pipeline.ingest())} chunks\n")

    answer = pipeline.ask(QUESTION)
    print(f"Q: {answer.question}\n")
    print(f"A: {answer.text}\n")
    print("Sources used:")
    for i, hit in enumerate(answer.sources, start=1):
        print(f"  [{i}] {hit.chunk.title} — {hit.chunk.source} (score {hit.score:.3f})")


if __name__ == "__main__":
    main()
