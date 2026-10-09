"""Demo CLI: ingest the sample docs, then ask questions.

Examples:
    python demo.py --ask "What is chunking?"
    python demo.py --retrieve-only --ask "What is RAG?"
    python demo.py --interactive
"""
from __future__ import annotations

import argparse

from src.rag_example.config import Settings
from src.rag_example.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal modular RAG pipeline demo.")
    parser.add_argument("--ask", help="answer a single question")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="skip the LLM and just show the retrieved chunks",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="ask questions in a loop"
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(Settings())
    chunks = pipeline.ingest()
    print(f"ingested {len(chunks)} chunks into the vector store")

    def handle(question: str) -> None:
        if args.retrieve_only:
            for hit in pipeline.retrieve(question):
                print(f"\n[{hit.score:.3f}] {hit.chunk.title} ({hit.chunk.chunk_id})")
                print(hit.chunk.text[:400])
        else:
            answer = pipeline.ask(question)
            print(f"\nQ: {answer.question}\n\nA: {answer.text}\n")
            print("Sources:")
            for i, hit in enumerate(answer.sources, start=1):
                print(f"  [{i}] {hit.chunk.title} — {hit.chunk.source}")

    if args.ask:
        handle(args.ask)
    elif args.interactive:
        while True:
            try:
                question = input("\nask> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"q", "quit", "exit"}:
                break
            if question:
                handle(question)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
