"""Stage 6 — Prompting: assemble retrieved context into an LLM prompt."""
from __future__ import annotations

from .vector_store import ScoredChunk


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context below. "
    "Cite the sources you used with bracketed numbers like [1], [2]. "
    "If the context does not contain the answer, say so instead of guessing."
)


def build_rag_prompt(question: str, hits: list[ScoredChunk]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the LLM."""
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        context_blocks.append(f"[{i}] ({hit.chunk.title})\n{hit.chunk.text}")
    context = "\n\n".join(context_blocks)
    user_prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer concisely and cite sources with bracketed numbers."
    )
    return SYSTEM_PROMPT, user_prompt
