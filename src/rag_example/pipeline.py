"""The full RAG pipeline.

Stages:
  1. ingestion      (ingestion.py)      load .md/.txt documents
  2. chunking       (chunking.py)       split into overlapping chunks
  3. embeddings     (embeddings.py)     text -> dense vectors
  4. vector storage (vector_store.py)   persist + cosine search
  5. retrieval      (retrieval.py)      top-k chunks for a question
  6. prompting      (prompting.py)      context -> LLM prompt
  7. llm            (llm.py)            generate the answer
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk, Chunker, FixedSizeChunker
from .config import Settings
from .embeddings import Embedder, SentenceTransformerEmbedder
from .ingestion import load_documents
from .llm import LLMClient, build_llm_client
from .prompting import build_rag_prompt
from .retrieval import Retriever
from .vector_store import InMemoryVectorStore, ScoredChunk


@dataclass
class Answer:
    question: str
    text: str
    sources: list[ScoredChunk]


class RAGPipeline:
    def __init__(
        self, settings: Settings | None = None, chunker: Chunker | None = None
    ) -> None:
        self.settings = settings or Settings()
        self.settings.validate()
        self.chunker: Chunker = chunker or FixedSizeChunker(
            self.settings.chunk_size, self.settings.chunk_overlap
        )
        self.embedder: Embedder = SentenceTransformerEmbedder(self.settings.embedding_model)
        self.store = InMemoryVectorStore()
        self.retriever = Retriever(self.embedder, self.store, top_k=self.settings.top_k)
        self.llm: LLMClient = build_llm_client(
            self.settings.llm_provider,
            model=self.settings.llm_model,
            ollama_host=self.settings.ollama_host,
            openai_api_key=self.settings.openai_api_key,
            openai_base_url=self.settings.openai_base_url,
        )
        self._chunks: list[Chunk] = []

    def ingest(self) -> list[Chunk]:
        """Run stages 1-4: load docs, chunk, embed, store."""
        documents = load_documents(self.settings.docs_dir)
        chunks = self.chunker.chunk(documents)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self._chunks = chunks
        return chunks

    def retrieve(self, question: str) -> list[ScoredChunk]:
        """Run stage 5 only — useful when no LLM is available."""
        return self.retriever.retrieve(question)

    def ask(self, question: str) -> Answer:
        """Run stages 5-7: retrieve, prompt, generate."""
        hits = self.retrieve(question)
        system_prompt, user_prompt = build_rag_prompt(question, hits)
        text = self.llm.generate(system_prompt, user_prompt)
        return Answer(question=question, text=text, sources=hits)
