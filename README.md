# rag-example

A minimal, modular **Retrieval-Augmented Generation (RAG)** pipeline you can read in one sitting.
Each stage of the pipeline is its own module with a tiny interface, so you can swap any piece
without touching the rest.

```
                  ┌─────────────┐
                  │  data/docs  │  *.md / *.txt
                  └──────┬──────┘
                         │ 1. ingestion      (ingestion.py)
                         ▼
                  ┌─────────────┐
                  │   chunks    │  overlapping text windows
                  └──────┬──────┘
                         │ 2. chunking       (chunking.py)
                         ▼
                  ┌─────────────┐
                  │   vectors   │  sentence-transformers, local
                  └──────┬──────┘
                         │ 3. embeddings     (embeddings.py)
                         ▼
                  ┌─────────────┐
                  │ vector store│  numpy, cosine similarity
                  └──────┬──────┘
                         │ 4. vector storage (vector_store.py)
                         ▼
   question ──►  ┌─────────────┐
                 │  retrieval  │  top-k chunks + scores
                 └──────┬──────┘
                        │ 5. retrieval      (retrieval.py)
                        ▼
                 ┌─────────────┐
                 │  prompting  │  context + citations -> prompt
                 └──────┬──────┘
                        │ 6. prompting      (prompting.py)
                        ▼
                 ┌─────────────┐
                 │     LLM     │  Ollama (local) or OpenAI-compatible
                 └─────────────┘
                        7. llm              (llm.py)

                 pipeline.py wires it all together.
```

## Quickstart

```bash
pip install -r requirements.txt

# Retrieval only (no LLM needed):
python demo.py --retrieve-only --ask "What is chunking?"

# Full RAG answer (needs Ollama running locally):
ollama serve   # in another terminal, then: ollama pull llama3.1
python demo.py --ask "What is chunking?"

# Interactive mode:
python demo.py --interactive
```

## Examples

Four runnable scripts under `examples/` (run from the repo root):

| Script | What it shows | LLM needed? |
|---|---|---|
| `01_retrieve.py` | Pull content out of RAG: ranked chunks, similarity scores, source metadata | No |
| `02_answer.py` | Full Q&A with cited sources | Yes |
| `03_my_docs.py` | Index your own folder: `RAG_DOCS_DIR=~/my-notes python examples/03_my_docs.py --ask "..."` | No |
| `04_custom_chunker.py` | Plug in your own chunking strategy via `RAGPipeline(settings, chunker=...)` | No |
| `05_shop_assistant.py` | RAG in a sales website: customer-service Q&A, semantic product search, policy Q&A over a mock store (`data/shop/`) | No |

A step-by-step Chinese tutorial covering all four examples — what each one implements
and what output to expect — is in [doc/rag-examples-tutorial.pdf](doc/rag-examples-tutorial.pdf),
with a landing page at [doc/index.html](doc/index.html).

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DOCS_DIR` | `data/docs` | where to load documents from |
| `RAG_CHUNK_SIZE` | `800` | max characters per chunk |
| `RAG_CHUNK_OVERLAP` | `120` | overlapping characters between chunks |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `RAG_TOP_K` | `4` | chunks retrieved per question |
| `RAG_LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `RAG_LLM_MODEL` | `llama3.1` | chat model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | — | for the `openai` provider |

Example with OpenAI:

```bash
RAG_LLM_PROVIDER=openai RAG_LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... \
  python demo.py --ask "What is a vector database?"
```

## Swapping components

- **Embeddings**: implement the `Embedder` protocol (`embed(texts) -> np.ndarray`) — e.g. wrap the OpenAI embeddings API.
- **Vector store**: replace `InMemoryVectorStore` with Chroma / pgvector / Qdrant behind the same `add` / `search` interface.
- **LLM**: implement the `LLMClient` protocol (`generate(system_prompt, user_prompt) -> str`).
- **Chunking**: subclass or replace `FixedSizeChunker` — try semantic or recursive strategies.

## Limitations

This is a learning example, not production code: the vector store is in-memory
(nothing persists between runs), chunking is character-based rather than
token-based, and there is no reranking, query rewriting, or evaluation harness.
Each of those is a natural next step — and a good exercise.
