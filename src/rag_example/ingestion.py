"""Stage 1 — Ingestion: load raw documents from disk into memory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str


SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(docs_dir: str | Path) -> list[Document]:
    """Read every supported file under docs_dir (recursively)."""
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"docs directory not found: {docs_dir}")
    documents: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = path.stem.replace("-", " ").replace("_", " ").title()
        documents.append(
            Document(doc_id=path.stem, title=title, text=text, source=str(path))
        )
    if not documents:
        raise ValueError(f"no supported documents found in {docs_dir}")
    return documents
