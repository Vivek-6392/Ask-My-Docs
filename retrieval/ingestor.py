"""
Document ingestor: load → split → embed → store.
Supports PDF, TXT, DOCX, Markdown.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
}


def _load_file(path: Path) -> list:
    ext = path.suffix.lower()
    loader_cls = _LOADERS.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")
    loader = loader_cls(str(path))
    return loader.load()


def _split(docs: list, chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        meta = doc.metadata
        for split in splits:
            if split.strip():
                chunks.append(
                    {
                        "text": split.strip(),
                        "metadata": {
                            "source": meta.get("source", str(meta.get("file_path", "unknown"))),
                            "page": meta.get("page", meta.get("page_number", "?")),
                        },
                    }
                )
    return chunks


def ingest_files(paths: list[Path], pipeline, config) -> dict:
    """
    Load, split, and index a list of file paths.
    Returns {files, chunks, elapsed_s}.
    """
    t0 = time.perf_counter()
    all_chunks: list[dict] = []

    for path in paths:
        try:
            docs = _load_file(path)
            chunks = _split(docs, config.chunk_size, config.chunk_overlap)
            all_chunks.extend(chunks)
            logger.info("Ingested %s → %d chunks", path.name, len(chunks))
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", path.name, exc)

    if all_chunks:
        pipeline.add_documents(all_chunks)

    return {
        "files": len(paths),
        "chunks": len(all_chunks),
        "elapsed_s": time.perf_counter() - t0,
    }
