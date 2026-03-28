"""
scripts/ingest.py — CLI tool to batch-ingest documents.

Usage:
    python scripts/ingest.py --dir ./my_docs
    python scripts/ingest.py --file report.pdf
    python scripts/ingest.py --dir ./docs --clear
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig
from retrieval.pipeline import RAGPipeline
from retrieval.ingestor import ingest_files

SUPPORTED = {".pdf", ".txt", ".docx", ".md"}


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED else []
    return [p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED]


def main():
    parser = argparse.ArgumentParser(description="Batch document ingestor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="Directory of documents to ingest")
    group.add_argument("--file", type=Path, help="Single file to ingest")
    parser.add_argument("--clear", action="store_true", help="Clear existing index first")
    args = parser.parse_args()

    config = AppConfig.from_env()
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"❌ {e}")
        sys.exit(1)

    pipeline = RAGPipeline(config)

    if args.clear:
        pipeline.vector_store.delete_collection()
        print("🗑️  Cleared existing vector store.")

    source = args.dir or args.file
    files = collect_files(source)

    if not files:
        print(f"❌ No supported files found in: {source}")
        sys.exit(1)

    print(f"📂 Found {len(files)} file(s) to ingest:")
    for f in files:
        print(f"   • {f.name}")

    stats = ingest_files(files, pipeline, config)
    print(
        f"\n✅ Ingested {stats['chunks']} chunks from "
        f"{stats['files']} file(s) in {stats['elapsed_s']:.1f}s"
    )


if __name__ == "__main__":
    main()
