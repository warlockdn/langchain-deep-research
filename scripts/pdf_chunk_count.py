#!/usr/bin/env python3
"""Save PDF text extracted by PyPDF into JSONL.

How it works:
1. Discover matching PDFs under an input directory.
2. Load each PDF through LangChain's `PyPDFLoader` in page mode.
3. Optionally split those page documents into smaller chunks.
4. Persist one JSONL row per returned LangChain `Document`.
5. Print per-file counts, total count, and fail if any discovered PDF produced no rows.

What it writes:
- `source_file`: basename of the originating PDF
- `chunk_index`: 1-based per-file row counter
- `page_content`: extracted page text
- `metadata`: loader metadata from PyPDF/LangChain

Usage:
```bash
uv run python scripts/pdf_chunk_count.py --input-dir docs
```

Write chunks to a specific file:
```bash
uv run python scripts/pdf_chunk_count.py \
  --input-dir docs \
  --output-file docs/pdf_chunks.jsonl
```

Split pages into smaller chunks before writing:
```bash
uv run python scripts/pdf_chunk_count.py \
  --input-dir docs \
  --split \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --output-file docs/pdf_chunks.jsonl
```

Store the final emitted chunks in PGVector:
```bash
uv run python scripts/pdf_chunk_count.py \
  --input-dir docs \
  --split \
  --pgvector
```
"""

import dotenv
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableLambda
from open_deep_research.tools.pgvector import store_documents_in_pgvector

dotenv.load_dotenv()

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:  # pragma: no cover - exercised via runtime guard
    PyPDFLoader = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - exercised via runtime guard
    RecursiveCharacterTextSplitter = None


class PyPDFLoaderKwargs(TypedDict):
    """Typed kwargs mirror for PyPDFLoader to improve editor autocomplete."""

    password: str | bytes | None
    headers: dict[str, str] | None
    extract_images: bool
    mode: Literal["single", "page"]
    images_parser: Any | None
    images_inner_format: Literal["text", "markdown-img", "html-img"]
    pages_delimiter: str
    extraction_mode: Literal["plain", "layout"]
    extraction_kwargs: dict[str, Any] | None


class TextSplitterKwargs(TypedDict):
    """Typed kwargs mirror for RecursiveCharacterTextSplitter."""

    chunk_size: int
    chunk_overlap: int
    length_function: Any
    is_separator_regex: bool


ALLOWED_METADATA_KEYS = ("source", "total_pages", "page", "page_label")


# Keep CLI parsing separate so the pipeline functions can stay pure and testable.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for chunk counting."""
    parser = argparse.ArgumentParser(
        description="Count LangChain documents produced by PyPDF page loading for PDFs."
    )
    parser.add_argument("--input-dir", default="docs", help="Directory containing PDFs.")
    parser.add_argument("--glob", default="*.pdf", help="Glob pattern for PDF discovery.")
    parser.add_argument(
        "--output-file",
        help="Path to the JSONL file where chunk records will be written.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split loaded page documents into smaller chunks before writing JSONL.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size used when --split is enabled.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap used when --split is enabled.",
    )
    parser.add_argument(
        "--pgvector",
        action="store_true",
        help="Store the final emitted documents in the configured PGVector collection.",
    )
    return parser.parse_args(argv)


# Spell out the current PyPDFLoader surface here so editors can autocomplete it.
def build_loader_kwargs(args: argparse.Namespace) -> PyPDFLoaderKwargs:
    """Build PyPDF loader kwargs from CLI arguments."""
    del args
    return {
        "password": None,
        "headers": None,
        "extract_images": False,
        "mode": "page",
        "images_parser": None,
        "images_inner_format": "text",
        "pages_delimiter": "\n\f",
        "extraction_mode": "plain",
        "extraction_kwargs": None,
    }


# Default output lives alongside the input PDFs unless the caller overrides it.
def resolve_output_path(args: argparse.Namespace) -> Path:
    """Resolve the output path for persisted chunk records."""
    if args.output_file:
        return Path(args.output_file)
    return Path(args.input_dir) / "pdf_chunks.jsonl"


def build_text_splitter_kwargs(args: argparse.Namespace) -> TextSplitterKwargs:
    """Build RecursiveCharacterTextSplitter kwargs from CLI arguments."""
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0.")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be 0 or greater.")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size.")
    return {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "length_function": len,
        "is_separator_regex": False,
    }


# Gather all runtime inputs up front so downstream pipeline steps only transform data.
def discover_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Find candidate PDF files and prepare loader configuration."""
    input_dir = Path(args.input_dir)
    files = sorted(path for path in input_dir.glob(args.glob) if path.is_file())
    if not files:
        raise ValueError(
            f"No PDF files matched '{args.glob}' under '{input_dir.resolve()}'."
        )
    return {
        "files": files,
        "loader_kwargs": build_loader_kwargs(args),
        "splitter_kwargs": build_text_splitter_kwargs(args) if args.split else None,
        "pgvector_enabled": args.pgvector,
        "output_path": resolve_output_path(args),
    }


def sanitize_document_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only the metadata fields needed downstream."""
    sanitized = {
        key: metadata[key] for key in ALLOWED_METADATA_KEYS if key in metadata
    }
    if "source" not in sanitized and metadata.get("filename"):
        sanitized["source"] = str(metadata["filename"])
    return sanitized


# Load every matched PDF and flatten the page documents into a single iterable.
def load_documents(payload: dict[str, Any]):
    """Load LangChain documents from local PDFs."""
    if PyPDFLoader is None:
        raise RuntimeError(
            "langchain-community or pypdf is not installed. Add them to your environment first."
        )

    documents = []
    for file_path in payload["files"]:
        loader = PyPDFLoader(str(file_path), **payload["loader_kwargs"])
        for document in loader.lazy_load():
            document.metadata = sanitize_document_metadata(document.metadata)
            documents.append(document)
    return {
        "files": payload["files"],
        "documents": documents,
        "splitter_kwargs": payload["splitter_kwargs"],
        "pgvector_enabled": payload["pgvector_enabled"],
        "output_path": payload["output_path"],
    }


def split_documents(payload: dict[str, Any]) -> dict[str, Any]:
    """Optionally split loaded page documents into smaller chunks."""
    splitter_kwargs = payload["splitter_kwargs"]
    documents = payload["documents"]
    if splitter_kwargs is not None:
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError(
                "langchain-text-splitters is not installed. Add it to your environment first."
            )
        splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
        documents = splitter.split_documents(documents)
    return {
        "files": payload["files"],
        "documents": documents,
        "pgvector_enabled": payload["pgvector_enabled"],
        "output_path": payload["output_path"],
    }


def persist_documents_to_pgvector(payload: dict[str, Any]) -> dict[str, Any]:
    """Optionally store the final emitted documents in PGVector."""
    pgvector_result = None
    if payload["pgvector_enabled"]:
        pgvector_result = store_documents_in_pgvector(payload["documents"])
    return {
        "files": payload["files"],
        "documents": payload["documents"],
        "pgvector_result": pgvector_result,
        "output_path": payload["output_path"],
    }


# Prefer explicit filename metadata, then fall back to the loader source path.
def document_name(metadata: dict[str, Any]) -> str:
    """Resolve a display name for a chunked document."""
    filename = metadata.get("filename")
    if filename:
        return str(filename)

    source = metadata.get("source")
    if source:
        return Path(str(source)).name

    return "unknown"


# Persist rows while counting so we do only one pass over the extracted documents.
def persist_and_aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    """Write chunk records to disk and aggregate counts by source file."""
    expected_names = [path.name for path in payload["files"]]
    counts: Counter[str] = Counter({name: 0 for name in expected_names})
    output_path = payload["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for document in payload["documents"]:
            source_file = document_name(document.metadata)
            counts[source_file] += 1
            record = {
                "source_file": source_file,
                "chunk_index": counts[source_file],
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    missing = [name for name in expected_names if counts[name] == 0]
    return {
        "counts": counts,
        "missing": missing,
        "output_path": output_path,
        "pgvector_result": payload.get("pgvector_result"),
    }


# The runnable chain keeps the top-level control flow easy to follow and test.
def build_pipeline():
    """Build the LangChain runnable pipeline for discovery, loading, and counting."""
    return (
        RunnableLambda(discover_inputs)
        | RunnableLambda(load_documents)
        | RunnableLambda(split_documents)
        | RunnableLambda(persist_documents_to_pgvector)
        | RunnableLambda(persist_and_aggregate)
    )


# The CLI wrapper is intentionally thin: run the pipeline, print counts, propagate failure.
def main(argv: list[str] | None = None) -> int:
    """Run the chunk counting CLI."""
    args = parse_args(argv)

    try:
        result = build_pipeline().invoke(args)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"Failed to count chunks: {exc}", file=sys.stderr)
        return 1

    counts = result["counts"]
    for name in sorted(counts):
        print(f"{name}: {counts[name]} chunks")

    print(f"TOTAL: {len(counts)} files, {sum(counts.values())} chunks")
    print(f"Saved chunks to {result['output_path']}")
    if result["pgvector_result"] is not None:
        print(
            "Saved "
            f"{result['pgvector_result']['count']} chunks to PGVector collection "
            f"{result['pgvector_result']['collection_name']}"
        )
    if result["missing"]:
        print(
            f"Missing chunk output for: {', '.join(result['missing'])}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
