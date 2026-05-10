#!/usr/bin/env python3
"""Save PDF text extracted by Docling into JSONL.

How it works:
1. Discover matching PDFs under an input directory.
2. Load each PDF through LangChain's `DoclingLoader`. Default export is full-document
   markdown (`--export-type markdown`); use `--export-type doc_chunks` with Docling's
   `HybridChunker` for layout-aware chunks.
3. For `markdown` export, split each document's markdown with `MarkdownHeaderTextSplitter`
   (`#` / `##` / `###`). For `doc_chunks`, keep Docling's chunks as-is.
4. Optionally split again with LangChain's `RecursiveCharacterTextSplitter` (`--split`).
5. Persist one JSONL row per returned LangChain `Document`.
6. Print per-file counts, total count, and fail if any discovered PDF produced no rows.

What it writes:
- `source_file`: basename of the originating PDF
- `chunk_index`: 1-based per-file row counter
- `page_content`: extracted page text
- `metadata`: loader metadata from Docling/LangChain, plus when available:
  `page_number` / optional `page_number_end` (from Docling provenance or `page`),
  `section_path` (heading breadcrumb), `section_number` (leading numbered prefix if detected)

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
  --export-type doc_chunks \
  --split \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --output-file docs/pdf_chunks.jsonl
```

Store the final emitted chunks in PGVector (OpenAI embeddings):
```bash
uv run python scripts/pdf_chunk_count.py \
  --input-dir docs \
  --split \
  --pgvector
```

Store in PGVector with local FastEmbed embeddings:
```bash
uv run python scripts/pdf_chunk_count.py \
  --input-dir docs \
  --split \
  --pgvector-fastembed
```
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

import dotenv
from docling.chunking import BaseChunker, HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from open_deep_research.tools.pgvector import (
    store_documents_in_pgvector,
    store_documents_in_pgvector_with_fastembed,
)

dotenv.load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if TYPE_CHECKING:
    from langchain_docling.loader import BaseMetaExtractor

try:
    from langchain_docling.loader import DoclingLoader
except ImportError:  # pragma: no cover - exercised via runtime guard
    DoclingLoader = None

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
except ImportError:  # pragma: no cover - exercised via runtime guard
    MarkdownHeaderTextSplitter = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - exercised via runtime guard
    RecursiveCharacterTextSplitter = None


# Default markdown heading levels for post-export splitting (markdown export only).
MARKDOWN_HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
]


class DoclingConvertKwargs(TypedDict, total=False):
    """Kwargs forwarded from `DoclingLoader` to `DocumentConverter.convert` (minus `source`)."""

    headers: dict[str, str] | None
    raises_on_error: bool
    max_num_pages: int
    max_file_size: int
    page_range: tuple[int, int]


class MarkdownExportKwargs(TypedDict, total=False):
    """Kwargs forwarded from `DoclingLoader` to `DoclingDocument.export_to_markdown`."""

    delim: str
    from_element: int
    to_element: int
    labels: set[DocItemLabel] | None
    strict_text: bool
    escape_html: bool
    escape_underscores: bool
    image_placeholder: str
    enable_chart_tables: bool
    image_mode: ImageRefMode
    indent: int
    text_width: int
    page_no: int | None
    included_content_layers: set[ContentLayer] | None
    page_break_placeholder: str | None
    include_annotations: bool
    mark_annotations: bool
    compact_tables: bool
    traverse_pictures: bool
    use_legacy_annotations: bool | None
    allowed_meta_names: set[str] | None
    blocked_meta_names: set[str] | None
    mark_meta: bool


ExportTypeName = Literal["markdown", "doc_chunks"]


class DoclingLoaderInitKwargs(TypedDict, total=False):
    """Mirror of keyword-only args accepted by `DoclingLoader.__init__`."""

    converter: DocumentConverter | None
    convert_kwargs: DoclingConvertKwargs | None
    export_type: ExportTypeName
    md_export_kwargs: MarkdownExportKwargs | None
    chunker: BaseChunker | None
    meta_extractor: BaseMetaExtractor | None


class SplitPipelineKwargs(TypedDict):
    """Splitter kwargs used by this script when `--split` is set."""

    chunk_size: int
    chunk_overlap: int
    length_function: Callable[[str], int]
    is_separator_regex: bool


class DiscoverInputsPayload(TypedDict):
    """Output of `discover_inputs`."""

    files: list[Path]
    loader_kwargs: DoclingLoaderInitKwargs
    export_type: ExportTypeName
    splitter_kwargs: SplitPipelineKwargs | None
    pgvector_enabled: bool
    pgvector_fastembed: bool
    fastembed_model: str | None
    output_path: Path


class LoadedDocumentsPayload(TypedDict):
    """Pipeline state after PDFs are loaded."""

    files: list[Path]
    documents: list[Document]
    export_type: ExportTypeName
    splitter_kwargs: SplitPipelineKwargs | None
    pgvector_enabled: bool
    pgvector_fastembed: bool
    fastembed_model: str | None
    output_path: Path


class SplitDocumentsPayload(TypedDict):
    """Pipeline state after optional chunking."""

    files: list[Path]
    documents: list[Document]
    pgvector_enabled: bool
    pgvector_fastembed: bool
    fastembed_model: str | None
    output_path: Path


class PgVectorStoreResult(TypedDict):
    """Return shape of PGVector store helpers."""

    collection_name: str
    count: int
    embedding_model: NotRequired[str]


class PgVectorPipelinePayload(TypedDict):
    """Pipeline state after optional PGVector persistence."""

    files: list[Path]
    documents: list[Document]
    pgvector_result: PgVectorStoreResult | None
    output_path: Path


class ChunkJsonlMetadata(TypedDict, total=False):
    """Subset of LangChain `Document.metadata` we retain in JSONL."""

    source: str
    total_pages: int
    page: int
    page_label: str
    page_number: int
    page_number_end: int
    section_path: str
    section_number: str
    Header_1: str
    Header_2: str
    Header_3: str


class ChunkJsonlRecord(TypedDict):
    """One JSONL row written by this script."""

    source_file: str
    chunk_index: int
    page_content: str
    metadata: dict[str, object]


class PersistAggregateResult(TypedDict):
    """Final aggregate returned by `persist_and_aggregate`."""

    counts: Counter[str]
    missing: list[str]
    output_path: Path
    pgvector_result: PgVectorStoreResult | None


ALLOWED_METADATA_KEYS = ("source", "total_pages", "page", "page_label")

# Markdown splitter keys merged into chunk metadata (markdown export only).
MARKDOWN_HEADER_METADATA_KEYS = ("Header_1", "Header_2", "Header_3")

# Docling HybridChunk / MetaExtractor chunk metadata includes nested provenance here.
SECTION_NUMBER_LEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+")


def _pages_from_dl_meta(dl_meta: object) -> list[int]:
    """Collect 1-based page indices from Docling chunk `dl_meta` provenance."""
    pages: list[int] = []
    if not isinstance(dl_meta, dict):
        return pages
    for item in dl_meta.get("doc_items") or []:
        if not isinstance(item, dict):
            continue
        for prov in item.get("prov") or []:
            if isinstance(prov, dict) and "page_no" in prov:
                try:
                    pages.append(int(prov["page_no"]))
                except (TypeError, ValueError):
                    pass
    return pages


def _headings_from_dl_meta(dl_meta: object) -> list[str]:
    if not isinstance(dl_meta, dict):
        return []
    headings = dl_meta.get("headings")
    if isinstance(headings, list):
        return [str(h) for h in headings if h is not None and str(h).strip()]
    return []


def extract_section_number(title: str) -> str | None:
    """Parse a leading section number like `1.2.3 Title` or `§ 402` when present."""
    if not title or not str(title).strip():
        return None
    s = str(title).strip()
    match = SECTION_NUMBER_LEADING_RE.match(s)
    if match:
        return match.group(1)
    section_symbol = re.match(r"^\s*§\s*([\d.]+)\b", s, re.IGNORECASE)
    if section_symbol:
        return section_symbol.group(1)
    return None


def finalize_chunk_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Whitelist loader fields, flatten Docling `dl_meta` into page/section fields."""
    meta = dict(metadata)
    dl_meta = meta.pop("dl_meta", None)

    dl_pages = _pages_from_dl_meta(dl_meta)
    dl_headings = _headings_from_dl_meta(dl_meta)

    filtered: dict[str, object] = {
        key: meta[key]
        for key in ALLOWED_METADATA_KEYS
        if key in meta
    }
    if "source" not in filtered and meta.get("filename"):
        filtered["source"] = str(meta["filename"])

    for key in MARKDOWN_HEADER_METADATA_KEYS:
        if key in meta:
            filtered[key] = meta[key]

    page_number: int | None = None
    page_number_end: int | None = None
    if dl_pages:
        uniq = sorted(set(dl_pages))
        page_number = uniq[0]
        if len(uniq) > 1:
            page_number_end = uniq[-1]
    elif isinstance(meta.get("page_number"), int):
        page_number = meta["page_number"]
        if isinstance(meta.get("page_number_end"), int):
            page_number_end = meta["page_number_end"]
    elif isinstance(meta.get("page"), int):
        page_number = meta["page"]

    if page_number is not None:
        filtered["page_number"] = page_number
    if page_number_end is not None:
        filtered["page_number_end"] = page_number_end

    header_titles = [
        str(meta[k]) for k in MARKDOWN_HEADER_METADATA_KEYS if k in meta and meta[k]
    ]

    deepest = ""
    if header_titles:
        filtered["section_path"] = " > ".join(header_titles)
        deepest = header_titles[-1]
    elif dl_headings:
        filtered["section_path"] = " > ".join(dl_headings)
        deepest = dl_headings[-1]
    elif isinstance(meta.get("section_path"), str) and meta["section_path"].strip():
        path_str = meta["section_path"].strip()
        filtered["section_path"] = path_str
        deepest = path_str.split(" > ")[-1]

    if deepest:
        section_num = extract_section_number(deepest)
        if section_num:
            filtered["section_number"] = section_num

    return filtered


# Keep CLI parsing separate so the pipeline functions can stay pure and testable.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for chunk counting."""
    parser = argparse.ArgumentParser(
        description="Count LangChain documents produced by Docling for PDFs."
    )
    parser.add_argument("--input-dir", default="docs", help="Directory containing PDFs.")
    parser.add_argument("--glob", default="*.pdf", help="Glob pattern for PDF discovery.")
    parser.add_argument(
        "--export-type",
        choices=("markdown", "doc_chunks"),
        default="markdown",
        help=(
            "Docling export: full-document markdown, or doc_chunks (HybridChunker "
            "layout-aware chunks)."
        ),
    )
    parser.add_argument(
        "--output-file",
        help="Path to the JSONL file where chunk records will be written.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split loaded documents into smaller chunks before writing JSONL.",
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
    parser.add_argument(
        "--pgvector-fastembed",
        action="store_true",
        help=(
            "Like --pgvector, but embed locally with FastEmbed (same DB env; use a "
            "matching embedding dimension for similarity search)."
        ),
    )
    parser.add_argument(
        "--fastembed-model",
        default=None,
        metavar="MODEL",
        help=(
            "FastEmbed model id (default: PGVECTOR_FASTEMBED_MODEL env or "
            "BAAI/bge-small-en-v1.5)."
        ),
    )
    return parser.parse_args(argv)


# Spell out the current DoclingLoader surface here so editors can autocomplete it.
def build_loader_kwargs(args: argparse.Namespace) -> DoclingLoaderInitKwargs:
    """Build Docling loader kwargs from CLI arguments."""
    export_type: ExportTypeName = args.export_type
    kwargs: DoclingLoaderInitKwargs = {"export_type": export_type}
    if export_type == "doc_chunks":
        # DoclingLoader would default to HybridChunker for doc_chunks; pass explicitly so
        # tokenizer / merge_peers / … can be tuned here later.
        kwargs["chunker"] = HybridChunker()
    return kwargs


# Default output lives alongside the input PDFs unless the caller overrides it.
def resolve_output_path(args: argparse.Namespace) -> Path:
    """Resolve the output path for persisted chunk records."""
    if args.output_file:
        return Path(args.output_file)
    return Path(args.input_dir) / "pdf_chunks.jsonl"


def build_text_splitter_kwargs(args: argparse.Namespace) -> SplitPipelineKwargs:
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
def discover_inputs(args: argparse.Namespace) -> DiscoverInputsPayload:
    """Find candidate PDF files and prepare loader configuration."""
    input_dir = Path(args.input_dir)
    files = sorted(path for path in input_dir.glob(args.glob) if path.is_file())
    if not files:
        raise ValueError(
            f"No PDF files matched '{args.glob}' under '{input_dir.resolve()}'."
        )
    if args.pgvector and args.pgvector_fastembed:
        raise ValueError("Use either --pgvector or --pgvector-fastembed, not both.")
    if args.fastembed_model and not args.pgvector_fastembed:
        raise ValueError("--fastembed-model is only valid with --pgvector-fastembed.")
    return {
        "files": files,
        "loader_kwargs": build_loader_kwargs(args),
        "export_type": args.export_type,
        "splitter_kwargs": build_text_splitter_kwargs(args) if args.split else None,
        "pgvector_enabled": args.pgvector,
        "pgvector_fastembed": args.pgvector_fastembed,
        "fastembed_model": args.fastembed_model,
        "output_path": resolve_output_path(args),
    }


# Load every matched PDF and flatten the extracted documents into a single iterable.
def load_documents(payload: DiscoverInputsPayload) -> LoadedDocumentsPayload:
    """Load LangChain documents from local PDFs."""
    if DoclingLoader is None:
        raise RuntimeError(
            "langchain-docling is not installed. Add it to your environment first."
        )

    documents: list[Document] = []
    for file_path in payload["files"]:
        loader = DoclingLoader(file_path=str(file_path), **payload["loader_kwargs"])
        for document in loader.lazy_load():
            document.metadata = finalize_chunk_metadata(dict(document.metadata))
            documents.append(document)
    return {
        "files": payload["files"],
        "documents": documents,
        "export_type": payload["export_type"],
        "splitter_kwargs": payload["splitter_kwargs"],
        "pgvector_enabled": payload["pgvector_enabled"],
        "pgvector_fastembed": payload["pgvector_fastembed"],
        "fastembed_model": payload["fastembed_model"],
        "output_path": payload["output_path"],
    }


def apply_markdown_header_splits(documents: list[Document]) -> list[Document]:
    """Split markdown `page_content` by headings; merge parent metadata into each chunk."""
    if MarkdownHeaderTextSplitter is None:
        raise RuntimeError(
            "langchain-text-splitters is not installed. Add it to your environment first."
        )
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON)
    out: list[Document] = []
    for doc in documents:
        for split in splitter.split_text(doc.page_content):
            merged: dict[str, object] = {**dict(doc.metadata), **dict(split.metadata)}
            out.append(Document(page_content=split.page_content, metadata=merged))
    return out


def split_documents(payload: LoadedDocumentsPayload) -> SplitDocumentsPayload:
    """Apply export-type splitting, then optional recursive character splitting."""
    export_type = payload["export_type"]
    documents = payload["documents"]

    if export_type == "markdown":
        documents = apply_markdown_header_splits(documents)
    elif export_type == "doc_chunks":
        pass
    else:
        raise ValueError(f"Unexpected export type: {export_type}")

    splitter_kwargs = payload["splitter_kwargs"]
    if splitter_kwargs is not None:
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError(
                "langchain-text-splitters is not installed. Add it to your environment first."
            )
        splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
        documents = splitter.split_documents(documents)

    documents = [
        Document(
            page_content=doc.page_content,
            metadata=finalize_chunk_metadata(dict(doc.metadata)),
        )
        for doc in documents
    ]
    return {
        "files": payload["files"],
        "documents": documents,
        "pgvector_enabled": payload["pgvector_enabled"],
        "pgvector_fastembed": payload["pgvector_fastembed"],
        "fastembed_model": payload["fastembed_model"],
        "output_path": payload["output_path"],
    }


def persist_documents_to_pgvector(payload: SplitDocumentsPayload) -> PgVectorPipelinePayload:
    """Optionally store the final emitted documents in PGVector."""
    pgvector_result: PgVectorStoreResult | None = None
    if payload["pgvector_fastembed"]:
        pgvector_result = store_documents_in_pgvector_with_fastembed(
            payload["documents"],
            model_name=payload["fastembed_model"],
        )
    elif payload["pgvector_enabled"]:
        pgvector_result = store_documents_in_pgvector(payload["documents"])
    return {
        "files": payload["files"],
        "documents": payload["documents"],
        "pgvector_result": pgvector_result,
        "output_path": payload["output_path"],
    }


# Prefer explicit filename metadata, then fall back to the loader source path.
def document_name(metadata: Mapping[str, object]) -> str:
    """Resolve a display name for a chunked document."""
    filename = metadata.get("filename")
    if filename:
        return str(filename)

    source = metadata.get("source")
    if source:
        return Path(str(source)).name

    return "unknown"


# Persist rows while counting so we do only one pass over the extracted documents.
def persist_and_aggregate(payload: PgVectorPipelinePayload) -> PersistAggregateResult:
    """Write chunk records to disk and aggregate counts by source file."""
    expected_names = [path.name for path in payload["files"]]
    counts: Counter[str] = Counter({name: 0 for name in expected_names})
    output_path = payload["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for document in payload["documents"]:
            source_file = document_name(dict(document.metadata))
            counts[source_file] += 1
            record: ChunkJsonlRecord = {
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
        "pgvector_result": payload["pgvector_result"],
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
        pg = result["pgvector_result"]
        line = (
            f"Saved {pg['count']} chunks to PGVector collection {pg['collection_name']}"
        )
        if "embedding_model" in pg:
            line += f" (FastEmbed: {pg['embedding_model']})"
        print(line)
    if result["missing"]:
        print(
            f"Missing chunk output for: {', '.join(result['missing'])}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
