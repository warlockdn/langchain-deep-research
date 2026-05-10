"""PGVector-backed document search tool."""

import asyncio
import json
import os
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

try:
    from langchain_community.embeddings import FastEmbedEmbeddings
except ImportError:  # pragma: no cover - optional until fastembed is installed
    FastEmbedEmbeddings = None  # type: ignore[misc, assignment]


def _strip_openai_prefix(model_name: str) -> str:
    """Normalize an openai:model identifier to the provider-local model id."""
    return model_name.split(":", 1)[1] if model_name.startswith("openai:") else model_name


def has_document_search_config() -> bool:
    """Return whether document search can be exposed to the agent."""
    return bool(os.getenv("CONNECTION_STRING") and os.getenv("COLLECTION_NAME"))


def _get_document_search_embedding_model() -> str:
    """Resolve the embedding model used for PGVector similarity queries."""
    return _strip_openai_prefix(
        os.getenv("OPENAI_EMBEDDING_MODEL")
        or os.getenv("PGVECTOR_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )


def _get_fastembed_model_name(explicit: str | None) -> str:
    """Resolve FastEmbed model id for local PGVector ingestion."""
    if explicit:
        return explicit
    return os.getenv("PGVECTOR_FASTEMBED_MODEL") or "BAAI/bge-small-en-v1.5"


def _validate_pgvector_connection_string(connection_string: str) -> str | None:
    """Validate PGVector connection string shape against langchain-postgres requirements."""
    if "psycopg2" in connection_string:
        return (
            "Document search is unavailable: CONNECTION_STRING must use the psycopg3 "
            "driver (`postgresql+psycopg://...`), not psycopg2."
        )
    return None


def _get_pgvector_env_config() -> tuple[str, str]:
    """Resolve and validate PGVector env configuration."""
    connection_string = os.getenv("CONNECTION_STRING")
    collection_name = os.getenv("COLLECTION_NAME")
    if not connection_string or not collection_name:
        raise RuntimeError(
            "Document ingestion is unavailable: CONNECTION_STRING and COLLECTION_NAME "
            "must both be set."
        )

    connection_error = _validate_pgvector_connection_string(connection_string)
    if connection_error:
        raise RuntimeError(connection_error)

    return connection_string, collection_name


@lru_cache(maxsize=4)
def _get_pgvector_store(
    connection_string: str,
    collection_name: str,
    embedding_model: str,
) -> PGVector:
    """Construct and cache a PGVector client for repeated document searches."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_ENDPOINT")
    return PGVector(
        embeddings=OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
        ),
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
        create_extension=False,
    )


@lru_cache(maxsize=8)
def _get_pgvector_store_fastembed(
    connection_string: str,
    collection_name: str,
    embedding_model: str,
) -> PGVector:
    """Construct and cache a PGVector client that uses FastEmbed local inference."""
    if FastEmbedEmbeddings is None:
        raise RuntimeError(
            "FastEmbed ingestion requires `langchain-community` with the `fastembed` "
            "package installed (`pip install fastembed`)."
        )
    return PGVector(
        embeddings=FastEmbedEmbeddings(model_name=embedding_model),
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
        create_extension=False,
    )


def _format_document_search_response(
    query: str,
    results: list[tuple[Any, float]],
) -> str:
    """Convert PGVector search hits into a deterministic text block for the model."""
    if not results:
        return f"No document search results found for query: {query}"

    formatted_results: list[str] = []
    for index, (document, score) in enumerate(results, start=1):
        metadata = getattr(document, "metadata", None) or {}
        metadata_json = json.dumps(metadata, sort_keys=True)
        formatted_results.append(
            "\n".join(
                [
                    f"Result {index}:",
                    f"Score: {score:.6f}",
                    f"Content: {getattr(document, 'page_content', '')}",
                    f"Metadata: {metadata_json}",
                ]
            )
        )

    return "\n\n".join(formatted_results)


def _document_id(document: Any, index: int) -> str:
    """Build a stable document id for PGVector ingestion."""
    metadata = getattr(document, "metadata", None) or {}
    source = metadata.get("source") or metadata.get("filename") or "document"
    return f"{source}:{index}"


def store_documents_in_pgvector(documents: list[Any]) -> dict[str, Any]:
    """Store documents in the configured PGVector collection."""
    connection_string, collection_name = _get_pgvector_env_config()
    vector_store = _get_pgvector_store(
        connection_string,
        collection_name,
        _get_document_search_embedding_model(),
    )
    ids = [_document_id(document, index) for index, document in enumerate(documents, start=1)]
    vector_store.add_documents(documents=documents, ids=ids)
    return {"collection_name": collection_name, "count": len(ids)}


def store_documents_in_pgvector_with_fastembed(
    documents: list[Any],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Store documents in PGVector using FastEmbed for local embedding inference.

    Uses the same CONNECTION_STRING and COLLECTION_NAME as OpenAI-backed ingestion.
    Query-time search (`document_search`) still uses the OpenAI embedding model unless
    you align collection dimensionality and reconfigure search separately.
    """
    connection_string, collection_name = _get_pgvector_env_config()
    resolved_model = _get_fastembed_model_name(model_name)
    vector_store = _get_pgvector_store_fastembed(
        connection_string,
        collection_name,
        resolved_model,
    )
    ids = [_document_id(document, index) for index, document in enumerate(documents, start=1)]
    vector_store.add_documents(documents=documents, ids=ids)
    return {
        "collection_name": collection_name,
        "count": len(ids),
        "embedding_model": resolved_model,
    }


@tool(
    "document_search",
    description=(
        "Search the configured PGVector document collection and return the most similar "
        "documents with scores and metadata."
    ),
)
async def document_search(
    query: str,
    k: int = 5,
    filter: dict[str, Any] | None = None,
) -> str:
    """Search the configured PGVector-backed document collection."""
    try:
        connection_string, collection_name = _get_pgvector_env_config()
        vector_store = _get_pgvector_store(
            connection_string,
            collection_name,
            _get_document_search_embedding_model(),
        )
        results = await asyncio.to_thread(
            vector_store.similarity_search_with_score,
            query=query,
            k=k,
            filter=filter,
        )
    except RuntimeError as exc:
        return str(exc).replace("Document ingestion", "Document search")
    except Exception as exc:
        return f"Document search failed: {exc}"

    return _format_document_search_response(query, results)
