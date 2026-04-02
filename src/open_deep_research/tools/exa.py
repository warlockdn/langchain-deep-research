"""Exa-backed web search tool."""

import os
from typing import Any

try:
    from exa_py import AsyncExa
except ImportError:  # pragma: no cover - exercised via runtime guard
    AsyncExa = None
from langchain_core.tools import tool


def _get_exa_result_value(result: Any, *field_names: str) -> Any:
    """Read a field from an Exa SDK result object or dict."""
    for field_name in field_names:
        if isinstance(result, dict) and field_name in result:
            return result[field_name]

        value = getattr(result, field_name, None)
        if value is not None:
            return value

    return None


def _format_exa_search_response(query: str, results: list[Any]) -> str:
    """Convert Exa search results into a deterministic text block for the model."""
    if not results:
        return f"No Exa results found for query: {query}"

    formatted_results: list[str] = []
    for index, result in enumerate(results, start=1):
        title = _get_exa_result_value(result, "title") or "Untitled"
        url = _get_exa_result_value(result, "url") or "Unknown URL"
        published_date = _get_exa_result_value(result, "published_date", "publishedDate")
        highlights = _get_exa_result_value(result, "highlights") or []
        if isinstance(highlights, str):
            highlights = [highlights]

        result_lines = [
            f"Result {index}:",
            f"Title: {title}",
            f"URL: {url}",
        ]
        if published_date:
            result_lines.append(f"Published: {published_date}")

        result_lines.append("Highlights:")
        if highlights:
            result_lines.extend(f"- {highlight}" for highlight in highlights)
        else:
            result_lines.append("- No highlights returned.")

        formatted_results.append("\n".join(result_lines))

    return "\n\n".join(formatted_results)


def is_exa_search_enabled() -> bool:
    """Return whether Exa search should be exposed to the agent."""
    value = os.getenv("EXA_ENABLED")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


@tool("exa_search", description="Search the web with Exa and return result metadata plus highlights.")
async def exa_search(
    query: str,
    num_results: int = 5,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> str:
    """Search the web using Exa and return a compact, citation-friendly result block."""
    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        return "Exa search is unavailable: EXA_API_KEY is not set."

    if AsyncExa is None:
        return "Exa search is unavailable: exa-py is not installed."

    try:
        exa_client = AsyncExa(api_key=exa_api_key)
        response = await exa_client.search(
            query,
            num_results=num_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            contents={"highlights": {"max_characters": 2000}},
        )
    except Exception as exc:
        return f"Exa search failed: {exc}"

    return _format_exa_search_response(query, getattr(response, "results", []))
