import asyncio
from types import SimpleNamespace

from open_deep_research import utils
from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import research_system_prompt
from open_deep_research.tools import exa as exa_tools
from open_deep_research.tools import pgvector as pgvector_tools


def test_configuration_defaults_to_exa_only():
    config = Configuration()

    assert config.search_api == SearchAPI.EXA
    assert SearchAPI.EXA.value == "exa"
    assert [member.value for member in SearchAPI] == ["exa"]


def test_get_search_tool_returns_exa_search_tool():
    for env_name in ("CONNECTION_STRING", "COLLECTION_NAME"):
        try:
            del __import__("os").environ[env_name]
        except KeyError:
            pass
    __import__("os").environ.pop("EXA_ENABLED", None)

    tools = asyncio.run(utils.get_search_tool(SearchAPI.EXA))

    assert len(tools) == 1
    assert tools[0].name == "exa_search"


def test_get_search_tool_returns_exa_and_document_search_when_pgvector_env_is_set(monkeypatch):
    monkeypatch.delenv("EXA_ENABLED", raising=False)
    monkeypatch.setenv("CONNECTION_STRING", "postgresql+psycopg://user:pass@localhost:5432/db")
    monkeypatch.setenv("COLLECTION_NAME", "research_docs")

    tools = asyncio.run(utils.get_search_tool(SearchAPI.EXA))

    assert [tool.name for tool in tools] == ["exa_search", "document_search"]


def test_get_search_tool_skips_exa_when_disabled(monkeypatch):
    monkeypatch.setenv("EXA_ENABLED", "false")
    monkeypatch.delenv("CONNECTION_STRING", raising=False)
    monkeypatch.delenv("COLLECTION_NAME", raising=False)

    tools = asyncio.run(utils.get_search_tool(SearchAPI.EXA))

    assert tools == []


def test_exa_search_formats_results(monkeypatch):
    captured: dict[str, str] = {}

    class FakeAsyncExa:
        def __init__(self, api_key: str, api_base: str = "https://api.exa.ai"):
            captured["api_key"] = api_key
            captured["api_base"] = api_base

        async def search(self, query, **kwargs):
            assert query == "latest AI chip news"
            assert kwargs["num_results"] == 3
            assert kwargs["include_domains"] == ["example.com"]
            assert kwargs["exclude_domains"] == ["spam.com"]
            assert kwargs["contents"] == {"highlights": {"max_characters": 2000}}
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        title="Chip launch",
                        url="https://example.com/chip",
                        published_date="2026-03-20",
                        highlights=["Fast", "Efficient"],
                    )
                ]
            )

    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setattr(exa_tools, "AsyncExa", FakeAsyncExa)

    tool = asyncio.run(utils.get_search_tool(SearchAPI.EXA))[0]
    result = asyncio.run(
        tool.ainvoke(
            {
                "query": "latest AI chip news",
                "num_results": 3,
                "include_domains": ["example.com"],
                "exclude_domains": ["spam.com"],
            }
        )
    )

    assert captured["api_key"] == "exa-key"
    assert captured["api_base"] == "https://api.exa.ai"
    assert "Title: Chip launch" in result
    assert "URL: https://example.com/chip" in result
    assert "Published: 2026-03-20" in result
    assert "Highlights:" in result
    assert "- Fast" in result
    assert "- Efficient" in result


def test_exa_search_returns_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    result = asyncio.run(exa_tools.exa_search.ainvoke({"query": "latest AI chip news"}))

    assert result == "Exa search is unavailable: EXA_API_KEY is not set."


def test_exa_enabled_defaults_true_and_supports_false_values(monkeypatch):
    monkeypatch.delenv("EXA_ENABLED", raising=False)
    assert exa_tools.is_exa_search_enabled() is True

    monkeypatch.setenv("EXA_ENABLED", "0")
    assert exa_tools.is_exa_search_enabled() is False

    monkeypatch.setenv("EXA_ENABLED", "false")
    assert exa_tools.is_exa_search_enabled() is False


def test_document_search_formats_scored_results(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOpenAIEmbeddings:
        def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
            captured["embedding_model"] = model
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    class FakePGVector:
        def __init__(self, *, embeddings, collection_name: str, connection: str, use_jsonb: bool, create_extension: bool):
            captured["collection_name"] = collection_name
            captured["connection"] = connection
            captured["use_jsonb"] = use_jsonb
            captured["create_extension"] = create_extension
            captured["embeddings_type"] = type(embeddings).__name__

        def similarity_search_with_score(self, *, query: str, k: int, filter=None):
            captured["query"] = query
            captured["k"] = k
            captured["filter"] = filter
            return [
                (
                    SimpleNamespace(
                        page_content="LangGraph supports durable execution.",
                        metadata={"source": "internal-docs", "section": "overview"},
                    ),
                    0.123456,
                )
            ]

    monkeypatch.setenv("CONNECTION_STRING", "postgresql+psycopg://user:pass@localhost:5432/db")
    monkeypatch.setenv("COLLECTION_NAME", "research_docs")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENAI_ENDPOINT",
        "https://deepa-min0ats1-eastus2.openai.azure.com/openai/v1/",
    )
    monkeypatch.setattr(pgvector_tools, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    monkeypatch.setattr(pgvector_tools, "PGVector", FakePGVector)
    pgvector_tools._get_pgvector_store.cache_clear()

    document_tool = asyncio.run(utils.get_search_tool(SearchAPI.EXA))[1]
    result = asyncio.run(
        document_tool.ainvoke(
            {
                "query": "durable execution",
                "k": 3,
                "filter": {"source": "internal-docs"},
            }
        )
    )

    assert captured["embedding_model"] == "text-embedding-3-small"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://deepa-min0ats1-eastus2.openai.azure.com/openai/v1/"
    assert captured["collection_name"] == "research_docs"
    assert captured["connection"] == "postgresql+psycopg://user:pass@localhost:5432/db"
    assert captured["use_jsonb"] is True
    assert captured["create_extension"] is False
    assert captured["query"] == "durable execution"
    assert captured["k"] == 3
    assert captured["filter"] == {"source": "internal-docs"}
    assert "Result 1:" in result
    assert "Score: 0.123456" in result
    assert "Content: LangGraph supports durable execution." in result
    assert 'Metadata: {"section": "overview", "source": "internal-docs"}' in result


def test_document_search_returns_clear_error_when_env_is_missing(monkeypatch):
    monkeypatch.delenv("CONNECTION_STRING", raising=False)
    monkeypatch.delenv("COLLECTION_NAME", raising=False)

    result = asyncio.run(pgvector_tools.document_search.ainvoke({"query": "durable execution"}))

    assert result == "Document search is unavailable: CONNECTION_STRING and COLLECTION_NAME must both be set."


def test_store_documents_in_pgvector_uses_openai_embedding_model_env(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOpenAIEmbeddings:
        def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
            captured["embedding_model"] = model
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    class FakePGVector:
        def __init__(
            self,
            *,
            embeddings,
            collection_name: str,
            connection: str,
            use_jsonb: bool,
            create_extension: bool,
        ):
            captured["collection_name"] = collection_name
            captured["connection"] = connection
            captured["use_jsonb"] = use_jsonb
            captured["create_extension"] = create_extension
            captured["embeddings_type"] = type(embeddings).__name__

        def add_documents(self, *, documents, ids):
            captured["documents"] = documents
            captured["ids"] = ids
            return ids

    monkeypatch.setenv("CONNECTION_STRING", "postgresql+psycopg://user:pass@localhost:5432/db")
    monkeypatch.setenv("COLLECTION_NAME", "research_docs")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENAI_ENDPOINT",
        "https://deepa-min0ats1-eastus2.openai.azure.com/openai/v1/",
    )
    monkeypatch.setattr(pgvector_tools, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    monkeypatch.setattr(pgvector_tools, "PGVector", FakePGVector)
    pgvector_tools._get_pgvector_store.cache_clear()

    result = pgvector_tools.store_documents_in_pgvector(
        [
            SimpleNamespace(page_content="doc-1", metadata={"source": "a.pdf"}),
            SimpleNamespace(page_content="doc-2", metadata={"source": "a.pdf"}),
        ]
    )

    assert captured["embedding_model"] == "text-embedding-3-small"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://deepa-min0ats1-eastus2.openai.azure.com/openai/v1/"
    assert captured["collection_name"] == "research_docs"
    assert captured["connection"] == "postgresql+psycopg://user:pass@localhost:5432/db"
    assert captured["use_jsonb"] is True
    assert captured["create_extension"] is False
    assert captured["ids"] == ["a.pdf:1", "a.pdf:2"]
    assert result == {"collection_name": "research_docs", "count": 2}


def test_research_prompt_mentions_document_search_usage():
    prompt = research_system_prompt.format(date="Tue Apr 1, 2026", mcp_prompt="")

    assert "**document_search**" in prompt
    assert "**exa_search**" in prompt
    assert (
        "Use **document_search** first for internal, pre-indexed, or repository-specific "
        "documents before searching the web."
    ) in prompt
    assert "Only call tools that are actually available in your runtime tool list." in prompt
