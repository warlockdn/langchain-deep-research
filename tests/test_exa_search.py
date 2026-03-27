import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from open_deep_research import deep_researcher, utils
from open_deep_research.configuration import Configuration, SearchAPI


def test_configuration_defaults_to_exa_only():
    config = Configuration()

    assert config.search_api == SearchAPI.EXA
    assert SearchAPI.EXA.value == "exa"
    assert [member.value for member in SearchAPI] == ["exa"]


def test_get_search_tool_returns_exa_search_tool():
    tools = asyncio.run(utils.get_search_tool(SearchAPI.EXA))

    assert len(tools) == 1
    assert tools[0].name == "exa_search"


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
    monkeypatch.setattr(utils, "AsyncExa", FakeAsyncExa)

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

    tool = asyncio.run(utils.get_search_tool(SearchAPI.EXA))[0]
    result = asyncio.run(tool.ainvoke({"query": "latest AI chip news"}))

    assert result == "Exa search is unavailable: EXA_API_KEY is not set."


def test_researcher_tools_ignores_old_openai_native_search_metadata():
    state = {
        "researcher_messages": [
            AIMessage(
                content="done",
                tool_calls=[],
                additional_kwargs={
                    "tool_outputs": [{"type": "web_search_call"}],
                },
            )
        ],
        "tool_call_iterations": 0,
    }
    config = {"configurable": {"search_api": "exa"}}

    result = asyncio.run(deep_researcher.researcher_tools(state, config, writer=lambda payload: None))

    assert result.goto == "compress_research"
    assert result.update is None


def test_researcher_tools_executes_exa_tool_calls(monkeypatch):
    class FakeTool:
        name = "exa_search"

        async def ainvoke(self, args, config):
            assert args == {"query": "latest AI chip news"}
            assert config["configurable"]["search_api"] == "exa"
            return "search output"

    monkeypatch.setattr(deep_researcher, "get_all_tools", lambda config: asyncio.sleep(0, result=[FakeTool()]))

    state = {
        "researcher_messages": [
            AIMessage(
                content="searching",
                tool_calls=[
                    {
                        "name": "exa_search",
                        "args": {"query": "latest AI chip news"},
                        "id": "call-1",
                    }
                ],
            )
        ],
        "tool_call_iterations": 1,
    }
    config = {"configurable": {"search_api": "exa", "max_react_tool_calls": 10}}

    result = asyncio.run(deep_researcher.researcher_tools(state, config, writer=lambda payload: None))

    assert result.goto == "researcher"
    assert result.update["researcher_messages"] == [
        ToolMessage(content="search output", name="exa_search", tool_call_id="call-1")
    ]
