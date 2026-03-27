import asyncio
from collections import deque

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from open_deep_research import deep_researcher
from open_deep_research.state import ClarifyWithUser, ResearchQuestion
from open_deep_research.stream_protocol import (
    astream_deep_researcher_protocol,
    extract_source_parts,
)


class FakeStructuredRunnable:
    def __init__(self, response):
        self.response = response

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        return self.response


class FakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class FakeModelFactory:
    def __init__(self, *, structured_responses=None, chat_responses=None):
        self.structured_responses = deque(structured_responses or [])
        self.chat_responses = deque(chat_responses or [])

    def __call__(self, *args, **kwargs):
        factory = self

        class Builder:
            def with_structured_output(self, schema):
                return FakeStructuredRunnable(factory.structured_responses.popleft())

            def bind_tools(self, tools, **kwargs):
                return FakeChatModel(responses=[factory.chat_responses.popleft()])

            async def ainvoke(self, messages, config=None):
                model = FakeChatModel(responses=[factory.chat_responses.popleft()])
                return await model.ainvoke(messages, config)

        return Builder()


class FakeExaTool:
    name = "exa_search"

    def __init__(self, output=""):
        self.output = output

    async def ainvoke(self, args, config):
        return self.output


async def _collect_stream(monkeypatch, model_factory, config=None, tool=None):
    monkeypatch.setattr(deep_researcher, "build_chat_model", model_factory)
    monkeypatch.setattr(
        deep_researcher,
        "get_all_tools",
        lambda config: asyncio.sleep(0, result=[tool] if tool else []),
    )

    parts = []
    async for part in astream_deep_researcher_protocol(
        {"messages": [HumanMessage(content="research something")]},
        config=config or {"configurable": {"thread_id": "thread-1", "search_api": "exa"}},
    ):
        parts.append(part)
    return parts


def test_astream_protocol_emits_run_and_node_events_for_clarification(monkeypatch):
    model_factory = FakeModelFactory(
        structured_responses=[
            ClarifyWithUser(
                need_clarification=True,
                question="Need more detail",
                verification="",
            )
        ]
    )

    parts = asyncio.run(_collect_stream(monkeypatch, model_factory))

    custom_types = [part["data"]["type"] for part in parts if part["type"] == "custom"]
    node_statuses = [
        part["data"]
        for part in parts
        if part["type"] == "custom" and part["data"]["type"] == "data-node-status"
    ]

    assert custom_types[0] == "start"
    assert "start-step" in custom_types
    assert "finish-step" in custom_types
    assert custom_types[-1] == "finish"
    assert [status["phase"] for status in node_statuses] == [
        "node_start",
        "llm_start",
        "llm_end",
        "node_end",
    ]
    assert all(status["node"] == "clarify_with_user" for status in node_statuses)


def test_astream_protocol_emits_subgraph_tool_source_and_message_events(monkeypatch):
    model_factory = FakeModelFactory(
        structured_responses=[ResearchQuestion(research_brief="brief")],
        chat_responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ConductResearch",
                        "args": {"research_topic": "topic"},
                        "id": "call-research",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "exa_search",
                        "args": {"query": "latest AI chip news"},
                        "id": "call-search",
                    }
                ],
            ),
            AIMessage(content="done", tool_calls=[]),
            AIMessage(content="compressed research"),
            AIMessage(
                content="",
                tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "call-complete"}],
            ),
            AIMessage(content="Final streamed report"),
        ],
    )
    tool = FakeExaTool(
        output=(
            "Result 1:\n"
            "Title: Chip launch\n"
            "URL: https://example.com/chip\n"
            "Published: 2026-03-20\n"
            "Highlights:\n"
            "- Fast"
        )
    )

    parts = asyncio.run(
        _collect_stream(
            monkeypatch,
            model_factory,
            config={"configurable": {"thread_id": "thread-2", "allow_clarification": False, "search_api": "exa"}},
            tool=tool,
        )
    )

    custom_parts = [part for part in parts if part["type"] == "custom"]
    custom_types = [part["data"]["type"] for part in custom_parts]
    message_parts = [part for part in parts if part["type"] == "messages"]

    assert "tool-input-start" in custom_types
    assert "tool-input-available" in custom_types
    assert "tool-output-available" in custom_types
    assert "source-url" in custom_types
    assert any(part["ns"] for part in custom_parts if part["data"].get("node") == "researcher_tools")
    assert any(message.content == "Final streamed report" for message, _ in [part["data"] for part in message_parts])
    assert all(part["data"]["type"] != "text-delta" for part in custom_parts)


def test_astream_protocol_emits_error_part_for_tool_failures(monkeypatch):
    model_factory = FakeModelFactory(
        structured_responses=[ResearchQuestion(research_brief="brief")],
        chat_responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ConductResearch",
                        "args": {"research_topic": "topic"},
                        "id": "call-research",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "exa_search",
                        "args": {"query": "latest AI chip news"},
                        "id": "call-search",
                    }
                ],
            ),
            AIMessage(content="done", tool_calls=[]),
            AIMessage(content="compressed research"),
            AIMessage(
                content="",
                tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "call-complete"}],
            ),
            AIMessage(content="Final streamed report"),
        ],
    )

    class FailingTool:
        name = "exa_search"

        async def ainvoke(self, args, config):
            raise RuntimeError("boom")

    parts = asyncio.run(
        _collect_stream(
            monkeypatch,
            model_factory,
            config={"configurable": {"thread_id": "thread-3", "allow_clarification": False, "search_api": "exa"}},
            tool=FailingTool(),
        )
    )

    error_parts = [
        part["data"]
        for part in parts
        if part["type"] == "custom" and part["data"]["type"] == "error"
    ]

    assert any(part["toolCallId"] == "call-search" for part in error_parts)
    assert any("boom" in part["errorText"] for part in error_parts)


def test_extract_source_parts_dedupes_urls_and_documents():
    parts = extract_source_parts(
        {
            "results": [
                {"url": "https://example.com/a", "title": "A"},
                {"url": "https://example.com/a", "title": "A duplicate"},
                {"source_id": "doc-1", "title": "Doc", "media_type": "file"},
            ]
        },
        seen_source_ids={"https://example.com/existing"},
    )

    assert parts == [
        {
            "type": "source-url",
            "sourceId": "https://example.com/a",
            "url": "https://example.com/a",
            "title": "A",
        },
        {
            "type": "source-document",
            "sourceId": "doc-1",
            "mediaType": "file",
            "title": "Doc",
        },
    ]
