import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from open_deep_research import deep_researcher
from open_deep_research.utils import get_notes_from_tool_calls


def _config(**overrides):
    configurable = {
        "allow_clarification": True,
        "research_model": "openai:gpt-5.4-mini",
        "research_model_max_tokens": 1024,
        "compression_model": "openai:gpt-5.4-mini",
        "compression_model_max_tokens": 1024,
        "final_report_model": "openai:gpt-5.4-mini",
        "final_report_model_max_tokens": 1024,
        "max_structured_output_retries": 2,
        "max_concurrent_research_units": 5,
        "max_researcher_iterations": 2,
        "apiKeys": {"OPENAI_API_KEY": "test-key"},
    }
    configurable.update(overrides)
    return {"configurable": configurable}


def _tool_call(name: str, tool_call_id: str, **args):
    return {
        "name": name,
        "id": tool_call_id,
        "args": args,
        "type": "tool_call",
    }


def test_get_notes_from_tool_calls_only_returns_conduct_research_results():
    messages = [
        ToolMessage(
            content="Reflection recorded: need another source",
            name="think_tool",
            tool_call_id="think-1",
        ),
        ToolMessage(
            content="Delegated findings with citations",
            name="ConductResearch",
            tool_call_id="research-1",
        ),
    ]

    assert get_notes_from_tool_calls(messages) == ["Delegated findings with citations"]


def test_supervisor_tools_stops_at_exact_iteration_limit():
    state = {
        "supervisor_messages": [
            ToolMessage(
                content="Reflection recorded: missing pricing page",
                name="think_tool",
                tool_call_id="think-1",
            ),
            ToolMessage(
                content="Delegated findings with citations",
                name="ConductResearch",
                tool_call_id="research-1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "ConductResearch",
                        "research-2",
                        research_topic="Research current pricing pages",
                    )
                ],
            ),
        ],
        "research_iterations": 2,
        "research_brief": "Compare pricing pages",
    }

    result = asyncio.run(deep_researcher.supervisor_tools(state, _config(max_researcher_iterations=2)))

    assert result.goto == END
    assert result.update["notes"] == ["Delegated findings with citations"]


def test_supervisor_tools_keeps_successful_results_when_one_child_fails(monkeypatch):
    class FakeResearcherSubgraph:
        async def ainvoke(self, payload, config):
            if payload["research_topic"] == "working topic":
                return {
                    "compressed_research": "Good delegated findings",
                    "raw_notes": ["Good raw note"],
                }
            raise RuntimeError("child exploded")

    monkeypatch.setattr(deep_researcher, "researcher_subgraph", FakeResearcherSubgraph())

    state = {
        "supervisor_messages": [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "ConductResearch",
                        "research-1",
                        research_topic="working topic",
                    ),
                    _tool_call(
                        "ConductResearch",
                        "research-2",
                        research_topic="failing topic",
                    ),
                ],
            )
        ],
        "research_iterations": 1,
        "research_brief": "Compare two topics",
    }

    result = asyncio.run(deep_researcher.supervisor_tools(state, _config()))

    assert result.goto == "supervisor"
    assert [message.content for message in result.update["supervisor_messages"]] == [
        "Good delegated findings",
        "Error running delegated research: child exploded",
    ]
    assert result.update["raw_notes"] == ["Good raw note"]


def test_final_report_generation_includes_compressed_findings_and_raw_evidence(monkeypatch):
    captured_prompt = {}

    class FakeChatModel:
        async def ainvoke(self, messages):
            captured_prompt["content"] = messages[0].content
            return AIMessage(content="Final report")

    monkeypatch.setattr(
        deep_researcher,
        "build_chat_model",
        lambda *args, **kwargs: FakeChatModel(),
    )

    result = asyncio.run(
        deep_researcher.final_report_generation(
            {
                "messages": [HumanMessage(content="Research this")],
                "research_brief": "Research this",
                "notes": ["Compressed finding A"],
                "raw_notes": ["Raw evidence B"],
            },
            _config(),
        )
    )

    assert result["final_report"] == "Final report"
    assert "Compressed findings:\nCompressed finding A" in captured_prompt["content"]
    assert "Raw evidence:\nRaw evidence B" in captured_prompt["content"]
    assert captured_prompt["content"].index("Compressed findings:") < captured_prompt["content"].index(
        "Raw evidence:"
    )
