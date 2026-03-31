import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from open_deep_research import deep_researcher
from open_deep_research.state import ClarifyWithUser, ResearchQuestion


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.retry_attempts = None

    def with_retry(self, *, stop_after_attempt):
        self.retry_attempts = stop_after_attempt
        return self

    async def ainvoke(self, messages):
        return self.response


class FakeModelWithStructuredOutput:
    def __init__(self, response):
        self.response = response
        self.include_raw = None

    def with_structured_output(self, schema, include_raw=False):
        self.include_raw = include_raw
        return FakeStructuredModel(self.response)


class FakeChatModel:
    def __init__(self, response):
        self.response = response

    async def ainvoke(self, messages):
        return self.response


def _config(**overrides):
    configurable = {
        "allow_clarification": True,
        "research_model": "openai:gpt-5.4-mini",
        "research_model_max_tokens": 1024,
        "compression_model": "openai:gpt-5.4-mini",
        "compression_model_max_tokens": 1024,
        "max_structured_output_retries": 2,
        "apiKeys": {"OPENAI_API_KEY": "test-key"},
    }
    configurable.update(overrides)
    return {"configurable": configurable}


def test_clarify_with_user_preserves_raw_llm_message_for_token_tracing(monkeypatch):
    raw_message = AIMessage(
        content="",
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )
    response = {
        "raw": raw_message,
        "parsed": ClarifyWithUser(
            need_clarification=True,
            question="Which market?",
            verification="Starting research.",
        ),
        "parsing_error": None,
    }
    fake_model = FakeModelWithStructuredOutput(response)

    monkeypatch.setattr(deep_researcher, "build_chat_model", lambda *args, **kwargs: fake_model)

    result = asyncio.run(
        deep_researcher.clarify_with_user(
            {"messages": [HumanMessage(content="Research AI chips")]},
            _config(),
        )
    )

    assert fake_model.include_raw is True
    assert result.update["messages"] == [AIMessage(content="Which market?")]
    assert result.update["trace_messages"] == {
        "type": "override",
        "value": [raw_message],
    }


def test_write_research_brief_preserves_raw_llm_message_for_token_tracing(monkeypatch):
    raw_message = AIMessage(
        content="",
        usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
    )
    response = {
        "raw": raw_message,
        "parsed": ResearchQuestion(research_brief="Compare AI chip roadmaps."),
        "parsing_error": None,
    }
    fake_model = FakeModelWithStructuredOutput(response)

    monkeypatch.setattr(deep_researcher, "build_chat_model", lambda *args, **kwargs: fake_model)

    result = asyncio.run(
        deep_researcher.write_research_brief(
            {"messages": [HumanMessage(content="Compare AI chip roadmaps")]},
            _config(max_concurrent_research_units=2, max_researcher_iterations=3),
        )
    )

    assert fake_model.include_raw is True
    assert result.update["research_brief"] == "Compare AI chip roadmaps."
    assert result.update["trace_messages"] == {
        "type": "override",
        "value": [raw_message],
    }
    assert isinstance(result.update["supervisor_messages"]["value"][0], SystemMessage)
    assert result.update["supervisor_messages"]["value"][1] == HumanMessage(
        content="Compare AI chip roadmaps."
    )


def test_compress_research_preserves_raw_llm_message_for_token_tracing(monkeypatch):
    raw_message = AIMessage(
        content="Compressed summary",
        usage_metadata={"input_tokens": 20, "output_tokens": 6, "total_tokens": 26},
    )

    monkeypatch.setattr(
        deep_researcher,
        "build_chat_model",
        lambda *args, **kwargs: FakeChatModel(raw_message),
    )

    result = asyncio.run(
        deep_researcher.compress_research(
            {"researcher_messages": [AIMessage(content="Existing note")]},
            _config(),
        )
    )

    assert result["compressed_research"] == "Compressed summary"
    assert result["trace_messages"] == {
        "type": "override",
        "value": [raw_message],
    }
