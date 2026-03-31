import importlib

from langgraph.constants import TAG_NOSTREAM

from open_deep_research import utils


def test_build_chat_model_uses_openai_endpoint_env(monkeypatch):
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv(
        "OPENAI_ENDPOINT",
        "https://deepa-min0ats1-eastus2.openai.azure.com/openai/v1/",
    )
    monkeypatch.setattr(utils, "ChatOpenAI", FakeChatOpenAI)

    utils.build_chat_model(
        "openai:gpt-5.4-mini",
        2048,
        "test-key",
        tags=["tagged"],
    )

    assert captured["model"] == "gpt-5.4-mini"
    assert captured["api_key"] == "test-key"
    assert captured["max_completion_tokens"] == 2048
    assert captured["base_url"] == "https://deepa-min0ats1-eastus2.openai.azure.com/openai/v1/"
    assert captured["tags"] == ["tagged"]


def test_build_chat_model_normalizes_legacy_nostream_tag_and_enables_stream_usage(
    monkeypatch,
):
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(utils, "ChatOpenAI", FakeChatOpenAI)

    utils.build_chat_model(
        "openai:gpt-5.4-mini",
        2048,
        "test-key",
        tags=["langsmith:nostream", "custom"],
    )

    assert captured["tags"] == [TAG_NOSTREAM, "custom"]
    assert captured["stream_usage"] is True


def test_configuration_defaults_follow_openai_model_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")

    configuration = importlib.import_module("open_deep_research.configuration")
    configuration = importlib.reload(configuration)

    config = configuration.Configuration()

    assert config.summarization_model == "openai:gpt-5.4-mini"
    assert config.research_model == "openai:gpt-5.4-mini"
    assert config.compression_model == "openai:gpt-5.4-mini"
    assert config.final_report_model == "openai:gpt-5.4-mini"
