"""FastAPI server exposing the deep_researcher LangGraph as an HTTP API."""

import uuid
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

from open_deep_research.deep_researcher import deep_researcher  # noqa: E402
from open_deep_research.sse import json_bytes, ms_stream_id, sse_frame  # noqa: E402

app = FastAPI(
    title="Open Deep Research API",
    description="HTTP API wrapping the deep_researcher LangGraph.",
    version="0.1.0",
)


# ─── models ───────────────────────────────────────────────────────────────────

class Message(BaseModel):  # noqa: D101
    role: str = Field(..., description="'user' or 'ai'")
    content: str


class ResearchRequest(BaseModel):
    """Payload for kicking off a research run."""

    messages: list[Message] = Field(..., description="Conversation messages in LangChain format.")
    thread_id: str | None = Field(
        default=None,
        description="Reuse an existing thread to continue a conversation. Omit to start fresh.",
    )


class ResearchResponse(BaseModel):
    """Final result returned by a completed research run."""

    thread_id: str
    final_report: str


# ─── helpers ──────────────────────────────────────────────────────────────────

def _build_config(thread_id: str, extra: dict[str, Any]) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id, **extra}}


def _to_lc_messages(messages: list[Message]) -> list:
    result = []
    for m in messages:
        if m.role == "ai":
            result.append(AIMessage(content=m.content))
        else:
            result.append(HumanMessage(content=m.content))
    return result


async def _graph_sse_stream(
    messages: list[Message],
    run_id: str,
    runnable_config: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Yield SSE bytes matching the LangGraph API wire format.

    Emits a ``metadata`` event first, then streams graph output events.
    Sub-graph events are prefixed: ``messages|<ns>``, ``values|<ns>``, etc.
    """
    yield sse_frame("metadata", json_bytes({"run_id": run_id, "attempt": 1}), ms_stream_id())

    async for event in deep_researcher.astream(
        {"messages": _to_lc_messages(messages)},
        config=runnable_config,
        stream_mode=["messages", "custom"],
        subgraphs=True,
    ):
        # subgraphs=True → each event is (ns_tuple, mode, chunk)
        ns, mode, chunk = event
        ns_str = "|".join(ns) if ns else ""
        event_name = f"{mode}|{ns_str}" if ns_str else mode
        yield sse_frame(event_name, json_bytes(chunk), ms_stream_id())


# ─── routes ───────────────────────────────────────────────────────────────────

@app.post("/research", response_model=ResearchResponse, summary="Run deep research (blocking)")
async def research(req: ResearchRequest) -> ResearchResponse:
    """Invoke the graph and wait for the final report."""
    thread_id = req.thread_id or str(uuid.uuid4())
    runnable_config = _build_config(thread_id, {})

    try:
        result = await deep_researcher.ainvoke(
            {"messages": _to_lc_messages(req.messages)},
            config=runnable_config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ResearchResponse(
        thread_id=thread_id,
        final_report=result.get("final_report", ""),
    )


@app.post("/research/stream", summary="Run deep research with LangGraph-compatible SSE")
async def research_stream(req: ResearchRequest) -> StreamingResponse:
    """Stream graph events as Server-Sent Events in LangGraph API wire format.

    Events: ``metadata`` (first), then ``values`` / ``messages`` / ``updates``
    and ``messages|<subgraph_ns>`` for sub-graph messages.
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    runnable_config = _build_config(thread_id, {})

    return StreamingResponse(
        _graph_sse_stream(req.messages, run_id, runnable_config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": thread_id,
            "X-Run-Id": run_id,
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
