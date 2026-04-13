"""FastAPI server exposing the deep_researcher LangGraph as an HTTP API."""

import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

from open_deep_research.deep_researcher import deep_researcher  # noqa: E402
from open_deep_research.sse import graph_sse_stream  # noqa: E402

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

def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _lc_messages(messages: list[Message]) -> list:
    return [
        AIMessage(content=m.content) if m.role == "ai" else HumanMessage(content=m.content)
        for m in messages
    ]


# ─── routes ───────────────────────────────────────────────────────────────────

@app.post("/research", response_model=ResearchResponse, summary="Run deep research (blocking)")
async def research(req: ResearchRequest) -> ResearchResponse:
    """Invoke the graph and wait for the final report."""
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        result = await deep_researcher.ainvoke(
            {"messages": _lc_messages(req.messages)},
            stream_mode=["custom", "messages"],
            version="v2",
            stream_subgraphs=True,
            config=_config(thread_id),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ResearchResponse(thread_id=thread_id, final_report=result.get("final_report", ""))


@app.post("/research/stream", summary="Run deep research with LangGraph-compatible SSE")
async def research_stream(req: ResearchRequest) -> StreamingResponse:
    """Stream graph events as Server-Sent Events in LangGraph API wire format."""
    thread_id = req.thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    return StreamingResponse(
        graph_sse_stream(
            graph=deep_researcher,
            input={"messages": _lc_messages(req.messages)},
            run_id=run_id,
            config=_config(thread_id),
        ),
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
