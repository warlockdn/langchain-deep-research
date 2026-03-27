"""Helpers for emitting AI SDK-compatible custom stream protocol parts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter
from typing_extensions import TypedDict

URL_PATTERN = re.compile(r"https?://[^\s>)]+")


class ProtocolPart(TypedDict, total=False):
    """Base protocol payload sent over LangGraph custom streaming."""

    type: str
    node: str
    ns: str
    phase: str
    timestamp: str
    run_id: str
    thread_id: str


class NodeStatusPart(ProtocolPart, total=False):
    """Node lifecycle update payload."""

    data: dict[str, Any]


class LogPart(ProtocolPart, total=False):
    """Structured log payload."""

    level: str
    message: str
    data: dict[str, Any]


class ToolInputStartPart(ProtocolPart, total=False):
    """Tool input start payload."""

    toolCallId: str
    toolName: str


class ToolInputDeltaPart(ProtocolPart, total=False):
    """Tool input delta payload."""

    toolCallId: str
    inputTextDelta: str


class ToolInputAvailablePart(ProtocolPart, total=False):
    """Tool input ready payload."""

    toolCallId: str
    toolName: str
    input: Any


class ToolOutputAvailablePart(ProtocolPart, total=False):
    """Tool output ready payload."""

    toolCallId: str
    output: Any


class SourceUrlPart(ProtocolPart, total=False):
    """Source URL payload."""

    sourceId: str
    url: str
    title: str


class SourceDocumentPart(ProtocolPart, total=False):
    """Source document payload."""

    sourceId: str
    mediaType: str
    title: str
    url: str


class ErrorPart(ProtocolPart, total=False):
    """Error payload."""

    errorText: str
    toolCallId: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_protocol_context(config: RunnableConfig | None) -> dict[str, Any]:
    if config is None:
        raise ValueError("RunnableConfig is required for stream protocol context.")

    configurable = config.setdefault("configurable", {})
    protocol = configurable.setdefault("__stream_protocol__", {})
    protocol.setdefault("run_id", configurable.get("thread_id") or str(uuid4()))
    protocol.setdefault("message_id", f"msg_{uuid4().hex}")
    protocol.setdefault("source_ids", set())
    return protocol


def _stringify_namespace(config: RunnableConfig | None) -> str:
    if not config:
        return ""

    configurable = config.get("configurable", {})
    metadata = config.get("metadata", {})
    namespace = (
        configurable.get("checkpoint_ns")
        or configurable.get("langgraph_checkpoint_ns")
        or metadata.get("checkpoint_ns")
        or metadata.get("langgraph_checkpoint_ns")
        or ""
    )
    return str(namespace)


def _base_part(
    config: RunnableConfig | None,
    *,
    part_type: str,
    node: str,
    phase: str,
) -> ProtocolPart:
    protocol = _ensure_protocol_context(config) if config is not None else {}
    configurable = config.get("configurable", {}) if config else {}
    return {
        "type": part_type,
        "node": node,
        "ns": _stringify_namespace(config),
        "phase": phase,
        "timestamp": _now_iso(),
        "run_id": protocol.get("run_id", ""),
        "thread_id": configurable.get("thread_id", ""),
    }


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def build_run_part(config: RunnableConfig, *, part_type: str) -> ProtocolPart:
    """Build a run-scoped protocol part for wrapper-owned start/finish events."""
    protocol = _ensure_protocol_context(config)
    part = _base_part(config, part_type=part_type, node="graph", phase="run")
    if part_type == "start":
        part["messageId"] = protocol["message_id"]
    return part


def extract_source_parts(
    payload: Any,
    *,
    seen_source_ids: set[str] | None = None,
) -> list[SourceUrlPart | SourceDocumentPart]:
    """Extract AI SDK-compatible source parts from tool outputs."""
    seen = seen_source_ids if seen_source_ids is not None else set()
    parts: list[SourceUrlPart | SourceDocumentPart] = []

    def add_url(url: str, title: str | None = None):
        if url in seen:
            return
        seen.add(url)
        part: SourceUrlPart = {
            "type": "source-url",
            "sourceId": url,
            "url": url,
        }
        if title:
            part["title"] = title
        parts.append(part)

    def add_document(source_id: str, media_type: str, title: str | None = None, url: str | None = None):
        if source_id in seen:
            return
        seen.add(source_id)
        part: SourceDocumentPart = {
            "type": "source-document",
            "sourceId": source_id,
            "mediaType": media_type,
        }
        if title:
            part["title"] = title
        if url:
            part["url"] = url
        parts.append(part)

    def walk(value: Any):
        if value is None:
            return

        if isinstance(value, str):
            lines = value.splitlines()
            current_title: str | None = None
            for line in lines:
                if line.startswith("Title: "):
                    current_title = line.removeprefix("Title: ").strip()
                elif line.startswith("URL: "):
                    add_url(line.removeprefix("URL: ").strip(), current_title)

            for url in URL_PATTERN.findall(value):
                add_url(url)
            return

        if isinstance(value, dict):
            url = value.get("url")
            title = value.get("title")
            source_id = value.get("source_id") or value.get("sourceId")
            media_type = value.get("media_type") or value.get("mediaType")

            if isinstance(url, str):
                add_url(url, title if isinstance(title, str) else None)

            if source_id and media_type:
                add_document(
                    str(source_id),
                    str(media_type),
                    title if isinstance(title, str) else None,
                    url if isinstance(url, str) else None,
                )

            for item in value.values():
                walk(item)
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)

    walk(payload)
    return parts


class NodeStreamEmitter:
    """Emit node-scoped custom stream payloads with stable metadata."""

    def __init__(self, writer: StreamWriter | None, *, node: str, config: RunnableConfig):
        """Initialize a node-scoped emitter with a real writer or a no-op fallback."""
        self.writer = writer or (lambda payload: None)
        self.node = node
        self.config = config
        self.protocol = _ensure_protocol_context(config)

    def emit(self, payload: ProtocolPart):
        """Write a payload into LangGraph custom streaming."""
        self.writer(payload)

    def start(self):
        """Emit the standard node-start step and status parts."""
        self.emit(_base_part(self.config, part_type="start-step", node=self.node, phase="node_start"))
        self.status("node_start")

    def finish(self):
        """Emit the standard node-end step and status parts."""
        self.emit(_base_part(self.config, part_type="finish-step", node=self.node, phase="node_end"))
        self.status("node_end")

    def status(self, phase: str, **data: Any):
        """Emit a structured node status part."""
        payload: NodeStatusPart = _base_part(
            self.config,
            part_type="data-node-status",
            node=self.node,
            phase=phase,
        )
        if data:
            payload["data"] = _safe_json(data)
        self.emit(payload)

    def log(self, phase: str, message: str, *, level: str = "info", **data: Any):
        """Emit a structured log part."""
        payload: LogPart = _base_part(
            self.config,
            part_type="data-log",
            node=self.node,
            phase=phase,
        )
        payload["level"] = level
        payload["message"] = message
        if data:
            payload["data"] = _safe_json(data)
        self.emit(payload)

    def error(self, phase: str, error: Exception | str, *, tool_call_id: str | None = None):
        """Emit an error part scoped to the current node."""
        payload: ErrorPart = _base_part(
            self.config,
            part_type="error",
            node=self.node,
            phase=phase,
        )
        payload["errorText"] = str(error)
        if tool_call_id:
            payload["toolCallId"] = tool_call_id
        self.emit(payload)

    def tool_input_start(self, tool_call_id: str, tool_name: str):
        """Emit the beginning of tool input generation."""
        payload: ToolInputStartPart = _base_part(
            self.config,
            part_type="tool-input-start",
            node=self.node,
            phase="tool_input_start",
        )
        payload["toolCallId"] = tool_call_id
        payload["toolName"] = tool_name
        self.emit(payload)

    def tool_input_delta(self, tool_call_id: str, delta: str):
        """Emit a best-effort tool input delta."""
        payload: ToolInputDeltaPart = _base_part(
            self.config,
            part_type="tool-input-delta",
            node=self.node,
            phase="tool_input_delta",
        )
        payload["toolCallId"] = tool_call_id
        payload["inputTextDelta"] = delta
        self.emit(payload)

    def tool_input_available(self, tool_call_id: str, tool_name: str, tool_input: Any):
        """Emit the finalized tool input before execution."""
        payload: ToolInputAvailablePart = _base_part(
            self.config,
            part_type="tool-input-available",
            node=self.node,
            phase="tool_input_available",
        )
        payload["toolCallId"] = tool_call_id
        payload["toolName"] = tool_name
        payload["input"] = _safe_json(tool_input)
        self.emit(payload)

    def tool_output_available(self, tool_call_id: str, output: Any):
        """Emit the tool result payload."""
        payload: ToolOutputAvailablePart = _base_part(
            self.config,
            part_type="tool-output-available",
            node=self.node,
            phase="tool_output_available",
        )
        payload["toolCallId"] = tool_call_id
        payload["output"] = _safe_json(output)
        self.emit(payload)

    def sources_from_output(self, output: Any):
        """Extract and emit deduplicated source parts from a tool output."""
        for part in extract_source_parts(output, seen_source_ids=self.protocol["source_ids"]):
            payload = _base_part(
                self.config,
                part_type=part["type"],
                node=self.node,
                phase="source_discovered",
            )
            payload.update(part)
            self.emit(payload)


async def astream_deep_researcher_protocol(
    inputs: dict[str, Any],
    *,
    config: RunnableConfig | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream the graph with `messages` and AI SDK-compatible `custom` parts."""
    from open_deep_research.deep_researcher import deep_researcher

    stream_config: RunnableConfig = config or {"configurable": {}}
    _ensure_protocol_context(stream_config)

    yield {"type": "custom", "ns": (), "data": build_run_part(stream_config, part_type="start")}
    try:
        async for part in deep_researcher.astream(
            inputs,
            config=stream_config,
            stream_mode=["messages", "custom"],
            subgraphs=True,
            version="v2",
        ):
            yield part
    except Exception as exc:
        error_part = _base_part(stream_config, part_type="error", node="graph", phase="run_error")
        error_part["errorText"] = str(exc)
        yield {"type": "custom", "ns": (), "data": error_part}
        abort_part = _base_part(stream_config, part_type="abort", node="graph", phase="run_abort")
        abort_part["reason"] = str(exc)
        yield {"type": "custom", "ns": (), "data": abort_part}
        raise
    else:
        yield {"type": "custom", "ns": (), "data": build_run_part(stream_config, part_type="finish")}
