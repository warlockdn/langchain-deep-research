"""SSE utilities matching the LangGraph API wire format.

Replicates langgraph_api/sse.py (json_to_sse) and the stream ID generation
from langgraph_runtime_inmem/inmem_stream.py (_generate_ms_seq_id).

Frame layout (CRLF separators)::

    event: <name>
    data: <json>
    id: <ms>-0

"""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import orjson
from langchain_core.messages import AIMessageChunk

_SEP = b"\r\n"


def ms_stream_id() -> bytes:
    """Redis-style millisecond-sequence ID, e.g. ``1775837488959-0``."""
    return f"{int(time.time() * 1000)}-0".encode()


def sse_frame(event: str, data: bytes, stream_id: bytes | None = None) -> bytes:
    """Encode one SSE event exactly as langgraph_api/sse.py ``json_to_sse`` does."""
    parts: list[bytes] = [b"event: ", event.encode(), _SEP, b"data: ", data, _SEP]
    if stream_id is not None:
        parts += [b"id: ", stream_id, _SEP]
    parts.append(_SEP)
    return b"".join(parts)


def json_bytes(obj: Any) -> bytes:
    """Serialise to JSON bytes, handling pydantic/LangChain models via ``model_dump``."""
    def _default(o: Any) -> Any:
        if hasattr(o, "model_dump") and callable(o.model_dump):
            # mode="json" coerces UUIDs, datetimes, etc. to JSON-native types
            return o.model_dump(mode="json")
        if hasattr(o, "__str__"):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return orjson.dumps(obj, default=_default)


async def graph_sse_stream(
    graph: Any,
    input: dict[str, Any],
    run_id: str,
    config: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Yield SSE bytes matching the LangGraph API wire format.

    Emits a ``metadata`` event first, then for each message in the graph:
    - ``messages/metadata`` with ``{msg_id: metadata_dict}`` (once per message id)
    - ``messages/partial`` with ``[AIMessageChunk]`` for token-by-token streaming
    - ``messages/complete`` with ``[msg_dict]`` for finalized messages
    - ``custom`` for any custom stream events
    """
    yield sse_frame("metadata", json_bytes({"run_id": run_id, "attempt": 1}), ms_stream_id())

    seen_msg_ids: set[str] = set()

    async for ns, mode, chunk in graph.astream(
        input,
        config=config,
        stream_mode=["messages", "custom"],
        subgraphs=True,
    ):
        if mode == "messages":
            msg, metadata = chunk
            msg_id: str = msg.id or str(uuid.uuid4())

            if msg_id not in seen_msg_ids:
                seen_msg_ids.add(msg_id)
                yield sse_frame("messages/metadata", json_bytes({msg_id: metadata}), ms_stream_id())

            event_name = "messages/partial" if isinstance(msg, AIMessageChunk) else "messages/complete"
            yield sse_frame(event_name, json_bytes([msg]), ms_stream_id())

        elif mode == "custom":
            yield sse_frame("custom", json_bytes(chunk), ms_stream_id())
