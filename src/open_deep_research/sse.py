"""SSE utilities matching the LangGraph API wire format.

Replicates langgraph_api/sse.py (json_to_sse) and the stream ID generation
from langgraph_runtime_inmem/inmem_stream.py (_generate_ms_seq_id).

Frame layout (CRLF separators)::

    event: <name>
    data: <json>
    id: <ms>-0

"""

import time
from typing import Any

import orjson

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
            return o.model_dump()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return orjson.dumps(obj, default=_default)
