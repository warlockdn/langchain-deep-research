import importlib
import sys
import types
from datetime import datetime, timezone

sys.modules.setdefault("aiohttp", types.SimpleNamespace(ClientSession=object))

utils = importlib.import_module("open_deep_research.utils")


class FakeStoreItem:
    def __init__(self, value, created_at):
        self.value = value
        self.created_at = created_at


class FakeStore:
    def __init__(self):
        self.items = {}

    async def aget(self, namespace, key):
        return self.items.get((namespace, key))

    async def aput(self, namespace, key, value):
        self.items[(namespace, key)] = FakeStoreItem(
            value=value,
            created_at=datetime.now(timezone.utc),
        )

    async def adelete(self, namespace, key):
        self.items.pop((namespace, key), None)


def test_token_storage_falls_back_to_thread_namespace_without_owner(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(utils, "get_store", lambda: store)

    config = {
        "configurable": {
            "thread_id": "thread-123",
        },
        "metadata": {},
    }
    tokens = {"access_token": "abc", "expires_in": 300}

    import asyncio

    asyncio.run(utils.set_tokens(config, tokens))
    stored = asyncio.run(utils.get_tokens(config))

    assert stored == tokens
    assert (("thread-123", "tokens"), "data") in store.items
