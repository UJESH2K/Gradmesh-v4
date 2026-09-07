"""In-process event bus used for the dashboard's live stream.

The v3 dashboard polled /nodes and /metrics on a timer, which made the UI feel a
second behind reality and put a constant load on the coordinator. v4 pushes
instead: every state transition publishes an event, and the Next.js dashboard
holds one Server-Sent Events connection per browser tab.

The bus keeps a bounded backlog so a tab that connects mid-run immediately
receives recent history rather than an empty screen.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any, AsyncIterator, Deque, Dict, List, Set

BACKLOG_SIZE = 400
QUEUE_SIZE = 256


class EventBus:
    def __init__(self) -> None:
        self._backlog: Deque[dict] = deque(maxlen=BACKLOG_SIZE)
        self._subscribers: Set[asyncio.Queue] = set()
        self._sequence = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the serving loop so worker threads can publish safely."""
        self._loop = loop

    def publish(self, kind: str, payload: Dict[str, Any] | None = None) -> dict:
        self._sequence += 1
        event = {
            "id": self._sequence,
            "kind": kind,
            "at": time.time(),
            "data": payload or {},
        }
        self._backlog.append(event)

        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled tab must never block the coordinator. Drop the tab's
                # oldest event and keep going; it will resync from /api state.
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except Exception:
                    pass
        return event

    def publish_threadsafe(self, kind: str, payload: Dict[str, Any] | None = None) -> None:
        """Publish from a non-async context, such as the aggregation thread."""
        if self._loop is None or self._loop.is_closed():
            self.publish(kind, payload)
            return
        self._loop.call_soon_threadsafe(self.publish, kind, payload)

    def recent(self, limit: int = 60) -> List[dict]:
        return list(self._backlog)[-limit:]

    async def subscribe(self, replay: int = 40) -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._subscribers.add(queue)
        try:
            for event in self.recent(replay):
                yield _format_sse(event)
            yield _format_sse({"id": 0, "kind": "stream.ready", "at": time.time(), "data": {}})

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Comment frame: keeps proxies and browsers from timing out.
                    yield ": keepalive\n\n"
                    continue
                yield _format_sse(event)
        finally:
            self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


def _format_sse(event: dict) -> str:
    return "event: %s\ndata: %s\n\n" % (event["kind"], json.dumps(event))


bus = EventBus()
