"""Request-ID echo/sanitization and graceful-shutdown drain behavior."""

import asyncio
import re
import types
from typing import Any

import pytest

from whisperx_api_server.main import GracefulShutdownMiddleware

pytestmark = pytest.mark.anyio

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


async def test_request_id_echoed_when_valid(client):
    resp = await client.get("/healthcheck", headers={"X-Request-ID": "abc.def_ghi-123"})
    assert resp.headers["x-request-id"] == "abc.def_ghi-123"


async def test_request_id_replaced_when_invalid_chars(client):
    resp = await client.get("/healthcheck", headers={"X-Request-ID": "bad id"})
    assert resp.headers["x-request-id"] != "bad id"
    assert _UUID.match(resp.headers["x-request-id"])


async def test_request_id_replaced_when_overlong(client):
    resp = await client.get("/healthcheck", headers={"X-Request-ID": "x" * 200})
    assert _UUID.match(resp.headers["x-request-id"])


async def _ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def _drive(mw, scope):
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    return sent


def _start_message(sent):
    return next(m for m in sent if m["type"] == "http.response.start")


async def test_shutting_down_returns_503_with_retry_after():
    state = types.SimpleNamespace(shutting_down=True, inflight=0, drain_event=None)
    mw = GracefulShutdownMiddleware(_ok_app, state)
    sent = await _drive(mw, {"type": "http", "path": "/v1/audio/transcriptions"})

    start = _start_message(sent)
    assert start["status"] == 503
    headers = {k.lower(): v for k, v in start["headers"]}
    assert headers[b"retry-after"] == b"5"


@pytest.mark.parametrize("path", ["/healthcheck", "/metrics"])
async def test_health_and_metrics_pass_through_during_drain(path):
    state = types.SimpleNamespace(shutting_down=True, inflight=0, drain_event=None)
    mw = GracefulShutdownMiddleware(_ok_app, state)
    sent = await _drive(mw, {"type": "http", "path": path})

    assert _start_message(sent)["status"] == 200
    assert state.inflight == 0


async def test_inflight_tracked_and_drain_event_set():
    state = types.SimpleNamespace(
        shutting_down=False, inflight=0, drain_event=asyncio.Event()
    )
    mw = GracefulShutdownMiddleware(_ok_app, state)
    sent = await _drive(mw, {"type": "http", "path": "/v1/audio/transcriptions"})

    assert _start_message(sent)["status"] == 200
    assert state.inflight == 0
    assert state.drain_event.is_set()


async def test_non_http_scope_passes_through():
    seen: list[str] = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    async def noop(*args) -> Any:
        return None

    state = types.SimpleNamespace(shutting_down=True, inflight=0, drain_event=None)
    mw = GracefulShutdownMiddleware(inner, state)
    await mw({"type": "lifespan"}, noop, noop)
    assert seen == ["lifespan"]
