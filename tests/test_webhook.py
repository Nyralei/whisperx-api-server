"""Unit tests for completion-webhook delivery (no network)."""

from types import SimpleNamespace

import pytest

from whisperx_api_server import webhook

pytestmark = pytest.mark.anyio


class _Poster:
    def __init__(self, *, status=None, exc=None):
        self.status = status
        self.exc = exc
        self.calls = 0

    async def __call__(self, url, body, *, connect_timeout, total_timeout):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.status


async def _deliver(**kw):
    return await webhook.deliver_webhook(
        "http://example.test/hook",
        {"job_id": "j", "status": "ok"},
        connect_timeout=1.0,
        total_timeout=1.0,
        allow_private_hosts=True,
        allowed_hosts=None,
        **kw,
    )


async def test_delivers_and_returns_true_on_2xx(monkeypatch):
    poster = _Poster(status=204)
    monkeypatch.setattr(webhook, "_post_once", poster)
    assert await _deliver() is True
    assert poster.calls == 1


async def test_retries_once_then_gives_up_on_error(monkeypatch):
    poster = _Poster(exc=ConnectionError("boom"))
    monkeypatch.setattr(webhook, "_post_once", poster)
    assert await _deliver() is False
    assert poster.calls == 2  # initial attempt + one retry


async def test_non_2xx_status_retries_then_false(monkeypatch):
    poster = _Poster(status=500)
    monkeypatch.setattr(webhook, "_post_once", poster)
    assert await _deliver() is False
    assert poster.calls == 2


async def test_ssrf_rejected_url_never_posts(monkeypatch):
    poster = _Poster(status=200)
    monkeypatch.setattr(webhook, "_post_once", poster)
    result = await webhook.deliver_webhook(
        "http://127.0.0.1:9/hook",
        {"job_id": "j"},
        connect_timeout=1.0,
        total_timeout=1.0,
        allow_private_hosts=False,
        allowed_hosts=None,
    )
    assert result is False
    assert poster.calls == 0


async def test_deliver_result_pulls_policy_from_config(monkeypatch):
    captured: dict = {}

    async def _fake(url, payload, **kw):
        captured["url"] = url
        captured["payload"] = payload
        captured["kw"] = kw
        return True

    monkeypatch.setattr(webhook, "deliver_webhook", _fake)
    config = SimpleNamespace(
        url_fetch_connect_timeout_seconds=3.0,
        webhook_timeout_seconds=9.0,
        url_fetch_allow_private_hosts=True,
        url_fetch_allowed_hosts=["a.example"],
    )
    assert await webhook.deliver_result("http://x/hook", "envelope", config) is True
    assert captured["url"] == "http://x/hook"
    assert captured["payload"] == "envelope"
    assert captured["kw"]["connect_timeout"] == 3.0
    assert captured["kw"]["total_timeout"] == 9.0
    assert captured["kw"]["allow_private_hosts"] is True
    assert captured["kw"]["allowed_hosts"] == ["a.example"]
