"""Direct-mode on-disk result store: unit behavior + GET .../result wiring."""

import os

import httpx
import pytest

from whisperx_api_server import result_store
from whisperx_api_server.dependencies import get_config

pytestmark = pytest.mark.anyio

RESULT = {"text": "hello world", "language": "en", "segments": []}


def _configure(monkeypatch, tmp_path, **env):
    for key, value in {"RESULT_STORE__DIR": str(tmp_path), **env}.items():
        monkeypatch.setenv(key, str(value))
    get_config.cache_clear()


def test_put_get_roundtrip(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    result_store.put("rid-1", RESULT)
    assert result_store.get("rid-1") == RESULT
    assert result_store.get("missing") is None


def test_disabled_is_noop(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, RESULT_STORE__ENABLED="false")
    result_store.put("rid-1", RESULT)
    assert result_store.get("rid-1") is None


def test_unsafe_id_ignored(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    result_store.put("../escape", RESULT)
    assert result_store.get("../escape") is None
    assert os.listdir(tmp_path) == []


def test_ttl_expiry_on_read(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, RESULT_STORE__TTL_SECONDS="300")
    result_store.put("rid-1", RESULT)
    path = tmp_path / "rid-1.json"
    old = os.path.getmtime(path) - 3600
    os.utime(path, (old, old))
    assert result_store.get("rid-1") is None
    assert not path.exists()


def test_evict_expired(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, RESULT_STORE__TTL_SECONDS="300")
    result_store.put("rid-1", RESULT)
    stored = os.path.getmtime(tmp_path / "rid-1.json")
    assert result_store.evict_expired(now=stored + 301) == 1
    assert result_store.get("rid-1") is None


def test_capacity_trims_oldest(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    for i in range(3):
        result_store.put(f"rid-{i}", RESULT)
        path = tmp_path / f"rid-{i}.json"
        os.utime(path, (1000 + i, 1000 + i))
    result_store._enforce_capacity(str(tmp_path), 2)
    remaining = sorted(os.listdir(tmp_path))
    assert remaining == ["rid-1.json", "rid-2.json"]


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def _run_direct(make_app, monkeypatch, tmp_path, **env):
    import whisperx_api_server.transcriber as transcriber

    async def fake(**kwargs):
        return dict(RESULT)

    monkeypatch.setattr(transcriber, "transcribe", fake)
    return make_app(RESULT_STORE__DIR=str(tmp_path), **env)


async def test_result_endpoint_serves_stored_result(make_app, monkeypatch, tmp_path):
    app = await _run_direct(make_app, monkeypatch, tmp_path)
    async with _client(app) as c:
        post = await c.post(
            "/v1/audio/transcriptions",
            data={"audio_url": "http://example.com/a.wav", "align": "false"},
            headers={"X-Request-ID": "stored-1"},
        )
        assert post.status_code == 200, post.text
        got = await c.get(
            "/v1/audio/transcriptions/stored-1/result?response_format=json"
        )
    assert got.status_code == 200, got.text
    assert got.json() == {"text": "hello world"}


async def test_result_endpoint_404_when_disabled(make_app, monkeypatch, tmp_path):
    app = await _run_direct(
        make_app, monkeypatch, tmp_path, RESULT_STORE__ENABLED="false"
    )
    async with _client(app) as c:
        await c.post(
            "/v1/audio/transcriptions",
            data={"audio_url": "http://example.com/a.wav", "align": "false"},
            headers={"X-Request-ID": "stored-2"},
        )
        got = await c.get("/v1/audio/transcriptions/stored-2/result")
    assert got.status_code == 404


async def test_result_endpoint_404_unknown(make_app, monkeypatch, tmp_path):
    app = await _run_direct(make_app, monkeypatch, tmp_path)
    async with _client(app) as c:
        got = await c.get("/v1/audio/transcriptions/never-seen/result")
    assert got.status_code == 404
