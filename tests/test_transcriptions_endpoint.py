"""Transcription endpoint: response formats, model aliasing, and error mapping."""

import asyncio
import json

import httpx
import pytest

import whisperx_api_server.transcriber as transcriber
from whisperx_api_server import webhook
from whisperx_api_server.routers.transcriptions import _CUDA_OOM_EXC
from whisperx_api_server.transcriber import (
    InvalidAudioError,
    QueueFullError,
    UploadTooLargeError,
)

pytestmark = pytest.mark.anyio

CANNED = {
    "text": "hello world",
    "language": "en",
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
}

try:
    import whisperx.utils  # noqa: F401

    HAS_WHISPERX = True
except Exception:
    HAS_WHISPERX = False

requires_whisperx = pytest.mark.skipif(
    not HAS_WHISPERX, reason="whisperx (ML extras) not installed"
)


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _patch_transcribe(monkeypatch, *, returns=None, raises=None, record=None):
    async def fake(**kwargs):
        if record is not None:
            record.update(kwargs)
        if raises is not None:
            raise raises
        return returns

    monkeypatch.setattr(transcriber, "transcribe", fake)


async def _post(client, **data):
    return await client.post("/v1/audio/transcriptions", data=data)


async def test_json_format_200(make_app, monkeypatch):
    _patch_transcribe(monkeypatch, returns=dict(CANNED))
    async with _client(make_app()) as c:
        resp = await _post(
            c,
            audio_url="http://example.com/a.wav",
            response_format="json",
            align="false",
        )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"text": "hello world"}


async def test_verbose_json_format_200(make_app, monkeypatch):
    _patch_transcribe(monkeypatch, returns=dict(CANNED))
    async with _client(make_app()) as c:
        resp = await _post(
            c,
            audio_url="http://example.com/a.wav",
            response_format="verbose_json",
            align="false",
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["segments"][0]["text"] == "hello world"


async def test_text_format_200(make_app, monkeypatch):
    _patch_transcribe(monkeypatch, returns=dict(CANNED))
    async with _client(make_app()) as c:
        resp = await _post(
            c,
            audio_url="http://example.com/a.wav",
            response_format="text",
            align="false",
        )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "hello world"


@requires_whisperx
@pytest.mark.parametrize(
    "fmt,content_type,needle",
    [
        ("srt", "text/plain", "-->"),
        ("vtt", "text/vtt", "WEBVTT"),
        ("aud", "text/plain", "hello world"),
    ],
)
async def test_subtitle_formats_200(make_app, monkeypatch, fmt, content_type, needle):
    _patch_transcribe(monkeypatch, returns=dict(CANNED))
    async with _client(make_app()) as c:
        resp = await _post(c, audio_url="http://example.com/a.wav", response_format=fmt)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith(content_type)
    assert needle in resp.text


async def test_model_placeholder_aliased_to_none(make_app, monkeypatch):
    record: dict = {}
    _patch_transcribe(monkeypatch, returns=dict(CANNED), record=record)
    async with _client(make_app()) as c:
        await _post(
            c,
            audio_url="http://example.com/a.wav",
            model="whisper-1",
            align="false",
        )
    assert record["model_name"] is None


async def test_explicit_model_passed_through(make_app, monkeypatch):
    record: dict = {}
    _patch_transcribe(monkeypatch, returns=dict(CANNED), record=record)
    async with _client(make_app()) as c:
        await _post(
            c,
            audio_url="http://example.com/a.wav",
            model="custom-model",
            align="false",
        )
    assert record["model_name"] == "custom-model"


@pytest.mark.parametrize(
    "exc,expected",
    [
        (InvalidAudioError("bad audio"), 422),
        (UploadTooLargeError("too big"), 413),
        (QueueFullError("busy"), 503),
        (TimeoutError("slow"), 504),
        (asyncio.TimeoutError(), 504),
        (_CUDA_OOM_EXC("CUDA out of memory"), 503),
    ],
)
async def test_error_mapping(make_app, monkeypatch, exc, expected):
    _patch_transcribe(monkeypatch, raises=exc)
    async with _client(make_app()) as c:
        resp = await _post(c, audio_url="http://example.com/a.wav", align="false")
    assert resp.status_code == expected, resp.text


async def test_missing_file_and_url_422(make_app):
    async with _client(make_app()) as c:
        resp = await _post(c, response_format="json")
    assert resp.status_code == 422
    assert "either" in resp.json()["detail"].lower()


async def test_both_file_and_url_422(make_app):
    async with _client(make_app()) as c:
        resp = await c.post(
            "/v1/audio/transcriptions",
            data={"audio_url": "http://example.com/a.wav"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )
    assert resp.status_code == 422
    assert "exactly one" in resp.json()["detail"].lower()


async def test_subtitle_without_alignment_422(make_app):
    async with _client(make_app()) as c:
        resp = await _post(
            c,
            audio_url="http://example.com/a.wav",
            response_format="srt",
            align="false",
        )
    assert resp.status_code == 422
    assert "alignment" in resp.json()["detail"].lower()


async def test_direct_mode_success_fires_completion_webhook(make_app, monkeypatch):
    _patch_transcribe(monkeypatch, returns=dict(CANNED))
    calls: list = []

    async def _spy(url, envelope, config):
        calls.append((url, envelope))
        return True

    monkeypatch.setattr(webhook, "deliver_result", _spy)
    app = make_app(URL_FETCH_ALLOW_PRIVATE_HOSTS="true")
    async with _client(app) as c:
        resp = await _post(
            c,
            audio_url="http://example.com/a.wav",
            response_format="json",
            align="false",
            callback_url="http://hook.example/done",
        )
    assert resp.status_code == 200, resp.text
    # Background task runs before ASGITransport returns the response.
    assert len(calls) == 1
    url, envelope = calls[0]
    assert url == "http://hook.example/done"
    env = json.loads(envelope)
    assert env["status"] == "ok"
    assert env["job_id"]
    assert env["result"]["text"] == "hello world"


async def test_callback_url_ssrf_rejected_422(make_app, monkeypatch):
    _patch_transcribe(monkeypatch, returns=dict(CANNED))
    async with _client(make_app()) as c:
        resp = await _post(
            c,
            audio_url="http://example.com/a.wav",
            align="false",
            callback_url="http://127.0.0.1:9/hook",
        )
    assert resp.status_code == 422, resp.text
    assert "callback_url" in resp.json()["detail"].lower()


def test_json_canned_is_serializable():
    # Guards against an accidentally non-JSON canned payload in this module.
    json.dumps(CANNED)
