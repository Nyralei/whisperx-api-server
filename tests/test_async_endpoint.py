"""Async submission (async=true) and the GET .../result fetch endpoint."""

import json

import httpx
import pytest

import whisperx_api_server.s3_client as s3_client
import whisperx_api_server.transcriber as transcriber

pytestmark = pytest.mark.anyio

CANNED = {
    "text": "hello world",
    "language": "en",
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
}


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _patch_get_result(monkeypatch, value):
    async def fake(job_id):
        return value

    monkeypatch.setattr(s3_client, "get_result", fake)


async def test_async_in_direct_mode_400(make_app):
    async with _client(make_app(MODE="direct")) as c:
        resp = await c.post(
            "/v1/audio/transcriptions",
            data={"audio_url": "http://example.com/a.wav", "async": "true"},
        )
    assert resp.status_code == 400
    assert "kafka" in resp.json()["detail"].lower()


async def test_async_submit_returns_202(make_app, monkeypatch):
    record: dict = {}

    async def fake_submit(**kwargs):
        record.update(kwargs)

    monkeypatch.setattr(transcriber, "submit_kafka_job", fake_submit)

    async with _client(make_app(MODE="kafka")) as c:
        resp = await c.post(
            "/v1/audio/transcriptions",
            data={"audio_url": "http://example.com/a.wav", "async": "true"},
            headers={"X-Request-ID": "async-job-1"},
        )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["request_id"] == "async-job-1"
    assert body["status_url"].endswith("/async-job-1/status")
    assert body["result_url"].endswith("/async-job-1/result")
    assert record["request_id"] == "async-job-1"
    assert record["source_url"] == "http://example.com/a.wav"


async def test_result_pending_404(make_app, monkeypatch):
    _patch_get_result(monkeypatch, None)
    async with _client(make_app(MODE="kafka")) as c:
        resp = await c.get("/v1/audio/transcriptions/job-x/result")
    assert resp.status_code == 404
    assert "pending" in resp.json()["detail"].lower()


async def test_result_success_json(make_app, monkeypatch):
    _patch_get_result(
        monkeypatch, json.dumps({"job_id": "job-ok", "status": "ok", "result": CANNED})
    )
    async with _client(make_app(MODE="kafka")) as c:
        resp = await c.get(
            "/v1/audio/transcriptions/job-ok/result", params={"response_format": "json"}
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"text": "hello world"}


async def test_result_success_verbose_json(make_app, monkeypatch):
    _patch_get_result(
        monkeypatch, json.dumps({"job_id": "job-ok", "status": "ok", "result": CANNED})
    )
    async with _client(make_app(MODE="kafka")) as c:
        resp = await c.get(
            "/v1/audio/transcriptions/job-ok/result",
            params={"response_format": "verbose_json"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["segments"][0]["text"] == "hello world"


@pytest.mark.parametrize(
    "error_type,expected",
    [
        ("InvalidAudioError", 422),
        ("UploadTooLargeError", 413),
        ("TimeoutError", 504),
        ("RuntimeError", 500),
        ("SomethingUnknown", 500),
    ],
)
async def test_result_error_envelope_mapped(
    make_app, monkeypatch, error_type, expected
):
    _patch_get_result(
        monkeypatch,
        json.dumps(
            {
                "job_id": "job-err",
                "status": "error",
                "error": "boom",
                "error_type": error_type,
            }
        ),
    )
    async with _client(make_app(MODE="kafka")) as c:
        resp = await c.get("/v1/audio/transcriptions/job-err/result")
    assert resp.status_code == expected, resp.text


async def test_result_corrupt_envelope_500(make_app, monkeypatch):
    _patch_get_result(monkeypatch, "not json {")
    async with _client(make_app(MODE="kafka")) as c:
        resp = await c.get("/v1/audio/transcriptions/job-bad/result")
    assert resp.status_code == 500


async def test_result_invalid_id_400(make_app):
    async with _client(make_app(MODE="kafka")) as c:
        resp = await c.get("/v1/audio/transcriptions/bad@id/result")
    assert resp.status_code == 400


async def test_result_direct_mode_404(make_app):
    async with _client(make_app(MODE="direct")) as c:
        resp = await c.get("/v1/audio/transcriptions/job-x/result")
    assert resp.status_code == 404
    assert "kafka" in resp.json()["detail"].lower()
