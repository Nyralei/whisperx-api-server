import shutil

import httpx
import pytest

pytestmark = pytest.mark.anyio

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


@requires_ffmpeg
async def test_transcription_with_fake_backend(client, sine_wav):
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", sine_wav, "audio/wav")},
        data={"align": "false", "response_format": "json"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"text": "hello world"}


async def test_invalid_api_key_rejected(make_app):
    transport = httpx.ASGITransport(app=make_app(API_KEY="sekret"))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer wrong"},
            data={"audio_url": "http://example.com/a.wav"},
        )
        assert resp.status_code == 403

        # HTTPBearer(auto_error=True) rejects a missing header with 401.
        resp = await c.post(
            "/v1/audio/transcriptions",
            data={"audio_url": "http://example.com/a.wav"},
        )
        assert resp.status_code == 401


async def test_status_unknown_id_404(client):
    resp = await client.get("/v1/audio/transcriptions/nope-123/status")
    assert resp.status_code == 404
