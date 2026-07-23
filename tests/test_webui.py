"""Web UI static mount: off by default, opt-in via WEBUI__ENABLED, and inert
with respect to the API contract when on."""

import shutil

import httpx
import pytest

pytestmark = pytest.mark.anyio

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


@pytest.fixture
def webui_dist(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><title>whisperx webui</title>", encoding="utf-8"
    )
    (dist / "assets" / "app.js").write_text("console.log('webui')", encoding="utf-8")
    return dist


def webui_env(webui_dist) -> dict:
    return {"WEBUI__ENABLED": "true", "WEBUI__DIST_DIR": str(webui_dist)}


async def test_webui_disabled_by_default(client):
    assert (await client.get("/webui/")).status_code == 404
    assert (await client.get("/")).status_code == 404


async def test_webui_enabled_serves_assets(make_app, webui_dist):
    transport = httpx.ASGITransport(app=make_app(**webui_env(webui_dist)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/webui/")
        assert resp.status_code == 200
        assert "whisperx webui" in resp.text

        resp = await c.get("/webui/assets/app.js")
        assert resp.status_code == 200

        resp = await c.get("/")
        assert resp.status_code == 307
        assert resp.headers["location"] == "/webui/"


async def test_webui_enabled_without_build_fails_fast(make_app, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no built UI was found"):
        make_app(WEBUI__ENABLED="true", WEBUI__DIST_DIR=str(empty))


async def test_webui_static_assets_not_behind_auth(make_app, webui_dist):
    app = make_app(API_KEY="sekret", **webui_env(webui_dist))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        assert (await c.get("/webui/")).status_code == 200
        # The API itself still requires the key.
        resp = await c.post(
            "/v1/audio/transcriptions",
            data={"audio_url": "http://example.com/a.wav"},
        )
        assert resp.status_code == 401


async def test_api_contract_identical_with_webui_enabled(make_app, webui_dist):
    plain = make_app()
    with_ui = make_app(**webui_env(webui_dist))

    schemas = []
    for app in (plain, with_ui):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            health = await c.get("/healthcheck")
            assert health.status_code == 200
            assert health.json() == {"status": "healthy"}
            schemas.append((await c.get("/openapi.json")).json())

    assert schemas[0] == schemas[1]


@requires_ffmpeg
async def test_transcription_identical_with_webui_enabled(
    make_app, webui_dist, sine_wav
):
    responses = []
    for env in ({}, webui_env(webui_dist)):
        transport = httpx.ASGITransport(app=make_app(**env))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", sine_wav, "audio/wav")},
                data={"align": "false", "response_format": "json"},
            )
            responses.append(resp)

    assert responses[0].status_code == responses[1].status_code == 200
    assert responses[0].json() == responses[1].json()
    assert responses[0].headers["content-type"] == responses[1].headers["content-type"]
