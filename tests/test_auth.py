"""Auth-mode tests: disabled when unset, enforced when configured."""

import json

import httpx
import pytest

pytestmark = pytest.mark.anyio

STATUS_PATH = "/v1/audio/transcriptions/known-format-id/status"


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_disabled_allows_requests_without_credentials(make_app):
    async with _client(make_app()) as c:
        resp = await c.get(STATUS_PATH)
    # Auth disabled → request reaches the handler; unknown id → 404, not 401/403.
    assert resp.status_code == 404


async def test_enabled_rejects_missing_credentials(make_app):
    async with _client(make_app(API_KEY="sekret")) as c:
        resp = await c.get(STATUS_PATH)
    assert resp.status_code == 401


async def test_enabled_rejects_invalid_key(make_app):
    async with _client(make_app(API_KEY="sekret")) as c:
        resp = await c.get(STATUS_PATH, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 403


async def test_enabled_accepts_valid_key(make_app):
    async with _client(make_app(API_KEY="sekret")) as c:
        resp = await c.get(STATUS_PATH, headers={"Authorization": "Bearer sekret"})
    # Auth passes → handler runs; unknown id → 404.
    assert resp.status_code == 404


async def test_keys_file_client_name_authorizes(make_app, tmp_path):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"file-key": "alice"}), encoding="utf-8")

    app = make_app(API_KEYS_FILE=str(keys_file))
    async with _client(app) as c:
        ok = await c.get(STATUS_PATH, headers={"Authorization": "Bearer file-key"})
        bad = await c.get(STATUS_PATH, headers={"Authorization": "Bearer nope"})
    assert ok.status_code == 404
    assert bad.status_code == 403


async def test_auth_required_without_keys_fails_startup(make_app):
    with pytest.raises(RuntimeError, match="AUTH_REQUIRED"):
        make_app(AUTH_REQUIRED="true")


async def test_auth_required_with_key_starts(make_app):
    async with _client(make_app(AUTH_REQUIRED="true", API_KEY="sekret")) as c:
        resp = await c.get(STATUS_PATH, headers={"Authorization": "Bearer sekret"})
    assert resp.status_code == 404
