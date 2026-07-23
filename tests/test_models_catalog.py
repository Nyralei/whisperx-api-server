"""GET /models/catalog: available in both modes, reports default + loaded."""

import httpx
import pytest

pytestmark = pytest.mark.anyio


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def test_catalog_direct_mode(client):
    body = (await client.get("/models/catalog")).json()
    assert "large-v3" in body["models"]
    assert "turbo" in body["models"]
    assert body["default"] == "large-v3"
    assert body["loaded"] == ["fake-tiny"]


async def test_catalog_available_in_kafka_mode(make_app):
    app = make_app(MODE="kafka")
    async with _client(app) as c:
        resp = await c.get("/models/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert "large-v3" in body["models"]
    assert body["loaded"] is None
