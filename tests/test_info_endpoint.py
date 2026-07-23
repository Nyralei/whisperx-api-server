"""GET /info: discoverability fields (upload cap, subtitle capability)."""

import pytest

pytestmark = pytest.mark.anyio


async def test_info_reports_unlimited_upload_by_default(client):
    body = (await client.get("/info")).json()
    assert body["max_upload_size_bytes"] is None
    assert isinstance(body["subtitle_formats_available"], bool)


async def test_info_reports_configured_upload_cap(make_app):
    import httpx

    app = make_app(MAX_UPLOAD_SIZE_BYTES="1048576")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        body = (await c.get("/info")).json()
    assert body["max_upload_size_bytes"] == 1048576
