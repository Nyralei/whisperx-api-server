"""GET /v1/audio/transcriptions/{id}/status: lifecycle, 404, 400, TTL expiry."""

import pytest

from whisperx_api_server import request_status

pytestmark = pytest.mark.anyio

STATUS = "/v1/audio/transcriptions/{}/status"
EVENTS = "/v1/audio/transcriptions/{}/events"


async def test_status_lifecycle(client):
    rid = "job-lifecycle-1"
    request_status.start(rid, mode="direct", filename="a.wav")

    resp = await client.get(STATUS.format(rid))
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"

    request_status.set_stage(rid, "transcribe")
    resp = await client.get(STATUS.format(rid))
    body = resp.json()
    assert body["status"] == "in_progress"
    assert body["stages"][-1]["name"] == "transcribe"
    assert body["stages"][-1]["in_progress"] is True

    request_status.mark_completed(rid)
    resp = await client.get(STATUS.format(rid))
    assert resp.json()["status"] == "completed"


async def test_status_unknown_id_404(client):
    resp = await client.get(STATUS.format("never-seen"))
    assert resp.status_code == 404


async def test_status_invalid_id_400(client):
    resp = await client.get(STATUS.format("bad@id"))
    assert resp.status_code == 400


async def test_events_terminal_emits_and_closes(client):
    rid = "job-events-1"
    request_status.start(rid, mode="direct", filename="a.wav")
    request_status.mark_completed(rid)

    resp = await client.get(EVENTS.format(rid))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.startswith("data:")
    assert "completed" in resp.text


async def test_events_unknown_id_404(client):
    resp = await client.get(EVENTS.format("never-seen"))
    assert resp.status_code == 404


async def test_events_invalid_id_400(client):
    resp = await client.get(EVENTS.format("bad@id"))
    assert resp.status_code == 400


async def test_status_ttl_expiry_404(client):
    rid = "job-expiry-1"
    request_status.start(rid, mode="direct")
    request_status.mark_completed(rid)
    st = request_status.get(rid)
    assert st is not None
    done = st["completed_at"]

    assert request_status.evict_expired(now=done + 301.0) == 1

    resp = await client.get(STATUS.format(rid))
    assert resp.status_code == 404
