"""Unit tests for the S3 job-lease primitives (fake boto client, no MinIO)."""

import json
import time

import pytest
from botocore.exceptions import ClientError

from whisperx_api_server import s3_client
from whisperx_api_server.config import S3Config

pytestmark = pytest.mark.anyio


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self):
        return self._data


def _client_error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code}}, op)


class FakeBotoS3:
    """Dict-backed stand-in honoring IfNoneMatch conditional creates."""

    def __init__(self, *, conditional_supported: bool = True):
        self.objects: dict[str, bytes] = {}
        self.conditional_supported = conditional_supported

    async def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None):
        if IfNoneMatch == "*":
            if not self.conditional_supported:
                raise _client_error("NotImplemented", "PutObject")
            if Key in self.objects:
                raise _client_error("PreconditionFailed", "PutObject")
        self.objects[Key] = Body

    async def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise _client_error("NoSuchKey", "GetObject")
        return {"Body": _Body(self.objects[Key])}

    async def delete_object(self, *, Bucket, Key):
        self.objects.pop(Key, None)


@pytest.fixture
def fake_s3(monkeypatch):
    fake = FakeBotoS3()
    monkeypatch.setattr(s3_client, "_client", fake)
    monkeypatch.setattr(s3_client, "_config", S3Config())
    monkeypatch.setattr(s3_client, "_conditional_writes_supported", True)
    return fake


def _lease_of(fake: FakeBotoS3, job_id: str) -> dict:
    return json.loads(fake.objects[f"claims/{job_id}"])


async def test_fresh_acquire_wins(fake_s3):
    acquired, attempts = await s3_client.acquire_job_lease("j1", "w1", 300.0)
    assert acquired is True
    assert attempts == 1
    lease = _lease_of(fake_s3, "j1")
    assert lease["owner"] == "w1"
    assert lease["expires_at"] > time.time()


async def test_live_foreign_lease_blocks_acquire(fake_s3):
    await s3_client.acquire_job_lease("j1", "w1", 300.0)

    acquired, attempts = await s3_client.acquire_job_lease("j1", "w2", 300.0)

    assert acquired is False
    assert attempts == 1
    assert _lease_of(fake_s3, "j1")["owner"] == "w1"


async def test_expired_lease_taken_over_with_attempt_bump(fake_s3):
    await s3_client.acquire_job_lease("j1", "w1", -1.0)  # born expired

    acquired, attempts = await s3_client.acquire_job_lease("j1", "w2", 300.0)

    assert acquired is True
    assert attempts == 2
    assert _lease_of(fake_s3, "j1")["owner"] == "w2"


async def test_own_lease_reacquired_after_restart(fake_s3):
    # Same worker_id after a crash-restart (containers reuse hostname+pid):
    # its own live lease must not block it.
    await s3_client.acquire_job_lease("j1", "w1", 300.0)

    acquired, attempts = await s3_client.acquire_job_lease("j1", "w1", 300.0)

    assert acquired is True
    assert attempts == 2


async def test_legacy_counter_claim_treated_as_expired(fake_s3):
    # Rolling deploy: an old worker wrote a bare-int delivery counter. Preserve
    # the count, take over.
    fake_s3.objects["claims/j1"] = b"2"

    acquired, attempts = await s3_client.acquire_job_lease("j1", "w1", 300.0)

    assert acquired is True
    assert attempts == 3


async def test_renew_by_owner_extends(fake_s3):
    await s3_client.acquire_job_lease("j1", "w1", 1.0)
    before = _lease_of(fake_s3, "j1")["expires_at"]

    assert await s3_client.renew_job_lease("j1", "w1", 300.0) is True
    assert _lease_of(fake_s3, "j1")["expires_at"] > before


async def test_renew_by_non_owner_or_missing_fails(fake_s3):
    await s3_client.acquire_job_lease("j1", "w1", 300.0)
    assert await s3_client.renew_job_lease("j1", "w2", 300.0) is False
    assert await s3_client.renew_job_lease("missing", "w1", 300.0) is False


async def test_release_then_reacquire_is_fresh(fake_s3):
    await s3_client.acquire_job_lease("j1", "w1", 300.0)
    await s3_client.delete_claim("j1")

    acquired, attempts = await s3_client.acquire_job_lease("j1", "w2", 300.0)

    assert acquired is True
    assert attempts == 1


async def test_fallback_without_conditional_writes(fake_s3):
    fake_s3.conditional_supported = False

    acquired, attempts = await s3_client.acquire_job_lease("j1", "w1", 300.0)
    assert acquired is True
    assert attempts == 1
    assert s3_client._conditional_writes_supported is False

    # Verified-write fallback still detects a live foreign lease.
    acquired, attempts = await s3_client.acquire_job_lease("j1", "w2", 300.0)
    assert acquired is False
    assert attempts == 1
