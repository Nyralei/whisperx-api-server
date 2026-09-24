"""Behavioural tests for the backend-agnostic job lease."""

import json
import time

import pytest

from storage_fakes import MemoryObjectStore
from whisperx_api_server.config import FsStorageConfig
from whisperx_api_server.storage.fs_store import FsObjectStore
from whisperx_api_server.storage.lease import LeaseManager

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["memory", "fs"])
async def store(request, tmp_path):
    if request.param == "memory":
        return MemoryObjectStore()
    fs = FsObjectStore(FsStorageConfig(root=str(tmp_path)))
    await fs.open()
    return fs


@pytest.fixture
def lease(store):
    return LeaseManager(store)


async def _lease_of(mgr: LeaseManager, job_id: str) -> dict:
    raw = await mgr._store.get_bytes(key=f"claims/{job_id}")
    assert raw is not None
    return json.loads(raw)


async def test_fresh_acquire_wins(lease):
    acquired, attempts = await lease.acquire("j1", "w1", 300.0)
    assert acquired is True
    assert attempts == 1
    stored = await _lease_of(lease, "j1")
    assert stored["owner"] == "w1"
    assert stored["expires_at"] > time.time()


async def test_live_foreign_lease_blocks_acquire(lease):
    await lease.acquire("j1", "w1", 300.0)

    acquired, attempts = await lease.acquire("j1", "w2", 300.0)

    assert acquired is False
    assert attempts == 1
    assert (await _lease_of(lease, "j1"))["owner"] == "w1"


async def test_expired_lease_taken_over_with_attempt_bump(lease):
    await lease.acquire("j1", "w1", -1.0)  # born expired

    acquired, attempts = await lease.acquire("j1", "w2", 300.0)

    assert acquired is True
    assert attempts == 2
    assert (await _lease_of(lease, "j1"))["owner"] == "w2"


async def test_own_lease_reacquired_after_restart(lease):
    # Same worker_id after a crash-restart (containers reuse hostname+pid):
    # its own live lease must not block it.
    await lease.acquire("j1", "w1", 300.0)

    acquired, attempts = await lease.acquire("j1", "w1", 300.0)

    assert acquired is True
    assert attempts == 2


async def test_legacy_counter_claim_treated_as_expired(lease, store):
    # Rolling deploy: an old worker wrote a bare-int delivery counter. Preserve
    # the count, take over.
    await store.put_bytes(key="claims/j1", data=b"2")

    acquired, attempts = await lease.acquire("j1", "w1", 300.0)

    assert acquired is True
    assert attempts == 3


async def test_renew_by_owner_extends(lease):
    await lease.acquire("j1", "w1", 1.0)
    before = (await _lease_of(lease, "j1"))["expires_at"]

    assert await lease.renew("j1", "w1", 300.0) is True
    assert (await _lease_of(lease, "j1"))["expires_at"] > before


async def test_renew_by_non_owner_or_missing_fails(lease):
    await lease.acquire("j1", "w1", 300.0)
    assert await lease.renew("j1", "w2", 300.0) is False
    assert await lease.renew("missing", "w1", 300.0) is False


async def test_release_then_reacquire_is_fresh(lease):
    await lease.acquire("j1", "w1", 300.0)
    await lease.release("j1")

    acquired, attempts = await lease.acquire("j1", "w2", 300.0)

    assert acquired is True
    assert attempts == 1
