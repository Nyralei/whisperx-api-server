"""Atomicity of create-if-absent on the filesystem backend.

The load-bearing property of the whole storage layer: exactly one caller may win
a claims/<job_id> create, or two workers transcribe the same job concurrently.
"""

import asyncio
import multiprocessing as mp
import os

import pytest

from fs_race_worker import run as race_run
from whisperx_api_server.config import FsStorageConfig
from whisperx_api_server.storage.fs_store import FsObjectStore
from whisperx_api_server.storage.lease import LeaseManager

pytestmark = pytest.mark.anyio


@pytest.fixture
async def store(tmp_path):
    s = FsObjectStore(FsStorageConfig(root=str(tmp_path)))
    await s.open()
    return s


async def test_concurrent_put_if_absent_has_exactly_one_winner(store):
    results = await asyncio.gather(
        *(store.put_if_absent(key="claims/j1", data=b'{"n":1}') for _ in range(64))
    )
    assert sum(results) == 1


async def test_concurrent_lease_acquire_has_exactly_one_winner(store):
    lease = LeaseManager(store)
    outcomes = await asyncio.gather(
        *(lease.acquire("j1", f"w{i}", 300.0) for i in range(32))
    )
    winners = [o for o in outcomes if o[0]]
    assert len(winners) == 1
    assert winners[0] == (True, 1)


def test_cross_process_exclusive_create_has_exactly_one_winner(tmp_path):
    """Threads can mask a race that separate processes expose."""
    ctx = mp.get_context("spawn")
    n = 8
    barrier = ctx.Barrier(n)
    results = ctx.Manager().list()
    target = str(tmp_path / "claim")

    procs = [
        ctx.Process(target=race_run, args=(barrier, target, results)) for _ in range(n)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    assert len(results) == n, "every child must have reported an outcome"
    assert sum(results) == 1
    assert os.path.exists(target)
