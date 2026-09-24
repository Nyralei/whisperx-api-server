"""Retention sweep: what it removes, and — more importantly — what it must not."""

import json
import os
import time

import pytest

from whisperx_api_server.config import FsStorageConfig
from whisperx_api_server.storage.fs_store import FsObjectStore

pytestmark = pytest.mark.anyio

DAY = 86400.0


@pytest.fixture
async def store(tmp_path):
    s = FsObjectStore(FsStorageConfig(root=str(tmp_path)))
    await s.open()
    return s


def _backdate(path, seconds: float) -> None:
    past = time.time() - seconds
    os.utime(path, (past, past))


async def test_old_audio_and_results_are_removed(store, tmp_path):
    await store.put_bytes(key="audio/j1/a.wav", data=b"x")
    await store.put_bytes(key="results/j1", data=b"envelope")
    _backdate(tmp_path / "whisperx" / "audio" / "j1" / "a.wav", 2 * DAY)
    _backdate(tmp_path / "whisperx" / "results" / "j1", 2 * DAY)

    removed = await store.sweep_expired(older_than_seconds=DAY, lease_grace_seconds=0.0)

    assert removed["audio"] == 1
    assert removed["results"] == 1
    assert not (tmp_path / "whisperx" / "audio" / "j1").exists()
    assert not (tmp_path / "whisperx" / "results" / "j1").exists()


async def test_fresh_objects_survive(store, tmp_path):
    await store.put_bytes(key="audio/j1/a.wav", data=b"x")
    await store.put_bytes(key="results/j1", data=b"envelope")

    removed = await store.sweep_expired(older_than_seconds=DAY, lease_grace_seconds=0.0)

    assert removed == {"audio": 0, "results": 0, "claims": 0}
    assert (tmp_path / "whisperx" / "results" / "j1").exists()


async def test_live_lease_survives_despite_an_old_mtime(store, tmp_path):
    """Claims are swept by content: deleting a live lease would let two workers
    process the same job."""
    lease = {"attempts": 1, "owner": "w1", "expires_at": time.time() + 3600}
    await store.put_bytes(key="claims/j1", data=json.dumps(lease).encode())
    _backdate(tmp_path / "whisperx" / "claims" / "j1", 30 * DAY)

    removed = await store.sweep_expired(older_than_seconds=DAY, lease_grace_seconds=0.0)

    assert removed["claims"] == 0
    assert (tmp_path / "whisperx" / "claims" / "j1").exists()


async def test_expired_lease_past_the_grace_window_is_removed(store, tmp_path):
    lease = {"attempts": 1, "owner": "w1", "expires_at": time.time() - 600}
    await store.put_bytes(key="claims/j1", data=json.dumps(lease).encode())

    removed = await store.sweep_expired(
        older_than_seconds=DAY, lease_grace_seconds=300.0
    )

    assert removed["claims"] == 1


async def test_expired_lease_inside_the_grace_window_survives(store, tmp_path):
    lease = {"attempts": 1, "owner": "w1", "expires_at": time.time() - 10}
    await store.put_bytes(key="claims/j1", data=json.dumps(lease).encode())

    removed = await store.sweep_expired(
        older_than_seconds=DAY, lease_grace_seconds=300.0
    )

    assert removed["claims"] == 0


async def test_third_party_files_outside_the_prefix_are_never_swept(store, tmp_path):
    """The single most dangerous failure mode: a sweep rooted at the mount root
    instead of the owned subtree wipes other services' data."""
    theirs = tmp_path / "files" / "wav"
    theirs.mkdir(parents=True)
    source = theirs / "123.wav"
    source.write_bytes(b"someone else's audio")
    _backdate(source, 365 * DAY)

    stray = tmp_path / "loose.txt"
    stray.write_bytes(b"also not ours")
    _backdate(stray, 365 * DAY)

    await store.sweep_expired(older_than_seconds=0.0, lease_grace_seconds=0.0)

    assert source.exists()
    assert source.read_bytes() == b"someone else's audio"
    assert stray.exists()


async def test_sweep_base_is_strictly_under_the_root(store, tmp_path):
    assert store.base != store.root
    assert store.base.startswith(store.root + os.sep)
    assert os.path.basename(store.base) == "whisperx"
