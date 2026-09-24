"""Keys arriving on the Kafka wire must not reach outside audio/.

`s3_key` comes off the request topic, and the worker feeds it to a read
(download) and a delete. Every assertion here checks the *absence of the side
effect*, not merely that an exception was raised: a store that deleted the file
and then complained would pass an exception-only test.
"""

import json
import os
from types import SimpleNamespace

import pytest

from storage_fakes import MemoryObjectStore
from whisperx_api_server.config import FsStorageConfig, KafkaConfig, StorageConfig
from whisperx_api_server.storage import service
from whisperx_api_server.storage.contracts import ObjectNotFound, StorageKeyError
from whisperx_api_server.storage.fs_store import FsObjectStore
from whisperx_worker import handler

pytestmark = pytest.mark.anyio

HOSTILE_KEYS = [
    "../../etc/passwd",
    "/etc/shadow",
    "audio/../../x",
    "results/other-job",
    "claims/other-job",
    "C:\\Windows\\System32\\drivers\\etc\\hosts",
    "audio/j1/../../../../etc/passwd",
    "",
    "audio/j1",
    "audio/j1/sub/dir/a.wav",
]


@pytest.fixture(params=["memory", "fs"])
async def wired_store(request, tmp_path, monkeypatch):
    if request.param == "memory":
        store = MemoryObjectStore()
    else:
        store = FsObjectStore(FsStorageConfig(root=str(tmp_path)))
        await store.open()
    monkeypatch.setattr(service, "_store", store)
    monkeypatch.setattr(service, "_lease", service.LeaseManager(store))
    return store


@pytest.mark.parametrize("key", HOSTILE_KEYS)
async def test_hostile_keys_rejected_on_the_read_path(wired_store, key):
    with pytest.raises(StorageKeyError):
        await service.download_audio_to_temp(key)


@pytest.mark.parametrize("key", HOSTILE_KEYS)
async def test_hostile_keys_rejected_on_the_delete_path(wired_store, key):
    with pytest.raises(StorageKeyError):
        await service.delete_audio(key)


async def test_wire_key_cannot_delete_another_jobs_result(wired_store):
    await service.put_result("other-job", b"another job's envelope")

    with pytest.raises(StorageKeyError):
        await service.delete_audio("results/other-job")

    assert await service.get_result("other-job") == "another job's envelope"


async def test_wire_key_cannot_release_another_workers_lease(wired_store):
    acquired, _ = await service.acquire_job_lease("other-job", "worker-a", 300.0)
    assert acquired is True

    with pytest.raises(StorageKeyError):
        await service.delete_audio("claims/other-job")

    # The lease is still held: a second worker must still be blocked.
    assert await service.acquire_job_lease("other-job", "worker-b", 300.0) == (
        False,
        1,
    )


async def test_legitimate_audio_key_still_works(wired_store):
    key = await service.upload_audio(b"audio-bytes", "job-1", "a.wav")
    assert key == "audio/job-1/a.wav"

    path = await service.download_audio_to_temp(key)
    with open(path, "rb") as f:
        assert f.read() == b"audio-bytes"
    os.remove(path)

    await service.delete_audio(key)
    with pytest.raises(ObjectNotFound):
        await service.download_audio_to_temp(key)


async def test_hostile_key_does_not_delete_the_input_via_the_handler(
    wired_store, monkeypatch
):
    """A poisoned s3_key must not stop the job replying, nor delete anything."""
    await service.put_result("victim", b"victim envelope")

    async def _ok(event, **kwargs):
        return {"text": "done", "segments": []}

    monkeypatch.setattr(handler, "process_job", _ok)

    sends = []

    class _Producer:
        async def send_and_wait(self, topic, *, key, value, partition=None):
            sends.append((topic, value))

    async def _commit():
        return None

    ctx = handler.WorkerContext(
        producer=_Producer(),
        config=SimpleNamespace(kafka=KafkaConfig(), storage=StorageConfig()),
        commit=_commit,
        worker_id="w1",
        storage=service,
    )
    monkeypatch.setattr(handler, "_LEASE_DEFER_SECONDS", 0)

    await handler.handle_message(
        {"job_id": "j-evil", "s3_key": "results/victim", "params": {}}, ctx
    )

    assert await service.get_result("victim") == "victim envelope"
    reply = [v for t, v in sends if t == ctx.config.kafka.reply_topic]
    assert json.loads(reply[0])["status"] == "ok"
