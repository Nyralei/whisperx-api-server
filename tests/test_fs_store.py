"""Filesystem store: key containment, copy semantics, and streaming."""

import os

import pytest

from whisperx_api_server.config import FsStorageConfig
from whisperx_api_server.storage.contracts import (
    ObjectNotFound,
    StorageKeyError,
    StorageSelectionError,
)
from whisperx_api_server.storage.fs_store import FsObjectStore

pytestmark = pytest.mark.anyio


@pytest.fixture
async def store(tmp_path):
    s = FsObjectStore(FsStorageConfig(root=str(tmp_path)))
    await s.open()
    return s


TRAVERSAL_KEYS = [
    "results/..",
    "audio/../../etc/passwd",
    "claims/../../../../etc/shadow",
    "audio/j/./x",
    "audio//x",
    "/abs/key",
    "C:/Windows/x",
    "C:\\Windows\\x",
    "\\\\server\\share",
    "claims/.",
    "audio/j/a\x00.wav",
    "..",
    "../outside",
    "audio/j/..%2f..%2fetc",
]


@pytest.mark.parametrize("key", TRAVERSAL_KEYS)
async def test_traversal_keys_are_rejected(store, key):
    with pytest.raises(StorageKeyError):
        await store.get_bytes(key=key)
    with pytest.raises(StorageKeyError):
        await store.put_bytes(key=key, data=b"x")
    with pytest.raises(StorageKeyError):
        await store.delete(key=key)


async def test_traversal_delete_leaves_the_target_alone(store, tmp_path):
    outside = tmp_path.parent / "victim.txt"
    outside.write_text("do not delete me", encoding="utf-8")
    try:
        with pytest.raises(StorageKeyError):
            await store.delete(key=f"audio/../../{outside.name}")
        assert outside.exists()
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.skipif(
    os.name == "nt", reason="symlink creation needs elevation on Windows"
)
async def test_symlink_inside_the_root_is_not_followed_out(store, tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = os.path.join(store.base, "results", "escape")
    try:
        os.symlink(str(outside), link)
        with pytest.raises(StorageKeyError):
            await store.get_bytes(key="results/escape")
    finally:
        outside.unlink(missing_ok=True)


async def test_empty_root_is_refused_at_construction():
    with pytest.raises(StorageSelectionError):
        FsObjectStore(FsStorageConfig(root="  "))


async def test_objects_land_under_the_owned_subtree_only(store, tmp_path):
    await store.put_bytes(key="results/j1", data=b"envelope")
    assert (tmp_path / "whisperx" / "results" / "j1").read_bytes() == b"envelope"


async def test_round_trip_and_missing_key(store):
    await store.put_bytes(key="results/j1", data=b"payload")
    assert await store.get_bytes(key="results/j1") == b"payload"
    assert await store.get_bytes(key="results/missing") is None


async def test_delete_of_missing_key_is_a_no_op(store):
    await store.delete(key="audio/j1/a.wav")


async def test_delete_prunes_the_emptied_audio_job_dir(store, tmp_path):
    await store.put_bytes(key="audio/j1/a.wav", data=b"x")
    await store.delete(key="audio/j1/a.wav")
    assert not (tmp_path / "whisperx" / "audio" / "j1").exists()
    assert (tmp_path / "whisperx" / "audio").exists()


async def test_download_copies_and_leaves_the_source_in_place(store, tmp_path):
    await store.put_bytes(key="audio/j1/a.wav", data=b"audio-bytes")
    src = tmp_path / "whisperx" / "audio" / "j1" / "a.wav"
    dst = tmp_path / "copy.wav"

    written = await store.download_to_path(key="audio/j1/a.wav", path=str(dst))

    assert written == len(b"audio-bytes")
    assert dst.read_bytes() == b"audio-bytes"
    assert src.exists(), "download must copy, never hand back or move the source"
    assert os.stat(src).st_ino != os.stat(dst).st_ino


async def test_download_of_missing_key_raises_object_not_found(store, tmp_path):
    with pytest.raises(ObjectNotFound):
        await store.download_to_path(key="audio/j1/a.wav", path=str(tmp_path / "o"))


async def test_put_stream_never_buffers_the_whole_payload(store, tmp_path):
    chunk = b"a" * (256 * 1024)
    count = 8
    live = {"max": 0}

    async def chunks():
        for _ in range(count):
            live["max"] = max(live["max"], len(chunk))
            yield chunk

    written = await store.put_stream(key="audio/j1/big.wav", chunks=chunks())

    assert written == len(chunk) * count
    assert live["max"] == len(chunk)
    assert (
        tmp_path / "whisperx" / "audio" / "j1" / "big.wav"
    ).stat().st_size == written


async def test_put_bytes_overwrites_atomically(store):
    await store.put_bytes(key="results/j1", data=b"first")
    await store.put_bytes(key="results/j1", data=b"second")
    assert await store.get_bytes(key="results/j1") == b"second"


async def test_no_temp_files_are_left_behind(store, tmp_path):
    await store.put_bytes(key="results/j1", data=b"x")
    await store.put_if_absent(key="claims/j1", data=b"{}")
    leftovers = [p for p in (tmp_path / "whisperx").rglob("*.tmp.*") if p.is_file()]
    assert leftovers == []
