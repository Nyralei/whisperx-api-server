"""Streaming an upload to its temp file: cleanup on cancellation and write errors."""

import asyncio
import io
import os
import threading
import time

import anyio
import pytest
from fastapi import UploadFile

import whisperx_api_server.transcriber as transcriber

pytestmark = pytest.mark.anyio


def _patch_temp_writes(monkeypatch, write) -> list[str]:
    real_open = open
    paths: list[str] = []

    class PatchedFile:
        def __init__(self, f):
            self.f = f

        def write(self, data):
            return write(self.f, data)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.f.close()

    def patched_open(path, *args, **kwargs):
        paths.append(path)
        return PatchedFile(real_open(path, *args, **kwargs))

    monkeypatch.setattr(transcriber, "open", patched_open, raising=False)
    return paths


@pytest.fixture
def slow_temp_writes(monkeypatch) -> list[str]:
    def slow(f, data):
        time.sleep(0.2)
        return f.write(data)

    return _patch_temp_writes(monkeypatch, slow)


@pytest.fixture
def failing_temp_writes(monkeypatch) -> list[str]:
    def fail(f, data):
        raise OSError("disk full")

    return _patch_temp_writes(monkeypatch, fail)


class _Upload(UploadFile):
    def __init__(self, chunks: int | None = None):
        super().__init__(io.BytesIO(), filename="a.wav")
        self.chunks = chunks
        self.drained = asyncio.Event()

    async def read(self, size: int = -1) -> bytes:
        await asyncio.sleep(0)
        if self.chunks is not None:
            if self.chunks == 0:
                self.drained.set()
                return b""
            self.chunks -= 1
        return b"x" * 1024


def _assert_writer_cleaned_up(path: str) -> None:
    writers = [
        t
        for t in threading.enumerate()
        if getattr(getattr(t, "_target", None), "__name__", "") == "_writer"
    ]
    assert writers == []
    assert not os.path.exists(path)


@pytest.mark.parametrize("via_scope", [False, True])
async def test_cancel_during_upload_save_stops_writer(slow_temp_writes, via_scope):
    if via_scope:
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(transcriber._save_upload_to_temp, _Upload(), "upload-1")
                await asyncio.sleep(0.3)
                tg.cancel_scope.cancel()
    else:
        task = asyncio.ensure_future(
            transcriber._save_upload_to_temp(_Upload(), "upload-1")
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)

    _assert_writer_cleaned_up(slow_temp_writes[0])


async def test_cancel_after_last_chunk_removes_temp_file(slow_temp_writes):
    upload = _Upload(chunks=2)
    task = asyncio.ensure_future(transcriber._save_upload_to_temp(upload, "upload-2"))
    await upload.drained.wait()
    await asyncio.sleep(0.05)
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)

    _assert_writer_cleaned_up(slow_temp_writes[0])


@pytest.mark.parametrize("chunks", [1, 50])
async def test_write_error_fails_the_upload(failing_temp_writes, chunks):
    upload = _Upload(chunks=chunks)

    with pytest.raises(OSError, match="disk full"):
        await asyncio.wait_for(transcriber._save_upload_to_temp(upload, "upload-3"), 5)

    _assert_writer_cleaned_up(failing_temp_writes[0])
    if chunks > 1:
        assert upload.chunks is not None and upload.chunks > 0
