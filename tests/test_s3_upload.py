"""Unit tests for audio upload chunking (fake boto client, no real server)."""

import asyncio
import io

import pytest

from whisperx_api_server.config import S3Config
from whisperx_api_server.storage import s3_store
from whisperx_api_server.storage.s3_store import S3ObjectStore

pytestmark = pytest.mark.anyio

PART = 1024


class FakeUploadFile:
    """Stand-in for Starlette's UploadFile: read(size) off a spooled buffer."""

    def __init__(self, data: bytes, *, expose_size: bool = True):
        self._buf = io.BytesIO(data)
        if expose_size:
            self.size = len(data)

    async def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)


class FakeBotoS3:
    """Dict-backed stand-in tracking multipart state."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, dict[int, bytes]] = {}
        self.aborted: list[str] = []
        self.completed: list[str] = []
        self.put_object_calls = 0
        self.part_sizes: list[int] = []
        self._next_id = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.fail_on_part: int | None = None

    async def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.put_object_calls += 1
        self.objects[Key] = Body

    async def create_multipart_upload(self, *, Bucket, Key):
        self._next_id += 1
        upload_id = f"upload-{self._next_id}"
        self.uploads[upload_id] = {}
        return {"UploadId": upload_id}

    async def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # Yield so concurrent parts actually overlap.
            await asyncio.sleep(0)
            if self.fail_on_part == PartNumber:
                raise RuntimeError(f"part {PartNumber} failed")
            self.uploads[UploadId][PartNumber] = Body
            self.part_sizes.append(len(Body))
            return {"ETag": f"etag-{PartNumber}"}
        finally:
            self.in_flight -= 1

    async def complete_multipart_upload(
        self, *, Bucket, Key, UploadId, MultipartUpload
    ):
        numbers = [p["PartNumber"] for p in MultipartUpload["Parts"]]
        assert numbers == sorted(numbers), "parts must be in ascending order"
        stored = self.uploads.pop(UploadId)
        assert set(numbers) == set(stored), "completed parts must match uploaded parts"
        self.objects[Key] = b"".join(stored[n] for n in numbers)
        self.completed.append(UploadId)

    async def abort_multipart_upload(self, *, Bucket, Key, UploadId):
        self.uploads.pop(UploadId, None)
        self.aborted.append(UploadId)


def _store(fake, **cfg) -> S3ObjectStore:
    store = S3ObjectStore(S3Config(multipart_part_size=PART, **cfg))
    store._client = fake
    return store


@pytest.fixture
def fake_s3(monkeypatch):
    # The real 5 MiB floor would make every fixture-sized body a single PUT.
    monkeypatch.setattr(s3_store, "_MIN_PART_SIZE", PART)
    return FakeBotoS3()


async def _upload(store, data, *, expose_size=True, job="job1", name="a.wav"):
    from whisperx_api_server.storage import service

    upload_file = FakeUploadFile(data, expose_size=expose_size)
    key = f"audio/{job}/{name}"
    return key, await store.put_stream(
        key=key,
        chunks=service._read_upload_chunks(upload_file),
        content_length=service._content_length(upload_file),
    )


async def test_small_upload_uses_single_put(fake_s3):
    store = _store(fake_s3)
    data = b"x" * (PART - 1)

    key, written = await _upload(store, data)

    assert written == len(data)
    assert fake_s3.objects[key] == data
    assert fake_s3.put_object_calls == 1
    assert not fake_s3.uploads and not fake_s3.completed


async def test_empty_upload_uses_single_put(fake_s3):
    store = _store(fake_s3)

    key, written = await _upload(store, b"")

    assert written == 0
    assert fake_s3.objects[key] == b""
    assert fake_s3.put_object_calls == 1
    assert not fake_s3.completed


async def test_large_upload_reassembles_exactly(fake_s3):
    store = _store(fake_s3)
    data = bytes(range(256)) * 20  # 5120 bytes = 5 parts

    key, written = await _upload(store, data)

    assert written == len(data)
    assert fake_s3.objects[key] == data
    assert fake_s3.put_object_calls == 0
    assert len(fake_s3.completed) == 1
    assert fake_s3.part_sizes == [PART] * 5


async def test_trailing_partial_part_preserved(fake_s3):
    store = _store(fake_s3)
    data = b"y" * (PART * 2 + 7)

    key, written = await _upload(store, data)

    assert written == len(data)
    assert fake_s3.objects[key] == data
    assert sorted(fake_s3.part_sizes) == [7, PART, PART]


async def test_upload_without_known_size_still_chunks(fake_s3):
    store = _store(fake_s3)
    data = b"z" * (PART * 3)

    key, _ = await _upload(store, data, expose_size=False)

    assert fake_s3.objects[key] == data
    assert len(fake_s3.part_sizes) == 3


async def test_parts_are_regrouped_across_reader_chunk_boundaries(fake_s3):
    """The reader's chunk size is independent of the S3 part size."""
    store = _store(fake_s3)
    data = bytes(range(256)) * 12  # 3072 bytes = 3 parts

    async def dribble():
        for i in range(0, len(data), 7):  # chunks that divide no part boundary
            yield data[i : i + 7]

    key = "audio/job1/a.wav"
    written = await store.put_stream(
        key=key, chunks=dribble(), content_length=len(data)
    )

    assert written == len(data)
    assert fake_s3.objects[key] == data
    assert fake_s3.part_sizes == [PART] * 3


async def test_concurrency_bounds_parts_in_flight(fake_s3):
    store = _store(fake_s3, multipart_concurrency=2)
    data = b"w" * (PART * 8)

    await _upload(store, data)

    assert fake_s3.max_in_flight <= 2


async def test_failed_part_aborts_upload(fake_s3):
    store = _store(fake_s3)
    fake_s3.fail_on_part = 3
    data = b"q" * (PART * 5)

    with pytest.raises(RuntimeError, match="part 3 failed"):
        await _upload(store, data)

    assert len(fake_s3.aborted) == 1
    assert not fake_s3.completed
    assert "audio/job1/a.wav" not in fake_s3.objects
    assert not fake_s3.uploads, "no multipart upload may be left dangling"


def test_part_size_respects_s3_minimum():
    store = S3ObjectStore(S3Config(multipart_part_size=1024))

    assert store._part_size_for(None) == s3_store._MIN_PART_SIZE


def test_part_size_scales_past_part_limit():
    store = S3ObjectStore(S3Config())
    configured = S3Config().multipart_part_size

    # A body that fits inside 10000 configured-size parts keeps that size.
    assert store._part_size_for(configured * s3_store._MAX_PARTS) == configured

    # Beyond that the part size grows so the count stays within the limit.
    huge = 5 * 1024**4  # 5 TiB, S3's largest object
    grown = store._part_size_for(huge)
    assert grown > configured
    assert -(-huge // grown) <= s3_store._MAX_PARTS
    assert grown % (1024 * 1024) == 0
