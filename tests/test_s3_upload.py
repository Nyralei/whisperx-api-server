"""Unit tests for audio upload chunking (fake boto client, no MinIO)."""

import asyncio
import io

import pytest

from whisperx_api_server import s3_client
from whisperx_api_server.config import S3Config

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


@pytest.fixture
def fake_s3(monkeypatch):
    fake = FakeBotoS3()
    monkeypatch.setattr(s3_client, "_client", fake)
    monkeypatch.setattr(s3_client, "_config", S3Config(multipart_part_size=PART))
    monkeypatch.setattr(s3_client, "_MIN_PART_SIZE", PART)
    return fake


async def test_small_upload_uses_single_put(fake_s3):
    data = b"x" * (PART - 1)
    key = await s3_client.upload_audio_stream(FakeUploadFile(data), "job1", "a.wav")

    assert key == "audio/job1/a.wav"
    assert fake_s3.objects[key] == data
    assert fake_s3.put_object_calls == 1
    assert not fake_s3.uploads and not fake_s3.completed


async def test_empty_upload_uses_single_put(fake_s3):
    key = await s3_client.upload_audio_stream(FakeUploadFile(b""), "job1", "a.wav")

    assert fake_s3.objects[key] == b""
    assert fake_s3.put_object_calls == 1
    assert not fake_s3.completed


async def test_large_upload_reassembles_exactly(fake_s3):
    data = bytes(range(256)) * 20  # 5120 bytes = 5 parts
    key = await s3_client.upload_audio_stream(FakeUploadFile(data), "job1", "a.wav")

    assert fake_s3.objects[key] == data
    assert fake_s3.put_object_calls == 0
    assert len(fake_s3.completed) == 1
    assert fake_s3.part_sizes == [PART] * 5


async def test_trailing_partial_part_preserved(fake_s3):
    data = b"y" * (PART * 2 + 7)
    key = await s3_client.upload_audio_stream(FakeUploadFile(data), "job1", "a.wav")

    assert fake_s3.objects[key] == data
    assert sorted(fake_s3.part_sizes) == [7, PART, PART]


async def test_upload_without_known_size_still_chunks(fake_s3):
    data = b"z" * (PART * 3)
    upload = FakeUploadFile(data, expose_size=False)
    key = await s3_client.upload_audio_stream(upload, "job1", "a.wav")

    assert fake_s3.objects[key] == data
    assert len(fake_s3.part_sizes) == 3


async def test_concurrency_bounds_parts_in_flight(fake_s3, monkeypatch):
    monkeypatch.setattr(
        s3_client,
        "_config",
        S3Config(multipart_part_size=PART, multipart_concurrency=2),
    )
    data = b"w" * (PART * 8)

    await s3_client.upload_audio_stream(FakeUploadFile(data), "job1", "a.wav")

    assert fake_s3.max_in_flight <= 2


async def test_failed_part_aborts_upload(fake_s3):
    fake_s3.fail_on_part = 3
    data = b"q" * (PART * 5)

    with pytest.raises(RuntimeError, match="part 3 failed"):
        await s3_client.upload_audio_stream(FakeUploadFile(data), "job1", "a.wav")

    assert len(fake_s3.aborted) == 1
    assert not fake_s3.completed
    assert "audio/job1/a.wav" not in fake_s3.objects
    assert not fake_s3.uploads, "no multipart upload may be left dangling"


def test_part_size_respects_s3_minimum(monkeypatch):
    monkeypatch.setattr(s3_client, "_config", S3Config(multipart_part_size=1024))

    assert s3_client._part_size_for(None) == s3_client._MIN_PART_SIZE


def test_part_size_scales_past_part_limit(monkeypatch):
    monkeypatch.setattr(s3_client, "_config", S3Config())
    configured = S3Config().multipart_part_size

    # A file that fits inside 10000 configured-size parts keeps that size.
    assert s3_client._part_size_for(configured * s3_client._MAX_PARTS) == configured

    # Beyond that the part size grows so the count stays within the limit.
    huge = 5 * 1024**4  # 5 TiB, S3's largest object
    grown = s3_client._part_size_for(huge)
    assert grown > configured
    assert -(-huge // grown) <= s3_client._MAX_PARTS
    assert grown % (1024 * 1024) == 0
