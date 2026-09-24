"""Multipart audio upload against a real S3 server.

The fake-client tests in test_s3_upload.py pin the chunking logic; they cannot
catch what only a real server rejects — an under-minimum non-final part, an ETag
echoed back in the wrong form, or a completed object whose bytes do not match
what went in. Multipart is also the path every large upload takes, so it is worth
one real round trip.

Marked ``kafka``: needs Docker. Run with ``pytest -m kafka``.
"""

import hashlib
import uuid

import pytest

from whisperx_api_server.config import S3Config
from whisperx_api_server.storage.s3_store import _MIN_PART_SIZE, S3ObjectStore

pytestmark = [pytest.mark.anyio, pytest.mark.kafka]

# The smallest part a real server accepts for a non-final part. Using exactly
# this keeps the test honest about the boundary and the transfer small.
_PART = _MIN_PART_SIZE


@pytest.fixture
async def store(s3_endpoint):
    store = S3ObjectStore(
        S3Config(
            endpoint_url=s3_endpoint,
            bucket=f"wx-mp-{uuid.uuid4().hex[:8]}",
            access_key_id="minioadmin",
            secret_access_key="minioadmin",
            region="us-east-1",
            multipart_part_size=_PART,
            multipart_concurrency=3,
        )
    )
    await store.open()
    try:
        yield store
    finally:
        await store.close()


def _body(size: int) -> bytes:
    """Pseudo-random so a misordered or duplicated part cannot compare equal."""
    out = bytearray()
    seed = b"whisperx-multipart-seed"
    while len(out) < size:
        seed = hashlib.sha256(seed).digest()
        out.extend(seed)
    return bytes(out[:size])


async def _chunks(data: bytes, chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


@pytest.mark.parametrize(
    "size, label",
    [
        (_PART * 2 + 12345, "two full parts and a short tail"),
        (_PART * 3, "an exact multiple of the part size"),
    ],
)
async def test_multipart_round_trips_byte_for_byte(store, tmp_path, size, label):
    data = _body(size)
    key = f"audio/{uuid.uuid4().hex}/big.wav"

    # 1 MiB reader chunks, as the upload path uses: nothing lines up with _PART.
    written = await store.put_stream(
        key=key, chunks=_chunks(data, 1024 * 1024), content_length=size
    )
    assert written == size, label

    assert await store.get_bytes(key=key) == data, label

    dest = tmp_path / "out.wav"
    assert await store.download_to_path(key=key, path=str(dest)) == size
    assert dest.read_bytes() == data, label


async def test_single_part_body_round_trips(store):
    """Below the threshold this must stay a plain PUT and still be exact."""
    data = _body(_PART - 1)
    key = f"audio/{uuid.uuid4().hex}/small.wav"

    assert await store.put_stream(key=key, chunks=_chunks(data, 64 * 1024)) == len(data)
    assert await store.get_bytes(key=key) == data


async def test_aborted_multipart_leaves_no_object(store):
    """A failed part must abort, not leave a partial object behind."""
    key = f"audio/{uuid.uuid4().hex}/doomed.wav"
    data = _body(_PART * 3)

    async def failing():
        yield data[:_PART]
        yield data[_PART : _PART * 2]
        raise RuntimeError("reader exploded mid-upload")

    with pytest.raises(RuntimeError, match="reader exploded"):
        await store.put_stream(key=key, chunks=failing(), content_length=len(data))

    assert await store.get_bytes(key=key) is None
