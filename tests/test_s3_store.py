"""S3ObjectStore error translation (fake boto client, no real server)."""

import pytest
from botocore.exceptions import ClientError

from whisperx_api_server.config import S3Config
from whisperx_api_server.storage.contracts import ObjectNotFound, StorageCapabilityError
from whisperx_api_server.storage.s3_store import S3ObjectStore

pytestmark = pytest.mark.anyio


class _Body:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self, size: int | None = None):
        if size is None:
            data, self._pos = self._data[self._pos :], len(self._data)
            return data
        data = self._data[self._pos : self._pos + size]
        self._pos += len(data)
        return data

    def close(self):
        pass


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
def store():
    s = S3ObjectStore(S3Config())
    s._client = FakeBotoS3()
    return s


async def test_missing_key_reads_as_none(store):
    assert await store.get_bytes(key="results/nope") is None


async def test_missing_key_download_raises_object_not_found(store, tmp_path):
    with pytest.raises(ObjectNotFound):
        await store.download_to_path(key="audio/j/a.wav", path=str(tmp_path / "out"))


async def test_precondition_failed_is_a_lost_race_not_an_error(store):
    assert await store.put_if_absent(key="claims/j", data=b"a") is True
    assert await store.put_if_absent(key="claims/j", data=b"b") is False
    assert store._client.objects["claims/j"] == b"a"


async def test_conditional_writes_unsupported_raises_capability_error(store):
    store._client.conditional_supported = False
    with pytest.raises(StorageCapabilityError):
        await store.put_if_absent(key="claims/j", data=b"a")


async def test_delete_of_missing_key_is_a_no_op(store):
    await store.delete(key="audio/gone/x.wav")


async def test_round_trip(store, tmp_path):
    await store.put_bytes(key="results/j", data=b"payload")
    assert await store.get_bytes(key="results/j") == b"payload"

    out = tmp_path / "copy.bin"
    written = await store.download_to_path(key="results/j", path=str(out))
    assert written == 7
    assert out.read_bytes() == b"payload"
