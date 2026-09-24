"""In-memory ObjectStore used by the storage unit tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from whisperx_api_server.storage.contracts import ObjectNotFound


class MemoryObjectStore:
    name = "memory"

    def __init__(self, *, atomic: bool = True) -> None:
        self.objects: dict[str, bytes] = {}
        self.atomic = atomic
        self.opened = False
        self.closed = False

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True

    async def put_bytes(self, *, key: str, data: bytes) -> None:
        self.objects[key] = data

    async def put_stream(
        self,
        *,
        key: str,
        chunks: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> int:
        buf = bytearray()
        async for chunk in chunks:
            buf.extend(chunk)
        self.objects[key] = bytes(buf)
        return len(buf)

    async def put_if_absent(self, *, key: str, data: bytes) -> bool:
        if self.atomic and key in self.objects:
            return False
        self.objects[key] = data
        return True

    async def get_bytes(self, *, key: str) -> bytes | None:
        return self.objects.get(key)

    async def download_to_path(self, *, key: str, path: str) -> int:
        data = self.objects.get(key)
        if data is None:
            raise ObjectNotFound(key)
        with open(path, "wb") as f:
            f.write(data)
        return len(data)

    async def delete(self, *, key: str) -> None:
        self.objects.pop(key, None)
