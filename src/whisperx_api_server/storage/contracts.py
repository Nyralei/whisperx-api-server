"""Storage backend interface and error taxonomy.

``ObjectStore`` is a dumb blob store: it knows keys and bytes. Every domain
concept — the audio/results/claims prefixes, lease JSON, temp files — lives in
``service.py`` so it is written once and shared by all backends.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class StorageError(RuntimeError):
    """Base class for storage-layer failures."""


class StorageSelectionError(ValueError):
    """Unknown or malformed storage backend name."""


class StorageKeyError(ValueError):
    """A key that is malformed, or would escape the backend's namespace."""


class ObjectNotFound(StorageError):
    """The requested object does not exist."""


class StorageCapabilityError(StorageError):
    """The backend cannot provide a primitive this deployment requires."""


class ObjectStore(Protocol):
    """Structural contract; implementations do not subclass this."""

    name: str

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def put_bytes(self, *, key: str, data: bytes) -> None: ...

    async def put_stream(
        self,
        *,
        key: str,
        chunks: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> int:
        """Write the stream to `key`, returning bytes written.

        `content_length` is a hint when the caller knows the total size; a
        backend may use it to size its chunking, and may ignore it.
        """
        ...

    async def put_if_absent(self, *, key: str, data: bytes) -> bool:
        """Create the object only if absent; True if this call created it.

        Raises StorageCapabilityError when the backend cannot do this
        atomically. There is deliberately no capability flag to branch on: the
        primitive is proven once at startup and assumed thereafter.
        """
        ...

    async def get_bytes(self, *, key: str) -> bytes | None: ...

    async def download_to_path(self, *, key: str, path: str) -> int:
        """Write the object to `path`, returning bytes written.

        Raises ObjectNotFound when the key does not exist.
        """
        ...

    async def delete(self, *, key: str) -> None:
        """Remove the object. A missing key is not an error."""
        ...
