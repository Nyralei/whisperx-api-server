"""Process-wide storage singleton and the domain layer over ObjectStore.

Owns everything the rest of the application knows about object storage: the
audio/results/claims key layout, the lease, temp-file handling, and the
validation applied to keys that arrive from untrusted sources.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
from collections.abc import AsyncIterator
from uuid import uuid4

from whisperx_api_server.config import Config

from .contracts import ObjectStore, StorageCapabilityError, StorageKeyError
from .lease import LeaseManager
from .registry import create_store

logger = logging.getLogger(__name__)

_AUDIO_PREFIX = "audio/"
_RESULTS_PREFIX = "results/"
_CLAIMS_PREFIX = "claims/"

_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB

_store: ObjectStore | None = None
_lease: LeaseManager | None = None


def _require_store() -> ObjectStore:
    if _store is None:
        raise RuntimeError("Storage backend not initialized")
    return _store


def _require_lease() -> LeaseManager:
    if _lease is None:
        raise RuntimeError("Storage backend not initialized")
    return _lease


def active_store() -> ObjectStore | None:
    return _store


# --------------------------------------------------------------------------
# Untrusted keys
# --------------------------------------------------------------------------

_AUDIO_KEY = re.compile(r"^audio/[A-Za-z0-9._-]{1,128}/[A-Za-z0-9._-]{1,255}$")


def validated_audio_key(key: str) -> str:
    """Confine a key that arrived on the Kafka wire to the audio/ subtree.

    The worker has no legitimate reason to read or delete anything outside
    audio/, so results/ and claims/ are unreachable from wire data by
    construction, and a filesystem backend cannot be walked out of its root.
    """
    if not isinstance(key, str) or not _AUDIO_KEY.match(key):
        raise StorageKeyError(
            "Rejected storage key from job event: keys must look like "
            f"'audio/<job_id>/<filename>', got {key!r}"
        )
    return key


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------

_CAPABILITY_FIX = {
    "s3": (
        "Fix: use Silo, upgrade MinIO to RELEASE.2024-08-* or later, use an "
        "S3 provider "
        "with conditional writes, or set STORAGE__BACKEND=fs with a shared "
        "POSIX mount."
    ),
    "fs": (
        "Fix: point STORAGE__FS__ROOT at a writable mount whose filesystem "
        "implements hard links (NFSv3+/NFSv4, SMB3, a cluster filesystem, or "
        "any local filesystem shared between containers on one host). FUSE "
        "object-storage gateways (s3fs, gcsfuse, rclone, blobfuse) cannot "
        "provide this."
    ),
}


def _capability_message(backend: str) -> str:
    return (
        f"Storage backend '{backend}' does not support atomic create-if-absent. "
        "Kafka mode requires it for the claims/<job_id> processing lease; "
        "without it two workers can process the same job concurrently. "
        + _CAPABILITY_FIX.get(backend, "")
    )


async def probe_atomic_create(store: ObjectStore) -> None:
    """Prove the create-if-absent primitive once, at startup.

    Probes inside claims/ rather than a directory of its own: that is the prefix
    whose semantics are under test, it leaves no empty directory behind on a
    filesystem backend, and a probe orphaned by a crash is reclaimed by the same
    retention sweep as a stale lease.
    """
    key = f"{_CLAIMS_PREFIX}_probe.{uuid4().hex}"
    try:
        first = await store.put_if_absent(key=key, data=b"probe")
        second = await store.put_if_absent(key=key, data=b"probe")
    finally:
        with contextlib.suppress(Exception):
            await store.delete(key=key)
    if not (first and not second):
        raise StorageCapabilityError(_capability_message(store.name))


async def init_storage(config: Config) -> None:
    global _store, _lease
    store = create_store(config.storage.backend, config)
    await store.open()
    try:
        await probe_atomic_create(store)
    except BaseException:
        with contextlib.suppress(Exception):
            await store.close()
        raise
    _store = store
    _lease = LeaseManager(store, prefix=_CLAIMS_PREFIX)
    logger.info("Storage backend '%s' initialized", store.name)


async def close_storage() -> None:
    global _store, _lease
    if _store is not None:
        await _store.close()
        _store = None
        _lease = None


# --------------------------------------------------------------------------
# Audio objects
# --------------------------------------------------------------------------


async def upload_audio(data: bytes, job_id: str, filename: str) -> str:
    key = f"{_AUDIO_PREFIX}{job_id}/{filename}"
    await _require_store().put_bytes(key=key, data=data)
    return key


async def _read_upload_chunks(upload_file) -> AsyncIterator[bytes]:
    while True:
        chunk = await upload_file.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


def _content_length(upload_file) -> int | None:
    size = getattr(upload_file, "size", None)
    return size if isinstance(size, int) and size >= 0 else None


async def upload_audio_stream(upload_file, job_id: str, filename: str) -> str:
    """Upload a FastAPI UploadFile without blocking the event loop.

    UploadFile.read() routes through anyio's thread executor, so concurrent
    uploads interleave instead of serializing on the loop.
    """
    key = f"{_AUDIO_PREFIX}{job_id}/{filename}"
    written = await _require_store().put_stream(
        key=key,
        chunks=_read_upload_chunks(upload_file),
        content_length=_content_length(upload_file),
    )
    logger.debug("Uploaded %s bytes to %s", written, key)
    return key


async def download_audio_to_temp(key: str, suffix: str = "") -> str:
    """Copy a stored audio object into a temp file; returns the file path.

    Always a copy: the caller deletes the returned path once the audio is
    decoded, long before the result is durable.
    """
    store = _require_store()
    key = validated_audio_key(key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_path = tmp.name
    try:
        total = await store.download_to_path(key=key, path=file_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(file_path)
        raise
    logger.debug("Downloaded %s bytes from %s to temp file", total, key)
    return file_path


async def delete_audio(key: str) -> None:
    await _require_store().delete(key=validated_audio_key(key))


# --------------------------------------------------------------------------
# Result envelopes
# --------------------------------------------------------------------------


async def put_result(job_id: str, envelope: bytes) -> None:
    """Store the terminal reply envelope so a redelivery can resend it."""
    await _require_store().put_bytes(key=f"{_RESULTS_PREFIX}{job_id}", data=envelope)


async def get_result(job_id: str) -> str | None:
    """Return the stored reply envelope, or None if the job hasn't finished."""
    data = await _require_store().get_bytes(key=f"{_RESULTS_PREFIX}{job_id}")
    return None if data is None else data.decode()


# --------------------------------------------------------------------------
# Job lease
# --------------------------------------------------------------------------


async def acquire_job_lease(
    job_id: str, worker_id: str, ttl_seconds: float
) -> tuple[bool, int]:
    return await _require_lease().acquire(job_id, worker_id, ttl_seconds)


async def renew_job_lease(job_id: str, worker_id: str, ttl_seconds: float) -> bool:
    return await _require_lease().renew(job_id, worker_id, ttl_seconds)


async def delete_claim(job_id: str) -> None:
    await _require_lease().release(job_id)
