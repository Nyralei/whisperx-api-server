"""Shared POSIX filesystem object store.

Keys map to paths under ``<root>/<prefix>/``. ``root`` is the confinement
boundary — nothing above it is reachable from any verb — and ``prefix`` is the
private subtree this service owns, so files belonging to other services sharing
the same mount are never written, deleted, or swept.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Iterable
from uuid import uuid4

from whisperx_api_server.config import Config, FsStorageConfig

from .contracts import (
    ObjectNotFound,
    StorageCapabilityError,
    StorageKeyError,
    StorageSelectionError,
)
from .mount_check import check_mount
from .registry import register_store

logger = logging.getLogger(__name__)

_SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,255}$")
_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")
_MAX_KEY_LENGTH = 1024
_MAX_KEY_SEGMENTS = 16

_COPY_CHUNK_SIZE = 1024 * 1024  # 1 MiB

_LINK_UNSUPPORTED = frozenset(
    e
    for e in (
        getattr(errno, "ENOSYS", None),
        getattr(errno, "EPERM", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOTSUP", None),
    )
    if e is not None
)


def _process_identity() -> str:
    getuid, getgid = getattr(os, "getuid", None), getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return "the current user"
    return f"uid {getuid()}:gid {getgid()}"


def _fchmod(fd: int, mode: int) -> None:
    # Absent on Windows; umask, NFS root_squash and SMB mount-level uid/gid make
    # this a no-op or EPERM often enough that failure is never fatal.
    fchmod = getattr(os, "fchmod", None)
    if fchmod is None:
        return
    try:
        fchmod(fd, mode)
    except OSError as e:
        logger.debug("fchmod(%o) failed: %s", mode, e)


class FsObjectStore:
    name = "fs"

    def __init__(self, cfg: FsStorageConfig) -> None:
        if not cfg.root.strip():
            raise StorageSelectionError(
                "STORAGE__FS__ROOT must be set when STORAGE__BACKEND=fs. It must "
                "name a directory visible at the same path in every API and "
                "worker container."
            )
        self._cfg = cfg
        self._root = os.path.abspath(cfg.root.strip())
        self._base = os.path.join(self._root, cfg.prefix)
        self._dir_mode = int(cfg.dir_mode, 8)
        self._file_mode = int(cfg.file_mode, 8)
        self._real_base = os.path.realpath(self._base)

    @property
    def root(self) -> str:
        return self._root

    @property
    def base(self) -> str:
        """The subtree this store owns. Nothing outside it is ever modified."""
        return self._base

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _resolve(self, key: str) -> str:
        """Map a key to an absolute path inside the owned subtree.

        The single entry point for path construction: no verb can bypass it.
        Runs in a worker thread because realpath() stats every component.
        """
        if not isinstance(key, str) or not key or len(key) > _MAX_KEY_LENGTH:
            raise StorageKeyError(f"Invalid storage key: {key!r}")
        if "\x00" in key or "\\" in key:
            raise StorageKeyError(
                f"Storage key must not contain backslashes or NUL bytes: {key!r}"
            )
        if key.startswith("/") or _DRIVE_LETTER.match(key):
            raise StorageKeyError(f"Storage key must be relative: {key!r}")

        segments = key.split("/")
        if len(segments) > _MAX_KEY_SEGMENTS:
            raise StorageKeyError(f"Storage key has too many segments: {key!r}")
        for segment in segments:
            if segment in ("", ".", "..") or not _SEGMENT.match(segment):
                raise StorageKeyError(
                    f"Invalid segment {segment!r} in storage key {key!r}"
                )

        path = os.path.join(self._base, *segments)
        real = os.path.realpath(path)
        if real != self._real_base and not real.startswith(self._real_base + os.sep):
            # A symlink planted inside the subtree pointing out of it.
            raise StorageKeyError(f"Storage key escapes the storage root: {key!r}")
        return path

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, mode=self._dir_mode, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(path, self._dir_mode)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        check_mount(self._root, allow_unsafe=self._cfg.allow_unsafe_mount)
        await asyncio.to_thread(self._open_sync)
        logger.info(
            "Filesystem storage initialized (root: %s, owned subtree: %s)",
            self._root,
            self._base,
        )

    def _open_sync(self) -> None:
        if not os.path.isdir(self._root):
            raise StorageCapabilityError(
                f"STORAGE__FS__ROOT={self._root} does not exist or is not a "
                "directory. It must be mounted before the process starts."
            )
        try:
            self._ensure_dir(self._base)
            self._real_base = os.path.realpath(self._base)
            for prefix in ("audio", "results", "claims"):
                self._ensure_dir(os.path.join(self._base, prefix))
        except OSError as e:
            raise StorageCapabilityError(
                f"Cannot create the storage subtree {self._base} ({e.strerror}). "
                f"This process runs as {_process_identity()}; the mount must be "
                "writable by it. A fresh Docker named volume is owned by root:root, "
                "so give the mount point the container's uid:gid in the image (the "
                "volume inherits it) or chown the export on the host."
            ) from e

    async def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def _write_temp(self, path: str, chunks: Iterable[bytes]) -> tuple[str, int]:
        self._ensure_dir(os.path.dirname(path))
        tmp = f"{path}.tmp.{uuid4().hex}"
        total = 0
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, self._file_mode)
        try:
            for chunk in chunks:
                if chunk:
                    total += os.write(fd, chunk)
            if self._cfg.fsync:
                os.fsync(fd)
            _fchmod(fd, self._file_mode)
        except BaseException:
            os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        os.close(fd)
        return tmp, total

    def _put_sync(self, key: str, data: bytes) -> int:
        path = self._resolve(key)
        tmp, total = self._write_temp(path, (data,))
        try:
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return total

    async def put_bytes(self, *, key: str, data: bytes) -> None:
        await asyncio.to_thread(self._put_sync, key, data)

    async def put_stream(
        self,
        *,
        key: str,
        chunks: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> int:
        """Write chunk-by-chunk to a temp file, then rename. Peak RAM is one chunk.

        content_length is irrelevant here: writes stream straight through.
        """
        path = await asyncio.to_thread(self._resolve, key)
        tmp = f"{path}.tmp.{uuid4().hex}"
        await asyncio.to_thread(self._ensure_dir, os.path.dirname(path))
        total = 0
        try:
            fd = await asyncio.to_thread(
                os.open, tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, self._file_mode
            )
            try:
                async for chunk in chunks:
                    if chunk:
                        total += await asyncio.to_thread(os.write, fd, chunk)
                if self._cfg.fsync:
                    await asyncio.to_thread(os.fsync, fd)
                await asyncio.to_thread(_fchmod, fd, self._file_mode)
            finally:
                await asyncio.to_thread(os.close, fd)
            await asyncio.to_thread(os.replace, tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                await asyncio.to_thread(os.unlink, tmp)
            raise
        return total

    def _put_if_absent_sync(self, key: str, data: bytes) -> bool:
        """Create the object only if absent, using the link() idiom.

        link() beats a bare O_EXCL create for two reasons. It stays correct when
        a networked filesystem loses the reply: the server performs the link, the
        response is dropped, the retry returns EEXIST, and a naive implementation
        concludes it lost a race it actually won — the st_nlink check
        disambiguates. And it fails closed on exactly the filesystems where
        O_EXCL fails open, since object-storage gateways return ENOSYS/EPERM for
        link() rather than silently breaking atomicity.
        """
        path = self._resolve(key)
        tmp, _ = self._write_temp(path, (data,))
        try:
            try:
                os.link(tmp, path)
            except FileExistsError:
                # Either another writer won, or our own link landed and the reply
                # was lost. st_nlink tells us which.
                pass
            except OSError as e:
                if e.errno in _LINK_UNSUPPORTED:
                    raise StorageCapabilityError(
                        f"The filesystem at {self._root} does not support hard "
                        f"links ({e.strerror}), so the claims/<job_id> processing "
                        "lease cannot be created atomically."
                    ) from e
                raise
            return os.stat(tmp).st_nlink == 2
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)

    async def put_if_absent(self, *, key: str, data: bytes) -> bool:
        return await asyncio.to_thread(self._put_if_absent_sync, key, data)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _read_sync(self, key: str) -> bytes | None:
        try:
            with open(self._resolve(key), "rb") as f:
                return f.read()
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            return None

    async def get_bytes(self, *, key: str) -> bytes | None:
        return await asyncio.to_thread(self._read_sync, key)

    def _copy_sync(self, key: str, dst: str) -> int:
        """Copy, never alias: the caller deletes what it is handed."""
        src = self._resolve(key)
        total = 0
        try:
            with open(src, "rb") as fin, open(dst, "wb") as fout:
                while True:
                    chunk = fin.read(_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    fout.write(chunk)
        except (FileNotFoundError, IsADirectoryError) as e:
            raise ObjectNotFound(f"No such object: {key}") from e
        return total

    async def download_to_path(self, *, key: str, path: str) -> int:
        return await asyncio.to_thread(self._copy_sync, key, path)

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------

    def _delete_sync(self, key: str) -> None:
        path = self._resolve(key)
        with contextlib.suppress(FileNotFoundError, IsADirectoryError, PermissionError):
            os.unlink(path)
        # Prune only emptied audio job directories; never any other directory,
        # and never the owned subtree itself.
        parent = os.path.dirname(path)
        if os.path.dirname(parent) == os.path.join(self._base, "audio"):
            with contextlib.suppress(OSError):
                os.rmdir(parent)

    async def delete(self, *, key: str) -> None:
        await asyncio.to_thread(self._delete_sync, key)

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    async def sweep_expired(
        self, *, older_than_seconds: float, lease_grace_seconds: float
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            self._sweep_sync, older_than_seconds, lease_grace_seconds
        )

    def _sweep_sync(
        self, older_than_seconds: float, lease_grace_seconds: float
    ) -> dict[str, int]:
        # Rooted at the owned subtree. Rooted at self._root it would delete files
        # belonging to other services sharing the mount.
        if self._base == self._root or not self._base.startswith(self._root + os.sep):
            raise StorageKeyError(
                "Refusing to sweep: the owned subtree must sit strictly under "
                f"the storage root (root={self._root!r}, subtree={self._base!r})"
            )
        cutoff = time.time() - older_than_seconds
        return {
            "audio": self._sweep_audio(cutoff),
            "results": self._sweep_flat(os.path.join(self._base, "results"), cutoff),
            "claims": self._sweep_claims(lease_grace_seconds),
        }

    def _sweep_audio(self, cutoff: float) -> int:
        removed = 0
        for job_dir in self._scandir(os.path.join(self._base, "audio")):
            if not job_dir.is_dir():
                continue
            for entry in self._scandir(job_dir.path):
                try:
                    if entry.stat().st_mtime <= cutoff:
                        os.unlink(entry.path)
                        removed += 1
                except OSError:
                    continue
            with contextlib.suppress(OSError):
                os.rmdir(job_dir.path)
        return removed

    def _sweep_flat(self, root: str, cutoff: float) -> int:
        removed = 0
        for entry in self._scandir(root):
            try:
                if entry.is_file() and entry.stat().st_mtime <= cutoff:
                    os.unlink(entry.path)
                    removed += 1
            except OSError:
                continue
        return removed

    def _sweep_claims(self, lease_grace_seconds: float) -> int:
        """Sweep claims by lease content, never by mtime alone.

        Deleting a live lease reintroduces exactly the concurrent-duplicate run
        the lease exists to prevent, so expiry is read from the object itself.
        """
        now = time.time()
        removed = 0
        for entry in self._scandir(os.path.join(self._base, "claims")):
            if not entry.is_file():
                continue
            try:
                with open(entry.path, "rb") as f:
                    lease = json.loads(f.read())
                expires_at = float(lease["expires_at"])
            except (OSError, ValueError, TypeError, KeyError):
                # Unparseable or legacy bare-int claim: fall back to mtime under
                # the same grace window.
                try:
                    expires_at = entry.stat().st_mtime
                except OSError:
                    continue
            if expires_at + lease_grace_seconds < now:
                try:
                    os.unlink(entry.path)
                    removed += 1
                except OSError:
                    continue
        return removed

    @staticmethod
    def _scandir(path: str) -> list[os.DirEntry]:
        try:
            with os.scandir(path) as it:
                return list(it)
        except OSError:
            return []


def _create(config: Config) -> FsObjectStore:
    return FsObjectStore(config.storage.fs)


def register_fs_store() -> None:
    register_store("fs", _create)
