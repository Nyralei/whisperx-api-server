"""S3-compatible object store (aiobotocore)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from whisperx_api_server.config import Config, S3Config

from .contracts import ObjectNotFound, StorageCapabilityError
from .registry import register_store

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client

logger = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_DOWNLOAD_WRITE_BUFFER_SIZE = 1024 * 1024  # 1 MiB

# A non-final part must be at least 5 MiB, and one upload may not exceed 10000 parts.
_MIN_PART_SIZE = 5 * 1024 * 1024
_MAX_PARTS = 10000


def _is_not_found(exc: Exception) -> bool:
    from botocore.exceptions import ClientError

    if not isinstance(exc, ClientError):
        return False
    return exc.response.get("Error", {}).get("Code") in (
        "404",
        "NoSuchKey",
        "NoSuchBucket",
    )


def _is_precondition_failed(exc: Exception) -> bool:
    from botocore.exceptions import ClientError

    if not isinstance(exc, ClientError):
        return False
    return exc.response.get("Error", {}).get("Code") in (
        "PreconditionFailed",
        "412",
        "ConditionalRequestConflict",
    )


def _is_conditional_unsupported(exc: Exception) -> bool:
    from botocore.exceptions import ClientError, ParamValidationError

    if isinstance(exc, ParamValidationError):
        return True
    if not isinstance(exc, ClientError):
        return False
    return exc.response.get("Error", {}).get("Code") in ("NotImplemented", "501")


class _PartAssembler:
    """Regroups an arbitrarily chunked stream into fixed-size upload parts.

    The caller's chunks are sized for its own convenience (1 MiB reads off an
    UploadFile); S3 constrains part sizes, so the two cannot be the same unit.
    """

    def __init__(self, chunks: AsyncIterator[bytes], part_size: int) -> None:
        self._chunks = chunks
        self._part_size = part_size
        self._buf = bytearray()
        self._exhausted = False

    async def next_part(self) -> bytes:
        """Return the next full part, a short final part, or b"" once drained."""
        while not self._exhausted and len(self._buf) < self._part_size:
            try:
                chunk = await self._chunks.__anext__()
            except StopAsyncIteration:
                self._exhausted = True
                break
            self._buf.extend(chunk)
        part = bytes(self._buf[: self._part_size])
        del self._buf[: self._part_size]
        return part


class S3ObjectStore:
    name = "s3"

    def __init__(self, cfg: S3Config) -> None:
        self._cfg = cfg
        self._client: S3Client | None = None
        self._ctx = None

    @property
    def _c(self) -> S3Client:
        if self._client is None:
            raise RuntimeError("S3 client not initialized")
        return self._client

    async def open(self) -> None:
        from aiobotocore.session import AioSession
        from botocore.config import Config as BotocoreConfig

        cfg = self._cfg
        session = AioSession()
        ctx = session.create_client(
            "s3",
            region_name=cfg.region,
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=cfg.access_key_id,
            aws_secret_access_key=cfg.secret_access_key,
            config=BotocoreConfig(
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=120,
            ),
        )
        client: S3Client = await ctx.__aenter__()
        self._client = client
        logger.info(
            "S3 client initialized (endpoint: %s, bucket: %s)",
            cfg.endpoint_url,
            cfg.bucket,
        )

        try:
            try:
                await client.head_bucket(Bucket=cfg.bucket)
            except Exception as exc:
                from botocore.exceptions import ClientError

                if not isinstance(exc, ClientError) or exc.response["Error"][
                    "Code"
                ] not in ("404", "NoSuchBucket"):
                    raise
                await client.create_bucket(Bucket=cfg.bucket)
                logger.info("Created S3 bucket: %s", cfg.bucket)

            if cfg.manage_lifecycle and cfg.object_expiry_days > 0:
                await client.put_bucket_lifecycle_configuration(
                    Bucket=cfg.bucket,
                    LifecycleConfiguration={
                        "Rules": [
                            {
                                "ID": "expire-audio",
                                "Status": "Enabled",
                                "Filter": {"Prefix": ""},
                                "Expiration": {"Days": cfg.object_expiry_days},
                            }
                        ]
                    },
                )
                logger.debug(
                    "S3 bucket lifecycle set: expire after %s day(s)",
                    cfg.object_expiry_days,
                )
        except Exception:
            await ctx.__aexit__(*sys.exc_info())
            self._client = None
            raise

        self._ctx = ctx

    async def close(self) -> None:
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._client = None
            self._ctx = None
            logger.info("S3 client closed")

    async def put_bytes(self, *, key: str, data: bytes) -> None:
        await self._c.put_object(Bucket=self._cfg.bucket, Key=key, Body=data)
        logger.debug(
            "Uploaded %s bytes to s3://%s/%s", len(data), self._cfg.bucket, key
        )

    def _part_size_for(self, content_length):
        """Part size that keeps the upload under the 10000-part limit.

        Scales past the configured size for bodies large enough to run the part
        count out, so the limit is never discovered at part 10001 with the
        transfer already wasted.
        """
        part_size = max(_MIN_PART_SIZE, self._cfg.multipart_part_size)
        if content_length is None:
            return part_size
        required = -(-content_length // _MAX_PARTS)
        if required <= part_size:
            return part_size
        return -(-required // (1024 * 1024)) * 1024 * 1024

    async def put_stream(
        self,
        *,
        key: str,
        chunks: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> int:
        """Upload the stream, switching to multipart once it outgrows one part.

        A single PutObject buffers the whole body in RAM, and on a plain-HTTP
        endpoint SigV4 cannot skip payload signing, so botocore hashes all of it
        inline on the event loop — a multi-GiB upload stalls the process for tens
        of seconds, timing out health probes and expiring Kafka consumer
        sessions. Multipart bounds both the buffer and the per-hash CPU burst to
        one part.

        Peak buffered bytes are roughly the part size times
        s3.multipart_concurrency, independent of body size.
        """
        part_size = self._part_size_for(content_length)
        parts = _PartAssembler(chunks, part_size)

        # A short first part means the body fits in one PUT; this also covers the
        # empty body, which multipart rejects.
        head = await parts.next_part()
        if len(head) < part_size:
            await self.put_bytes(key=key, data=head)
            return len(head)

        total = await self._upload_multipart(key, parts, head, part_size)
        logger.debug(
            "Uploaded %s bytes to s3://%s/%s (multipart, %s byte parts)",
            total,
            self._cfg.bucket,
            key,
            part_size,
        )
        return total

    async def _upload_multipart(self, key, parts, head, part_size) -> int:
        """Run the multipart upload for key; returns bytes written.

        head is the already-read first part. Aborts on failure so stored parts do
        not linger as billable storage.
        """
        created = await self._c.create_multipart_upload(
            Bucket=self._cfg.bucket, Key=key
        )
        upload_id = created["UploadId"]
        try:
            sent = await self._send_parts(key, upload_id, parts, head, part_size)
            await self._c.complete_multipart_upload(
                Bucket=self._cfg.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": [descriptor for _, descriptor in sent]},
            )
        except BaseException:
            with contextlib.suppress(Exception):
                await self._c.abort_multipart_upload(
                    Bucket=self._cfg.bucket, Key=key, UploadId=upload_id
                )
            raise
        return sum(size for size, _ in sent)

    async def _send_parts(self, key, upload_id, parts, head, part_size):
        """Upload every part with bounded concurrency.

        Returns (size, part-descriptor) pairs ordered by part number, the order
        complete_multipart_upload requires.
        """
        client, bucket = self._c, self._cfg.bucket
        semaphore = asyncio.Semaphore(max(1, self._cfg.multipart_concurrency))
        tasks: list[asyncio.Task] = []

        async def send(part_number: int, chunk: bytes):
            try:
                response = await client.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk,
                )
                return len(chunk), {
                    "ETag": response["ETag"],
                    "PartNumber": part_number,
                }
            finally:
                semaphore.release()

        async def dispatch(part_number: int, chunk: bytes) -> None:
            # Acquired here, not inside send(), so back-pressure applies before
            # the next read allocates another part.
            await semaphore.acquire()
            tasks.append(asyncio.create_task(send(part_number, chunk)))

        try:
            await dispatch(1, head)
            part_number = 2
            while True:
                chunk = await parts.next_part()
                if not chunk:
                    break
                if part_number > _MAX_PARTS:
                    # Only reachable when the content length was unknown up front.
                    raise ValueError(
                        f"Upload exceeds the {_MAX_PARTS}-part S3 limit at "
                        f"{part_size} bytes per part; raise s3.multipart_part_size"
                    )
                await dispatch(part_number, chunk)
                part_number += 1
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def put_if_absent(self, *, key: str, data: bytes) -> bool:
        try:
            await self._c.put_object(
                Bucket=self._cfg.bucket, Key=key, Body=data, IfNoneMatch="*"
            )
            return True
        except Exception as exc:
            if _is_precondition_failed(exc):
                return False
            if _is_conditional_unsupported(exc):
                raise StorageCapabilityError(
                    "The configured S3 backend does not support atomic "
                    "create-if-absent (conditional PUT with If-None-Match)."
                ) from exc
            raise

    async def get_bytes(self, *, key: str) -> bytes | None:
        try:
            response = await self._c.get_object(Bucket=self._cfg.bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        async with response["Body"] as stream:
            return await stream.read()

    async def download_to_path(self, *, key: str, path: str) -> int:
        total = 0
        try:
            response = await self._c.get_object(Bucket=self._cfg.bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                raise ObjectNotFound(f"No such object: {key}") from exc
            raise
        # keep the StreamingBody wrapper: `async with` unwraps to the raw
        # aiohttp response, whose read() takes no size argument
        body = response["Body"]
        try:
            with open(path, "wb", buffering=_DOWNLOAD_WRITE_BUFFER_SIZE) as f:
                while True:
                    chunk = await body.read(_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    f.write(chunk)
        finally:
            body.close()
        logger.debug(
            "Downloaded %s bytes from s3://%s/%s", total, self._cfg.bucket, key
        )
        return total

    async def delete(self, *, key: str) -> None:
        await self._c.delete_object(Bucket=self._cfg.bucket, Key=key)
        logger.debug("Deleted s3://%s/%s", self._cfg.bucket, key)


def _create(config: Config) -> S3ObjectStore:
    return S3ObjectStore(config.s3)


def register_s3_store() -> None:
    register_store("s3", _create)
