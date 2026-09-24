import asyncio
import contextlib
import json
import logging
import os
import random
import sys
import tempfile
import time
from typing import TYPE_CHECKING

from whisperx_api_server.config import S3Config

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client

logger = logging.getLogger(__name__)

_client: "S3Client | None" = None
_config: S3Config | None = None
_ctx = None


async def init_client(cfg: S3Config) -> None:
    global _client, _config, _ctx
    from aiobotocore.session import AioSession

    _config = cfg
    from botocore.config import Config as BotocoreConfig

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
    _client = client
    logger.info(
        "S3 client initialized (endpoint: %s, bucket: %s)", cfg.endpoint_url, cfg.bucket
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
        _client = None
        raise

    _ctx = ctx


async def close_client() -> None:
    global _client, _ctx
    if _ctx is not None:
        await _ctx.__aexit__(None, None, None)
        _client = None
        _ctx = None
        logger.info("S3 client closed")


async def upload_audio(data: bytes, job_id: str, filename: str) -> str:
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    key = f"audio/{job_id}/{filename}"
    await _client.put_object(Bucket=_config.bucket, Key=key, Body=data)
    logger.debug("Uploaded %s bytes to s3://%s/%s", len(data), _config.bucket, key)
    return key


# A non-final part must be at least 5 MiB, and one upload may not exceed 10000 parts.
_MIN_PART_SIZE = 5 * 1024 * 1024
_MAX_PARTS = 10000


def _part_size_for(content_length: int | None) -> int:
    """Part size that keeps the upload under the 10000-part limit.

    Scales past the configured size for files large enough to run the part count
    out, so the limit is never discovered at part 10001 with the transfer wasted.
    """
    assert _config is not None
    part_size = max(_MIN_PART_SIZE, _config.multipart_part_size)
    if content_length is None:
        return part_size
    required = -(-content_length // _MAX_PARTS)
    if required <= part_size:
        return part_size
    return -(-required // (1024 * 1024)) * 1024 * 1024


def _content_length(upload_file) -> int | None:
    size = getattr(upload_file, "size", None)
    return size if isinstance(size, int) and size >= 0 else None


async def upload_audio_stream(upload_file, job_id: str, filename: str) -> str:
    """Upload a FastAPI UploadFile to S3 without blocking the event loop.

    Reads go through UploadFile.read(size), which routes to anyio's thread
    executor, so bytes are never pulled off the SpooledTemporaryFile inline in
    the async path.

    Large files go out as a multipart upload. A single PutObject buffers the
    whole body in RAM, and on a plain-HTTP endpoint SigV4 cannot skip payload
    signing, so botocore hashes all of it inline on the event loop — a multi-GiB
    upload stalls the process for tens of seconds, timing out health probes and
    expiring Kafka consumer sessions. Multipart bounds both the buffer and the
    per-hash CPU burst to one part.

    Peak buffered bytes per upload are roughly the part size times
    s3.multipart_concurrency, independent of file size.
    """
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    key = f"audio/{job_id}/{filename}"
    part_size = _part_size_for(_content_length(upload_file))

    # A short first read means the body fits in one part; also covers the empty
    # upload, which multipart rejects.
    head = await upload_file.read(part_size)
    if len(head) < part_size:
        await _client.put_object(Bucket=_config.bucket, Key=key, Body=head)
        logger.debug("Uploaded %s bytes to s3://%s/%s", len(head), _config.bucket, key)
        return key

    total = await _upload_multipart(upload_file, key, head, part_size)
    logger.debug(
        "Uploaded %s bytes to s3://%s/%s (multipart, %s byte parts)",
        total,
        _config.bucket,
        key,
        part_size,
    )
    return key


async def _upload_multipart(upload_file, key: str, head: bytes, part_size: int) -> int:
    """Stream upload_file to key as a multipart upload; returns bytes written.

    head is the already-read first part. Aborts on failure so stored parts do
    not linger as billable storage.
    """
    assert _client is not None and _config is not None
    created = await _client.create_multipart_upload(Bucket=_config.bucket, Key=key)
    upload_id = created["UploadId"]
    try:
        parts = await _send_parts(upload_file, key, upload_id, head, part_size)
        await _client.complete_multipart_upload(
            Bucket=_config.bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": [descriptor for _, descriptor in parts]},
        )
    except BaseException:
        with contextlib.suppress(Exception):
            await _client.abort_multipart_upload(
                Bucket=_config.bucket, Key=key, UploadId=upload_id
            )
        raise
    return sum(size for size, _ in parts)


async def _send_parts(
    upload_file, key: str, upload_id: str, head: bytes, part_size: int
) -> list[tuple[int, dict]]:
    """Upload every part with bounded concurrency.

    Returns (size, part-descriptor) pairs ordered by part number, the order
    complete_multipart_upload requires.
    """
    assert _client is not None and _config is not None
    client, bucket = _client, _config.bucket
    semaphore = asyncio.Semaphore(max(1, _config.multipart_concurrency))
    tasks: list[asyncio.Task[tuple[int, dict]]] = []

    async def send(part_number: int, chunk: bytes) -> tuple[int, dict]:
        try:
            response = await client.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=chunk,
            )
            return len(chunk), {"ETag": response["ETag"], "PartNumber": part_number}
        finally:
            semaphore.release()

    async def dispatch(part_number: int, chunk: bytes) -> None:
        # Acquired here, not inside send(), so back-pressure applies before the
        # next read allocates another chunk.
        await semaphore.acquire()
        tasks.append(asyncio.create_task(send(part_number, chunk)))

    try:
        await dispatch(1, head)
        part_number = 2
        while True:
            chunk = await upload_file.read(part_size)
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


_DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_DOWNLOAD_WRITE_BUFFER_SIZE = 1024 * 1024  # 1 MiB


async def download_audio_to_temp(key: str, suffix: str = "") -> str:
    """Stream an S3 object into a temp file; returns the file path."""
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_path = tmp.name
    total = 0
    try:
        response = await _client.get_object(Bucket=_config.bucket, Key=key)
        # keep the StreamingBody wrapper: `async with` unwraps to the raw
        # aiohttp response, whose read() takes no size argument
        body = response["Body"]
        try:
            with open(file_path, "wb", buffering=_DOWNLOAD_WRITE_BUFFER_SIZE) as f:
                while True:
                    chunk = await body.read(_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    f.write(chunk)
        finally:
            body.close()
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(file_path)
        raise
    logger.debug(
        "Downloaded %s bytes from s3://%s/%s to temp file", total, _config.bucket, key
    )
    return file_path


async def delete_audio(key: str) -> None:
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    await _client.delete_object(Bucket=_config.bucket, Key=key)
    logger.debug("Deleted s3://%s/%s", _config.bucket, key)


_RESULTS_PREFIX = "results/"
_CLAIMS_PREFIX = "claims/"


def _is_not_found(exc: Exception) -> bool:
    from botocore.exceptions import ClientError

    if not isinstance(exc, ClientError):
        return False
    return exc.response.get("Error", {}).get("Code") in (
        "404",
        "NoSuchKey",
        "NoSuchBucket",
    )


async def put_result(job_id: str, envelope: bytes) -> None:
    """Store the terminal reply envelope so a redelivery can resend it."""
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    await _client.put_object(
        Bucket=_config.bucket,
        Key=f"{_RESULTS_PREFIX}{job_id}",
        Body=envelope,
    )


async def get_result(job_id: str) -> str | None:
    """Return the stored reply envelope, or None if the job hasn't finished."""
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    try:
        response = await _client.get_object(
            Bucket=_config.bucket, Key=f"{_RESULTS_PREFIX}{job_id}"
        )
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise
    async with response["Body"] as stream:
        data = await stream.read()
    return data.decode()


# The claims/{job_id} object is a processing lease: it makes a redelivered copy
# of an in-flight job defer instead of starting a concurrent duplicate run, and
# it carries the delivery-attempt counter used for DLQ routing. Acquisition is
# atomic via conditional PUT (If-None-Match); backends without conditional-write
# support degrade to a verified write (small race window, logged once).
_conditional_writes_supported = True


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


async def _get_lease(job_id: str) -> dict | None:
    assert _client is not None and _config is not None
    try:
        response = await _client.get_object(
            Bucket=_config.bucket, Key=f"{_CLAIMS_PREFIX}{job_id}"
        )
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise
    async with response["Body"] as stream:
        data = await stream.read()
    try:
        lease = json.loads(data)
        if isinstance(lease, dict):
            return lease
    except ValueError:
        pass
    # Legacy bare-int claim counter from a pre-lease worker: preserve the
    # attempt count, treat as expired so a lease-aware worker can take over.
    try:
        attempts = int(data.decode().strip() or "0")
    except ValueError:
        attempts = 0
    return {"attempts": attempts, "owner": None, "expires_at": 0.0}


async def _put_lease_if_absent(job_id: str, lease: dict) -> bool:
    """Create the lease object only if none exists; True if this call created it."""
    global _conditional_writes_supported
    assert _client is not None and _config is not None
    key = f"{_CLAIMS_PREFIX}{job_id}"
    body = json.dumps(lease).encode()
    if _conditional_writes_supported:
        try:
            await _client.put_object(
                Bucket=_config.bucket, Key=key, Body=body, IfNoneMatch="*"
            )
            return True
        except Exception as exc:
            if _is_precondition_failed(exc):
                return False
            if not _is_conditional_unsupported(exc):
                raise
            _conditional_writes_supported = False
            logger.warning(
                "S3 backend does not support conditional writes (If-None-Match); "
                "job lease acquisition degrades to verified writes with a small "
                "race window — upgrade the S3/MinIO backend for atomic leases"
            )
    # Fallback: emulate if-absent — check, write, then read back after a jitter
    # so concurrent writers converge on a single surviving owner.
    if await _get_lease(job_id) is not None:
        return False
    await _client.put_object(Bucket=_config.bucket, Key=key, Body=body)
    await asyncio.sleep(random.uniform(0.05, 0.15))
    current = await _get_lease(job_id)
    return current is not None and current.get("owner") == lease["owner"]


async def acquire_job_lease(
    job_id: str, worker_id: str, ttl_seconds: float
) -> tuple[bool, int]:
    """Claim exclusive processing rights for a job; return (acquired, attempts).

    A live lease held by another worker means the job is being processed right
    now (e.g. a rebalance redelivered its uncommitted message) — the caller must
    defer, not process. An expired lease or one left by a previous incarnation
    of this worker (same worker_id after a crash-restart) is taken over, which
    also advances the attempts counter used for DLQ routing.
    """
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    fresh = {"attempts": 1, "owner": worker_id, "expires_at": time.time() + ttl_seconds}
    if await _put_lease_if_absent(job_id, fresh):
        return True, 1

    current = await _get_lease(job_id)
    if current is None:
        # Deleted between the failed create and the read (holder just released);
        # one retry, then give up and let the requeued copy sort it out.
        if await _put_lease_if_absent(job_id, fresh):
            return True, 1
        current = await _get_lease(job_id) or {"attempts": 0}

    attempts = int(current.get("attempts") or 0)
    is_live = (
        float(current.get("expires_at") or 0.0) > time.time()
        and current.get("owner") != worker_id
    )
    if is_live:
        return False, attempts

    takeover = {
        "attempts": attempts + 1,
        "owner": worker_id,
        "expires_at": time.time() + ttl_seconds,
    }
    await delete_claim(job_id)
    if await _put_lease_if_absent(job_id, takeover):
        return True, attempts + 1
    return False, attempts


async def renew_job_lease(job_id: str, worker_id: str, ttl_seconds: float) -> bool:
    """Extend a held lease; False if it is gone or owned by someone else."""
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    current = await _get_lease(job_id)
    if current is None or current.get("owner") != worker_id:
        return False
    current["expires_at"] = time.time() + ttl_seconds
    await _client.put_object(
        Bucket=_config.bucket,
        Key=f"{_CLAIMS_PREFIX}{job_id}",
        Body=json.dumps(current).encode(),
    )
    return True


async def delete_claim(job_id: str) -> None:
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    await _client.delete_object(Bucket=_config.bucket, Key=f"{_CLAIMS_PREFIX}{job_id}")
