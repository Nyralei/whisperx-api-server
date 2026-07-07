import contextlib
import logging
import os
import sys
import tempfile
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


async def upload_audio_stream(upload_file, job_id: str, filename: str) -> str:
    """Upload a FastAPI UploadFile to S3 without blocking the event loop.

    Passing the SpooledTemporaryFile directly as aiobotocore's Body serializes
    concurrent uploads: aiohttp's request body iteration calls fileobj.read()
    inline in the async path, so the loop stalls while bytes are pulled. Reading
    via UploadFile.read() instead routes through anyio's thread executor, so
    multiple concurrent uploads interleave. We then hand aiobotocore a plain
    bytes Body, which aiohttp sends without any further blocking I/O.

    Memory: peak buffered bytes are bounded by kafka.max_pending_jobs * payload
    size (default 100 * audio file size). SpooledTemporaryFile spills to disk
    above 1MB so the source was never strictly disk-only for small files; this
    swap turns the small-file case into RAM-only and pulls large files fully
    into RAM at upload time. Adjust kafka.max_pending_jobs if that ceiling is
    too high for the deployment's typical file sizes.
    """
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    key = f"audio/{job_id}/{filename}"

    data = await upload_file.read()
    await _client.put_object(Bucket=_config.bucket, Key=key, Body=data)
    logger.debug("Uploaded %s bytes to s3://%s/%s", len(data), _config.bucket, key)
    return key


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


async def put_result(job_id: str, envelope: str) -> None:
    """Store the terminal reply envelope so a redelivery can resend it."""
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    await _client.put_object(
        Bucket=_config.bucket,
        Key=f"{_RESULTS_PREFIX}{job_id}",
        Body=envelope.encode(),
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


async def increment_claim(job_id: str) -> int:
    """Read-modify-write the per-job delivery counter; return the new count."""
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    key = f"{_CLAIMS_PREFIX}{job_id}"
    count = 0
    try:
        response = await _client.get_object(Bucket=_config.bucket, Key=key)
    except Exception as exc:
        if not _is_not_found(exc):
            raise
    else:
        async with response["Body"] as stream:
            data = await stream.read()
        try:
            count = int(data.decode().strip() or "0")
        except ValueError:
            count = 0
    count += 1
    await _client.put_object(Bucket=_config.bucket, Key=key, Body=str(count).encode())
    return count


async def delete_claim(job_id: str) -> None:
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    await _client.delete_object(Bucket=_config.bucket, Key=f"{_CLAIMS_PREFIX}{job_id}")
