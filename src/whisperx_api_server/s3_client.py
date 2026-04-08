import logging
import sys
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
    session = AioSession()
    ctx = session.create_client(
        "s3",
        region_name=cfg.region,
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
    )
    _client = await ctx.__aenter__()
    logger.info(
        f"S3 client initialized (endpoint: {cfg.endpoint_url}, bucket: {cfg.bucket})")

    try:
        try:
            await _client.head_bucket(Bucket=cfg.bucket)
        except Exception as exc:
            from botocore.exceptions import ClientError
            if not isinstance(exc, ClientError) or exc.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
                raise
            await _client.create_bucket(Bucket=cfg.bucket)
            logger.info(f"Created S3 bucket: {cfg.bucket}")

        if cfg.object_expiry_days > 0:
            await _client.put_bucket_lifecycle_configuration(
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
                f"S3 bucket lifecycle set: expire after {cfg.object_expiry_days} day(s)")
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
    logger.debug(f"Uploaded {len(data)} bytes to s3://{_config.bucket}/{key}")
    return key


async def download_audio(key: str) -> bytes:
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    response = await _client.get_object(Bucket=_config.bucket, Key=key)
    async with response["Body"] as stream:
        data = await stream.read()
    logger.debug(
        f"Downloaded {len(data)} bytes from s3://{_config.bucket}/{key}")
    return data


async def delete_audio(key: str) -> None:
    if _client is None or _config is None:
        raise RuntimeError("S3 client not initialized")
    await _client.delete_object(Bucket=_config.bucket, Key=key)
    logger.debug(f"Deleted s3://{_config.bucket}/{key}")
