from functools import lru_cache
from typing import Annotated
import json
import logging
import os
import threading
from fastapi import (
    Depends,
    HTTPException,
    status
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from whisperx_api_server.config import Config


@lru_cache
def get_config() -> Config:
    return Config()


ConfigDependency = Annotated[Config, Depends(get_config)]

security = HTTPBearer()

logger = logging.getLogger(__name__)
_api_keys_cache: dict[str, str] = {}
_api_keys_cache_mtime_ns: int | None = None
_api_keys_cache_path: str | None = None
_api_keys_cache_lock = threading.Lock()


def _load_api_keys(api_keys_file: str) -> dict[str, str]:
    global _api_keys_cache
    global _api_keys_cache_mtime_ns
    global _api_keys_cache_path

    stat = os.stat(api_keys_file)
    current_mtime_ns = stat.st_mtime_ns

    with _api_keys_cache_lock:
        if (
            _api_keys_cache_path == api_keys_file
            and _api_keys_cache_mtime_ns == current_mtime_ns
        ):
            return _api_keys_cache

    with open(api_keys_file, "r", encoding="utf-8") as f:
        loaded_api_keys = json.load(f)
    if not isinstance(loaded_api_keys, dict):
        raise ValueError("API keys file must contain a JSON object")

    with _api_keys_cache_lock:
        _api_keys_cache = loaded_api_keys
        _api_keys_cache_mtime_ns = current_mtime_ns
        _api_keys_cache_path = api_keys_file
        return _api_keys_cache


async def verify_api_key(
    config: ConfigDependency, credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> None:
    api_keys = {}

    if config.api_keys_file:
        try:
            api_keys = _load_api_keys(config.api_keys_file)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API keys file error",
            ) from e

    client_name = api_keys.get(credentials.credentials)

    if credentials.credentials != config.api_key and client_name is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")

    if client_name:
        logger.info(f"Authorized request from client: '{client_name}'")
    else:
        logger.info("Authorized request using the default API key")

ApiKeyDependency = Depends(verify_api_key)
