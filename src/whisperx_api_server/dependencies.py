from functools import lru_cache
from typing import Annotated
import json
import logging
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


async def verify_api_key(
    config: ConfigDependency, credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> None:
    api_keys = {}

    if config.api_keys_file:
        try:
            with open(config.api_keys_file, 'r') as f:
                api_keys = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
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
