from functools import lru_cache
from typing import Annotated

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

async def verify_api_key(
    config: ConfigDependency, credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> None:
    if credentials.credentials != config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

ApiKeyDependency = Depends(verify_api_key)