import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

MCP_SERVER_NAME = "whisperx-api-server"
DEFAULT_BASE_URL = "http://127.0.0.1:8001"
BASE_URL_ENV = "WHISPERX_API_BASE_URL"
API_KEY_ENV = "WHISPERX_API_KEY"
API_TIMEOUT_ENV = "WHISPERX_API_TIMEOUT_SECONDS"

mcp = FastMCP(MCP_SERVER_NAME)


def _api_timeout() -> float:
    raw = os.getenv(API_TIMEOUT_ENV, "600")
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 600.0


def _api_base_url() -> str:
    return os.getenv(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")


def _auth_headers() -> dict[str, str]:
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _decode_response(response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
    else:
        body = response.text
    return {
        "status_code": response.status_code,
        "content_type": content_type,
        "body": body,
    }


def _ensure_success(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"WhisperX API request failed: {exc.response.status_code} {exc.response.text}"
        ) from exc


def _post_form(path: str, data: dict[str, Any]) -> dict[str, Any]:
    url = f"{_api_base_url()}{path}"
    with httpx.Client(timeout=_api_timeout()) as client:
        response = client.post(url, headers=_auth_headers(), data=data)
    _ensure_success(response)
    return _decode_response(response)


def _get_json(path: str) -> dict[str, Any]:
    url = f"{_api_base_url()}{path}"
    with httpx.Client(timeout=_api_timeout()) as client:
        response = client.get(url, headers=_auth_headers())
    _ensure_success(response)
    return _decode_response(response)


def _validate_audio_path(audio_path: str) -> Path:
    path = Path(audio_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Audio file does not exist: {audio_path}")
    return path


@mcp.tool()
def whisperx_healthcheck() -> dict[str, Any]:
    """Check that the WhisperX API is healthy."""
    return _get_json("/healthcheck")


@mcp.tool()
def whisperx_list_models() -> dict[str, Any]:
    """List loaded transcription models."""
    return _get_json("/models/list")


@mcp.tool()
def whisperx_load_model(model: str) -> dict[str, Any]:
    """Load a transcription model into memory."""
    return _post_form("/models/load", {"model": model})


@mcp.tool()
def whisperx_unload_model(model: str) -> dict[str, Any]:
    """Unload a transcription model from memory."""
    return _post_form("/models/unload", {"model": model})


@mcp.tool()
def whisperx_list_align_models() -> dict[str, Any]:
    """List loaded alignment models."""
    return _get_json("/align_models/list")


@mcp.tool()
def whisperx_load_align_model(language: str) -> dict[str, Any]:
    """Load an alignment model by language code."""
    return _post_form("/align_models/load", {"language": language})


@mcp.tool()
def whisperx_unload_align_model(language: str) -> dict[str, Any]:
    """Unload an alignment model by language code."""
    return _post_form("/align_models/unload", {"language": language})


@mcp.tool()
def whisperx_list_diarize_models() -> dict[str, Any]:
    """List loaded diarization models."""
    return _get_json("/diarize_models/list")


@mcp.tool()
def whisperx_load_diarize_model(model: str) -> dict[str, Any]:
    """Load a diarization model into memory."""
    return _post_form("/diarize_models/load", {"model": model})


@mcp.tool()
def whisperx_unload_diarize_model(model: str) -> dict[str, Any]:
    """Unload a diarization model from memory."""
    return _post_form("/diarize_models/unload", {"model": model})


@mcp.tool()
def whisperx_transcribe_audio(
    audio_path: str,
    model: str = "whisper-1",
    language: str | None = None,
    prompt: str | None = None,
    response_format: str = "json",
    temperature: float = 0.0,
    timestamp_granularities: list[str] | None = None,
    stream: bool = False,
    hotwords: str | None = None,
    suppress_numerals: bool = True,
    highlight_words: bool = False,
    align: bool = True,
    diarize: bool = False,
    speaker_embeddings: bool = False,
    chunk_size: int = 30,
    batch_size: int = 12,
) -> dict[str, Any]:
    """Transcribe an audio file using WhisperX HTTP API."""
    path = _validate_audio_path(audio_path)
    form_fields: dict[str, Any] = {
        "model": model,
        "response_format": response_format,
        "temperature": temperature,
        "stream": stream,
        "suppress_numerals": suppress_numerals,
        "highlight_words": highlight_words,
        "align": align,
        "diarize": diarize,
        "speaker_embeddings": speaker_embeddings,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
    }
    if language is not None:
        form_fields["language"] = language
    if prompt is not None:
        form_fields["prompt"] = prompt
    if hotwords is not None:
        form_fields["hotwords"] = hotwords

    for value in (timestamp_granularities or ["segment"]):
        form_fields.setdefault("timestamp_granularities[]", [])
        form_fields["timestamp_granularities[]"].append(value)

    with path.open("rb") as handle:
        files = {"file": (path.name, handle.read())}

    url = f"{_api_base_url()}/v1/audio/transcriptions"
    with httpx.Client(timeout=_api_timeout()) as client:
        response = client.post(url, headers=_auth_headers(), files=files, data=form_fields)
    _ensure_success(response)
    return _decode_response(response)


@mcp.tool()
def whisperx_translate_audio(
    audio_path: str,
    model: str = "whisper-1",
    prompt: str = "",
    response_format: str = "json",
    temperature: float = 0.0,
    chunk_size: int = 30,
    batch_size: int = 12,
) -> dict[str, Any]:
    """Translate an audio file using WhisperX HTTP API."""
    path = _validate_audio_path(audio_path)
    form_fields = {
        "model": model,
        "prompt": prompt,
        "response_format": response_format,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
    }
    with path.open("rb") as handle:
        files = {"file": (path.name, handle.read())}

    url = f"{_api_base_url()}/v1/audio/translations"
    with httpx.Client(timeout=_api_timeout()) as client:
        response = client.post(url, headers=_auth_headers(), files=files, data=form_fields)
    _ensure_success(response)
    return _decode_response(response)


@mcp.tool()
def whisperx_server_info() -> dict[str, str]:
    """Return MCP bridge runtime settings."""
    return {
        "name": MCP_SERVER_NAME,
        "mode": "http-bridge",
        "api_base_url": _api_base_url(),
        "api_key_configured": str(bool(os.getenv(API_KEY_ENV))).lower(),
        "timeout_seconds": str(_api_timeout()),
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
