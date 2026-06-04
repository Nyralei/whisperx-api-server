import asyncio
import logging
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
    status
)
from fastapi.responses import Response
from typing import Literal, Annotated
import time


import whisperx_api_server.transcriber as transcriber
from whisperx_api_server.transcriber import (
    InvalidAudioError,
    QueueFullError,
    UploadTooLargeError,
)
from whisperx_api_server.url_fetch import filename_from_url
from whisperx_api_server import request_status
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.formatters import format_transcription
from whisperx_api_server.config import (
    DistributedMode,
    Language,
    ResponseFormat,
)

try:
    import torch as _torch
    _CUDA_OOM_EXC: type[BaseException] = _torch.cuda.OutOfMemoryError
except Exception:  # torch missing or no CUDA build
    _CUDA_OOM_EXC = RuntimeError  # benign fallback; isinstance still works


def _raise_for_transcription_error(request_id: str, e: Exception, kind: str) -> None:
    """Map domain exceptions to specific HTTP codes; fall through to 500."""
    if isinstance(e, InvalidAudioError):
        logger.info(f"Request ID: {request_id} - Invalid audio: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    if isinstance(e, UploadTooLargeError):
        logger.info(f"Request ID: {request_id} - Upload too large: {e}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e)) from e
    if isinstance(e, QueueFullError):
        logger.warning(f"Request ID: {request_id} - Queue full: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is busy, please try again later.",
        ) from e
    if isinstance(e, asyncio.TimeoutError) or isinstance(e, TimeoutError):
        logger.warning(f"Request ID: {request_id} - Timeout: {e}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e)) from e
    if isinstance(e, _CUDA_OOM_EXC) and "out of memory" in str(e).lower():
        logger.error(f"Request ID: {request_id} - CUDA OOM: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Out of GPU memory; retry shortly.",
        ) from e
    logger.exception(f"Request ID: {request_id} - {kind} failed: {e}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"An unexpected error occurred while processing the {kind} request.",
    ) from e


logger = logging.getLogger(__name__)

config = get_config()

router = APIRouter()

# OpenAI clients send model="whisper-1" as a placeholder; treat that — and an omitted
# model — as "unspecified" so the transcription backend's own default is used. The
# default is resolved where the backend actually runs: the worker in Kafka mode
# (whisperx_worker/processor.py) and transcriber.transcribe in direct mode. This keeps
# the API independent of which transcription backend the worker is configured with.
_OPENAI_PLACEHOLDER_MODEL = "whisper-1"


def _requested_model(model: str | None) -> str | None:
    if model is None or model == _OPENAI_PLACEHOLDER_MODEL:
        return None
    return model


def get_timestamp_granularities(
    timestamp_granularities: list[Literal["segment", "word"]] | None,
) -> list[Literal["segment", "word"]]:
    TIMESTAMP_GRANULARITIES_COMBINATIONS = [
        [],
        ["segment"],
        ["word"],
        ["word", "segment"],
        ["segment", "word"],
    ]
    if timestamp_granularities is None:
        return ["segment"]
    if timestamp_granularities not in TIMESTAMP_GRANULARITIES_COMBINATIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`.",
        )
    return timestamp_granularities


"""
OpenAI-like endpoint to transcribe audio files using the configured transcription backend.

Args:
    request (Request): The HTTP request object.
    file (UploadFile): The audio file to transcribe.
    model (str | None): Transcription model name; omit (or send "whisper-1") to use the backend default.
    language (Language): The language to use for the transcription. Defaults to "en".
    prompt (str): The prompt to use for the transcription.
    response_format (ResponseFormat): The response format to use for the transcription. Defaults to "json".
    temperature (float): The temperature to use for the transcription. Defaults to 0.0.
    timestamp_granularities (list[Literal["segment", "word"]]): The timestamp granularities to use for the transcription. Defaults to ["segment"].
    stream (bool): Whether to enable streaming mode. Defaults to False.
    hotwords (str): The hotwords to use for the transcription.
    suppress_numerals (bool): Whether to suppress numerals in the transcription. Defaults to True.
    highlight_words (bool): Whether to highlight words in the transcription (Applies only to VTT and SRT). Defaults to False.
    align (bool): Whether to do transcription timings alignment. Defaults to True.
    diarize (bool): Whether to diarize the transcription. Defaults to False.
    chunk_size (int): Chunk size in seconds for merging VAD segments. Defaults to 30.

Returns:
    Transcription: The transcription of the audio file.
"""


@router.post(
    "/v1/audio/transcriptions",
    description="Transcribe audio files using the configured transcription backend.",
    tags=["Transcription"],
)
async def transcribe_audio(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    audio_url: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    language: Annotated[Language, Form()] = config.default_language,
    prompt: Annotated[str, Form()] = None,
    response_format: Annotated[ResponseFormat,
                               Form()] = config.default_response_format,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        list[Literal["segment", "word"]],
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str, Form()] = None,
    suppress_numerals: Annotated[bool, Form()] = True,
    highlight_words: Annotated[bool, Form()] = False,
    align: Annotated[bool, Form()] = True,
    diarize: Annotated[bool, Form()] = False,
    speaker_embeddings: Annotated[bool, Form()] = False,
    chunk_size: Annotated[int, Form()] = config.whisper.chunk_size,
    batch_size: Annotated[int, Form()] = config.whisper.batch_size
) -> Response:
    timestamp_granularities = get_timestamp_granularities(
        timestamp_granularities)
    request_id = request.state.request_id
    logger.info(f"Request ID: {request_id} - Received transcription request")
    start_time = time.time()
    model_name = _requested_model(model)
    use_url = bool(audio_url)
    if not use_url and (file is None or not file.filename):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either 'file' or 'audio_url'.",
        )
    if use_url and file is not None and file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide exactly one of 'file' or 'audio_url', not both.",
        )
    if use_url:
        source_filename = filename_from_url(audio_url)
    else:
        source_filename = file.filename
    request_status.start(
        request_id,
        mode=config.mode.value,
        filename=source_filename,
        params={
            "endpoint": "transcriptions",
            "model": model_name,
            "language": language.value if language else None,
            "align": align,
            "diarize": diarize,
        },
    )
    logger.info(f"Request ID: {request_id} - Received request to transcribe {source_filename} with parameters: \
        model: {model_name}, \
        language: {language}, \
        prompt: {prompt}, \
        response_format: {response_format}, \
        temperature: {temperature}, \
        timestamp_granularities: {timestamp_granularities}, \
        stream: {stream}, \
        hotwords: {hotwords}, \
        suppress_numerals: {suppress_numerals}, \
        highlight_words: {highlight_words}, \
        align: {align}, \
        diarize: {diarize}, \
        speaker_embeddings: {speaker_embeddings}, \
        chunk_size: {chunk_size}, \
        batch_size: {batch_size}")

    if not align:
        if response_format in ('vtt', 'srt', 'aud', 'vtt_json'):
            detail = "Subtitles format ('vtt', 'srt', 'aud', 'vtt_json') requires alignment to be enabled."
            request_status.mark_failed(request_id, detail, "HTTPException")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=detail,
            )

        if diarize:
            detail = "Diarization requires alignment to be enabled."
            request_status.mark_failed(request_id, detail, "HTTPException")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=detail,
            )

    word_timestamps = "word" in timestamp_granularities

    asr_options = {
        "suppress_numerals": suppress_numerals,
        "temperatures": temperature,
        "word_timestamps": word_timestamps,
        "initial_prompt": prompt,
        "hotwords": hotwords,
    }

    try:
        if config.mode == DistributedMode.KAFKA:
            params = {
                "model_name": model_name,
                "language": language.value if language else None,
                "task": "transcribe",
                "align": align,
                "diarize": diarize,
                "speaker_embeddings": speaker_embeddings,
                "batch_size": batch_size,
                "chunk_size": chunk_size,
                "asr_options": asr_options,
            }
            transcription = await transcriber.transcribe_via_kafka(
                audio_file=None if use_url else file,
                source_url=audio_url if use_url else None,
                params=params,
                request_id=request_id,
            )
        else:
            transcription = await transcriber.transcribe(
                audio_file=None if use_url else file,
                source_url=audio_url if use_url else None,
                batch_size=batch_size,
                asr_options=asr_options,
                language=language,
                model_name=model_name,
                align=align,
                diarize=diarize,
                speaker_embeddings=speaker_embeddings,
                chunk_size=chunk_size,
                request_id=request_id,
            )
    except asyncio.CancelledError:
        logger.info(
            f"Request ID: {request_id} - Client disconnected; cancelling")
        raise
    except Exception as e:
        _raise_for_transcription_error(request_id, e, "transcription")

    total_time = time.time() - start_time
    logger.info(
        f"Request ID: {request_id} - Transcription process took {total_time:.2f} seconds")

    return format_transcription(transcription, response_format, highlight_words=highlight_words)

"""
OpenAI-like endpoint to translate audio files using the configured transcription backend.

Args:
    request (Request): The HTTP request object.
    file (UploadFile): The audio file to translate.
    model (str | None): Translation model name; omit (or send "whisper-1") to use the backend default.
    prompt (str): The prompt to use for the translation.
    response_format (ResponseFormat): The response format to use for the translation. Defaults to "json".
    temperature (float): The temperature to use for the translation. Defaults to 0.0.
    chunk_size (int): Chunk size in seconds for merging VAD segments. Defaults to 30.

Returns:
    Translation: The translation of the audio file.
"""


@router.post(
    "/v1/audio/translations",
    description="Translate audio files using the configured transcription backend",
    tags=["Translation"],
)
async def translate_audio(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    audio_url: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    prompt: Annotated[str, Form()] = "",
    response_format: Annotated[ResponseFormat,
                               Form()] = config.default_response_format,
    temperature: Annotated[float, Form()] = 0.0,
    chunk_size: Annotated[int, Form()] = config.whisper.chunk_size,
    batch_size: Annotated[int, Form()] = config.whisper.batch_size
) -> Response:
    request_id = request.state.request_id
    logger.info(f"Request ID: {request_id} - Received translation request")
    start_time = time.time()
    model_name = _requested_model(model)
    use_url = bool(audio_url)
    if not use_url and (file is None or not file.filename):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either 'file' or 'audio_url'.",
        )
    if use_url and file is not None and file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide exactly one of 'file' or 'audio_url', not both.",
        )
    if use_url:
        source_filename = filename_from_url(audio_url)
    else:
        source_filename = file.filename
    request_status.start(
        request_id,
        mode=config.mode.value,
        filename=source_filename,
        params={
            "endpoint": "translations",
            "model": model_name,
        },
    )
    logger.info(f"Request ID: {request_id} - Received request to translate {source_filename} with parameters: \
        model: {model_name}, \
        prompt: {prompt}, \
        response_format: {response_format}, \
        temperature: {temperature}, \
        chunk_size: {chunk_size}, \
        batch_size: {batch_size}")

    asr_options = {
        "initial_prompt": prompt,
        "temperatures": temperature,
    }

    try:
        if config.mode == DistributedMode.KAFKA:
            params = {
                "model_name": model_name,
                "language": None,
                "task": "translate",
                "align": False,
                "diarize": False,
                "speaker_embeddings": False,
                "batch_size": batch_size,
                "chunk_size": chunk_size,
                "asr_options": asr_options,
            }
            translation = await transcriber.transcribe_via_kafka(
                audio_file=None if use_url else file,
                source_url=audio_url if use_url else None,
                params=params,
                request_id=request_id,
            )
        else:
            translation = await transcriber.transcribe(
                audio_file=None if use_url else file,
                source_url=audio_url if use_url else None,
                batch_size=batch_size,
                asr_options=asr_options,
                model_name=model_name,
                chunk_size=chunk_size,
                request_id=request_id,
                task="translate",
            )
    except asyncio.CancelledError:
        logger.info(
            f"Request ID: {request_id} - Client disconnected; cancelling")
        raise
    except Exception as e:
        _raise_for_transcription_error(request_id, e, "translation")

    total_time = time.time() - start_time
    logger.info(
        f"Request ID: {request_id} - Translation process took {total_time:.2f} seconds")

    return format_transcription(translation, response_format)
