import logging
import uuid
from .models import handle_default_openai_model
from fastapi import (
    APIRouter,
    UploadFile,
    Form,
    HTTPException,
    Request,
    status
)
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Literal, Annotated
from pydantic import AfterValidator
import time


import whisperx_api_server.transcriber as transcriber
from whisperx_api_server.dependencies import ConfigDependency
from whisperx_api_server.formatters import format_transcription
from whisperx_api_server.config import (
    Language,
    ResponseFormat,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Annotated ModelName for validation and defaults
ModelName = Annotated[str, AfterValidator(handle_default_openai_model)]


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


async def get_timestamp_granularities(request: Request) -> list[Literal["segment", "word"]]:
    TIMESTAMP_GRANULARITIES_COMBINATIONS = [
        [],
        ["segment"],
        ["word"],
        ["word", "segment"],
        ["segment", "word"],
    ]
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return ["segment"]
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS, (
        f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`."
    )
    return timestamp_granularities


def apply_defaults(config, model, language=None, response_format=None):
    if model is None:
        model = config.whisper.model
    if language is None:
        language = config.default_language
    if response_format is None:
        response_format = config.default_response_format
    return model, language, response_format


"""
OpenAI-like endpoint to transcribe audio files using the Whisper ASR model.

Args:
    request (Request): The HTTP request object.
    file (UploadFile): The audio file to transcribe.
    model (ModelName): The model to use for the transcription.
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
    description="Transcribe audio files using the Whisper ASR model.",
    tags=["Transcription"],
)
async def transcribe_audio(
    config: ConfigDependency,
    request: Request,
    file: UploadFile,
    model: Annotated[ModelName, Form()] = None,
    language: Annotated[Language, Form()] = None,
    prompt: Annotated[str, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = None,
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
    chunk_size: Annotated[int, Form()] = 30,
) -> Response:
    model, language, response_format = apply_defaults(
        config, model, language, response_format)
    timestamp_granularities = await get_timestamp_granularities(request)
    request_id = request.state.request_id
    logger.info(f"Request ID: {request_id} - Received transcription request")
    start_time = time.time()  # Start the timer
    logger.info(f"Request ID: {request_id} - Received request to transcribe {file.filename} with parameters: \
        model: {model}, \
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
        chunk_size: {chunk_size}")

    if not align:
        if response_format in ('vtt', 'srt', 'aud', 'vtt_json'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Subtitles format ('vtt', 'srt', 'aud', 'vtt_json') requires alignment to be enabled."
            )

        if diarize:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Diarization requires alignment to be enabled."
            )

    # Determine if word timestamps are required
    word_timestamps = "word" in timestamp_granularities

    # Build ASR options
    asr_options = {
        "suppress_numerals": suppress_numerals,
        "temperatures": temperature,
        "word_timestamps": word_timestamps,
        "initial_prompt": prompt,
        "hotwords": hotwords,
    }

    model_load_time = time.time()

    logger.info(
        f"Loaded model {model} in {time.time() - model_load_time:.2f} seconds")

    try:
        transcription = await transcriber.transcribe(
            audio_file=file,
            asr_options=asr_options,
            language=language,
            model_name=model,
            align=align,
            diarize=diarize,
            speaker_embeddings=speaker_embeddings,
            chunk_size=chunk_size,
            request_id=request_id
        )
    except Exception as e:
        logger.exception(
            f"Request ID: {request_id} - Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the transcription request."
        ) from e

    total_time = time.time() - start_time
    logger.info(
        f"Request ID: {request_id} - Transcription process took {total_time:.2f} seconds")

    return format_transcription(transcription, response_format, highlight_words=highlight_words)

"""
OpenAI-like endpoint to translate audio files using the Whisper ASR model.

Args:
    request (Request): The HTTP request object.
    file (UploadFile): The audio file to translate.
    model (ModelName): The model to use for the translation.
    prompt (str): The prompt to use for the translation.
    response_format (ResponseFormat): The response format to use for the translation. Defaults to "json".
    temperature (float): The temperature to use for the translation. Defaults to 0.0.
    chunk_size (int): Chunk size in seconds for merging VAD segments. Defaults to 30.

Returns:
    Translation: The translation of the audio file.
"""


@router.post(
    "/v1/audio/translations",
    description="Translate audio files using the Whisper ASR model",
    tags=["Translation"],
)
async def translate_audio(
    config: ConfigDependency,
    request: Request,
    file: UploadFile,
    model: Annotated[ModelName, Form()] = None,
    prompt: Annotated[str, Form()] = "",
    response_format: Annotated[ResponseFormat, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    chunk_size: Annotated[int, Form()] = 30,
) -> Response:
    model, _, response_format = apply_defaults(
        config, model, language=None, response_format=response_format)
    request_id = request.state.request_id
    logger.info(f"Request ID: {request_id} - Received translation request")
    start_time = time.time()  # Start the timer
    logger.info(f"Request ID: {request_id} - Received request to translate {file.filename} with parameters: \
        model: {model}, \
        prompt: {prompt}, \
        response_format: {response_format}, \
        temperature: {temperature}, \
        chunk_size: {chunk_size}")

    # Build ASR options
    asr_options = {
        "initial_prompt": prompt,
        "temperatures": temperature,
    }

    model_load_time = time.time()

    logger.info(
        f"Loaded model {model} in {time.time() - model_load_time:.2f} seconds")

    try:
        translation = await transcriber.transcribe(
            audio_file=file,
            asr_options=asr_options,
            model_name=model,
            chunk_size=chunk_size,
            request_id=request_id,
            task="translate"
        )
    except Exception as e:
        logger.exception(f"Request ID: {request_id} - Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the translation request."
        ) from e

    total_time = time.time() - start_time
    logger.info(
        f"Request ID: {request_id} - Translation process took {total_time:.2f} seconds")

    return format_transcription(translation, response_format)
