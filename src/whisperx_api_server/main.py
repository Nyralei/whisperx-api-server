import logging
import uuid
from fastapi import FastAPI, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Literal, Annotated
from pydantic import AfterValidator
import time

import whisperx_api_server.transcriber as transcriber
from whisperx_api_server.config import (
    Language,
    ResponseFormat,
    MediaType,
    config,
)
from whisperx_api_server.models import (
    load_model_instance,
    load_align_model_cached,
    load_diarize_model_cached,
    model_instances,
    align_model_instances,
    diarize_model_instances,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("api_logger")
logger.setLevel(config.log_level)

def handle_default_openai_model(model_name: str) -> str:
    """Adjust the model name if it defaults to 'whisper-1'."""
    if model_name == "whisper-1":
        logger.info(f"{model_name} is not a valid model name. Using {config.whisper.model} instead.")
        return config.whisper.model
    return model_name

# Annotated ModelName for validation and defaults
ModelName = Annotated[str, AfterValidator(handle_default_openai_model)]

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

# FastAPI app instance
app = FastAPI()
app.add_middleware(RequestIDMiddleware)

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
    diarize (bool): Whether to diarize the transcription. Defaults to False.

Returns:
    Transcription: The transcription of the audio file.
"""
@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    request: Request,
    file: Annotated[UploadFile, Form()],
    model: Annotated[ModelName, Form()] = config.whisper.model,
    language: Annotated[Language, Form()] = config.default_language,
    prompt: Annotated[str, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = config.default_response_format,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        list[Literal["segment", "word"]],
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str, Form()] = None,
    suppress_numerals: Annotated[bool, Form()] = True,
    highlight_words: Annotated[bool, Form()] = False,
    diarize: Annotated[bool, Form()] = False,
) -> Response:
    request_id = request.state.request_id
    logger.debug(f"Request ID: {request_id} - Received transcription request")
    start_time = time.time()  # Start the timer
    logger.debug(f"Request ID: {request_id} - Received request to transcribe {file.filename} with parameters: \
        model: {model}, \
        language: {language}, \
        prompt: {prompt}, \
        response_format: {response_format}, \
        temperature: {temperature}, \
        timestamp_granularities: {timestamp_granularities}, \
        stream: {stream}, \
        hotwords: {hotwords}, \
        suppress_numerals: {suppress_numerals} \
        highlight_words: {highlight_words} \
        diarize: {diarize}")

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
    # Get model instance (reuse if cached)
    model_instance = await load_model_instance(model)

    logger.info(f"Loaded model {model} in {time.time() - model_load_time:.2f} seconds")

    try:
        transcription = await transcriber.transcribe(
            audio_file=file,
            batch_size=config.batch_size,
            asr_options=asr_options,
            language=language,
            response_format=response_format,
            whispermodel=model_instance,
            highlight_words=highlight_words,
            diarize=diarize,
            request_id=request_id
        )
    except Exception as e:
        logger.exception(f"Request ID: {request_id} - Transcription failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

    total_time = time.time() - start_time
    logger.info(f"Request ID: {request_id} - Transcription process took {total_time:.2f} seconds")

    return transcription

@app.get("/healthcheck")
def health_check():
    return JSONResponse(content={"status": "healthy"}, media_type=MediaType.APPLICATION_JSON)

@app.get("/models/list")
def list_models():
    global model_instances
    return JSONResponse(content={"models": list(model_instances.keys())}, media_type=MediaType.APPLICATION_JSON)

@app.post("/models/unload")
def unload_model(model: Annotated[ModelName, Form()]):
    try:
        if model in model_instances:
            del model_instances[model]
            response_data = {"status": "success"}
        else:
            response_data = {"status": "error", "message": f"Model {model} not found"}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)

@app.post("/models/load")
async def load_model(model: Annotated[ModelName, Form()]):
    try:
        await load_model_instance(model)
        return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)

@app.get("/align_models/list")
def list_align_models():
    global align_model_instances
    return JSONResponse(content={"models": list(align_model_instances.keys())}, media_type=MediaType.APPLICATION_JSON)

@app.post("/align_models/unload")
def unload_align_model(language: Annotated[Language, Form()]):
    try:
        if language in align_model_instances:
            del align_model_instances[language]
            response_data = {"status": "success"}
        else:
            response_data = {"status": "error", "message": f"Model with language {language} not found"}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)

@app.post("/align_models/load")
def load_align_model(language: Annotated[Language, Form()]):
    try:
        load_align_model_cached(language, transcriber.check_device())
        return JSONResponse(content={"status": "success", "model": language}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)
    
@app.get("/diarize_models/list")
def list_diarize_models():
    global diarize_model_instances
    return JSONResponse(content={"models": list(diarize_model_instances.keys())}, media_type=MediaType.APPLICATION_JSON)

@app.post("/diarize_models/unload")
def unload_diarize_model(model: Annotated[ModelName, Form()]):
    try:
        if model in diarize_model_instances:
            del diarize_model_instances[model]
            response_data = {"status": "success"}
        else:
            response_data = {"status": "error", "message": f"Model {model} not found"}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)

@app.post("/diarize_models/load")
def load_diarize_model(model: Annotated[ModelName, Form()]):
    try:
        load_diarize_model_cached(model, transcriber.check_device())
        return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)