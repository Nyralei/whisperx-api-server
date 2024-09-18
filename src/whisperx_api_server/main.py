import logging
from collections import defaultdict
from asyncio import Lock
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Literal, Annotated
from pydantic import AfterValidator
import time

import whisperx_api_server.transcriber as transcriber
from whisperx_api_server.config import (
    Language,
    ResponseFormat,
    config,
)

# Global cache and locks
model_instances = {}
model_locks = defaultdict(Lock)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("api_logger")
logger.setLevel(logging.DEBUG)

def handle_default_openai_model(model_name: str) -> str:
    """Adjust the model name if it defaults to 'whisper-1'."""
    if model_name == "whisper-1":
        logger.info(f"{model_name} is not a valid model name. Using {config.whisper.model} instead.")
        return config.whisper.model
    return model_name

# Annotated ModelName for validation and defaults
ModelName = Annotated[str, AfterValidator(handle_default_openai_model)]

# FastAPI app instance
app = FastAPI()

# Async function to get the model instance
async def load_model_instance(model_name: str):
    global model_locks
    global model_instances

    # Check if the model is already in memory cache (without locking)
    if model_name in model_instances:
        logger.info(f"Reusing cached model instance for {model_name}")
        return model_instances[model_name]

    # Use lock to avoid reloading the model concurrently
    async with model_locks[model_name]:
        # Double-check if model was loaded during waiting for the lock
        if model_name not in model_instances:
            logger.info(f"Initializing model: {model_name}")
            model_instances[model_name] = await transcriber.initialize_model(model_name)
        return model_instances[model_name]
    
    
"""
OpenAI-like endpoint to transcribe audio files using the Whisper ASR model.

Args:
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

Returns:
    Transcription: The transcription of the audio file.
"""
@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
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
):
    start_time = time.time()  # Start the timer
    logger.debug(f"Received request to transcribe {file.filename} with parameters: \
        model: {model}, \
        language: {language}, \
        prompt: {prompt}, \
        response_format: {response_format}, \
        temperature: {temperature}, \
        timestamp_granularities: {timestamp_granularities}, \
        stream: {stream}, \
        hotwords: {hotwords}, \
        suppress_numerals: {suppress_numerals} \
    ")

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
            highlight_words=highlight_words
        )
    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

    total_time = time.time() - start_time  # Calculate the total time taken
    logger.info(f"Transcription process took {total_time:.2f} seconds")

    if stream:
        return StreamingResponse(transcription.stream(), media_type="text/event-stream")

    return transcription

@app.get("/healthcheck")
def health_check():
    return {"status": "healthy"}

@app.get("/models/list")
def list_models():
    global model_instances
    return {
        "models": list(model_instances.keys())
    }

@app.post("/models/unload")
def unload_model(model: Annotated[ModelName, Form()]):
    if model not in model_instances:
        return {"status": "error", "message": f"Model {model} not found"}
    del model_instances[model]
    return {"status": "success"}

@app.post("/models/load")
async def load_model(model: Annotated[ModelName, Form()]):
    try:
        await load_model_instance(model)
        return {"status": "success", "model": model}
    except Exception as e:
        return {"status": "error", "message": str(e)}
