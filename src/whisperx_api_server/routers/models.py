import logging
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import AfterValidator

from whisperx_api_server.config import (
    Language,
    MediaType,
)
from whisperx_api_server.backends.registry import (
    get_alignment_backend,
    get_default_transcription_model_name,
    get_diarization_backend,
    get_transcription_backend,
    resolve_stage_backends,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def handle_default_openai_model(
    model_name: str
) -> str:
    """Adjust the model name if it defaults to 'whisper-1'."""
    if model_name == "whisper-1":
        default_model = get_default_transcription_model_name()
        logger.info(
            f"{model_name} is not a valid model name. Using {default_model} instead.")
        return default_model
    return model_name


ModelName = Annotated[str, AfterValidator(handle_default_openai_model)]


@router.get(
    "/models/list",
    description="List loaded models",
    tags=["models", "transcribe"],
)
def list_models_endpoint():
    try:
        selected_backends = resolve_stage_backends()
        backend = get_transcription_backend(selected_backends.transcription)
        models = backend.list_loaded_models()
        return JSONResponse(content={"models": models}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/models/unload",
    description="Unload a model",
    tags=["models", "transcribe"],
)
def unload_model_endpoint(model: Annotated[ModelName, Form()]):
    logger.info(f"Received request to unload model {model}")
    try:
        selected_backends = resolve_stage_backends()
        backend = get_transcription_backend(selected_backends.transcription)
        unloaded = backend.unload_model(model)
        response_data = {"status": "success"} if unloaded else {
            "status": "error", "message": f"Model {model} not found"}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/models/load",
    description="Load a model",
    tags=["models", "transcribe"],
)
async def load_model(model: Annotated[ModelName, Form()]):
    logger.info(f"Received request to load model {model}")
    try:
        selected_backends = resolve_stage_backends()
        backend = get_transcription_backend(selected_backends.transcription)
        if model in backend.list_loaded_models():
            return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
        await backend.load_model(model_name=model)
        return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.get(
    "/align_models/list",
    description="List loaded align models",
    tags=["models", "align"],
)
def list_align_models_endpoint():
    try:
        selected_backends = resolve_stage_backends()
        backend = get_alignment_backend(selected_backends.alignment)
        return JSONResponse(content={"models": backend.list_loaded_models()}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/align_models/unload",
    description="Unload an align model",
    tags=["models", "align"],
)
def unload_align_model_endpoint(language: Annotated[Language, Form()]):
    logger.info(f"Received request to unload align model {language}")
    try:
        selected_backends = resolve_stage_backends()
        backend = get_alignment_backend(selected_backends.alignment)
        language_code = getattr(language, "value", language)
        unloaded = backend.unload_model(str(language_code))
        response_data = {"status": "success"} if unloaded else {
            "status": "error", "message": f"Model with language {language_code} not found"}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/align_models/load",
    description="Load an align model",
    tags=["models", "align"],
)
async def load_alignment_endpoint(language: Annotated[Language, Form()]):
    logger.info(f"Received request to load align model {language}")
    try:
        selected_backends = resolve_stage_backends()
        backend = get_alignment_backend(selected_backends.alignment)
        language_code = str(getattr(language, "value", language))
        if language_code in backend.list_loaded_models():
            return JSONResponse(content={"status": "success", "model": language_code}, media_type=MediaType.APPLICATION_JSON)
        await backend.load_model(language_code)
        return JSONResponse(content={"status": "success", "model": language_code}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.get(
    "/diarize_models/list",
    description="List loaded diarize models",
    tags=["models", "diarize"],
)
def list_diarize_models_endpoint():
    try:
        selected_backends = resolve_stage_backends()
        backend = get_diarization_backend(selected_backends.diarization)
        return JSONResponse(content={"models": backend.list_loaded_models()}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/diarize_models/unload",
    description="Unload a diarize model",
    tags=["models", "diarize"],
)
def unload_diarize_model(model: Annotated[ModelName, Form()]):
    logger.info(f"Received request to unload diarize model {model}")
    try:
        selected_backends = resolve_stage_backends()
        backend = get_diarization_backend(selected_backends.diarization)
        unloaded = backend.unload_model(model)
        response_data = {"status": "success"} if unloaded else {
            "status": "error", "message": f"Model {model} not found"}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/diarize_models/load",
    description="Load a diarize model",
    tags=["models", "diarize"],
)
async def load_diarization_endpoint(model: Annotated[ModelName, Form()]):
    logger.info(f"Received request to load diarize model {model}")
    try:
        selected_backends = resolve_stage_backends()
        backend = get_diarization_backend(selected_backends.diarization)
        if model in backend.list_loaded_models():
            return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
        await backend.load_model(model_name=model)
        return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)
