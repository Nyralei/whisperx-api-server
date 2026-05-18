import logging
from fastapi import APIRouter, Form, HTTPException, status
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
        logger.exception("Failed to list transcription models")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post(
    "/models/unload",
    description="Unload a model",
    tags=["models", "transcribe"],
)
async def unload_model_endpoint(model: Annotated[ModelName, Form()]):
    logger.info(f"Received request to unload model {model}")
    try:
        selected_backends = resolve_stage_backends()
        backend = get_transcription_backend(selected_backends.transcription)
        unloaded = await backend.unload_model(model)
    except Exception as e:
        logger.exception("Failed to unload transcription model %s", model)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    if not unloaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model} not found",
        )
    return JSONResponse(
        content={"status": "success"},
        media_type=MediaType.APPLICATION_JSON,
    )


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
        logger.exception("Failed to load transcription model %s", model)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


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
        logger.exception("Failed to list alignment models")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post(
    "/align_models/unload",
    description="Unload an align model",
    tags=["models", "align"],
)
async def unload_align_model_endpoint(language: Annotated[Language, Form()]):
    logger.info(f"Received request to unload align model {language}")
    language_code = getattr(language, "value", language)
    try:
        selected_backends = resolve_stage_backends()
        backend = get_alignment_backend(selected_backends.alignment)
        unloaded = await backend.unload_model(str(language_code))
    except Exception as e:
        logger.exception("Failed to unload align model %s", language_code)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    if not unloaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with language {language_code} not found",
        )
    return JSONResponse(
        content={"status": "success"},
        media_type=MediaType.APPLICATION_JSON,
    )


@router.post(
    "/align_models/load",
    description="Load an align model",
    tags=["models", "align"],
)
async def load_alignment_endpoint(language: Annotated[Language, Form()]):
    logger.info(f"Received request to load align model {language}")
    language_code = str(getattr(language, "value", language))
    try:
        selected_backends = resolve_stage_backends()
        backend = get_alignment_backend(selected_backends.alignment)
        if language_code in backend.list_loaded_models():
            return JSONResponse(content={"status": "success", "model": language_code}, media_type=MediaType.APPLICATION_JSON)
        await backend.load_model(language_code)
        return JSONResponse(content={"status": "success", "model": language_code}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        logger.exception("Failed to load align model %s", language_code)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


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
        logger.exception("Failed to list diarization models")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post(
    "/diarize_models/unload",
    description="Unload a diarize model",
    tags=["models", "diarize"],
)
async def unload_diarize_model(model: Annotated[ModelName, Form()]):
    logger.info(f"Received request to unload diarize model {model}")
    try:
        selected_backends = resolve_stage_backends()
        backend = get_diarization_backend(selected_backends.diarization)
        unloaded = await backend.unload_model(model)
    except Exception as e:
        logger.exception("Failed to unload diarize model %s", model)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    if not unloaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model} not found",
        )
    return JSONResponse(
        content={"status": "success"},
        media_type=MediaType.APPLICATION_JSON,
    )


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
        logger.exception("Failed to load diarize model %s", model)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
