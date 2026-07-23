import logging
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import AfterValidator

from whisperx_api_server.backends.registry import (
    get_alignment_backend,
    get_default_transcription_model_name,
    get_diarization_backend,
    get_transcription_backend,
    resolve_stage_backends,
)
from whisperx_api_server.config import (
    DistributedMode,
    Language,
    MediaType,
)
from whisperx_api_server.dependencies import get_config

logger = logging.getLogger(__name__)

router = APIRouter()
catalog_router = APIRouter()

# Fallback for a slim install without faster-whisper; the live list is preferred
# via faster_whisper.available_models(). Mirrors faster-whisper 1.2.x.
KNOWN_TRANSCRIPTION_MODELS = (
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-medium.en",
    "distil-small.en",
    "distil-large-v3",
    "distil-large-v3.5",
    "large-v3-turbo",
    "turbo",
)


def _known_models() -> list[str]:
    try:
        from faster_whisper import available_models

        return list(available_models())
    except Exception:
        return list(KNOWN_TRANSCRIPTION_MODELS)


@catalog_router.get(
    "/models/catalog",
    description=(
        "Curated catalog of known transcription model names plus the configured "
        "default. In direct mode `loaded` lists models currently in this process's "
        "cache; in Kafka mode `loaded` is null (models live in worker processes). "
        "Available in both modes, unlike /models/list."
    ),
    tags=["models", "transcribe"],
)
def models_catalog():
    config = get_config()
    loaded: list[str] | None = None
    if config.mode != DistributedMode.KAFKA:
        try:
            selected_backends = resolve_stage_backends()
            loaded = get_transcription_backend(
                selected_backends.transcription
            ).list_loaded_models()
        except Exception:
            logger.warning("models_catalog: loaded-model enumeration failed")
            loaded = None
    return JSONResponse(
        content={
            "models": _known_models(),
            "default": config.whisper.model,
            "loaded": loaded,
        },
        media_type=MediaType.APPLICATION_JSON,
    )


def handle_default_openai_model(model_name: str) -> str:
    """Adjust the model name if it defaults to 'whisper-1'."""
    if model_name == "whisper-1":
        default_model = get_default_transcription_model_name()
        logger.info(
            "%s is not a valid model name. Using %s instead.", model_name, default_model
        )
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
        return JSONResponse(
            content={"models": models}, media_type=MediaType.APPLICATION_JSON
        )
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
    logger.info("Received request to unload model %s", model)
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
    logger.info("Received request to load model %s", model)
    try:
        selected_backends = resolve_stage_backends()
        backend = get_transcription_backend(selected_backends.transcription)
        if model in backend.list_loaded_models():
            return JSONResponse(
                content={"status": "success", "model": model},
                media_type=MediaType.APPLICATION_JSON,
            )
        await backend.load_model(model_name=model)
        return JSONResponse(
            content={"status": "success", "model": model},
            media_type=MediaType.APPLICATION_JSON,
        )
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
        return JSONResponse(
            content={"models": backend.list_loaded_models()},
            media_type=MediaType.APPLICATION_JSON,
        )
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
    logger.info("Received request to unload align model %s", language)
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
    logger.info("Received request to load align model %s", language)
    language_code = str(getattr(language, "value", language))
    try:
        selected_backends = resolve_stage_backends()
        backend = get_alignment_backend(selected_backends.alignment)
        if language_code in backend.list_loaded_models():
            return JSONResponse(
                content={"status": "success", "model": language_code},
                media_type=MediaType.APPLICATION_JSON,
            )
        await backend.load_model(language_code)
        return JSONResponse(
            content={"status": "success", "model": language_code},
            media_type=MediaType.APPLICATION_JSON,
        )
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
        return JSONResponse(
            content={"models": backend.list_loaded_models()},
            media_type=MediaType.APPLICATION_JSON,
        )
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
    logger.info("Received request to unload diarize model %s", model)
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
    logger.info("Received request to load diarize model %s", model)
    try:
        selected_backends = resolve_stage_backends()
        backend = get_diarization_backend(selected_backends.diarization)
        if model in backend.list_loaded_models():
            return JSONResponse(
                content={"status": "success", "model": model},
                media_type=MediaType.APPLICATION_JSON,
            )
        await backend.load_model(model_name=model)
        return JSONResponse(
            content={"status": "success", "model": model},
            media_type=MediaType.APPLICATION_JSON,
        )
    except Exception as e:
        logger.exception("Failed to load diarize model %s", model)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
