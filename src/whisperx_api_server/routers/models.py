import logging
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import AfterValidator

import whisperx_api_server.transcriber as transcriber
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.config import (
    Language,
    MediaType,
)
from whisperx_api_server.models import (
    load_model_instance,
    load_align_model_cached,
    load_diarize_model_cached,
    model_instances,
    align_model_instances,
    diarize_model_instances,
)

logger = logging.getLogger(__name__)

router = APIRouter()

def handle_default_openai_model(
        model_name: str
    ) -> str:
    """Adjust the model name if it defaults to 'whisper-1'."""
    config = get_config()
    if model_name == "whisper-1":
        logger.info(f"{model_name} is not a valid model name. Using {config.whisper.model} instead.")
        return config.whisper.model
    return model_name

ModelName = Annotated[str, AfterValidator(handle_default_openai_model)]

@router.get(
    "/models/list",
    description="List loaded models",
    tags=["models", "transcribe"],
)
def list_models():
    global model_instances
    return JSONResponse(content={"models": list(model_instances.keys())}, media_type=MediaType.APPLICATION_JSON)

@router.post(
    "/models/unload",
    description="Unload a model",
    tags=["models", "transcribe"],
)
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

@router.post(
    "/models/load",
    description="Load a model",
    tags=["models", "transcribe"],
)
async def load_model(model: Annotated[ModelName, Form()]):
    try:
        await load_model_instance(model)
        return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)

@router.get(
    "/align_models/list",
    description="List loaded align models",
    tags=["models", "align"],
)
def list_align_models():
    global align_model_instances
    return JSONResponse(content={"models": list(align_model_instances.keys())}, media_type=MediaType.APPLICATION_JSON)

@router.post(
    "/align_models/unload",
    description="Unload an align model",
    tags=["models", "align"],
)
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

@router.post(
    "/align_models/load",
    description="Load an align model",
    tags=["models", "align"],
)
def load_align_model(language: Annotated[Language, Form()]):
    try:
        load_align_model_cached(language, transcriber.check_device())
        return JSONResponse(content={"status": "success", "model": language}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)
    
@router.get(
    "/diarize_models/list",
    description="List loaded diarize models",
    tags=["models", "diarize"],
)
def list_diarize_models():
    global diarize_model_instances
    return JSONResponse(content={"models": list(diarize_model_instances.keys())}, media_type=MediaType.APPLICATION_JSON)

@router.post(
    "/diarize_models/unload",
    description="Unload a diarize model",
    tags=["models", "diarize"],
)
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

@router.post(
    "/diarize_models/load",
    description="Load a diarize model",
    tags=["models", "diarize"],
)
def load_diarize_model(model: Annotated[ModelName, Form()]):
    try:
        load_diarize_model_cached(model, transcriber.check_device())
        return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)