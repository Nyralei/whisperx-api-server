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
    load_transcribe_pipeline,
    load_align_model,
    load_diarize_pipeline,
    transcribe_pipeline_instances,
    align_model_instances,
    diarize_pipeline_instances,
    unload_model_object,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def handle_default_openai_model(
    model_name: str
) -> str:
    """Adjust the model name if it defaults to 'whisper-1'."""
    config = get_config()
    if model_name == "whisper-1":
        logger.info(
            f"{model_name} is not a valid model name. Using {config.whisper.model} instead.")
        return config.whisper.model
    return model_name


ModelName = Annotated[str, AfterValidator(handle_default_openai_model)]


@router.get(
    "/models/list",
    description="List loaded models",
    tags=["models", "transcribe"],
)
def list_models_endpoint():
    models = list({key[0] for key in transcribe_pipeline_instances.keys()})
    return JSONResponse(content={"models": models}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/models/unload",
    description="Unload a model",
    tags=["models", "transcribe"],
)
def unload_model_endpoint(model: Annotated[ModelName, Form()]):
    try:
        to_remove = [k for k in transcribe_pipeline_instances if k[0] == model]
        for k in to_remove:
            pipeline = transcribe_pipeline_instances.pop(k, None)
            if pipeline is not None:
                unload_model_object(pipeline)
        response_data = {"status": "success"} if to_remove else {
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
    try:
        await load_transcribe_pipeline(model_name=model, language=None, task="transcribe")
        return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.get(
    "/align_models/list",
    description="List loaded align models",
    tags=["models", "align"],
)
def list_align_models_endpoint():
    return JSONResponse(content={"models": list(align_model_instances.keys())}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/align_models/unload",
    description="Unload an align model",
    tags=["models", "align"],
)
def unload_align_model_endpoint(language: Annotated[Language, Form()]):
    try:
        if language in align_model_instances:
            align_model_data = align_model_instances.pop(language, None)
            if align_model_data is not None:
                unload_model_object(align_model_data.get("model"))
                del align_model_data
            response_data = {"status": "success"}
        else:
            response_data = {
                "status": "error", "message": f"Model with language {language} not found"}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/align_models/load",
    description="Load an align model",
    tags=["models", "align"],
)
async def load_alignment_endpoint(language: Annotated[Language, Form()]):
    try:
        await load_align_model(language)
        return JSONResponse(content={"status": "success", "model": language}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.get(
    "/diarize_models/list",
    description="List loaded diarize models",
    tags=["models", "diarize"],
)
def list_diarize_models_endpoint():
    return JSONResponse(content={"models": list(diarize_pipeline_instances.keys())}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/diarize_models/unload",
    description="Unload a diarize model",
    tags=["models", "diarize"],
)
def unload_diarize_model(model: Annotated[ModelName, Form()]):
    try:
        if model in diarize_pipeline_instances:
            diarize_model_data = diarize_pipeline_instances.pop(model, None)
            if diarize_model_data is not None:
                unload_model_object(diarize_model_data)
            response_data = {"status": "success"}
        else:
            response_data = {"status": "error",
                             "message": f"Model {model} not found"}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)


@router.post(
    "/diarize_models/load",
    description="Load a diarize model",
    tags=["models", "diarize"],
)
async def load_diarization_endpoint(model: Annotated[ModelName, Form()]):
    try:
        config = get_config()
        await load_diarize_pipeline(model)
        return JSONResponse(content={"status": "success", "model": model}, media_type=MediaType.APPLICATION_JSON)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, media_type=MediaType.APPLICATION_JSON)
