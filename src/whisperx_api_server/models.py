import logging
import asyncio
import contextlib
import gc
from collections import defaultdict
from asyncio import Lock
from typing import Dict, Optional, Tuple, Any

import torch
from whisperx import transcribe as whisperx_transcribe
from whisperx import alignment as whisperx_alignment
from whisperx import diarize as whisperx_diarize
from whisperx.asr import FasterWhisperPipeline

from whisperx_api_server.dependencies import get_config

logger = logging.getLogger(__name__)


def _hashable_options(opts: Any) -> Any:
    if opts is None:
        return None
    if isinstance(opts, dict):
        return tuple(sorted((k, _hashable_options(v)) for k, v in opts.items()))
    if isinstance(opts, (list, tuple)):
        return tuple(_hashable_options(v) for v in opts)
    return opts


# caches
transcribe_pipeline_instances: Dict[Tuple[Any, ...],
                                    FasterWhisperPipeline] = {}
transcribe_pipeline_locks = defaultdict(Lock)
align_model_instances = {}
alignment_locks = defaultdict(Lock)
alignment_cache_mod_lock = Lock()
diarize_pipeline_instances = {}
diarize_locks = defaultdict(Lock)


def unload_model_object(model_obj: Any):
    if model_obj is None:
        return
    with contextlib.suppress(Exception):
        if hasattr(model_obj, "to"):
            model_obj.to("cpu")
        elif hasattr(model_obj, "model") and hasattr(model_obj.model, "to"):
            model_obj.model.to("cpu")
    del model_obj
    gc.collect()
    torch.cuda.empty_cache()


def check_device():
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        logger.error("Could not determine device. Using 'cpu' instead.")
        return "cpu"


def determine_inference_device():
    config = get_config()
    inference_device = config.whisper.inference_device.value
    if inference_device == "auto":
        inference_device = check_device()
    return inference_device


async def _get_or_init_model(
    key,
    cache_dict: dict,
    lock_dict: dict,
    init_func,
    log_reuse: str = "Reusing cached model instance for {key}",
    log_init: str = "Initializing model: {key}",
) -> Any:
    if key in cache_dict:
        logger.info(log_reuse.format(key=key))
        return cache_dict[key]

    async with lock_dict[key]:
        if key in cache_dict:
            return cache_dict[key]
        try:
            instance = await init_func()
        except Exception as e:
            logger.error(f"Failed to initialize model {key}: {e}")
            raise
        cache_dict[key] = instance
        return instance


async def load_transcribe_pipeline(
    model_name: str,
) -> FasterWhisperPipeline:
    config = get_config()
    inference_device = determine_inference_device()
    compute_type = config.whisper.compute_type.value
    vad_method = getattr(config.whisper.vad_method,
                         "value", config.whisper.vad_method)
    cache_key = (
        model_name,
        inference_device,
        compute_type,
        vad_method,
        config.whisper.vad_model,
        _hashable_options(config.whisper.vad_options),
    )
    pipeline_opts = {
        "suppress_numerals": True,
        "temperatures": 0.0,
        "word_timestamps": False,
        "initial_prompt": None,
        "hotwords": None,
    }

    def _create_pipeline():
        return whisperx_transcribe.load_model(
            whisper_arch=model_name,
            device=inference_device,
            device_index=config.whisper.device_index,
            compute_type=compute_type,
            language=None,
            asr_options=pipeline_opts,
            vad_model=config.whisper.vad_model,
            vad_method=config.whisper.vad_method,
            vad_options=config.whisper.vad_options,
            download_root=config.whisper.download_root,
            local_files_only=config.whisper.local_files_only,
            threads=config.whisper.cpu_threads or 4,
            use_auth_token=config.hf_token,
        )

    pipeline = await _get_or_init_model(
        key=cache_key,
        cache_dict=transcribe_pipeline_instances,
        lock_dict=transcribe_pipeline_locks,
        init_func=lambda: asyncio.to_thread(_create_pipeline),
        log_reuse="Reusing cached transcribe pipeline: {key}",
        log_init="Initializing transcribe pipeline: {key}",
    )

    if not config.whisper.cache:
        transcribe_pipeline_instances.pop(cache_key, None)

    pipeline._transcribe_lock = transcribe_pipeline_locks[cache_key]
    return pipeline


async def _cleanup_alignment_cache_whitelist(keep_key: Optional[str] = None):
    config = get_config()
    whitelist = config.alignment.whitelist
    if not whitelist:
        return
    async with alignment_cache_mod_lock:
        keys_to_remove = [
            k for k in align_model_instances
            if k not in whitelist and k != keep_key]
        for key in keys_to_remove:
            logger.info(
                f"Unloading alignment model for {key} (not in whitelist).")
            align_model_data = align_model_instances.pop(key, None)
            if align_model_data:
                unload_model_object(align_model_data.get("model"))
                del align_model_data


async def load_align_model(
    language_code: str,
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:

    config = get_config()
    inference_device = determine_inference_device()
    selected_model_name = model_name
    if "multilingual" in config.alignment.models:
        selected_model_name = config.alignment.models["multilingual"]
        logger.info(
            f"Overriding with 'multilingual' alignment model: {selected_model_name}")
    elif language_code in config.alignment.models:
        selected_model_name = config.alignment.models[language_code]
        logger.info(
            f"Using configured alignment model for '{language_code}': {selected_model_name}")

    if (selected_model_name is not None
            and selected_model_name == config.alignment.models.get("multilingual")):
        cache_key = "multilingual"
    else:
        cache_key = language_code

    await _cleanup_alignment_cache_whitelist(keep_key=cache_key)

    logger.debug(f"config.alignment.models = {config.alignment.models}")
    logger.debug(
        f"Incoming language_code = {language_code}, model_name param = {model_name}")

    async def _init_alignment():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: whisperx_alignment.load_align_model(
                language_code=language_code,
                device=inference_device,
                model_name=selected_model_name,
                model_dir=model_dir
            )
        )

    async def _init_wrapper():
        model, metadata = await _init_alignment()
        return {"model": model, "metadata": metadata}

    model_data = await _get_or_init_model(
        key=cache_key,
        cache_dict=align_model_instances,
        lock_dict=alignment_locks,
        init_func=_init_wrapper,
        log_reuse="Reusing cached alignment model for: {key}",
        log_init="Initializing alignment model for: {key}",
    )

    if not config.alignment.cache:
        async with alignment_cache_mod_lock:
            removed = align_model_instances.pop(cache_key, None)
            if removed:
                unload_model_object(removed.get("model"))
    return model_data["model"], model_data["metadata"]


async def load_diarize_pipeline(model_name: str) -> whisperx_diarize.DiarizationPipeline:
    config = get_config()
    inference_device = determine_inference_device()

    def _init_diarization():
        logger.info(f"Loading diarization pipeline: {model_name}")
        return whisperx_diarize.DiarizationPipeline(
            model_name=model_name,
            token=config.hf_token,
            device=inference_device
        )

    diarize_model = await _get_or_init_model(
        key=model_name,
        cache_dict=diarize_pipeline_instances,
        lock_dict=diarize_locks,
        init_func=lambda: asyncio.to_thread(_init_diarization),
        log_reuse="Reusing cached diarization model for: {key}",
        log_init="Initializing diarization model: {key}",
    )

    if not config.diarization.cache:
        removed = diarize_pipeline_instances.pop(model_name, None)
        if removed is not None:
            unload_model_object(removed)
    return diarize_model
