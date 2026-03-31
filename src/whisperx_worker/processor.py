import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Any

import numpy as np

from whisperx_api_server.backends.registry import (
    get_alignment_backend,
    get_diarization_backend,
    get_transcription_backend,
    get_default_transcription_model_name,
    resolve_stage_backends,
)
from whisperx_api_server.config import Language
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.transcriber import (
    _load_audio,
    _finalize_text,
    _cleanup_cache_only,
    _get_concurrency_semaphore,
)
import whisperx_api_server.s3_client as s3_client

logger = logging.getLogger(__name__)
config = get_config()


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def serialize_result(result: dict) -> str:
    return json.dumps(result, cls=_NumpyEncoder)


async def process_job(event: dict[str, Any]) -> dict[str, Any]:
    job_id = event["job_id"]
    s3_key = event["s3_key"]
    filename = event.get("filename", "audio")
    params = event["params"]

    align = params.get("align", False)
    diarize = params.get("diarize", False)

    start_time = time.perf_counter()
    file_path = None
    audio = None
    concurrency_sem = _get_concurrency_semaphore()
    semaphore_acquired = False
    profile: dict[str, float] = {}

    try:
        t0 = time.perf_counter()
        logger.info(f"Job {job_id}: downloading audio from S3 (key: {s3_key})")
        audio_bytes = await s3_client.download_audio(s3_key)
        profile["s3_download"] = time.perf_counter() - t0
        logger.info(
            f"Job {job_id}: S3 download took {profile['s3_download']:.2f} seconds")

        t0 = time.perf_counter()
        suffix = f"_{filename}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            await asyncio.to_thread(tmp.write, audio_bytes)
            file_path = tmp.name
        del audio_bytes
        profile["file_write"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        audio = await _load_audio(file_path, job_id)
        profile["audio_load"] = time.perf_counter() - t0
        logger.info(
            f"Job {job_id}: loading audio took {profile['audio_load']:.2f} seconds")

        os.remove(file_path)
        file_path = None

        if concurrency_sem:
            await concurrency_sem.acquire()
            semaphore_acquired = True
            logger.debug(f"Job {job_id}: acquired GPU concurrency semaphore")

        selected_backends = resolve_stage_backends()
        transcription_backend = get_transcription_backend(
            selected_backends.transcription)
        alignment_backend = get_alignment_backend(
            selected_backends.alignment) if (align or diarize) else None
        diarization_backend = get_diarization_backend(
            selected_backends.diarization) if diarize else None

        model_name = params.get(
            "model_name") or get_default_transcription_model_name()
        raw_language = params.get("language")
        language = Language(
            raw_language) if raw_language else config.default_language
        asr_options = params.get("asr_options")
        task = params.get("task", "transcribe")

        logger.info(
            f"Job {job_id}: transcribing {filename} with model: {model_name}, "
            f"options: {asr_options}, language: {language}, task: {task}, "
            f"stage_backends: {selected_backends}"
        )

        t0 = time.perf_counter()
        result = await transcription_backend.transcribe(
            model_name=model_name,
            audio=audio,
            batch_size=params.get("batch_size", config.whisper.batch_size),
            chunk_size=params.get("chunk_size", config.whisper.chunk_size),
            language=language,
            task=task,
            asr_options=asr_options,
            request_id=job_id,
        )
        profile["transcribe"] = time.perf_counter() - t0
        logger.info(
            f"Job {job_id}: transcription took {profile['transcribe']:.2f} seconds")

        if align or diarize:
            t0 = time.perf_counter()
            result = await alignment_backend.align(
                result=result, audio=audio, request_id=job_id
            )
            profile["align"] = time.perf_counter() - t0
            logger.debug(
                f"Job {job_id}: alignment took {profile['align']:.2f} seconds")

        if diarize:
            t0 = time.perf_counter()
            result = await diarization_backend.diarize(
                result=result,
                audio=audio,
                speaker_embeddings=params.get("speaker_embeddings", False),
                request_id=job_id,
            )
            profile["diarize"] = time.perf_counter() - t0
            logger.debug(
                f"Job {job_id}: diarization took {profile['diarize']:.2f} seconds")

        t0 = time.perf_counter()
        result = _finalize_text(result, align or diarize)
        profile["finalize"] = time.perf_counter() - t0

        total = time.perf_counter() - start_time
        logger.info(f"Job {job_id}: transcription complete for {filename}")
        logger.debug(
            f"Job {job_id}: profile: total={total:.2f}s | "
            + " | ".join(f"{k}={v:.2f}s" for k, v in profile.items())
            + f" | (other={total - sum(profile.values()):.2f}s)"
        )
        return result

    finally:
        if concurrency_sem and semaphore_acquired:
            concurrency_sem.release()
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        if config.audio_cleanup and audio is not None:
            del audio
            logger.info(f"Job {job_id}: audio data cleaned up")
        if config.cache_cleanup:
            _cleanup_cache_only()
            logger.info(f"Job {job_id}: cache cleanup completed")
