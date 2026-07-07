import contextlib
import json
import logging
import os
import time
from typing import Any

import numpy as np

import whisperx_api_server.s3_client as s3_client
from whisperx_api_server import url_fetch
from whisperx_api_server.backends.registry import (
    get_alignment_backend,
    get_default_transcription_model_name,
    get_diarization_backend,
    get_transcription_backend,
    resolve_stage_backends,
)
from whisperx_api_server.config import Language
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.transcriber import (
    _cleanup_cache_only,
    _finalize_text,
    _get_concurrency_semaphore,
    _safe_filename_suffix,
    load_audio_from_path,
)
from whisperx_worker.progress import publish_stage

logger = logging.getLogger(__name__)


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


async def process_job(
    event: dict[str, Any],
    *,
    progress_producer: Any = None,
    progress_topic: str | None = None,
    timeline_out: dict[str, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    config = get_config()
    job_id = event["job_id"]
    s3_key = event.get("s3_key")
    audio_url = event.get("audio_url")
    if bool(s3_key) == bool(audio_url):
        raise ValueError(
            f"Job {job_id}: event must specify exactly one of s3_key or audio_url"
        )
    filename = event.get("filename", "audio")
    params = event["params"]

    align = params.get("align", False)
    diarize = params.get("diarize", False)

    start_time = time.perf_counter()
    audio = None
    file_path = None
    concurrency_sem = _get_concurrency_semaphore()
    profile: dict[str, float] = {}
    # timeline carries wall-clock (time.time()) per-stage timestamps back to the
    # API server in the reply, so the request_status tracker has an authoritative
    # source of truth that doesn't depend on the progress topic winning the race
    # against the reply topic. Dict preserves insertion order.
    if timeline_out is None:
        timeline_out = {}

    async def _progress(stage: str) -> None:
        t = time.time()
        if timeline_out:
            last_name = next(reversed(timeline_out))
            if timeline_out[last_name].get("completed_at") is None:
                timeline_out[last_name]["completed_at"] = t
        timeline_out[stage] = {"started_at": t, "completed_at": None}
        await publish_stage(progress_producer, progress_topic or "", job_id, stage)

    def _close_timeline() -> None:
        if not timeline_out:
            return
        last_name = next(reversed(timeline_out))
        if timeline_out[last_name].get("completed_at") is None:
            timeline_out[last_name]["completed_at"] = time.time()

    try:
        if audio_url is not None:
            await _progress("url_download")
            t0 = time.perf_counter()
            logger.info("Job %s: downloading audio from URL", job_id)
            file_path = await url_fetch.download_url_to_temp(
                audio_url,
                job_id,
                max_bytes=config.max_upload_size_bytes,
                connect_timeout=config.url_fetch_connect_timeout_seconds,
                total_timeout=config.url_fetch_timeout_seconds,
                allow_private_hosts=config.url_fetch_allow_private_hosts,
                allowed_hosts=config.url_fetch_allowed_hosts,
            )
            profile["url_download"] = time.perf_counter() - t0
            logger.info(
                "Job %s: URL download took %.2f seconds",
                job_id,
                profile["url_download"],
            )
        else:
            await _progress("s3_download")
            t0 = time.perf_counter()
            logger.info("Job %s: downloading audio from S3 (key: %s)", job_id, s3_key)
            assert s3_key is not None
            file_path = await s3_client.download_audio_to_temp(
                s3_key, suffix=_safe_filename_suffix(filename)
            )
            profile["s3_download"] = time.perf_counter() - t0
            logger.info(
                "Job %s: S3 download took %.2f seconds", job_id, profile["s3_download"]
            )

        await _progress("audio_load")
        t0 = time.perf_counter()
        audio = await load_audio_from_path(file_path, job_id)
        profile["audio_load"] = time.perf_counter() - t0
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        file_path = None
        logger.info(
            "Job %s: loading audio took %.2f seconds", job_id, profile["audio_load"]
        )

        if concurrency_sem:
            await _progress("awaiting_gpu")
            await concurrency_sem.acquire()
            logger.debug("Job %s: acquired GPU concurrency semaphore", job_id)

        try:
            selected_backends = resolve_stage_backends()
            transcription_backend = get_transcription_backend(
                selected_backends.transcription
            )
            alignment_backend = (
                get_alignment_backend(selected_backends.alignment)
                if (align or diarize)
                else None
            )
            diarization_backend = (
                get_diarization_backend(selected_backends.diarization)
                if diarize
                else None
            )

            model_name = (
                params.get("model_name") or get_default_transcription_model_name()
            )
            raw_language = params.get("language")
            language = (
                Language(raw_language) if raw_language else config.default_language
            )
            asr_options = params.get("asr_options")
            task = params.get("task", "transcribe")

            logger.info(
                "Job %s: transcribing %s with model: %s, options: %s, language: %s, task: %s, stage_backends: %s",
                job_id,
                filename,
                model_name,
                asr_options,
                language,
                task,
                selected_backends,
            )

            await _progress("transcribe")
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
                "Job %s: transcription took %.2f seconds", job_id, profile["transcribe"]
            )

            if align or diarize:
                if alignment_backend is None:
                    raise RuntimeError(
                        "Alignment backend is not initialized but alignment or diarization was requested"
                    )
                await _progress("align")
                t0 = time.perf_counter()
                result = await alignment_backend.align(
                    result=result, audio=audio, request_id=job_id
                )
                profile["align"] = time.perf_counter() - t0
                logger.debug(
                    "Job %s: alignment took %.2f seconds", job_id, profile["align"]
                )

            if diarize:
                assert diarization_backend is not None
                await _progress("diarize")
                t0 = time.perf_counter()
                result = await diarization_backend.diarize(
                    result=result,
                    audio=audio,
                    speaker_embeddings=params.get("speaker_embeddings", False),
                    request_id=job_id,
                )
                profile["diarize"] = time.perf_counter() - t0
                logger.debug(
                    "Job %s: diarization took %.2f seconds", job_id, profile["diarize"]
                )

            await _progress("finalize")
            t0 = time.perf_counter()
            result = _finalize_text(result, align or diarize)
            profile["finalize"] = time.perf_counter() - t0

            total = time.perf_counter() - start_time
            logger.info("Job %s: transcription complete for %s", job_id, filename)
            logger.debug(
                "Job %s: profile: total=%.2fs | %s | (other=%.2fs)",
                job_id,
                total,
                " | ".join(f"{k}={v:.2f}s" for k, v in profile.items()),
                total - sum(profile.values()),
            )
            return result
        finally:
            if concurrency_sem:
                concurrency_sem.release()

    finally:
        _close_timeline()
        with contextlib.suppress(Exception):
            if file_path is not None and os.path.exists(file_path):
                os.remove(file_path)
        if config.audio_cleanup and audio is not None:
            del audio
            logger.info("Job %s: audio data cleaned up", job_id)
        if config.cache_cleanup:
            _cleanup_cache_only()
            logger.info("Job %s: cache cleanup completed", job_id)
