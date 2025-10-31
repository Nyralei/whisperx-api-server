import contextlib
import os
from whisperx import transcribe as whisperx_transcribe
from whisperx import audio as whisperx_audio
from whisperx import alignment as whisperx_alignment
from whisperx import diarize as whisperx_diarize
from whisperx import types as whisperx_types
from fastapi import UploadFile
import logging
import time
import tempfile
import asyncio
import torch
import gc

from whisperx_api_server.config import (
    Language,
)
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.models import (
    CustomWhisperModel,
    load_align_model_cached,
    load_diarize_model_cached,
    load_transcribe_pipeline_cached,
)

logger = logging.getLogger(__name__)

config = get_config()

_concurrency_semaphore = None

def _get_concurrency_semaphore() -> asyncio.Semaphore | None:
    """Return a semaphore only if running on GPU."""
    global _concurrency_semaphore
    if not torch.cuda.is_available():
        return None
    if _concurrency_semaphore is None:
        max_concurrent = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "1"))
        _concurrency_semaphore = asyncio.Semaphore(max_concurrent)
    return _concurrency_semaphore

def _cleanup_cache_only():
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

async def _save_upload_to_temp(audio_file: UploadFile, request_id: str) -> str:
    loop = asyncio.get_running_loop()
    try:
        file_bytes = await audio_file.read()
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Failed to read uploaded file: {e}")
        raise

    def _write_temp_file(data: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as temp_file:
            temp_file.write(data)
            return temp_file.name
        
    try:
        file_path = await loop.run_in_executor(None, _write_temp_file, file_bytes)
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Failed to write temp file: {e}")
        raise

    return file_path


async def _load_audio(file_path: str, request_id: str):
    loop = asyncio.get_running_loop()
    try:
        audio = await loop.run_in_executor(None, whisperx_audio.load_audio, file_path)
        logger.info(f"Request ID: {request_id} - Audio loaded from {file_path}")
        return audio
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Failed to load audio: {e}")
        raise

async def _transcribe_audio(model, audio, batch_size, chunk_size, language, task, request_id):
    loop = asyncio.get_running_loop()

    def _run_transcription():
        with torch.inference_mode():
            return model.transcribe(
                audio=audio,
                batch_size=batch_size,
                chunk_size=chunk_size,
                num_workers=config.whisper.num_workers,
                language=language,
                task=task,
            )

    result = await loop.run_in_executor(None, _run_transcription)

    logger.info(f"Request ID: {request_id} - Transcription completed")
    return result  


async def _align_audio(result, audio, whispermodel, request_id):
    loop = asyncio.get_running_loop()
    try:
        alignment_model_start = time.time()
        logger.info(f"Request ID: {request_id} - Loading alignment model")
        model_a, metadata = await load_align_model_cached(language_code=result["language"])
        logger.info(f"Request ID: {request_id} - Alignment model loaded")
        logger.info(f"Request ID: {request_id} - Loading alignment model took {time.time() - alignment_model_start:.2f} seconds")

        def _run_alignment():
            with torch.inference_mode():
                return whisperx_alignment.align(
                    transcript=result["segments"],
                    model=model_a,
                    align_model_metadata=metadata,
                    audio=audio,
                    device=whispermodel.device,
                    return_char_alignments=False
                )
        alignment_start = time.time()
        result["segments"] = await loop.run_in_executor(None, _run_alignment)
        logger.info(f"Request ID: {request_id} - Alignment took {time.time() - alignment_start:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Alignment failed: {e}")
        raise


async def _diarize_audio(result, audio, request_id):
    loop = asyncio.get_running_loop()
    try:
        diarization_model_start = time.time()
        logger.info(f"Request ID: {request_id} - Loading diarization model")
        diarize_model = await load_diarize_model_cached(model_name="tensorlake/speaker-diarization-3.1")
        logger.info(f"Request ID: {request_id} - Diarization model loaded. Loading took {time.time() - diarization_model_start:.2f} seconds. Starting diarization")

        def _run_diarization():
            with torch.inference_mode():
                return diarize_model(audio)
        diarize_start = time.time()
        diarize_segments = await loop.run_in_executor(None, _run_diarization)
        result["segments"] = whisperx_diarize.assign_word_speakers(diarize_segments, result["segments"])
        logger.info(f"Request ID: {request_id} - Diarization took {time.time() - diarize_start:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Diarization failed: {e}")
        raise

def _finalize_text(result, align_or_diarize: bool):
    segments = result.get("segments", [])
    if align_or_diarize and isinstance(segments, dict):
        segments = segments.get("segments", [])

    result["text"] = '\n'.join([s.get("text", "").strip() for s in segments if s.get("text")])
    return result

async def transcribe(
    audio_file: UploadFile,
    batch_size: int = config.batch_size,
    chunk_size: int = 30,
    asr_options: dict = {},
    language: Language = config.default_language,
    whispermodel: CustomWhisperModel = config.whisper.model,
    align: bool = False,
    diarize: bool = False,
    request_id: str = "",
    task: str = "transcribe",
) -> whisperx_types.TranscriptionResult:
    start_time = time.time()
    file_path = None
    audio = None
    concurrency_sem = _get_concurrency_semaphore()

    try:
        file_path = await _save_upload_to_temp(audio_file, request_id)
        logger.info(f"Request ID: {request_id} - Saving uploaded file took {time.time() - start_time:.2f} seconds")

        if concurrency_sem:
            await concurrency_sem.acquire()
            logger.debug(f"Request ID: {request_id} - Acquired GPU concurrency semaphore")

        logger.info(f"Request ID: {request_id} - Transcribing {audio_file.filename} with model: {whispermodel.model_size_or_path} and options: {asr_options}, language: {language}, task: {task}")
        
        model_loading_start = time.time()

        model = await load_transcribe_pipeline_cached(
            whispermodel=whispermodel,
            language=language,
            task=task,
        )

        logger.info(f"Request ID: {request_id} - Loading model took {time.time() - model_loading_start:.2f} seconds (cached)")

        audio_loading_start = time.time()

        audio = await _load_audio(file_path, request_id)
        
        logger.info(f"Request ID: {request_id} - Loading audio took {time.time() - audio_loading_start:.2f} seconds")

        transcription_start = time.time()

        result = await _transcribe_audio(model, audio, batch_size, chunk_size, language, task, request_id)

        logger.info(f"Request ID: {request_id} - Transcription took {time.time() - transcription_start:.2f} seconds")

        if align or diarize:
            result = await _align_audio(result, audio, whispermodel, request_id)

        if diarize:
            result = await _diarize_audio(result, audio, request_id)

        result = _finalize_text(result, align or diarize)

        logger.info(f"Request ID: {request_id} - Transcription completed for {audio_file.filename}")

        return result
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Transcription failed for {audio_file.filename} with error: {e}")
        raise
    finally:
        with contextlib.suppress(Exception):
            if concurrency_sem:
                concurrency_sem.release()
        with contextlib.suppress(Exception):
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        if config.audio_cleanup and audio is not None:
            del audio
            logger.debug(f"Request ID: {request_id} - Audio data cleaned up")
        if config.cache_cleanup:
            _cleanup_cache_only()
            logger.debug(f"Request ID: {request_id} - Cache cleanup completed")