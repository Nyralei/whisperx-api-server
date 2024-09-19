
import logging
from collections import defaultdict
from asyncio import Lock
from whisperx_api_server.config import (
    config,
)
import whisperx_api_server.transcriber as transcriber

from whisperx import alignment as whisperx_alignment
from whisperx import diarize as whisperx_diarize

logger = logging.getLogger("transcriber_logger")
logger.setLevel(config.log_level)

model_instances = {}
model_locks = defaultdict(Lock)

align_model_instances = {}
diarize_model_instances = {}

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
    
def load_align_model_cached(language_code, device, model_name=None, model_dir=None):
    global align_model_instances

    # Check if the model for this language is already cached
    if language_code in align_model_instances:
        logger.info(f"Reusing cached model for language: {language_code}")
        return align_model_instances[language_code]["model"], align_model_instances[language_code]["metadata"]

    # If not cached, call the original load_align_model function
    align_model, align_metadata = whisperx_alignment.load_align_model(
        language_code=language_code,
        device=device,
        model_name=model_name,
        model_dir=model_dir
    )

    # Cache the loaded model and metadata
    align_model_instances[language_code] = {
        "model": align_model,
        "metadata": align_metadata
    }

    return align_model, align_metadata

def load_diarize_model_cached(model_name, device):
    global diarize_model_instances

    if model_name in diarize_model_instances:
        logger.info(f"Reusing cached model for diarization: {model_name}")
        return diarize_model_instances[model_name]
    
    logger.info(f"Loading diarization pipeline for model: {model_name} with device: {device}")

    diarize_model = whisperx_diarize.DiarizationPipeline(model_name=model_name, device=device)

    diarize_model_instances[model_name] = diarize_model

    return diarize_model