import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_model_executor: ThreadPoolExecutor | None = None
_io_executor: ThreadPoolExecutor | None = None


def get_model_executor() -> ThreadPoolExecutor:
    global _model_executor
    if _model_executor is None:
        from whisperx_api_server.dependencies import get_config
        config = get_config()
        n = max(1, config.max_concurrent_transcriptions)
        _model_executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="wx-model")
        logger.info("model thread pool: %d workers", n)
    return _model_executor


def get_io_executor() -> ThreadPoolExecutor:
    global _io_executor
    if _io_executor is None:
        from whisperx_api_server.dependencies import get_config
        config = get_config()
        n = min(32, 4 * max(1, config.max_concurrent_transcriptions))
        _io_executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="wx-io")
        logger.info("IO thread pool: %d workers", n)
    return _io_executor


def shutdown_executors() -> None:
    global _model_executor, _io_executor
    for name, ex in [("model", _model_executor), ("io", _io_executor)]:
        if ex is not None:
            ex.shutdown(wait=False)
            logger.info("%s executor shut down", name)
    _model_executor = None
    _io_executor = None
