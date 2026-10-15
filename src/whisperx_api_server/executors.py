import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import anyio

logger = logging.getLogger(__name__)

T = TypeVar("T")

_model_executor: ThreadPoolExecutor | None = None
_io_executor: ThreadPoolExecutor | None = None


class ModelCallStopped(Exception):
    pass


def get_model_executor() -> ThreadPoolExecutor:
    global _model_executor
    if _model_executor is None:
        from whisperx_api_server.dependencies import get_config

        config = get_config()
        n = max(1, config.max_concurrent_transcriptions)
        _model_executor = ThreadPoolExecutor(
            max_workers=n, thread_name_prefix="wx-model"
        )
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


async def await_to_completion(
    future: asyncio.Future[T], on_cancel: Callable[[], None] | None = None
) -> T:
    """Await future; if the caller is cancelled, call on_cancel and keep waiting
    until future is done before re-raising, so what it uses stays held until then."""
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        if on_cancel is not None:
            on_cancel()
        with anyio.CancelScope(shield=True):
            while not future.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait({future})
        if not future.cancelled():
            exc = future.exception()
            if exc is not None and not isinstance(exc, ModelCallStopped):
                logger.warning("Call failed after its caller was cancelled: %r", exc)
        raise


async def run_stoppable(fn: Callable[[Callable[..., None]], T]) -> T:
    """Run fn(checkpoint) on the model executor; once the caller is cancelled,
    the next checkpoint() raises inside the thread."""
    stop = threading.Event()

    def checkpoint(*_: Any) -> None:
        if stop.is_set():
            raise ModelCallStopped

    def call() -> T:
        checkpoint()
        return fn(checkpoint)

    future = asyncio.get_running_loop().run_in_executor(get_model_executor(), call)
    return await await_to_completion(future, on_cancel=stop.set)


def shutdown_executors() -> None:
    global _model_executor, _io_executor
    for name, ex in [("model", _model_executor), ("io", _io_executor)]:
        if ex is not None:
            ex.shutdown(wait=False)
            logger.info("%s executor shut down", name)
    _model_executor = None
    _io_executor = None
