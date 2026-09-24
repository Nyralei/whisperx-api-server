"""Retention loop for backends that have no lifecycle policy of their own.

Three classes of object accumulate otherwise: unconsumed audio, result envelopes
(nothing deletes these on the happy path), and claims left by workers that died
holding a lease.
"""

from __future__ import annotations

import asyncio
import logging
import random

from whisperx_api_server.config import Config

from .contracts import ObjectStore

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86400.0
_MAX_STARTUP_JITTER_SECONDS = 300.0


async def sweep_loop(
    store: ObjectStore,
    *,
    interval_seconds: float,
    retention_days: int,
    lease_grace_seconds: float,
) -> None:
    sweep = getattr(store, "sweep_expired", None)
    if sweep is None or interval_seconds <= 0 or retention_days <= 0:
        return

    older_than = retention_days * _SECONDS_PER_DAY
    # Stagger replicas so they don't all readdir a large network directory at
    # the same instant.
    await asyncio.sleep(
        random.uniform(0.0, min(interval_seconds, _MAX_STARTUP_JITTER_SECONDS))
    )
    while True:
        try:
            removed = await sweep(
                older_than_seconds=older_than,
                lease_grace_seconds=lease_grace_seconds,
            )
            if any(removed.values()):
                logger.info("Storage retention sweep removed %s", removed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Storage retention sweep failed", exc_info=True)
        await asyncio.sleep(interval_seconds)


def start_sweeper(store: ObjectStore, config: Config) -> asyncio.Task | None:
    """Start the retention loop, or return None when the backend manages its own."""
    if not hasattr(store, "sweep_expired"):
        return None
    fs = config.storage.fs
    if fs.sweep_interval_seconds <= 0 or fs.retention_days <= 0:
        logger.info("Storage retention sweep disabled by configuration")
        return None
    return asyncio.create_task(
        sweep_loop(
            store,
            interval_seconds=fs.sweep_interval_seconds,
            retention_days=fs.retention_days,
            lease_grace_seconds=config.kafka.job_lease_ttl_seconds,
        ),
        name="storage-retention-sweep",
    )
