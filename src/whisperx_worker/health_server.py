"""Worker HTTP health server.

Runs alongside the Kafka consumer loop and exposes:

  - GET /healthcheck — liveness, always 200 while the process is alive.
  - GET /ready       — readiness, 200 only after the worker has loaded models,
                       initialized the S3 client, and subscribed to the Kafka
                       request topic. 503 otherwise, with a JSON body listing
                       which gates are still pending.

Kubernetes startup/readiness probes use /ready so OrderedReady StatefulSet
rollout actually waits for each pod to be able to process jobs. Liveness uses
/healthcheck so the process isn't killed during long model preloads.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass
class WorkerReadiness:
    models_loaded: asyncio.Event = field(default_factory=asyncio.Event)
    s3_initialized: asyncio.Event = field(default_factory=asyncio.Event)
    kafka_subscribed: asyncio.Event = field(default_factory=asyncio.Event)

    def is_ready(self) -> bool:
        return all(
            e.is_set()
            for e in (self.models_loaded, self.s3_initialized, self.kafka_subscribed)
        )

    def pending(self) -> list[str]:
        return [
            name
            for name, e in (
                ("models_loaded", self.models_loaded),
                ("s3_initialized", self.s3_initialized),
                ("kafka_subscribed", self.kafka_subscribed),
            )
            if not e.is_set()
        ]


async def start_health_server(readiness: WorkerReadiness, port: int) -> web.AppRunner:
    app = web.Application()

    async def healthcheck(_: web.Request) -> web.Response:
        return web.json_response({"status": "alive"})

    async def ready(_: web.Request) -> web.Response:
        if readiness.is_ready():
            return web.json_response({"status": "ready"})
        return web.json_response(
            {"status": "not_ready", "pending": readiness.pending()},
            status=503,
        )

    app.add_routes(
        [
            web.get("/healthcheck", healthcheck),
            web.get("/ready", ready),
        ]
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    logger.info("Worker health server started on port %d", port)
    return runner
