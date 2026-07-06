"""Completion-webhook delivery for the optional ``callback_url`` field.

When a request carries ``callback_url``, the terminal result envelope is POSTed
there once the job finishes. In Kafka mode the worker owns delivery (it is the
single execution point that holds the envelope and has URL egress); in direct
mode the API delivers in-process after the response is returned.

Delivery is best-effort: the URL is re-checked against the shared SSRF policy,
posted with one retry, and failures are logged rather than raised. The durable
result stays in S3 (Kafka) or the synchronous response (direct) regardless, so a
dropped webhook never loses data. It never fires on the redelivery/marker-resend
path, so a receiver sees at most one notification per fresh completion.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from whisperx_api_server.url_fetch import validate_url_for_fetch

logger = logging.getLogger(__name__)


async def _post_once(
    url: str, body: str | bytes, *, connect_timeout: float, total_timeout: float
) -> int:
    timeout = aiohttp.ClientTimeout(total=total_timeout, connect=connect_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            allow_redirects=False,
        ) as response:
            return response.status


async def deliver_webhook(
    url: str,
    payload: dict[str, Any] | str | bytes,
    *,
    connect_timeout: float,
    total_timeout: float,
    allow_private_hosts: bool,
    allowed_hosts: list[str] | None,
    retries: int = 1,
) -> bool:
    """POST the result envelope to ``url``; return True on a 2xx response.

    Re-validates the URL against the SSRF policy up front and never raises:
    rejection, connection failure, or a non-2xx status all return False.
    """
    try:
        await validate_url_for_fetch(
            url, allow_private_hosts=allow_private_hosts, allowed_hosts=allowed_hosts
        )
    except Exception as exc:
        logger.warning("Webhook URL rejected: %s", exc)
        return False

    body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)

    for attempt in range(1, retries + 2):
        try:
            status = await _post_once(
                url, body, connect_timeout=connect_timeout, total_timeout=total_timeout
            )
        except Exception as exc:
            logger.warning(
                "Webhook delivery attempt %d failed: %s", attempt, type(exc).__name__
            )
            continue
        if 200 <= status < 300:
            return True
        logger.warning("Webhook delivery attempt %d returned HTTP %d", attempt, status)
    return False


async def deliver_result(url: str, envelope: str | bytes, config: Any) -> bool:
    """Deliver a stored result envelope, pulling timeout/SSRF policy from config."""
    return await deliver_webhook(
        url,
        envelope,
        connect_timeout=config.url_fetch_connect_timeout_seconds,
        total_timeout=config.webhook_timeout_seconds,
        allow_private_hosts=config.url_fetch_allow_private_hosts,
        allowed_hosts=config.url_fetch_allowed_hosts,
    )
