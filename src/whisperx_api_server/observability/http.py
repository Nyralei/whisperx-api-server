"""HTTP-level instrument shims"""

import contextlib
import time
from typing import Any

from starlette.types import ASGIApp, Scope, Receive, Send

EXCLUDED_PATHS: frozenset[str] = frozenset({"/metrics"})

ERROR_TYPE_MAP: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    413: "payload_too_large",
    422: "invalid_audio",
    500: "pipeline_error",
    503: "queue_full",
    504: "timeout",
}

# Static allowlist used as the `endpoint` label for the in-flight gauge.
# Any path not in this set is normalized to "other" so unbounded paths
# (typos, scanners, future routes) cannot grow Prometheus cardinality.
KNOWN_INFLIGHT_PATHS: frozenset[str] = frozenset(
    {
        "/healthcheck",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/models/list",
        "/models/load",
        "/models/unload",
        "/align_models/list",
        "/align_models/load",
        "/align_models/unload",
        "/diarize_models/list",
        "/diarize_models/load",
        "/diarize_models/unload",
    }
)


def _inflight_label(path: str) -> str:
    return path if path in KNOWN_INFLIGHT_PATHS else "other"


class _NoOpHistogram:
    def labels(self, **kwargs: Any) -> "_NoOpHistogram":
        return self

    def observe(self, amount: float) -> None:
        pass

    def time(self):
        return contextlib.nullcontext()


class _NoOpCounter:
    def labels(self, **kwargs: Any) -> "_NoOpCounter":
        return self

    def inc(self, amount: float = 1) -> None:
        pass


class _NoOpGauge:
    def labels(self, **kwargs: Any) -> "_NoOpGauge":
        return self

    def set(self, value: float) -> None:
        pass

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set_function(self, f) -> None:
        pass


# Module-level singletons — shims by default, replaced by setup_metrics() when enabled.
requests_total: Any = _NoOpCounter()
request_duration: Any = _NoOpHistogram()
requests_in_flight: Any = _NoOpGauge()
errors_total: Any = _NoOpCounter()


class MetricsMiddleware:
    """Raw ASGI middleware that records HTTP request metrics.
    Records four instruments per request (defined in observability/http.py
    module-level singletons, replaced with real prometheus_client objects
    by registry._setup_http_instruments()):
      - whisperx_http_requests_total      (Counter)
      - whisperx_http_request_duration_seconds  (Histogram)
      - whisperx_http_requests_in_flight  (Gauge)
      - whisperx_http_errors_total        (Counter, status_code >= 400 only)

    Path exclusion: /metrics passes through untouched (D-03).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Pass-through for non-HTTP scopes (lifespan, websocket).
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Read scope['path'] ONCE — used as the in-flight gauge label key
        # for BOTH inc and dec (Pitfall 2: mismatched labels leak the gauge).
        path: str = scope["path"]
        if path in EXCLUDED_PATHS:
            # /metrics: pass through with NO instrument calls (D-03).
            await self.app(scope, receive, send)
            return

        # Deferred import so the module read is the CURRENT module state —
        # picks up real instruments after registry._setup_http_instruments()
        # has replaced the singletons.
        from whisperx_api_server.observability import http as _http

        inflight_label: str = _inflight_label(path)
        start_time = time.perf_counter()
        status_holder: list[int] = []

        async def wrapped_send(message) -> None:
            # Capture the status code from http.response.start; pass every
            # message through unchanged so the response body streams.
            if message["type"] == "http.response.start":
                status_holder.append(message["status"])
            await send(message)

        _http.requests_in_flight.labels(endpoint=inflight_label).inc()
        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            route = scope.get("route")
            endpoint: str = getattr(route, "path", None) or "unknown"
            elapsed = time.perf_counter() - start_time
            sc = status_holder[0] if status_holder else 500
            sc_class = f"{sc // 100}xx"
            method: str = scope.get("method", "")

            _http.requests_in_flight.labels(endpoint=inflight_label).dec()
            _http.requests_total.labels(
                endpoint=endpoint, method=method, status_code=str(sc)
            ).inc()
            _http.request_duration.labels(
                endpoint=endpoint, method=method, status_code_class=sc_class
            ).observe(elapsed)
            if sc >= 400:
                error_type = ERROR_TYPE_MAP.get(sc, "unknown")
                _http.errors_total.labels(
                    endpoint=endpoint, status_code=str(sc), error_type=error_type
                ).inc()
