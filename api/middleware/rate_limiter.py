from __future__ import annotations

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from api.core.config import get_settings
from api.core.metrics import RATE_LIMIT_HITS_TOTAL
from api.core.redis import acquire_redis, check_rate_limit

log = structlog.get_logger(__name__)

# Paths that bypass rate limiting entirely.
_EXEMPT_PREFIXES = ("/health", "/metrics", "/docs", "/redoc", "/openapi.json")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter (per-minute and per-hour) keyed on X-API-Key.

    Falls back to client IP when no API key header is present.
    Skips rate limiting gracefully when the Redis pool is unavailable.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._settings = get_settings()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # OPTIONS (CORS preflight) and exempt paths skip rate limiting.
        if request.method == "OPTIONS" or any(
            request.url.path.startswith(p) for p in _EXEMPT_PREFIXES
        ):
            return await call_next(request)

        api_key = (
            request.headers.get("X-API-Key")
            or (request.client.host if request.client else None)
            or "anonymous"
        )
        bound_log = log.bind(api_key=api_key, path=request.url.path)

        try:
            async with acquire_redis() as redis:
                if not await check_rate_limit(
                    redis, api_key, self._settings.rate_limit_per_minute, 60
                ):
                    bound_log.warning("rate_limit_exceeded", window="per_minute")
                    RATE_LIMIT_HITS_TOTAL.labels(window="per_minute").inc()
                    return JSONResponse(
                        {"detail": "Rate limit exceeded. Try again later."},
                        status_code=429,
                        headers={"Retry-After": "60"},
                    )

                if not await check_rate_limit(
                    redis, api_key, self._settings.rate_limit_per_hour, 3600
                ):
                    bound_log.warning("rate_limit_exceeded", window="per_hour")
                    RATE_LIMIT_HITS_TOTAL.labels(window="per_hour").inc()
                    return JSONResponse(
                        {"detail": "Hourly rate limit exceeded. Try again later."},
                        status_code=429,
                        headers={"Retry-After": "3600"},
                    )
        except RuntimeError:
            # Pool not yet initialised (e.g. during tests) — pass through.
            bound_log.debug("rate_limiter_skipped_no_pool")

        return await call_next(request)
