from __future__ import annotations

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from api.core import database as database_module
from api.core import redis as redis_module
from api.core.config import get_settings
from api.middleware.rate_limiter import RateLimitMiddleware
from api.models import job_record as _job_record_models  # noqa: F401 — registers ORM model with Base.metadata
from api.models import user as _user_models  # noqa: F401 — registers ORM model with Base.metadata
from api.routers import admin, auth, health, jobs, segment, websocket

log = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Connect shared resources on startup; tear them down on shutdown."""
    log.info("startup_begin", env=settings.app_env)
    redis_module.init_redis_pool()
    await database_module.init_db()
    log.info("startup_complete")
    yield
    log.info("shutdown_begin")
    await redis_module.close_redis_pool()
    await database_module.close_db()
    log.info("shutdown_complete")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

# ── Middleware (last-added = outermost; CORS wraps RateLimiter) ────────────────

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus ─────────────────────────────────────────────────────────────────

app.mount("/metrics", make_asgi_app())

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(admin.router, prefix="/api/v1/admin")
app.include_router(auth.router, prefix="/api/v1/auth")
app.include_router(jobs.router, prefix="/api/v1/jobs")
app.include_router(segment.router, prefix="/api/v1/segment")
app.include_router(websocket.router, prefix="/ws")
app.include_router(health.router, prefix="/health")
