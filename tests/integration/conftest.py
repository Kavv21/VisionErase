"""Integration test configuration.

All services must be reachable at localhost (docker-compose mapped ports):
  Redis    → localhost:6379
  Postgres → localhost:5432
  MinIO    → localhost:9000

Env vars are overridden at module level so pydantic-settings picks up
localhost addresses instead of the docker-compose service names in .env.
"""
from __future__ import annotations

import os

# Override docker service hostnames before any app module is imported.
# pydantic-settings gives environment variables precedence over .env file values.
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/1"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/2"
os.environ["DATABASE_URL"] = (
    "postgresql+asyncpg://visionerase:secret@localhost:5432/visionerase"
)
os.environ["S3_ENDPOINT_URL"] = "http://localhost:9000"

import boto3
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from botocore.config import Config
from botocore.exceptions import ClientError
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from api.core import redis as redis_module
from api.core.config import get_settings
from api.main import app


# ── Session-level service bootstrapping ───────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def ensure_minio_bucket() -> None:
    """Create the uploads S3 bucket in MinIO once for the session if absent."""
    settings = get_settings()
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    try:
        s3.head_bucket(Bucket=settings.s3_bucket)
    except ClientError:
        try:
            s3.create_bucket(Bucket=settings.s3_bucket)
        except ClientError:
            pass  # already exists under a different code


# ── Per-test fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture()
async def client() -> AsyncClient:
    """Async HTTP test client with a fresh Redis pool bound to this test's event loop.

    ASGITransport bypasses the ASGI lifespan, so we own the pool lifecycle
    here.  Each async test runs in its own event loop (pytest-asyncio STRICT
    mode); re-initialising the pool per test prevents stale connections from
    a previous loop from leaking into the next test.
    """
    redis_module.init_redis_pool()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            follow_redirects=True,
        ) as ac:
            yield ac
    finally:
        await redis_module.close_redis_pool()


@pytest_asyncio.fixture()
async def redis_direct() -> aioredis.Redis:
    """Direct Redis client for test-side assertions and manual state injection."""
    r = aioredis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
    yield r
    await r.aclose()


@pytest.fixture()
def ws_client() -> TestClient:
    """Synchronous Starlette test client for WebSocket tests.

    Used as a context manager so the ASGI lifespan is triggered, which
    initialises its own Redis pool bound to the TestClient's event loop.
    The pool is closed when the context manager exits.
    """
    with TestClient(app, raise_server_exceptions=True) as tc:
        yield tc
