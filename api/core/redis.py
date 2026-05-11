from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
import structlog
from redis.exceptions import RedisError

from api.core.config import get_settings

log = structlog.get_logger(__name__)

_pool: aioredis.ConnectionPool | None = None

# ── Pool lifecycle ─────────────────────────────────────────────────────────────

def init_redis_pool() -> None:
    """Create the module-level async connection pool; call once at app startup."""
    global _pool
    settings = get_settings()
    _pool = aioredis.ConnectionPool.from_url(
        settings.redis_url,
        max_connections=20,
        decode_responses=True,
    )
    log.info("redis_pool_initialized", url=settings.redis_url, max_connections=20)


async def close_redis_pool() -> None:
    """Drain the connection pool; call once at app shutdown."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        log.info("redis_pool_closed")


async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """Yield a Redis client from the shared pool; use as FastAPI Depends()."""
    if _pool is None:
        raise RuntimeError("Redis pool not initialized — call init_redis_pool() at startup.")
    client = aioredis.Redis(connection_pool=_pool)
    try:
        yield client
    finally:
        await client.aclose()


@asynccontextmanager
async def acquire_redis() -> AsyncIterator[aioredis.Redis]:
    """Async context manager for Redis access outside of FastAPI Depends (e.g. middleware)."""
    if _pool is None:
        raise RuntimeError("Redis pool not initialized — call init_redis_pool() at startup.")
    client = aioredis.Redis(connection_pool=_pool)
    try:
        yield client
    finally:
        await client.aclose()


# ── Rate limiting (sliding window) ────────────────────────────────────────────

async def check_rate_limit(
    redis: aioredis.Redis,
    api_key: str,
    limit: int,
    window_sec: int,
) -> bool:
    """Return True if request is within limit, False if rate limit exceeded."""
    now = time.time()
    window_start = now - window_sec
    key = f"ratelimit:{api_key}:{window_sec}"
    member = str(uuid.uuid4())

    async with redis.pipeline(transaction=True) as pipe:
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zadd(key, {member: now})
        pipe.zcard(key)
        pipe.expire(key, window_sec + 1)
        results = await pipe.execute()

    count: int = results[2]
    allowed = count <= limit
    log.debug(
        "rate_limit_check",
        api_key=api_key,
        window_sec=window_sec,
        count=count,
        limit=limit,
        allowed=allowed,
    )
    return allowed


# ── Job deduplication ──────────────────────────────────────────────────────────

def compute_job_hash(video_s3_key: str, mask_data: dict[str, Any]) -> str:
    """Return SHA-256 hex digest of the canonical (video_s3_key + mask_data) input."""
    canonical = video_s3_key + json.dumps(mask_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def get_cached_result(
    redis: aioredis.Redis, job_hash: str
) -> dict[str, Any] | None:
    """Return cached result for job_hash, or None on miss."""
    key = f"result:{job_hash}"
    try:
        raw = await redis.get(key)
    except RedisError as exc:
        log.warning("cache_get_error", job_hash=job_hash, error=str(exc))
        return None
    if raw is None:
        log.debug("cache_miss", job_hash=job_hash)
        return None
    log.info("cache_hit", job_hash=job_hash)
    return json.loads(raw)


async def cache_result(
    redis: aioredis.Redis, job_hash: str, result: dict[str, Any]
) -> None:
    """Store result under result:{job_hash} with redis_result_ttl TTL."""
    key = f"result:{job_hash}"
    settings = get_settings()
    try:
        await redis.setex(key, settings.redis_result_ttl, json.dumps(result))
    except RedisError as exc:
        log.error("cache_set_error", job_hash=job_hash, error=str(exc))
        raise
    log.debug("result_cached", job_hash=job_hash, ttl=settings.redis_result_ttl)


# ── Priority queue ─────────────────────────────────────────────────────────────

_QUEUE_KEY = "job:queue"


async def enqueue_job(
    redis: aioredis.Redis,
    job_id: str,
    priority: int | float,
    payload: dict[str, Any],
) -> None:
    """Store job payload and add job_id to the priority sorted set.

    Higher score = higher priority. Workers dequeue with ZPOPMAX.
    """
    settings = get_settings()
    payload_key = f"job:payload:{job_id}"

    async with redis.pipeline(transaction=True) as pipe:
        pipe.setex(payload_key, 3600, json.dumps(payload))
        pipe.zadd(_QUEUE_KEY, {job_id: priority})
        pipe.expire(_QUEUE_KEY, settings.redis_result_ttl)
        await pipe.execute()

    log.info("job_enqueued", job_id=job_id, priority=priority)


# ── Progress pub/sub ───────────────────────────────────────────────────────────

async def publish_progress(
    redis: aioredis.Redis, job_id: str, data: dict[str, Any]
) -> None:
    """Publish an in-progress update on channel progress:{job_id}."""
    channel = f"progress:{job_id}"
    message = json.dumps({"type": "progress", **data})
    try:
        await redis.publish(channel, message)
    except RedisError as exc:
        log.warning("publish_progress_error", job_id=job_id, error=str(exc))


async def publish_job_complete(
    redis: aioredis.Redis, job_id: str, result: dict[str, Any]
) -> None:
    """Publish a completion event on channel progress:{job_id}."""
    channel = f"progress:{job_id}"
    message = json.dumps({"type": "complete", "result": result})
    try:
        await redis.publish(channel, message)
    except RedisError as exc:
        log.warning("publish_complete_error", job_id=job_id, error=str(exc))


# ── Job status ─────────────────────────────────────────────────────────────────

async def get_job_status(
    redis: aioredis.Redis, job_id: str
) -> dict[str, Any] | None:
    """Return current job status dict, or None if not found."""
    key = f"job:status:{job_id}"
    try:
        raw = await redis.get(key)
    except RedisError as exc:
        log.warning("get_status_error", job_id=job_id, error=str(exc))
        return None
    if raw is None:
        return None
    return json.loads(raw)


async def set_job_status(
    redis: aioredis.Redis, job_id: str, status: dict[str, Any]
) -> None:
    """Write job status under job:status:{job_id} with redis_result_ttl TTL."""
    key = f"job:status:{job_id}"
    settings = get_settings()
    try:
        await redis.setex(key, settings.redis_result_ttl, json.dumps(status))
    except RedisError as exc:
        log.error("set_status_error", job_id=job_id, error=str(exc))
        raise
    log.debug("job_status_set", job_id=job_id, state=status.get("state"))
