"""Unit tests for api/core/redis.py — uses fakeredis, no real Redis needed."""
from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

from api.core.redis import (
    cache_result,
    check_rate_limit,
    compute_job_hash,
    enqueue_job,
    get_cached_result,
    get_job_status,
    publish_job_complete,
    publish_progress,
    set_job_status,
)

# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def r() -> FakeRedis:
    client = FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


# ── compute_job_hash ───────────────────────────────────────────────────────────

def test_compute_job_hash_deterministic():
    h1 = compute_job_hash("s3://bucket/video.mp4", {"x": 1, "y": 2})
    h2 = compute_job_hash("s3://bucket/video.mp4", {"x": 1, "y": 2})
    assert h1 == h2


def test_compute_job_hash_key_order_independent():
    h1 = compute_job_hash("s3://v.mp4", {"b": 2, "a": 1})
    h2 = compute_job_hash("s3://v.mp4", {"a": 1, "b": 2})
    assert h1 == h2


def test_compute_job_hash_different_inputs():
    h1 = compute_job_hash("s3://v1.mp4", {"x": 1})
    h2 = compute_job_hash("s3://v2.mp4", {"x": 1})
    assert h1 != h2


def test_compute_job_hash_is_hex_string():
    h = compute_job_hash("key", {})
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ── check_rate_limit ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limit_allows_up_to_limit(r):
    for _ in range(5):
        assert await check_rate_limit(r, "api-key-1", 5, 60) is True


@pytest.mark.asyncio
async def test_rate_limit_blocks_on_exceed(r):
    for _ in range(5):
        await check_rate_limit(r, "api-key-2", 5, 60)
    assert await check_rate_limit(r, "api-key-2", 5, 60) is False


@pytest.mark.asyncio
async def test_rate_limit_key_namespaced_per_api_key(r):
    for _ in range(5):
        await check_rate_limit(r, "key-a", 5, 60)
    # different api_key should be unaffected
    assert await check_rate_limit(r, "key-b", 5, 60) is True


@pytest.mark.asyncio
async def test_rate_limit_key_has_ttl(r):
    await check_rate_limit(r, "ttl-key", 10, 60)
    ttl = await r.ttl("ratelimit:ttl-key:60")
    assert 0 < ttl <= 61


# ── get_cached_result / cache_result ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_miss_returns_none(r):
    assert await get_cached_result(r, "nonexistent") is None


@pytest.mark.asyncio
async def test_cache_round_trip(r):
    result = {"output_url": "s3://out/video.mp4", "duration": 12.3}
    await cache_result(r, "abc123", result)
    cached = await get_cached_result(r, "abc123")
    assert cached == result


@pytest.mark.asyncio
async def test_cache_result_has_ttl(r):
    await cache_result(r, "ttlhash", {"x": 1})
    ttl = await r.ttl("result:ttlhash")
    assert ttl > 0


# ── enqueue_job ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enqueue_job_stores_payload(r):
    await enqueue_job(r, "job-1", priority=1, payload={"video": "s3://v.mp4"})
    raw = await r.get("job:payload:job-1")
    assert json.loads(raw) == {"video": "s3://v.mp4"}


@pytest.mark.asyncio
async def test_enqueue_job_adds_to_queue(r):
    await enqueue_job(r, "job-2", priority=5, payload={})
    score = await r.zscore("job:queue", "job-2")
    assert score == 5.0


@pytest.mark.asyncio
async def test_enqueue_job_payload_has_ttl(r):
    await enqueue_job(r, "job-3", priority=1, payload={})
    ttl = await r.ttl("job:payload:job-3")
    assert 0 < ttl <= 3600


@pytest.mark.asyncio
async def test_enqueue_job_queue_has_ttl(r):
    await enqueue_job(r, "job-4", priority=1, payload={})
    ttl = await r.ttl("job:queue")
    assert ttl > 0


@pytest.mark.asyncio
async def test_enqueue_job_priority_ordering(r):
    await enqueue_job(r, "low", priority=1, payload={})
    await enqueue_job(r, "high", priority=10, payload={})
    # ZPOPMAX returns highest score first
    result = await r.zpopmax("job:queue")
    assert result[0][0] == "high"


# ── publish_progress / publish_job_complete ───────────────────────────────────

@pytest.mark.asyncio
async def test_publish_progress_does_not_raise(r):
    # No subscribers — publish returns 0 receivers, should not raise
    await publish_progress(r, "job-5", {"step": "segmentation", "pct": 25})


@pytest.mark.asyncio
async def test_publish_job_complete_does_not_raise(r):
    await publish_job_complete(r, "job-6", {"output_url": "s3://out/done.mp4"})


# ── get_job_status / set_job_status ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_job_status_returns_none_for_unknown(r):
    assert await get_job_status(r, "no-such-job") is None


@pytest.mark.asyncio
async def test_job_status_round_trip(r):
    status = {"state": "processing", "progress": 42}
    await set_job_status(r, "job-7", status)
    result = await get_job_status(r, "job-7")
    assert result == status


@pytest.mark.asyncio
async def test_job_status_has_ttl(r):
    await set_job_status(r, "job-8", {"state": "done"})
    ttl = await r.ttl("job:status:job-8")
    assert ttl > 0


@pytest.mark.asyncio
async def test_job_status_overwrite(r):
    await set_job_status(r, "job-9", {"state": "queued"})
    await set_job_status(r, "job-9", {"state": "done"})
    result = await get_job_status(r, "job-9")
    assert result["state"] == "done"
