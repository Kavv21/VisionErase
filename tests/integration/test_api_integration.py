"""Integration tests for the VisionErase API.

Exercises all modules together against real Docker services:
  Redis → localhost:6379, PostgreSQL → localhost:5432, MinIO → localhost:9000

Each test uses a unique job ID / API key and cleans up its own Redis keys
so the server state stays predictable across the suite.

Run with:
  python -m pytest tests/integration/ -v --timeout=30
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
import redis.asyncio as aioredis
from httpx import AsyncClient
from starlette.testclient import TestClient

from api.core.redis import cache_result, compute_job_hash


# ── Helpers ────────────────────────────────────────────────────────────────────


def _job_payload(s3_key: str | None = None) -> dict:
    """Return a minimal valid CreateJobRequest body."""
    return {
        "video_s3_key": s3_key or f"uploads/integ-{uuid.uuid4()}.mp4",
        "mask": {"points": [{"x": 0.5, "y": 0.5}], "frame_index": 0},
    }


# ── Health ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_endpoint_returns_ok(client: AsyncClient) -> None:
    """GET /health must return 200 {"status": "ok"}."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Metrics ────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_data(client: AsyncClient) -> None:
    """GET /metrics must serve Prometheus-format text containing our metric prefix."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "visionerase_" in resp.text


# ── Upload URL ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_url_endpoint_returns_presigned_url(client: AsyncClient) -> None:
    """POST /api/v1/jobs/upload-url must return an http presigned URL and uploads/ key."""
    resp = await client.post(
        "/api/v1/jobs/upload-url",
        json={"filename": "test-video.mp4", "size_bytes": 1024 * 1024},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["upload_url"].startswith("http"), (
        f"Expected presigned URL starting with 'http', got: {data['upload_url'][:80]}"
    )
    assert data["s3_key"].startswith("uploads/"), (
        f"Expected s3_key starting with 'uploads/', got: {data['s3_key']}"
    )
    assert data["expires_in"] == 3600


# ── Job submission ─────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_job_submission_returns_202(
    client: AsyncClient, redis_direct: aioredis.Redis
) -> None:
    """POST /api/v1/jobs with a valid payload must return 202 with job_id and status=pending."""
    payload = _job_payload()
    resp = await client.post("/api/v1/jobs", json=payload)

    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data, "Response missing 'job_id'"
    assert data["status"] == "pending"
    assert data["cached"] is False

    job_id = data["job_id"]
    await redis_direct.delete(f"job:status:{job_id}", f"job:payload:{job_id}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_job_submission_stored_in_redis(
    client: AsyncClient, redis_direct: aioredis.Redis
) -> None:
    """After POST /api/v1/jobs the job status document must be readable from Redis."""
    payload = _job_payload()
    resp = await client.post("/api/v1/jobs", json=payload)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    key = f"job:status:{job_id}"
    try:
        raw = await redis_direct.get(key)
        assert raw is not None, f"Redis key {key!r} was not found after job submission"
        doc = json.loads(raw)
        assert doc["job_id"] == job_id
        assert doc["status"] == "pending"
    finally:
        await redis_direct.delete(f"job:status:{job_id}", f"job:payload:{job_id}")


# ── Deduplication ──────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_job_returns_cached_true(
    client: AsyncClient, redis_direct: aioredis.Redis
) -> None:
    """Submitting the same job a second time must return cached=True with the same job_id."""
    unique_key = f"uploads/integ-dedup-{uuid.uuid4()}.mp4"
    payload = _job_payload(unique_key)

    # First submission — new job accepted
    r1 = await client.post("/api/v1/jobs", json=payload)
    assert r1.status_code == 202
    job_id = r1.json()["job_id"]

    job_hash = compute_job_hash(unique_key, payload["mask"])

    # Simulate a worker completing the job so the dedup cache is populated.
    await cache_result(
        redis_direct,
        job_hash,
        {
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result_s3_key": None,
        },
    )

    try:
        # Second submission — dedup hit
        r2 = await client.post("/api/v1/jobs", json=payload)
        assert r2.status_code == 200, (
            f"Expected 200 for dedup hit, got {r2.status_code}"
        )
        assert r2.json()["cached"] is True
        assert r2.json()["job_id"] == job_id
    finally:
        await redis_direct.delete(
            f"result:{job_hash}",
            f"job:status:{job_id}",
            f"job:payload:{job_id}",
        )


# ── Rate limiting ──────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rate_limiter_blocks_on_21st_request(
    client: AsyncClient, redis_direct: aioredis.Redis
) -> None:
    """The per-minute sliding-window rate limiter must reject the 21st request."""
    api_key = f"integ-ratelimit-{uuid.uuid4()}"
    headers = {"X-API-Key": api_key}
    upload_payload = {"filename": "rl-probe.mp4", "size_bytes": 512}

    try:
        for i in range(20):
            resp = await client.post(
                "/api/v1/jobs/upload-url",
                json=upload_payload,
                headers=headers,
            )
            assert resp.status_code == 200, (
                f"Request {i + 1}/20 was unexpectedly rejected: {resp.status_code}"
            )

        resp_21 = await client.post(
            "/api/v1/jobs/upload-url",
            json=upload_payload,
            headers=headers,
        )
        assert resp_21.status_code == 429
        assert "Rate limit" in resp_21.json().get("detail", "")
    finally:
        await redis_direct.delete(
            f"ratelimit:{api_key}:60", f"ratelimit:{api_key}:3600"
        )


# ── Job not found ──────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unknown_job_returns_404(client: AsyncClient) -> None:
    """GET /api/v1/jobs/{unknown_id} must return 404."""
    resp = await client.get(f"/api/v1/jobs/nonexistent-{uuid.uuid4()}")
    assert resp.status_code == 404


# ── CORS ───────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cors_header_present_on_api_response(client: AsyncClient) -> None:
    """Requests from an allowed origin must receive Access-Control-Allow-Origin."""
    resp = await client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers, (
        "CORS header missing; allowed origins may not include http://localhost:3000"
    )


# ── WebSocket ──────────────────────────────────────────────────────────────────
# Note: this test is synchronous — httpx does not support WebSocket.
# ws_client is a bare TestClient (no context manager) so the ASGI lifespan
# is NOT re-triggered and the session Redis pool remains alive.


@pytest.mark.integration
def test_websocket_connects_and_sends_error_for_unknown_job(
    ws_client: TestClient,
) -> None:
    """Connecting to /ws/{unknown_id} must immediately yield a job-not-found error."""
    job_id = f"nonexistent-{uuid.uuid4()}"
    with ws_client.websocket_connect(f"/ws/{job_id}") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["message"] == "job not found"
