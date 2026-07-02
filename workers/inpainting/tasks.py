from __future__ import annotations

import asyncio
from typing import Any

import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.metrics import SEGMENTS_TOTAL
from api.core.redis import acquire_redis, publish_progress, set_job_status
from api.models.job import JobStatus
from workers.celery_app import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(
    bind=True,
    queue="inpainting",
    max_retries=2,
    soft_time_limit=600,
    time_limit=660,
    name="workers.inpainting.tasks.inpaint_segment",
)
def inpaint_segment(
    self,
    job_id: str,
    segment_s3_key: str,
    tracked_masks_s3_key: str,
) -> dict[str, Any]:
    """Stub: ProPainter inpainting (Month 2 integration)."""
    bound_log = log.bind(job_id=job_id, task_id=self.request.id)
    bound_log.info("inpainting_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "inpainting", "pct": 0})
        )
        loop.run_until_complete(
            _update_status(job_id, JobStatus.INPAINTING, None)
        )
        SEGMENTS_TOTAL.labels(status="inpainting_started").inc()

        inpainted_s3_key = f"jobs/{job_id}/inpainted/stub.mp4"

        loop.run_until_complete(
            _publish(job_id, {"stage": "inpainting", "pct": 100})
        )
        SEGMENTS_TOTAL.labels(status="inpainting_complete").inc()

        from workers.stitching.tasks import stitch_segments
        stitch_segments.apply_async(
            args=[job_id, [inpainted_s3_key]],
            queue="stitching",
        )

        return {
            "job_id": job_id,
            "inpainted_s3_key": inpainted_s3_key,
            "status": "stub",
        }

    except SoftTimeLimitExceeded:
        bound_log.error("inpainting_soft_timeout")
        SEGMENTS_TOTAL.labels(status="inpainting_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "inpaint_segment timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("inpainting_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="inpainting_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _publish(job_id: str, data: dict) -> None:
    from redis import asyncio as _aioredis
    import json as _json
    from api.core.config import get_settings as _get_settings
    _s = _get_settings()
    _r = _aioredis.Redis.from_url(_s.redis_url, decode_responses=True)
    try:
        channel = f"progress:{job_id}"
        message = _json.dumps({"type": "progress", **data})
        await _r.publish(channel, message)
    finally:
        await _r.aclose()


async def _update_status(job_id: str, status: str, error: str | None = None, extra: dict | None = None) -> None:
    import json as _json
    from redis import asyncio as _aioredis
    from api.core.config import get_settings as _get_settings
    _s = _get_settings()
    status_str = status.value if hasattr(status, "value") else str(status)
    _r = _aioredis.Redis.from_url(_s.redis_url, decode_responses=True)
    try:
        key = f"job:status:{job_id}"
        existing_raw = await _r.get(key)
        existing = _json.loads(existing_raw) if existing_raw else {}
        existing["status"] = status_str
        if error is not None:
            existing["error"] = error
        if extra:
            existing.update(extra)
        await _r.setex(key, _s.redis_result_ttl, _json.dumps(existing))
    finally:
        await _r.aclose()
