from __future__ import annotations

import asyncio
from typing import Any

import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.metrics import SEGMENTS_TOTAL
from api.core.redis import acquire_redis, publish_progress, set_job_status

log = structlog.get_logger(__name__)

# Import celery_app lazily to avoid circular imports at module load time.
from workers.celery_app import celery_app  # noqa: E402


@celery_app.task(
    bind=True,
    queue="segmentation",
    max_retries=2,
    soft_time_limit=300,
    time_limit=360,
    name="workers.segmentation.tasks.segment_first_frame",
)
def segment_first_frame(
    self,
    job_id: str,
    segment_s3_key: str,
    mask_data: dict[str, Any],
) -> dict[str, Any]:
    """Stub: segment the first frame with SAM 2 (Month 2 integration)."""
    bound_log = log.bind(
        job_id=job_id,
        segment_s3_key=segment_s3_key,
        task_id=self.request.id,
    )
    bound_log.info("segmentation_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "segmenting", "pct": 0})
        )

        SEGMENTS_TOTAL.labels(status="segmentation_started").inc()

        loop.run_until_complete(
            _publish(job_id, {"stage": "segmenting", "pct": 100})
        )

        return {
            "job_id": job_id,
            "segment_s3_key": segment_s3_key,
            "mask_s3_key": f"jobs/{job_id}/masks/mask_stub.npy",
            "status": "stub",
        }

    except SoftTimeLimitExceeded:
        bound_log.error("segmentation_soft_timeout")
        SEGMENTS_TOTAL.labels(status="segmentation_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "segment_first_frame timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("segmentation_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="segmentation_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    queue="segmentation",
    max_retries=2,
    soft_time_limit=300,
    time_limit=360,
    name="workers.segmentation.tasks.track_masks",
)
def track_masks(
    self,
    job_id: str,
    segment_s3_key: str,
    mask_s3_key: str,
) -> dict[str, Any]:
    """Stub: propagate masks across frames with XMem++ (Month 2 integration)."""
    bound_log = log.bind(
        job_id=job_id,
        task_id=self.request.id,
    )
    bound_log.info("tracking_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "tracking", "pct": 0})
        )

        SEGMENTS_TOTAL.labels(status="tracking_started").inc()

        loop.run_until_complete(
            _publish(job_id, {"stage": "tracking", "pct": 100})
        )

        return {
            "job_id": job_id,
            "segment_s3_key": segment_s3_key,
            "tracked_masks_s3_key": f"jobs/{job_id}/masks/tracked_stub.npy",
            "status": "stub",
        }

    except SoftTimeLimitExceeded:
        bound_log.error("tracking_soft_timeout")
        SEGMENTS_TOTAL.labels(status="tracking_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "track_masks timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("tracking_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="tracking_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _publish(job_id: str, data: dict) -> None:
    async with acquire_redis() as redis:
        await publish_progress(redis, job_id, data)


async def _update_status(job_id: str, status: str, error: str) -> None:
    async with acquire_redis() as redis:
        await set_job_status(redis, job_id, {"status": status, "error": error})
