from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from typing import Any

import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.config import get_settings
from api.core.metrics import SEGMENTS_TOTAL
from api.core.redis import acquire_redis, publish_progress, set_job_status
from api.models.job import JobStatus
from workers.celery_app import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(
    bind=True,
    queue="stitching",
    max_retries=2,
    soft_time_limit=300,
    time_limit=360,
    name="workers.stitching.tasks.stitch_segments",
)
def stitch_segments(
    self,
    job_id: str,
    inpainted_s3_keys: list[str],
) -> dict[str, Any]:
    """Stub: combine inpainted segments into final video (Month 2 integration)."""
    bound_log = log.bind(job_id=job_id, task_id=self.request.id)
    bound_log.info("stitching_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "stitching", "pct": 0})
        )
        loop.run_until_complete(
            _update_status(job_id, JobStatus.STITCHING, None)
        )
        SEGMENTS_TOTAL.labels(status="stitching_started").inc()

        # Pass through the real inpainted video from ProPainter
        result_s3_key = inpainted_s3_keys[0] if inpainted_s3_keys else f"jobs/{job_id}/result/output_stub.mp4"

        loop.run_until_complete(
            _publish(job_id, {"stage": "stitching", "pct": 100})
        )
        SEGMENTS_TOTAL.labels(status="stitching_complete").inc()

        from workers.quality.tasks import quality_check_final
        quality_check_final.apply_async(
            args=[job_id, result_s3_key],
            queue="quality",
        )

        return {
            "job_id": job_id,
            "result_s3_key": result_s3_key,
            "status": "stub",
        }

    except SoftTimeLimitExceeded:
        bound_log.error("stitching_soft_timeout")
        SEGMENTS_TOTAL.labels(status="stitching_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "stitch_segments timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("stitching_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="stitching_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    queue="stitching",
    max_retries=2,
    soft_time_limit=300,
    time_limit=360,
    name="workers.stitching.tasks.stitch_all_chunks",
)
def stitch_all_chunks(
    self,
    job_id: str,
    total_chunks: int,
) -> dict[str, Any]:
    """Concatenate every inpainted chunk into the final result video via ffmpeg."""
    settings = get_settings()
    bound_log = log.bind(job_id=job_id, total_chunks=total_chunks, task_id=self.request.id)
    bound_log.info("chunk_stitching_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "stitching", "pct": 0})
        )
        loop.run_until_complete(
            _update_status(job_id, JobStatus.STITCHING, None)
        )
        SEGMENTS_TOTAL.labels(status="chunk_stitching_started").inc()

        s3 = _make_s3_client(settings)

        with tempfile.TemporaryDirectory() as tmp:
            concat_list_path = os.path.join(tmp, "concat_list.txt")
            with open(concat_list_path, "w") as concat_list:
                for i in range(total_chunks):
                    chunk_key = f"jobs/{job_id}/chunks/{i}/inpainted_chunk.mp4"
                    chunk_path = os.path.join(tmp, f"chunk_{i:04d}.mp4")
                    s3.download_file(settings.s3_bucket, chunk_key, chunk_path)
                    concat_list.write(f"file '{chunk_path}'\n")

            output_path = os.path.join(tmp, "output.mp4")
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_list_path,
                    "-c", "copy",
                    output_path,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

            result_s3_key = f"jobs/{job_id}/result/output.mp4"
            s3.upload_file(output_path, settings.s3_bucket, result_s3_key)

        loop.run_until_complete(
            _publish(job_id, {"stage": "stitching", "pct": 100})
        )
        SEGMENTS_TOTAL.labels(status="chunk_stitching_complete").inc()

        from workers.boundary.tasks import apply_boundary_fusion
        apply_boundary_fusion.apply_async(
            args=[job_id, result_s3_key],
            queue="boundary",
        )

        return {
            "job_id": job_id,
            "result_s3_key": result_s3_key,
            "total_chunks": total_chunks,
            "status": "real",
        }

    except SoftTimeLimitExceeded:
        bound_log.error("chunk_stitching_soft_timeout")
        SEGMENTS_TOTAL.labels(status="chunk_stitching_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "stitch_all_chunks timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("chunk_stitching_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="chunk_stitching_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_s3_client(settings: Any) -> Any:
    """Return a boto3 S3 client configured for the MinIO endpoint."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
    )


async def _publish(job_id: str, data: dict) -> None:
    from redis import asyncio as _aioredis
    import json as _json
    from api.core.config import get_settings as _get_settings
    _s = _get_settings()
    _r = _aioredis.Redis.from_url(_s.redis_url, decode_responses=True)
    try:
        channel = f"progress:{job_id}"
        message = _json.dumps({"type": "progress", **data})
        # Persist the latest progress event so REST polling and WebSocket
        # snapshots can report it (read back via api.core.redis.get_job_progress)
        await _r.setex(f"job:progress:{job_id}", _s.redis_result_ttl, message)
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
