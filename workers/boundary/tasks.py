from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import Any

import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.config import get_settings
from api.core.metrics import BOUNDARY_FUSION_TIME, SEGMENTS_TOTAL
from workers.celery_app import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(
    bind=True,
    queue="boundary",
    max_retries=2,
    soft_time_limit=300,
    time_limit=360,
    name="workers.boundary.tasks.apply_boundary_fusion",
)
def apply_boundary_fusion(
    self,
    job_id: str,
    stitched_s3_key: str,
) -> dict[str, Any]:
    """Correct temporal discontinuities at every chunk boundary of the stitched video.

    Downloads all inpainted chunks, runs BoundaryFusion on each adjacent pair's
    boundary frames, uploads the corrected full video, then chains to
    quality_check_final. If only one chunk exists there are no boundaries to
    fix; if fusion fails, the stitched video is passed through unchanged.
    """
    settings = get_settings()
    bound_log = log.bind(job_id=job_id, task_id=self.request.id)
    bound_log.info("boundary_fusion_started", stitched_s3_key=stitched_s3_key)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "boundary_fusion", "pct": 0})
        )
        SEGMENTS_TOTAL.labels(status="boundary_fusion_started").inc()

        s3 = _make_s3_client(settings)
        result_s3_key = stitched_s3_key
        status = "boundary_fusion_applied"

        try:
            chunk_keys = _list_chunk_keys(s3, settings.s3_bucket, job_id)
            bound_log.info("boundary_chunks_found", num_chunks=len(chunk_keys))

            if len(chunk_keys) < 2:
                # Single chunk (or none) → no boundaries to correct
                status = "skipped_no_boundaries"
                SEGMENTS_TOTAL.labels(status="boundary_fusion_skipped").inc()
                bound_log.info("boundary_fusion_skipped_single_chunk")
            else:
                t0 = time.perf_counter()
                with tempfile.TemporaryDirectory() as tmp:
                    chunk_paths = []
                    for i, chunk_key in enumerate(chunk_keys):
                        chunk_path = os.path.join(tmp, f"chunk_{i:04d}.mp4")
                        s3.download_file(settings.s3_bucket, chunk_key, chunk_path)
                        chunk_paths.append(chunk_path)

                    output_path = os.path.join(tmp, "output.mp4")
                    _run_fusion_on_chunks(chunk_paths, output_path, settings, bound_log)

                    result_s3_key = f"jobs/{job_id}/result/output.mp4"
                    s3.upload_file(output_path, settings.s3_bucket, result_s3_key)

                elapsed = time.perf_counter() - t0
                BOUNDARY_FUSION_TIME.observe(elapsed)
                SEGMENTS_TOTAL.labels(status="boundary_fusion_complete").inc()
                bound_log.info(
                    "boundary_fusion_complete",
                    elapsed_sec=round(elapsed, 3),
                    result_s3_key=result_s3_key,
                    num_boundaries=len(chunk_keys) - 1,
                )
        except SoftTimeLimitExceeded:
            raise
        except Exception as exc:
            # Graceful degradation: keep the uncorrected stitched video
            bound_log.exception("boundary_fusion_failed_falling_back", error=str(exc))
            SEGMENTS_TOTAL.labels(status="boundary_fusion_fallback").inc()
            result_s3_key = stitched_s3_key
            status = "boundary_fusion_failed_passthrough"

        loop.run_until_complete(
            _publish(job_id, {"stage": "boundary_fusion", "pct": 100})
        )

        from workers.quality.tasks import quality_check_final
        quality_check_final.apply_async(
            args=[job_id, result_s3_key],
            queue="quality",
        )

        return {
            "job_id": job_id,
            "result_s3_key": result_s3_key,
            "status": status,
        }

    except SoftTimeLimitExceeded:
        bound_log.error("boundary_fusion_soft_timeout")
        SEGMENTS_TOTAL.labels(status="boundary_fusion_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "apply_boundary_fusion timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("boundary_fusion_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="boundary_fusion_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_fusion_on_chunks(
    chunk_paths: list[str],
    output_path: str,
    settings: Any,
    bound_log: Any,
) -> None:
    """Acquire BoundaryFusion from the model pool and fuse all chunk boundaries."""
    from visionerase_models.boundary_fusion.inference import (
        WEIGHTS_FILENAME,
        load_boundary_fusion_model,
        run_boundary_fusion_on_chunks,
    )

    # Rule: always load models through the model pool
    from pipeline.pool.model_pool import get_model_pool

    weights_path = os.path.join(settings.model_cache_dir, WEIGHTS_FILENAME)
    pool = get_model_pool()
    # Small model (890K params) — CPU inference is fast, no VRAM needed
    pool.register_loader(
        "boundary_fusion",
        lambda: load_boundary_fusion_model(weights_path, device="cpu"),
    )
    with pool.acquire("boundary_fusion") as model:
        bound_log.info("boundary_fusion_model_acquired")
        run_boundary_fusion_on_chunks(chunk_paths, output_path, model=model)


def _list_chunk_keys(s3: Any, bucket: str, job_id: str) -> list[str]:
    """Return all inpainted chunk keys for a job, ordered by chunk index."""
    prefix = f"jobs/{job_id}/chunks/"
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/inpainted_chunk.mp4"):
                keys.append(key)
    return sorted(keys, key=lambda k: int(k[len(prefix):].split("/")[0]))


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
