from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any

import numpy as np
import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.config import get_settings
from api.core.metrics import BOUNDARY_FUSION_TIME, BOUNDARY_QUALITY_SCORE, SEGMENTS_TOTAL
from api.core.redis import acquire_redis, publish_progress, set_job_status

log = structlog.get_logger(__name__)

# Import celery_app lazily to avoid circular imports at module load time.
# tasks.py is collected by the worker process which already has the app configured.
from workers.celery_app import celery_app  # noqa: E402


@celery_app.task(
    bind=True,
    queue="boundary",
    max_retries=2,
    soft_time_limit=120,
    time_limit=150,
    name="workers.boundary.tasks.apply_boundary_fusion",
)
def apply_boundary_fusion(
    self,
    job_id: str,
    segment_a_s3_key: str,
    segment_b_s3_key: str,
    segment_index: int,
) -> dict[str, Any]:
    """Correct temporal discontinuities at the boundary between two adjacent segments.

    Downloads the tail of segment A and the head of segment B, runs BoundaryFusion
    cross-segment attention, and uploads the corrected boundary frames to S3.
    Returns a dict with corrected_frames_s3_key.
    """
    import asyncio

    import boto3
    from botocore.config import Config

    settings = get_settings()
    bound_log = log.bind(
        job_id=job_id,
        segment_index=segment_index,
        task_id=self.request.id,
    )
    bound_log.info("boundary_fusion_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "boundary_fusion", "segment": segment_index, "pct": 0})
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )

        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmp:
            seg_a_path = os.path.join(tmp, "seg_a.mp4")
            seg_b_path = os.path.join(tmp, "seg_b.mp4")

            s3.download_file(settings.s3_bucket, segment_a_s3_key, seg_a_path)
            s3.download_file(settings.s3_bucket, segment_b_s3_key, seg_b_path)

            frames_a = _read_tail_frames(seg_a_path, settings.boundary_window_frames)
            frames_b = _read_head_frames(seg_b_path, settings.boundary_window_frames)

            if settings.boundary_fusion_enabled:
                corrected = _run_fusion_model(frames_a, frames_b, settings, bound_log)
                method = "boundary_fusion"
            else:
                corrected = _overlap_blend(frames_a, frames_b)
                method = "overlap_blend"
                bound_log.info("boundary_fusion_disabled_using_blend")

            dest_key = f"jobs/{job_id}/boundaries/boundary_{segment_index}.npy"
            npy_path = os.path.join(tmp, "boundary.npy")
            np.save(npy_path, corrected)
            s3.upload_file(npy_path, settings.s3_bucket, dest_key)

        elapsed = time.perf_counter() - t0
        BOUNDARY_FUSION_TIME.observe(elapsed)
        SEGMENTS_TOTAL.labels(status="boundary_corrected").inc()

        # Score boundary quality (mean SSIM proxy: pixel variance ratio)
        quality_score = float(np.clip(1.0 - np.std(corrected.astype(float) / 255.0), 0.0, 1.0))
        BOUNDARY_QUALITY_SCORE.observe(quality_score)

        loop.run_until_complete(
            _publish(job_id, {"stage": "boundary_fusion", "segment": segment_index, "pct": 100})
        )

        bound_log.info(
            "boundary_fusion_complete",
            method=method,
            elapsed_sec=round(elapsed, 3),
            dest_key=dest_key,
            quality_score=round(quality_score, 4),
        )
        return {"corrected_frames_s3_key": dest_key, "method": method}

    except SoftTimeLimitExceeded:
        bound_log.error("boundary_fusion_soft_timeout")
        SEGMENTS_TOTAL.labels(status="boundary_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "BoundaryFusion timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("boundary_fusion_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="boundary_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_tail_frames(video_path: str, n: int) -> np.ndarray:
    """Return the last n frames of a video as (n, H, W, 3) uint8 array."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, total - n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def _read_head_frames(video_path: str, n: int) -> np.ndarray:
    """Return the first n frames of a video as (n, H, W, 3) uint8 array."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def _run_fusion_model(
    frames_a: np.ndarray,
    frames_b: np.ndarray,
    settings: Any,
    bound_log: Any,
) -> np.ndarray:
    """Load BoundaryFusion from the model pool and run inference."""
    from models.boundary_fusion.inference import run_boundary_fusion

    # Rule: always load models through the model pool
    from pipeline.pool.model_pool import get_model_pool

    pool = get_model_pool()
    with pool.acquire("boundary_fusion", settings.boundary_fusion_weights) as model:
        bound_log.info("boundary_fusion_model_acquired")
        return run_boundary_fusion(model, frames_a, frames_b)


def _overlap_blend(frames_a: np.ndarray, frames_b: np.ndarray) -> np.ndarray:
    """Fallback: linearly blend the tail of A into the head of B."""
    n = len(frames_a)
    blended = []
    for i in range(n):
        alpha = i / max(n - 1, 1)
        frame = ((1 - alpha) * frames_a[i] + alpha * frames_b[i]).astype(np.uint8)
        blended.append(frame)
    # Append segment B frames unchanged after the blend window
    return np.concatenate([np.stack(blended), frames_b], axis=0)


async def _publish(job_id: str, data: dict) -> None:
    async with acquire_redis() as redis:
        await publish_progress(redis, job_id, data)


async def _update_status(job_id: str, status: str, error: str) -> None:
    async with acquire_redis() as redis:
        await set_job_status(redis, job_id, {"status": status, "error": error})
