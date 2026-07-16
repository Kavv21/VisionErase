from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import numpy as np
import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.metrics import (
    QUALITY_PSNR_DB,
    QUALITY_SSIM_SCORE,
    QUALITY_TEMPORAL_CONSISTENCY,
    SEGMENTS_TOTAL,
)
from api.models.job import JobStatus
from workers.celery_app import celery_app

log = structlog.get_logger(__name__)

MAX_METRIC_FRAMES = 30   # sample at most this many frame pairs for SSIM/PSNR

# Quality-warning gates (poor quality thresholds)
SSIM_WARN_THRESHOLD = 0.5
PSNR_WARN_THRESHOLD = 20.0


@celery_app.task(
    bind=True,
    queue="quality",
    max_retries=2,
    soft_time_limit=300,
    time_limit=330,
    name="workers.quality.tasks.quality_check_final",
)
def quality_check_final(
    self,
    job_id: str,
    result_s3_key: str,
) -> dict[str, Any]:
    """Score the result video (SSIM/PSNR vs input, temporal consistency) and complete the job."""
    from api.core.config import get_settings
    from workers.db import get_job_video_s3_key, mark_job_completed

    settings = get_settings()
    bound_log = log.bind(job_id=job_id, task_id=self.request.id)
    bound_log.info("quality_check_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "quality_check", "pct": 0})
        )
        loop.run_until_complete(
            _update_status(job_id, JobStatus.QUALITY_CHECK, None)
        )
        SEGMENTS_TOTAL.labels(status="quality_check_started").inc()

        s3 = _make_s3_client(settings)
        video_s3_key = get_job_video_s3_key(job_id)

        metrics: dict[str, Any] = {}
        quality_warning = False
        with tempfile.TemporaryDirectory() as tmp:
            result_path = os.path.join(tmp, "result.mp4")
            s3.download_file(settings.s3_bucket, result_s3_key, result_path)
            result_frames = _read_frames_sampled(result_path, MAX_METRIC_FRAMES)
            if not result_frames:
                raise RuntimeError(f"no frames decoded from result {result_s3_key}")

            loop.run_until_complete(
                _publish(job_id, {"stage": "quality_check", "pct": 40})
            )

            metrics["temporal_consistency"] = _temporal_consistency(result_frames)
            QUALITY_TEMPORAL_CONSISTENCY.observe(metrics["temporal_consistency"])

            if video_s3_key:
                original_path = os.path.join(tmp, "original.mp4")
                s3.download_file(settings.s3_bucket, video_s3_key, original_path)
                original_frames = _read_frames_sampled(original_path, MAX_METRIC_FRAMES)

                mean_ssim, mean_psnr = _ssim_psnr(original_frames, result_frames)
                if mean_ssim is not None and mean_psnr is not None:
                    metrics["ssim"] = mean_ssim
                    metrics["psnr"] = mean_psnr
                    QUALITY_SSIM_SCORE.observe(mean_ssim)
                    QUALITY_PSNR_DB.observe(mean_psnr)
                    quality_warning = (
                        mean_ssim < SSIM_WARN_THRESHOLD or mean_psnr < PSNR_WARN_THRESHOLD
                    )
            else:
                bound_log.warning("quality_no_input_video_key", job_id=job_id)

        bound_log.info(
            "quality_metrics",
            ssim=metrics.get("ssim"),
            psnr=metrics.get("psnr"),
            temporal_consistency=metrics.get("temporal_consistency"),
            quality_warning=quality_warning,
            job_id=job_id,
        )

        loop.run_until_complete(
            _publish(job_id, {"stage": "quality_check", "pct": 100})
        )

        loop.run_until_complete(_complete_job(job_id, result_s3_key))
        mark_job_completed(
            job_id,
            result_s3_key,
            quality_metrics=metrics or None,
            status="quality_warning" if quality_warning else "completed",
        )
        SEGMENTS_TOTAL.labels(status="quality_check_complete").inc()
        bound_log.info("job_completed", result_s3_key=result_s3_key)

        return {
            "job_id": job_id,
            "result_s3_key": result_s3_key,
            "status": "quality_warning" if quality_warning else "completed",
            "quality_metrics": metrics,
        }

    except SoftTimeLimitExceeded:
        bound_log.error("quality_check_soft_timeout")
        SEGMENTS_TOTAL.labels(status="quality_check_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "quality_check_final timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("quality_check_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="quality_check_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Metric helpers ────────────────────────────────────────────────────────────

def _read_frames_sampled(video_path: str, max_frames: int) -> list[np.ndarray]:
    """Read up to max_frames evenly spaced RGB frames from a video."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.unique(np.linspace(0, total - 1, min(max_frames, total)).astype(int))
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _ssim_psnr(
    original_frames: list[np.ndarray],
    result_frames: list[np.ndarray],
) -> tuple[float | None, float | None]:
    """Return (mean SSIM, mean PSNR) over matching frame pairs.

    The result video can be at a different resolution (tracking/inpainting
    resize), so original frames are resized to the result's shape first.
    """
    import cv2
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    n = min(len(original_frames), len(result_frames))
    if n == 0:
        return None, None

    ssim_scores: list[float] = []
    psnr_scores: list[float] = []
    for orig, res in zip(original_frames[:n], result_frames[:n]):
        if orig.shape != res.shape:
            orig = cv2.resize(orig, (res.shape[1], res.shape[0]))
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        res_gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        ssim_scores.append(
            float(structural_similarity(orig_gray, res_gray, data_range=255))
        )
        # Identical frames give infinite PSNR — cap so the mean stays finite.
        if np.array_equal(orig_gray, res_gray):
            psnr_scores.append(100.0)
        else:
            psnr_scores.append(
                float(peak_signal_noise_ratio(orig_gray, res_gray, data_range=255))
            )

    return float(np.mean(ssim_scores)), float(np.mean(psnr_scores))


def _temporal_consistency(frames: list[np.ndarray]) -> float:
    """Mean cosine similarity between consecutive frames — measures smoothness."""
    if len(frames) < 2:
        return 1.0

    similarities: list[float] = []
    for prev, cur in zip(frames[:-1], frames[1:]):
        a = prev.astype(np.float32).ravel()
        b = cur.astype(np.float32).ravel()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            similarities.append(1.0)
        else:
            similarities.append(float(np.dot(a, b) / denom))
    return float(np.mean(similarities))


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


async def _complete_job(job_id: str, result_s3_key: str) -> None:
    import json as _json
    from redis import asyncio as _aioredis
    from api.core.config import get_settings as _get_settings
    _s = _get_settings()
    _r = _aioredis.Redis.from_url(_s.redis_url, decode_responses=True)
    try:
        key = f"job:status:{job_id}"
        status = {"status": JobStatus.COMPLETED, "result_s3_key": result_s3_key}
        await _r.setex(key, _s.redis_result_ttl, _json.dumps(status))
        channel = f"progress:{job_id}"
        message = _json.dumps({
            "type": "progress",
            "stage": "completed",
            "pct": 100,
            "result_s3_key": result_s3_key,
        })
        await _r.publish(channel, message)
    finally:
        await _r.aclose()
