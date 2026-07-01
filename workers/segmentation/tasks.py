from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import numpy as np
import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.config import get_settings
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
    """Segment the target frame with SAM 2 given user-clicked points."""
    settings = get_settings()
    bound_log = log.bind(
        job_id=job_id,
        segment_s3_key=segment_s3_key,
        task_id=self.request.id,
    )
    bound_log.info("segmentation_started")

    points = mask_data.get("points") or []
    frame_index = mask_data.get("frame_index", 0)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "segmenting", "pct": 0})
        )
        SEGMENTS_TOTAL.labels(status="segmentation_started").inc()

        s3 = _make_s3_client(settings)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "segment.mp4")
            s3.download_file(settings.s3_bucket, segment_s3_key, video_path)

            frame_rgb = _extract_frame(video_path, frame_index)

            loop.run_until_complete(
                _publish(job_id, {"stage": "segmenting", "pct": 50})
            )

            if not points:
                bound_log.warning("segmentation_no_points")
                mask_array = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
                status = "empty"
            else:
                best_mask = _run_sam2_segmentation(frame_rgb, points, settings, bound_log)
                mask_array = (best_mask.astype(np.uint8) * 255)
                status = "real"

            mask_s3_key = f"jobs/{job_id}/masks/frame_{frame_index}_mask.npy"
            mask_path = os.path.join(tmp, "mask.npy")
            np.save(mask_path, mask_array)
            s3.upload_file(mask_path, settings.s3_bucket, mask_s3_key)

        loop.run_until_complete(
            _publish(job_id, {"stage": "segmenting", "pct": 100})
        )
        SEGMENTS_TOTAL.labels(status=f"segmentation_{status}").inc()

        return {
            "job_id": job_id,
            "segment_s3_key": segment_s3_key,
            "mask_s3_key": mask_s3_key,
            "status": status,
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


def _extract_frame(video_path: str, frame_index: int) -> np.ndarray:
    """Seek to frame_index and return it as an (H, W, 3) uint8 RGB array."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _load_sam2_model(settings: Any) -> Any:
    """Build the SAM2 model from the checkpoint in settings.model_cache_dir."""
    from sam2.build_sam import build_sam2

    checkpoint_path = os.path.join(settings.model_cache_dir, "sam2_hiera_small.pt")
    return build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint_path,
        device=settings.device,
    )


def _run_sam2_segmentation(
    frame_rgb: np.ndarray,
    points: list[dict],
    settings: Any,
    bound_log: Any,
) -> np.ndarray:
    """Run SAM2 point-based segmentation on a single frame, via the model pool."""
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Rule: always load models through the model pool
    from pipeline.pool.model_pool import get_model_pool

    pool = get_model_pool()
    pool.register_loader("sam2", lambda: _load_sam2_model(settings))

    with pool.acquire("sam2") as sam2_model:
        bound_log.info("sam2_model_acquired")
        predictor = SAM2ImagePredictor(sam2_model)
        with torch.no_grad():
            predictor.set_image(frame_rgb)
            point_coords = np.array([[p["x"], p["y"]] for p in points])
            point_labels = np.ones(len(point_coords), dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )

    bound_log.info("sam2_inference_complete")
    return masks[0]


async def _publish(job_id: str, data: dict) -> None:
    async with acquire_redis() as redis:
        await publish_progress(redis, job_id, data)


async def _update_status(job_id: str, status: str, error: str) -> None:
    async with acquire_redis() as redis:
        await set_job_status(redis, job_id, {"status": status, "error": error})
