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
from api.models.job import JobStatus

log = structlog.get_logger(__name__)

MAX_TRACKING_FRAMES = 60    # process 60 frames per chunk to avoid RAM OOM
TRACKING_WIDTH      = 854   # resize to 480p
TRACKING_HEIGHT     = 480

# Chunks overlap so the stitcher can cross-fade the seam instead of hard-cutting.
# CHUNK_SIZE is capped by RAFT's correlation volume inside ProPainter, which is
# allocated over the whole chunk as (T-1) x (H*W/64)^2 x 4 bytes. Measured on
# the A10G at 854x480: T=80 -> 11.5GB OOM, T=50 -> 7.1GB OOM, T=49 -> only just
# fits. 40 leaves real headroom (see pipeline/modal_propainter.py).
# OVERLAP must stay < STRIDE, otherwise the final chunk can be shorter than the
# blend region and the stitcher cannot cross-fade it.
CHUNK_SIZE = 40                    # larger chunk = better ProPainter temporal context
OVERLAP    = 10                    # frames shared between adjacent chunks (25%, as at 80/20)
STRIDE     = CHUNK_SIZE - OVERLAP  # = 30 frames advance per chunk

OVERLAP_KEY_TTL = 86400  # 24h — must outlive inpainting so stitching can read it


# Import celery_app lazily to avoid circular imports at module load time.
from workers.celery_app import celery_app  # noqa: E402


def detect_and_add_shadow(frame_rgb, object_mask, dilation_px=20):
    """
    Expand mask to include shadow region near the object.
    Uses two approaches:
    1. Simple dilation (catches nearby shadows)
    2. HSV darkness detection connected to object
    """
    import cv2

    H, W = frame_rgb.shape[:2]

    # Method 1: Dilate mask to catch nearby shadow
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    dilated = cv2.dilate(object_mask, kernel, iterations=1)

    # Method 2: Find dark regions connected to object
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    V = hsv[:, :, 2]  # brightness channel

    # Estimate background brightness (median outside mask). A mask covering the
    # whole frame leaves nothing to sample — np.median would return nan and taint
    # the threshold, so fall back to dilation-only in that case.
    bg_pixels = V[object_mask == 0]
    bg_brightness = float(np.median(bg_pixels)) if bg_pixels.size else 0.0

    # Shadow = darker than background by >20%
    shadow_threshold = bg_brightness * 0.80
    dark_regions = (V < shadow_threshold).astype(np.uint8) * 255

    # Shadow must be connected to the dilated object mask
    combined = cv2.bitwise_or(dark_regions, dilated)

    # Find connected components containing the object
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)

    # Keep components that overlap with original object mask
    final_mask = np.zeros_like(object_mask)
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        overlap = cv2.bitwise_and(
            component,
            (object_mask > 0).astype(np.uint8),
        )
        if overlap.sum() > 0:
            final_mask = cv2.bitwise_or(final_mask, component * 255)

    # Safety cap: reject expansion if > 4x original area
    # Prevents swallowing large dark surfaces (asphalt, shaded floor)
    original_area = float((object_mask > 0).sum())
    expanded_area = float((final_mask > 0).sum())
    if original_area > 0 and expanded_area > original_area * 4.0:
        import logging
        logging.warning(
            f"shadow_expansion_capped: {expanded_area/original_area:.1f}x "
            f"exceeds 4x limit, using dilation-only"
        )
        # Fall back to simple dilation only (no dark region detection)
        final_mask = dilated.copy()

    return final_mask.astype(np.uint8)


@celery_app.task(
    bind=True,
    queue="segmentation",
    max_retries=2,
    # Generous limits: this task now also runs SAM2 VP tracking over the full
    # video (~30s per 360 frames) plus CPU OIV refinement.
    soft_time_limit=900,
    time_limit=1020,
    name="workers.segmentation.tasks.segment_first_frame",
)
def segment_first_frame(
    self,
    job_id: str,
    segment_s3_key: str,
    mask_data: dict[str, Any],
    include_shadow: bool = True,
) -> dict[str, Any]:
    """Segment the target frame with SAM 2 given user-clicked points."""
    settings = get_settings()
    bound_log = log.bind(
        job_id=job_id,
        segment_s3_key=segment_s3_key,
        task_id=self.request.id,
    )
    bound_log.info("segmentation_started")

    from workers.db import mark_job_processing
    mark_job_processing(job_id)

    points = mask_data.get("points") or []
    frame_index = mask_data.get("frame_index", 0)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "segmenting", "pct": 0})
        )
        loop.run_until_complete(
            _update_status(job_id, JobStatus.SEGMENTING, None)
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
                # frame_rgb comes back from _extract_frame already RGB-converted.
                mask_array = (best_mask.astype(np.uint8) * 255)
                if include_shadow:
                    # Auto-detect and include shadow region
                    original_coverage = float((mask_array > 0).mean() * 100)
                    mask_array = detect_and_add_shadow(
                        frame_rgb=frame_rgb,
                        object_mask=mask_array,
                        dilation_px=20,
                    )
                    expanded_coverage = float((mask_array > 0).mean() * 100)
                    bound_log.info(
                        "shadow_detection_complete",
                        original_coverage_pct=round(original_coverage, 2),
                        expanded_coverage_pct=round(expanded_coverage, 2),
                        shadow_added_pct=round(expanded_coverage - original_coverage, 2),
                    )
                else:
                    bound_log.info("shadow_detection_skipped")
                status = "real"

            mask_s3_key = f"jobs/{job_id}/masks/frame_{frame_index}_mask.npy"
            mask_path = os.path.join(tmp, "mask.npy")
            np.save(mask_path, mask_array)
            s3.upload_file(mask_path, settings.s3_bucket, mask_s3_key)

            total_frames, _fps = _get_frame_count_and_fps(video_path)

            loop.run_until_complete(
                _publish(job_id, {"stage": "segmenting", "pct": 100})
            )
            SEGMENTS_TOTAL.labels(status=f"segmentation_{status}").inc()

            # Evict SAM2 image predictor from GPU before SAM2 VP tracking
            try:
                from pipeline.pool.model_pool import get_model_pool
                get_model_pool().evict("sam2")
                import torch as _torch
                _torch.cuda.empty_cache()
                bound_log.info("sam2_image_predictor_evicted")
            except Exception as _e:
                bound_log.warning("sam2_evict_failed", error=str(_e))

            # Track ALL frames in one pass with SAM2 Video Predictor on the
            # local GPU — replaces the sequential per-chunk XMem++ Modal calls.
            from workers.segmentation.sam2_video_tracker import (
                track_with_sam2_video_predictor,
            )

            loop.run_until_complete(
                _update_status(job_id, JobStatus.TRACKING, None)
            )
            loop.run_until_complete(
                _publish(job_id, {"stage": "tracking", "pct": 0})
            )
            SEGMENTS_TOTAL.labels(status="sam2_vp_tracking_started").inc()
            bound_log.info("sam2_vp_tracking_started", total_frames=total_frames)

            def _tracking_progress(pct: int) -> None:
                loop.run_until_complete(
                    _publish(job_id, {"stage": "tracking", "pct": pct})
                )

            tracked_masks = track_with_sam2_video_predictor(
                video_path=video_path,
                first_frame_mask=mask_array,
                model_cache_dir=settings.model_cache_dir,
                device=settings.device,
                # on_progress removed — subprocess wrapper handles progress
            )
            # Trust the decoded frame count over container metadata so chunk
            # ranges always line up with the mask array.
            total_frames = len(tracked_masks)

            masks_key = f"jobs/{job_id}/masks/tracked_masks.npy"
            tracked_path = os.path.join(tmp, "tracked_masks.npy")
            np.save(tracked_path, tracked_masks)
            s3.upload_file(tracked_path, settings.s3_bucket, masks_key)

            loop.run_until_complete(
                _publish(job_id, {"stage": "tracking", "pct": 100})
            )
            SEGMENTS_TOTAL.labels(status="sam2_vp_tracking_complete").inc()
            bound_log.info(
                "sam2_vp_tracking_complete",
                total_frames=total_frames,
                masks_key=masks_key,
            )

        chunks = []
        start = 0
        while start < total_frames:
            end = min(start + CHUNK_SIZE, total_frames)
            chunks.append((start, end))
            if end >= total_frames:
                break
            start += STRIDE
        total_chunks = len(chunks)

        # The stitcher needs OVERLAP to know how many frames to cross-fade, and
        # total_frames to verify it reassembled the video without gaps.
        # A single chunk has no seam, so publish overlap=0 and let the stitcher
        # take its plain-concat path.
        effective_overlap = OVERLAP if total_chunks > 1 else 0
        loop.run_until_complete(
            _store_stitch_params(job_id, effective_overlap, total_frames)
        )

        bound_log.info(
            "chunking_video",
            total_frames=total_frames,
            total_chunks=total_chunks,
            chunk_size=CHUNK_SIZE,
            overlap=effective_overlap,
            stride=STRIDE,
        )
        SEGMENTS_TOTAL.labels(status="chunking_dispatched").inc()

        from workers.db import set_job_total_chunks
        set_job_total_chunks(job_id, total_chunks)

        from workers.inpainting.chunk_tasks import inpaint_chunk
        for chunk_index, (start_frame, end_frame) in enumerate(chunks):
            inpaint_chunk.apply_async(
                args=[
                    job_id,
                    chunk_index,
                    total_chunks,
                    segment_s3_key,
                    masks_key,  # full tracked masks; each chunk slices its range
                    start_frame,
                    end_frame,
                ],
                queue="inpainting",
            )

        loop.run_until_complete(
            _update_status(
                job_id, JobStatus.INPAINTING, extra={"total_chunks": total_chunks}
            )
        )

        return {
            "job_id": job_id,
            "segment_s3_key": segment_s3_key,
            "mask_s3_key": mask_s3_key,
            "tracked_masks_s3_key": masks_key,
            "total_chunks": total_chunks,
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
    soft_time_limit=600,
    time_limit=720,
    name="workers.segmentation.tasks.track_masks",
)
def track_masks(
    self,
    job_id: str,
    segment_s3_key: str,
    mask_s3_key: str,
) -> dict[str, Any]:
    """Propagate the SAM2 first-frame mask across all frames with XMem.

    DEPRECATED: no longer dispatched — segment_first_frame now tracks the full
    video with SAM2 Video Predictor (see sam2_video_tracker.py). Kept for
    reference only.
    """
    settings = get_settings()
    bound_log = log.bind(
        job_id=job_id,
        segment_s3_key=segment_s3_key,
        task_id=self.request.id,
    )
    bound_log.info("tracking_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "tracking", "pct": 0})
        )
        loop.run_until_complete(
            _update_status(job_id, JobStatus.TRACKING, None)
        )
        SEGMENTS_TOTAL.labels(status="tracking_started").inc()

        s3 = _make_s3_client(settings)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "segment.mp4")
            mask_path = os.path.join(tmp, "mask.npy")
            s3.download_file(settings.s3_bucket, segment_s3_key, video_path)
            s3.download_file(settings.s3_bucket, mask_s3_key, mask_path)

            mask = np.load(mask_path)  # (H, W), values 0 or 255
            mask_binary = (mask > 127).astype(np.uint8)

            frames = _extract_all_frames(video_path)
            if not frames:
                raise RuntimeError(f"no frames decoded from {segment_s3_key}")
            bound_log.info("frames_extracted", num_frames=len(frames))

            # 50%: video + mask loaded, model about to run
            loop.run_until_complete(
                _publish(job_id, {"stage": "tracking", "pct": 50})
            )

            tracked_array = _run_xmem_tracking(frames, mask_binary, settings, bound_log)

            from workers.segmentation.oiv_refiner import refine_tracked_masks

            bound_log.info("oiv_refinement_started", num_frames=len(frames))
            tracked_array = refine_tracked_masks(
                frames=frames,
                tracked_masks=tracked_array,
                model_cache_dir=settings.model_cache_dir,
                device="cpu",  # CPU to avoid VRAM conflict with SAM2
            )
            bound_log.info("oiv_refinement_complete")

            tracked_key = f"jobs/{job_id}/masks/tracked_masks.npy"
            tracked_path = os.path.join(tmp, "tracked_masks.npy")
            np.save(tracked_path, tracked_array)
            s3.upload_file(tracked_path, settings.s3_bucket, tracked_key)

        loop.run_until_complete(
            _publish(job_id, {"stage": "tracking", "pct": 100})
        )
        SEGMENTS_TOTAL.labels(status="tracking_real").inc()

        from workers.inpainting.tasks import inpaint_segment
        inpaint_segment.apply_async(
            args=[job_id, segment_s3_key, tracked_key],
            queue="inpainting",
        )

        return {
            "job_id": job_id,
            "segment_s3_key": segment_s3_key,
            "tracked_masks_s3_key": tracked_key,
            "status": "real",
            "num_frames": len(frames),
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




def _get_frame_count_and_fps(video_path: str) -> tuple[int, float]:
    """Return (total_frames, fps) for the full video via cv2."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps


def _extract_all_frames(video_path: str) -> list[np.ndarray]:
    """Read up to MAX_TRACKING_FRAMES frames as (H, W, 3) uint8 RGB arrays."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while len(frames) < MAX_TRACKING_FRAMES:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _run_xmem_tracking(
    frames: list[np.ndarray],
    mask_binary: np.ndarray,
    settings: Any,
    bound_log: Any,
) -> np.ndarray:
    """Propagate a first-frame mask across all frames with XMem.

    XMem is loaded directly (not via the model pool): segment_first_frame and
    track_masks run sequentially, so SAM2 has already released GPU memory before
    XMem starts, and the pool's VRAM management would only get in the way here.
    """
    import sys

    sys.path.insert(0, "/home/kavish/XMem")

    import torch
    # Clear GPU memory left by SAM2 before loading XMem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    from model.network import XMem as XMemNetwork
    from inference.inference_core import InferenceCore
    from inference.interact.interactive_utils import (
        image_to_torch,
        index_numpy_to_one_hot_torch,
    )

    xmem_config = {
        "top_k": 30,
        "mem_every": 5,
        "deep_update_every": -1,
        "enable_long_term": True,
        "enable_long_term_count_usage": True,
        "num_prototypes": 128,
        "min_mid_term_frames": 5,
        "max_mid_term_frames": 10,
        "max_long_term_elements": 10000,
    }
    network = XMemNetwork(
        xmem_config,
        os.path.join(settings.model_cache_dir, "XMem.pth"),
    ).cuda().eval()
    processor = InferenceCore(network, config=xmem_config)
    processor.set_all_labels(list(range(1, 2)))  # single object
    bound_log.info("xmem_model_loaded")

    tracked_masks: list[np.ndarray] = []
    with torch.no_grad():
        for i, frame_rgb in enumerate(frames):
            frame_torch, _ = image_to_torch(frame_rgb, device="cuda")
            if i == 0:
                # First frame: seed propagation with the SAM2 mask.
                mask_torch = index_numpy_to_one_hot_torch(mask_binary, 2).cuda()
                output_prob = processor.step(frame_torch, mask_torch[1:2])
            else:
                output_prob = processor.step(frame_torch)

            out_mask = (output_prob[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            tracked_masks.append(out_mask)

    bound_log.info("xmem_tracking_complete", num_frames=len(frames))
    return np.stack(tracked_masks)  # (T, H, W)


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


async def _store_stitch_params(job_id: str, overlap: int, total_frames: int) -> None:
    """Publish the chunk overlap and frame count for stitch_all_chunks to read.

    Rule 6: every Redis key gets a TTL, so these use setex rather than set.
    """
    from redis import asyncio as _aioredis
    from api.core.config import get_settings as _get_settings
    _r = _aioredis.Redis.from_url(_get_settings().redis_url, decode_responses=True)
    try:
        await _r.setex(f"job:{job_id}:overlap", OVERLAP_KEY_TTL, overlap)
        await _r.setex(f"job:{job_id}:total_frames", OVERLAP_KEY_TTL, total_frames)
    finally:
        await _r.aclose()


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
