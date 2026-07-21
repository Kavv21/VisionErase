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
from workers.celery_app import celery_app
from workers.inpainting.tasks import (
    MODAL_MIN_HEIGHT,
    MODAL_MIN_WIDTH,
    _encode_frames_jpeg,
    _encode_masks_png,
    _make_s3_client,
    _read_fps,
    _read_resolution,
    _run_modal_inpainting,
    _run_propainter_inpainting,
    _write_video,
)

log = structlog.get_logger(__name__)

COMPLETED_CHUNKS_TTL = 86400  # 24hr


@celery_app.task(
    bind=True,
    queue="inpainting",
    max_retries=2,
    soft_time_limit=600,
    time_limit=720,
    name="workers.inpainting.chunk_tasks.inpaint_chunk",
)
def inpaint_chunk(
    self,
    job_id: str,
    chunk_index: int,
    total_chunks: int,
    segment_s3_key: str,
    masks_s3_key: str,
    chunk_start_frame: int,
    chunk_end_frame: int,
) -> dict[str, Any]:
    """Inpaint one chunk of frames with ProPainter, guided by its tracked masks.

    masks_s3_key points at the full-video (T, H, W) mask array produced by
    SAM2 Video Predictor; this task slices out its own frame range.

    Once every chunk's inpainting has completed (tracked via an atomic Redis
    counter), triggers stitch_all_chunks to assemble the final video.
    """
    settings = get_settings()
    bound_log = log.bind(
        job_id=job_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        task_id=self.request.id,
    )
    bound_log.info("chunk_inpainting_started")

    loop = asyncio.new_event_loop()

    def _report_progress(pct: int) -> None:
        # Progress events use 1-based chunk numbers (see api/core/progress.py)
        loop.run_until_complete(
            _publish(job_id, {"stage": "inpainting", "chunk": chunk_index + 1, "total": total_chunks, "pct": pct})
        )

    try:
        _report_progress(0)
        SEGMENTS_TOTAL.labels(status="chunk_inpainting_started").inc()

        s3 = _make_s3_client(settings)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "video.mp4")
            masks_path = os.path.join(tmp, "tracked_masks.npy")
            s3.download_file(settings.s3_bucket, segment_s3_key, video_path)
            s3.download_file(settings.s3_bucket, masks_s3_key, masks_path)

            all_masks = np.load(masks_path)  # (T, H, W) uint8 full-video masks
            masks = all_masks[chunk_start_frame:chunk_end_frame]
            width, height = _read_resolution(video_path)
            fps = _read_fps(video_path)

            frames = _extract_chunk_frames(video_path, chunk_start_frame, chunk_end_frame)
            if len(frames) != len(masks):
                raise RuntimeError(
                    f"frame/mask count mismatch for chunk {chunk_index}: "
                    f"got {len(frames)} frames for {len(masks)} masks from {segment_s3_key}"
                )
            bound_log.info("chunk_frames_extracted", num_frames=len(frames))

            use_modal = settings.modal_enabled and (
                width > MODAL_MIN_WIDTH or height > MODAL_MIN_HEIGHT
            )

            output_key = f"jobs/{job_id}/chunks/{chunk_index}/inpainted_chunk.mp4"
            output_path = os.path.join(tmp, "inpainted_chunk.mp4")

            if use_modal:
                bound_log.info(
                    "using_modal_propainter",
                    resolution=f"{width}x{height}",
                    num_frames=len(frames),
                )
                frames_bytes = _encode_frames_jpeg(frames)
                masks_bytes = _encode_masks_png(masks)
                _report_progress(20)

                video_bytes = _run_modal_inpainting(frames_bytes, masks_bytes, fps)
                _report_progress(90)

                with open(output_path, "wb") as f:
                    f.write(video_bytes)
                s3.upload_file(output_path, settings.s3_bucket, output_key)
                status = "modal_hires"
            else:
                bound_log.info(
                    "using_local_propainter",
                    resolution=f"{width}x{height}",
                    num_frames=len(frames),
                )
                comp_frames = _run_propainter_inpainting(
                    frames, masks, settings, bound_log, _report_progress
                )
                _write_video(comp_frames, output_path, fps)
                s3.upload_file(output_path, settings.s3_bucket, output_key)
                status = "real"

        _report_progress(100)
        SEGMENTS_TOTAL.labels(status="chunk_inpainting_complete").inc()

        completed_count = loop.run_until_complete(
            _increment_completed_chunks(job_id, total_chunks)
        )
        bound_log.info("chunk_inpainting_done", completed_chunks=completed_count)

        if completed_count == total_chunks:
            from workers.stitching.tasks import stitch_all_chunks
            stitch_all_chunks.apply_async(
                # segment_s3_key = the original video, used to restore audio
                args=[job_id, total_chunks, segment_s3_key],
                queue="stitching",
            )

        return {
            "job_id": job_id,
            "chunk_index": chunk_index,
            "inpainted_chunk_s3_key": output_key,
            "status": status,
        }

    except SoftTimeLimitExceeded:
        bound_log.error("chunk_inpainting_soft_timeout")
        SEGMENTS_TOTAL.labels(status="chunk_inpainting_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", f"inpaint_chunk timed out on chunk {chunk_index}")
        )
        raise
    except Exception as exc:
        bound_log.exception("chunk_inpainting_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="chunk_inpainting_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_chunk_frames(video_path: str, start_frame: int, end_frame: int) -> list[np.ndarray]:
    """Read frames [start_frame, end_frame) as (H, W, 3) uint8 RGB arrays."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(end_frame - start_frame):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


async def _increment_completed_chunks(job_id: str, total_chunks: int) -> int:
    """Atomically increment the completed-chunk counter and return the new count."""
    from redis import asyncio as _aioredis
    from api.core.config import get_settings as _gs
    _r = _aioredis.Redis.from_url(_gs().redis_url, decode_responses=True)
    try:
        key = f"jobs:{job_id}:completed_chunks"
        count = await _r.incr(key)
        await _r.expire(key, COMPLETED_CHUNKS_TTL)
        return count
    finally:
        await _r.aclose()


async def _publish(job_id: str, data: dict) -> None:
    import json as _json
    from redis import asyncio as _aioredis
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
