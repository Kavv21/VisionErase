from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import Any

import numpy as np
import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.config import get_settings
from api.core.metrics import SEGMENTS_TOTAL
from api.core.redis import acquire_redis, publish_progress, set_job_status
from workers.celery_app import celery_app
from workers.segmentation.oiv_refiner import refine_tracked_masks
from workers.segmentation.tasks import _make_s3_client, _run_xmem_tracking

log = structlog.get_logger(__name__)

MAX_CHUNK_WAIT_SECONDS = 600     # max time to wait for the previous chunk's last_mask
CHUNK_WAIT_POLL_SECONDS = 5


@celery_app.task(
    bind=True,
    queue="segmentation",
    max_retries=2,
    soft_time_limit=600,
    time_limit=720,
    name="workers.segmentation.chunk_tasks.track_masks_chunk",
)
def track_masks_chunk(
    self,
    job_id: str,
    chunk_index: int,
    total_chunks: int,
    segment_s3_key: str,
    first_frame_mask_s3_key: str,
    chunk_start_frame: int,
    chunk_end_frame: int,
) -> dict[str, Any]:
    """Propagate a tracked mask across one chunk of frames with XMem++.

    Chunk 0 seeds from the SAM2 first-frame mask. Later chunks seed from the
    previous chunk's last tracked mask, so object identity carries across
    chunk boundaries even though chunks are dispatched to the queue in
    parallel and may not finish in order.
    """
    settings = get_settings()
    bound_log = log.bind(
        job_id=job_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        task_id=self.request.id,
    )
    bound_log.info("chunk_tracking_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "tracking", "chunk": chunk_index, "total": total_chunks, "pct": 0})
        )
        SEGMENTS_TOTAL.labels(status="chunk_tracking_started").inc()

        s3 = _make_s3_client(settings)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "video.mp4")
            s3.download_file(settings.s3_bucket, segment_s3_key, video_path)

            seed_mask_path = os.path.join(tmp, "seed_mask.npy")
            if chunk_index > 0:
                prev_mask_key = f"jobs/{job_id}/chunks/{chunk_index - 1}/last_mask.npy"
                _wait_for_s3_object(s3, settings.s3_bucket, prev_mask_key, bound_log)
                s3.download_file(settings.s3_bucket, prev_mask_key, seed_mask_path)
            else:
                s3.download_file(settings.s3_bucket, first_frame_mask_s3_key, seed_mask_path)

            seed_mask = np.load(seed_mask_path)
            seed_mask_binary = (seed_mask > 127).astype(np.uint8)

            frames = _extract_chunk_frames(video_path, chunk_start_frame, chunk_end_frame)
            if not frames:
                raise RuntimeError(
                    f"no frames decoded for chunk {chunk_index} "
                    f"[{chunk_start_frame}, {chunk_end_frame}) from {segment_s3_key}"
                )
            bound_log.info("chunk_frames_extracted", num_frames=len(frames))

            loop.run_until_complete(
                _publish(job_id, {"stage": "tracking", "chunk": chunk_index, "total": total_chunks, "pct": 40})
            )

            # Run XMem++ on Modal T4 GPU
            bound_log.info("xmem_modal_started", chunk_index=chunk_index, num_frames=len(frames))
            import io as _io
            import cv2 as _cv2
            import modal as _modal

            # Encode frames as JPEG bytes
            frames_bytes = []
            for f in frames:
                bgr = _cv2.cvtColor(f, _cv2.COLOR_RGB2BGR)
                _, buf = _cv2.imencode('.jpg', bgr, [_cv2.IMWRITE_JPEG_QUALITY, 95])
                frames_bytes.append(buf.tobytes())

            # Encode seed mask as PNG bytes
            from PIL import Image as _Image
            mask_pil = _Image.fromarray(seed_mask_binary * 255)
            mask_buf = _io.BytesIO()
            mask_pil.save(mask_buf, format='PNG')
            seed_mask_bytes = mask_buf.getvalue()

            # Call Modal XMem
            xmem_fn = _modal.Function.from_name("visionerase-xmem", "track_masks")
            tracked_npy_bytes = xmem_fn.remote(frames_bytes, seed_mask_bytes)

            # Decode result
            tracked_array = np.load(_io.BytesIO(tracked_npy_bytes))
            bound_log.info("xmem_modal_complete", chunk_index=chunk_index,
                          tracked_shape=str(tracked_array.shape))

            bound_log.info("oiv_refinement_started", num_frames=len(frames))
            tracked_array = refine_tracked_masks(
                frames=frames,
                tracked_masks=tracked_array,
                model_cache_dir=settings.model_cache_dir,
                device="cpu",  # CPU to avoid VRAM conflict with SAM2/XMem
            )
            bound_log.info("oiv_refinement_complete")

            tracked_key = f"jobs/{job_id}/chunks/{chunk_index}/tracked_masks.npy"
            tracked_path = os.path.join(tmp, "tracked_masks.npy")
            np.save(tracked_path, tracked_array)
            s3.upload_file(tracked_path, settings.s3_bucket, tracked_key)

            last_mask_key = f"jobs/{job_id}/chunks/{chunk_index}/last_mask.npy"
            last_mask_path = os.path.join(tmp, "last_mask.npy")
            np.save(last_mask_path, tracked_array[-1])
            s3.upload_file(last_mask_path, settings.s3_bucket, last_mask_key)

        loop.run_until_complete(
            _publish(job_id, {"stage": "tracking", "chunk": chunk_index, "total": total_chunks, "pct": 100})
        )
        SEGMENTS_TOTAL.labels(status="chunk_tracking_complete").inc()

        from workers.inpainting.chunk_tasks import inpaint_chunk
        inpaint_chunk.apply_async(
            args=[
                job_id,
                chunk_index,
                total_chunks,
                segment_s3_key,
                tracked_key,
                chunk_start_frame,
                chunk_end_frame,
            ],
            queue="inpainting",
        )

        return {
            "job_id": job_id,
            "chunk_index": chunk_index,
            "chunk_s3_key": segment_s3_key,
            "tracked_masks_s3_key": tracked_key,
            "chunk_start_frame": chunk_start_frame,
            "chunk_end_frame": chunk_end_frame,
        }

    except SoftTimeLimitExceeded:
        bound_log.error("chunk_tracking_soft_timeout")
        SEGMENTS_TOTAL.labels(status="chunk_tracking_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", f"track_masks_chunk timed out on chunk {chunk_index}")
        )
        raise
    except Exception as exc:
        bound_log.exception("chunk_tracking_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="chunk_tracking_error").inc()
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


def _wait_for_s3_object(s3: Any, bucket: str, key: str, bound_log: Any) -> None:
    """Poll until an S3 object exists, or raise TimeoutError after MAX_CHUNK_WAIT_SECONDS.

    Chunks are dispatched to the queue all at once, so chunk N can reach the
    front of the queue before chunk N-1 has finished writing its last_mask.
    """
    waited = 0
    while waited < MAX_CHUNK_WAIT_SECONDS:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return
        except Exception:
            bound_log.info("waiting_for_previous_chunk_mask", key=key, waited=waited)
            time.sleep(CHUNK_WAIT_POLL_SECONDS)
            waited += CHUNK_WAIT_POLL_SECONDS
    raise TimeoutError(f"{key} not available after {MAX_CHUNK_WAIT_SECONDS}s")


async def _publish(job_id: str, data: dict) -> None:
    import json as _json
    from redis import asyncio as _aioredis
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
