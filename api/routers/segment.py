from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import time
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException

from api.core.auth import get_current_user
from api.core.config import get_settings
from api.core.metrics import (
    SEGMENT_PREVIEW_ERRORS_TOTAL,
    SEGMENT_PREVIEW_LATENCY_SECONDS,
    SEGMENT_PREVIEW_REQUESTS_TOTAL,
)
from api.models.job import MaskPoint, SegmentPreviewRequest, SegmentPreviewResponse
from api.models.user import User
from api.services.storage import _s3_client

log = structlog.get_logger(__name__)
router = APIRouter(tags=["segment"])

AuthDep = Annotated[User, Depends(get_current_user)]


@router.post("/preview", response_model=SegmentPreviewResponse)
async def segment_preview(
    req: SegmentPreviewRequest,
    current_user: AuthDep,
) -> SegmentPreviewResponse:
    """Return a real SAM2 mask for the given points on the given frame."""
    log.info(
        "segment_preview_requested",
        user_id=str(current_user.id),
        video_s3_key=req.video_s3_key,
        num_points=len(req.points),
        frame_index=req.frame_index,
    )
    SEGMENT_PREVIEW_REQUESTS_TOTAL.inc()

    if not req.points:
        raise HTTPException(status_code=422, detail="points must be a non-empty list")

    settings = get_settings()
    start = time.monotonic()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, _run_sam2_preview, req.video_s3_key, req.frame_index, req.points, settings
        )
    except Exception as exc:
        SEGMENT_PREVIEW_ERRORS_TOTAL.inc()
        log.exception("segment_preview_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="segmentation failed") from exc
    finally:
        SEGMENT_PREVIEW_LATENCY_SECONDS.observe(time.monotonic() - start)

    return SegmentPreviewResponse(**result, stub=False)


# ── Blocking work, run in a thread pool executor ────────────────────────────────

def _run_sam2_preview(
    video_s3_key: str,
    frame_index: int,
    points: list[MaskPoint],
    settings: Any,
) -> dict[str, Any]:
    """Download the frame, run SAM2 with the given points, return a mask PNG.

    Runs on the FastAPI thread pool executor — this whole function is blocking
    (S3 download, cv2 decode, SAM2 inference) and must never run on the event loop.
    """
    import cv2
    import numpy as np
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Rule: always load models through the model pool, never directly.
    from pipeline.pool.model_pool import get_model_pool
    from workers.segmentation.tasks import _extract_frame, _load_sam2_model

    s3 = _s3_client()
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "frame_source.mp4")
        s3.download_file(settings.s3_bucket, video_s3_key, video_path)
        frame_rgb = _extract_frame(video_path, frame_index)

    pool = get_model_pool()
    pool.register_loader("sam2", lambda: _load_sam2_model(settings))

    point_coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    point_labels = np.array([p.label for p in points], dtype=np.int32)

    with pool.acquire("sam2") as sam2_model:
        predictor = SAM2ImagePredictor(sam2_model)
        with torch.no_grad():
            predictor.set_image(frame_rgb)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]
    score = float(scores[best_idx])

    mask_uint8 = (best_mask.astype(np.uint8) * 255)
    ok, encoded = cv2.imencode(".png", mask_uint8)
    if not ok:
        raise RuntimeError("failed to encode SAM2 mask as PNG")

    height, width = mask_uint8.shape
    return {
        "mask_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "score": score,
        "width": int(width),
        "height": int(height),
    }
