from __future__ import annotations

import math
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends

from api.core.auth import get_current_user
from api.core.metrics import SEGMENT_PREVIEW_REQUESTS_TOTAL
from api.models.job import MaskPoint, SegmentPreviewRequest, SegmentPreviewResponse
from api.models.user import User

log = structlog.get_logger(__name__)
router = APIRouter(tags=["segment"])

AuthDep = Annotated[User, Depends(get_current_user)]

_PREVIEW_RADIUS = 40.0
_PREVIEW_POINTS = 24


@router.post("/preview", response_model=SegmentPreviewResponse)
async def segment_preview(
    req: SegmentPreviewRequest,
    current_user: AuthDep,
) -> SegmentPreviewResponse:
    """Return a placeholder circular mask around the clicked point.

    SAM 2 integration is planned for Month 2. Until then this stub returns a
    fixed-radius polygon so the frontend mask editor can demonstrate the UX.
    stub=True in the response signals that this is not a real segmentation.
    """
    log.info(
        "segment_preview_requested",
        user_id=str(current_user.id),
        video_s3_key=req.video_s3_key,
        point_x=req.point.x,
        point_y=req.point.y,
        frame_index=req.frame_index,
    )
    SEGMENT_PREVIEW_REQUESTS_TOTAL.inc()

    step = 2 * math.pi / _PREVIEW_POINTS
    mask_points = [
        MaskPoint(
            x=req.point.x + _PREVIEW_RADIUS * math.cos(i * step),
            y=req.point.y + _PREVIEW_RADIUS * math.sin(i * step),
        )
        for i in range(_PREVIEW_POINTS)
    ]

    return SegmentPreviewResponse(mask_points=mask_points, stub=True)
