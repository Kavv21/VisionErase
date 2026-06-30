from __future__ import annotations

from datetime import datetime
from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field


class MaskPoint(BaseModel):
    x: float
    y: float


class MaskData(BaseModel):
    points: list[MaskPoint]
    frame_index: int = 0


class JobPriority(IntEnum):
    URGENT = 0
    HIGH = 1
    NORMAL = 2
    BATCH = 3


class JobStatus(StrEnum):
    PENDING = "pending"
    SEGMENTING = "segmenting"
    TRACKING = "tracking"
    INPAINTING = "inpainting"
    STITCHING = "stitching"
    QUALITY_CHECK = "quality_check"
    COMPLETED = "completed"
    FAILED = "failed"


class CreateJobRequest(BaseModel):
    video_s3_key: str
    mask: MaskData
    priority: JobPriority = JobPriority.NORMAL
    webhook_url: str | None = None
    output_format: str | None = None  # None → preserve input container format


class SegmentStatus(BaseModel):
    segment_index: int
    worker_id: str
    status: JobStatus
    start_frame: int
    end_frame: int
    progress_pct: float = 0.0
    ssim_score: float | None = None
    error: str | None = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_pct: float = 0.0
    created_at: datetime
    result_s3_key: str | None = None
    error: str | None = None
    cached: bool = False
    num_segments: int | None = None
    segments: list[SegmentStatus] = []
    boundary_fusion_applied: bool = False


class SegmentPreviewRequest(BaseModel):
    video_s3_key: str
    point: MaskPoint
    frame_index: int = 0


class SegmentPreviewResponse(BaseModel):
    mask_points: list[MaskPoint]
    stub: bool = True


class UploadURLRequest(BaseModel):
    filename: str
    content_type: str = "video/mp4"
    size_bytes: int


class UploadURLResponse(BaseModel):
    upload_url: str
    s3_key: str
    expires_in: int = Field(default=3600)


class VideoUploadResponse(BaseModel):
    s3_key: str
    size_bytes: int
