from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from redis.asyncio import Redis

from api.core.auth import get_current_user
from api.core.config import get_settings
from api.core.metrics import (
    JOB_STATUS_REQUESTS_TOTAL,
    JOB_SUBMISSIONS_TOTAL,
    UPLOAD_REQUESTS_TOTAL,
    UPLOAD_URL_REQUESTS_TOTAL,
)
from api.core.redis import (
    compute_job_hash,
    enqueue_job,
    get_cached_result,
    get_job_status,
    get_redis,
    set_job_status,
)
from api.models.job import (
    CreateJobRequest,
    JobPriority,
    JobResponse,
    JobStatus,
    UploadURLRequest,
    UploadURLResponse,
    VideoUploadResponse,
)
from api.models.user import User
from api.services.storage import _s3_client, generate_upload_url

log = structlog.get_logger(__name__)
router = APIRouter(tags=["jobs"])

RedisDep = Annotated[Redis, Depends(get_redis)]
AuthDep = Annotated[User, Depends(get_current_user)]

# ZPOPMAX dequeues highest score first.  Invert the IntEnum (URGENT=0 → score=3).
_MAX_PRIORITY_SCORE = int(JobPriority.BATCH)

_ALLOWED_VIDEO_TYPES = frozenset({"video/mp4", "video/quicktime", "video/webm"})


@router.post("/", response_model=JobResponse)
async def create_job(
    req: CreateJobRequest,
    redis: RedisDep,
    response: Response,
    _current_user: AuthDep,
) -> JobResponse:
    """Submit a job; returns 200 + cached=True on dedup hit, 202 on new enqueue."""
    mask_dict = req.mask.model_dump()
    job_hash = compute_job_hash(req.video_s3_key, mask_dict)
    bound_log = log.bind(job_hash=job_hash, video_s3_key=req.video_s3_key)

    cached = await get_cached_result(redis, job_hash)
    if cached is not None:
        bound_log.info("job_deduplicated")
        JOB_SUBMISSIONS_TOTAL.labels(outcome="deduplicated").inc()
        return JobResponse(
            job_id=cached["job_id"],
            status=JobStatus.COMPLETED,
            progress_pct=100.0,
            created_at=datetime.fromisoformat(cached["created_at"]),
            result_s3_key=cached.get("result_s3_key"),
            cached=True,
        )

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    bound_log = bound_log.bind(job_id=job_id)

    await set_job_status(
        redis,
        job_id,
        {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress_pct": 0.0,
            "created_at": now.isoformat(),
            "result_s3_key": None,
            "error": None,
        },
    )

    payload = {
        "job_id": job_id,
        "video_s3_key": req.video_s3_key,
        "mask": mask_dict,
        "webhook_url": req.webhook_url,
        "output_format": req.output_format,
        "job_hash": job_hash,
        "created_at": now.isoformat(),
    }
    # URGENT=0 → score=3, BATCH=3 → score=0 so ZPOPMAX always pops most urgent.
    queue_score = _MAX_PRIORITY_SCORE - int(req.priority)
    await enqueue_job(redis, job_id, queue_score, payload)

    bound_log.info("job_accepted", priority=req.priority.name)
    JOB_SUBMISSIONS_TOTAL.labels(outcome="accepted").inc()

    response.status_code = status.HTTP_202_ACCEPTED
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress_pct=0.0,
        created_at=now,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, redis: RedisDep) -> JobResponse:
    """Return the current status of a job by ID."""
    JOB_STATUS_REQUESTS_TOTAL.inc()
    bound_log = log.bind(job_id=job_id)

    status_data = await get_job_status(redis, job_id)
    if status_data is None:
        bound_log.info("job_not_found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    bound_log.debug("job_status_fetched", job_status=status_data.get("status"))
    return JobResponse(
        job_id=status_data["job_id"],
        status=JobStatus(status_data["status"]),
        progress_pct=status_data.get("progress_pct", 0.0),
        created_at=datetime.fromisoformat(status_data["created_at"]),
        result_s3_key=status_data.get("result_s3_key"),
        error=status_data.get("error"),
        cached=status_data.get("cached", False),
    )


@router.post("/upload-url", response_model=UploadURLResponse)
async def request_upload_url(req: UploadURLRequest, _current_user: AuthDep) -> UploadURLResponse:
    """Generate a presigned S3 PUT URL for direct client-to-storage upload."""
    UPLOAD_URL_REQUESTS_TOTAL.inc()
    log.info("upload_url_requested", filename=req.filename, size_bytes=req.size_bytes)

    result = await generate_upload_url(req.filename, req.content_type, req.size_bytes)
    return UploadURLResponse(
        upload_url=result["upload_url"],
        s3_key=result["s3_key"],
        expires_in=result["expires_in"],
    )


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: Annotated[UploadFile, File()],
    _current_user: AuthDep,
) -> VideoUploadResponse:
    """Accept a multipart video upload and stream it directly to MinIO."""
    settings = get_settings()
    bound_log = log.bind(filename=file.filename, content_type=file.content_type)

    if file.content_type not in _ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported media type '{file.content_type}'. "
                "Accepted: video/mp4, video/quicktime, video/webm"
            ),
        )

    max_bytes = settings.max_video_size_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.max_video_size_mb} MB.",
        )

    s3_key = f"uploads/{uuid.uuid4()}/{file.filename}"
    bound_log = bound_log.bind(s3_key=s3_key)
    bound_log.info("upload_started")

    file.file.seek(0)
    s3 = _s3_client()
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: s3.upload_fileobj(file.file, settings.s3_bucket, s3_key),
        )
    except Exception as exc:
        bound_log.error("upload_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload to storage failed.",
        ) from exc

    size_bytes = file.size or 0
    bound_log.info("upload_complete", size_bytes=size_bytes)
    UPLOAD_REQUESTS_TOTAL.inc()

    return VideoUploadResponse(s3_key=s3_key, size_bytes=size_bytes)
