from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response, status
from redis.asyncio import Redis

from api.core.metrics import (
    JOB_STATUS_REQUESTS_TOTAL,
    JOB_SUBMISSIONS_TOTAL,
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
)
from api.services.storage import generate_upload_url

log = structlog.get_logger(__name__)
router = APIRouter(tags=["jobs"])

RedisDep = Annotated[Redis, Depends(get_redis)]

# ZPOPMAX dequeues highest score first.  Invert the IntEnum (URGENT=0 → score=3).
_MAX_PRIORITY_SCORE = int(JobPriority.BATCH)


@router.post("/", response_model=JobResponse)
async def create_job(
    req: CreateJobRequest,
    redis: RedisDep,
    response: Response,
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
async def request_upload_url(req: UploadURLRequest) -> UploadURLResponse:
    """Generate a presigned S3 PUT URL for direct client-to-storage upload."""
    UPLOAD_URL_REQUESTS_TOTAL.inc()
    log.info("upload_url_requested", filename=req.filename, size_bytes=req.size_bytes)

    result = await generate_upload_url(req.filename, req.content_type, req.size_bytes)
    return UploadURLResponse(
        upload_url=result["upload_url"],
        s3_key=result["s3_key"],
        expires_in=result["expires_in"],
    )
