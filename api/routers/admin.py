from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.metrics import ADMIN_JOBS_REQUESTS_TOTAL
from api.models.job_record import JobRecord

log = structlog.get_logger(__name__)
router = APIRouter(tags=["admin"])

DbDep = Annotated[AsyncSession, Depends(get_db)]


@router.get("/jobs")
async def list_jobs(
    db: DbDep,
    status: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[dict[str, Any]]:
    """Return all jobs ordered by created_at DESC. Internal use — no auth for now."""
    ADMIN_JOBS_REQUESTS_TOTAL.inc()

    stmt = select(JobRecord).order_by(JobRecord.created_at.desc()).limit(limit)
    if status is not None:
        stmt = stmt.where(JobRecord.status == status)

    result = await db.execute(stmt)
    jobs = result.scalars().all()
    log.debug("admin_jobs_listed", count=len(jobs), status_filter=status)

    return [
        {
            "id": str(job.id),
            "user_id": str(job.user_id) if job.user_id else None,
            "status": job.status,
            "video_s3_key": job.video_s3_key,
            "result_s3_key": job.result_s3_key,
            "priority": job.priority,
            "total_chunks": job.total_chunks,
            "progress_pct": job.progress_pct,
            "error_message": job.error_message,
            "processing_time_seconds": job.processing_time_seconds,
            "quality_metrics": job.quality_metrics,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
        for job in jobs
    ]
