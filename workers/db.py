"""Synchronous jobs-table access for Celery workers.

Celery tasks run outside the FastAPI event loop, so they talk to Postgres
through a plain psycopg2 engine instead of the async engine in
api/core/database.py. Every helper swallows and logs DB errors: the jobs
table is bookkeeping, and a Postgres hiccup must never fail a pipeline task.
"""
from __future__ import annotations

import json
from typing import Any

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from api.core.config import get_settings

log = structlog.get_logger(__name__)

_engine: Engine | None = None


def _get_sync_engine() -> Engine:
    """Return a lazily created process-wide synchronous engine."""
    global _engine
    if _engine is None:
        sync_url = get_settings().database_url.replace(
            "postgresql+asyncpg://", "postgresql+psycopg2://"
        )
        _engine = create_engine(sync_url, pool_pre_ping=True, pool_size=2, max_overflow=2)
    return _engine


def _execute(sql: str, params: dict[str, Any], action: str) -> None:
    try:
        engine = _get_sync_engine()
        with engine.begin() as conn:
            conn.execute(text(sql), params)
    except Exception as exc:
        log.error("job_db_update_failed", action=action, error=str(exc), **params)


def mark_job_processing(job_id: str) -> None:
    """Mark a job as processing when the first pipeline task picks it up."""
    _execute(
        "UPDATE jobs SET status = 'processing', updated_at = NOW() WHERE id = :job_id",
        {"job_id": job_id},
        action="mark_processing",
    )


def set_job_total_chunks(job_id: str, total_chunks: int) -> None:
    """Record how many chunks the video was split into."""
    _execute(
        "UPDATE jobs SET total_chunks = :total_chunks, updated_at = NOW() WHERE id = :job_id",
        {"job_id": job_id, "total_chunks": total_chunks},
        action="set_total_chunks",
    )


def mark_job_completed(
    job_id: str,
    result_s3_key: str,
    quality_metrics: dict[str, Any] | None = None,
    status: str = "completed",
) -> None:
    """Finalize a job: result key, timing, progress, and optional quality metrics.

    status may be "quality_warning" when the result failed the quality gates.
    """
    _execute(
        """
        UPDATE jobs SET
            status = :status,
            result_s3_key = :result_s3_key,
            quality_metrics = CAST(:quality_metrics AS jsonb),
            completed_at = NOW(),
            processing_time_seconds = EXTRACT(EPOCH FROM (NOW() - created_at)),
            progress_pct = 100.0,
            updated_at = NOW()
        WHERE id = :job_id
        """,
        {
            "job_id": job_id,
            "status": status,
            "result_s3_key": result_s3_key,
            "quality_metrics": json.dumps(quality_metrics) if quality_metrics else None,
        },
        action="mark_completed",
    )


def mark_job_failed(job_id: str, error: str) -> None:
    """Mark a job as failed with a truncated error message."""
    _execute(
        """
        UPDATE jobs SET
            status = 'failed',
            error_message = :error_message,
            updated_at = NOW()
        WHERE id = :job_id
        """,
        {"job_id": job_id, "error_message": str(error)[:1000]},
        action="mark_failed",
    )


def get_job_video_s3_key(job_id: str) -> str | None:
    """Return the original input video key for a job, or None if unknown."""
    try:
        engine = _get_sync_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT video_s3_key FROM jobs WHERE id = :job_id"),
                {"job_id": job_id},
            ).first()
        return row[0] if row else None
    except Exception as exc:
        log.error("job_db_read_failed", action="get_video_s3_key", job_id=job_id, error=str(exc))
        return None
