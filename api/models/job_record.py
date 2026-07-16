from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from api.core.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobRecord(Base):
    """Persistent row per video-removal job — mirrors the manually created jobs table."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=True
    )
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    video_s3_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    result_s3_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    priority: Mapped[str] = mapped_column(String(20), default="normal")
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    processing_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
