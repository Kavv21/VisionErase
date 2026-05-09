from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from api.core.config import get_settings

log = structlog.get_logger(__name__)

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    echo=_settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # required for async: avoids lazy-load after commit
)


class Base(DeclarativeBase):
    """Declarative base — all ORM models inherit from this."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session; use as FastAPI Depends()."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all tables from metadata; use Alembic for production migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("database_tables_created")


async def close_db() -> None:
    """Dispose the engine connection pool; call once at app shutdown."""
    await engine.dispose()
    log.info("database_engine_disposed")
