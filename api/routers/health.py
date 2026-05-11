from __future__ import annotations

import structlog
from fastapi import APIRouter

from api.core.metrics import HEALTH_CHECKS_TOTAL

router = APIRouter(tags=["health"])
log = structlog.get_logger(__name__)


@router.get("", summary="Liveness probe")
async def health_check() -> dict[str, str]:
    """Return service liveness status."""
    HEALTH_CHECKS_TOTAL.inc()
    log.debug("health_check")
    return {"status": "ok"}
