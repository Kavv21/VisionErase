"""Celery application definition and full pipeline canvas for VisionErase."""
from __future__ import annotations

from typing import Any

import structlog
from celery import Celery, chain, chord, group
from celery.canvas import Signature

from api.core.config import get_settings
from api.core.metrics import SEGMENTS_TOTAL

log = structlog.get_logger(__name__)

settings = get_settings()

# ── App ────────────────────────────────────────────────────────────────────────

celery_app = Celery(
    "visionerase",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_max_tasks_per_child=50,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=settings.redis_result_ttl,
    task_routes={
        "workers.segmentation.*": {"queue": "segmentation"},
        "workers.inpainting.*": {"queue": "inpainting"},
        "workers.stitching.*": {"queue": "stitching"},
        "workers.quality.*": {"queue": "quality"},
        "workers.boundary.*": {"queue": "boundary"},
    },
)

celery_app.autodiscover_tasks([
    "workers.segmentation",
    "workers.inpainting",
    "workers.stitching",
    "workers.quality",
    "workers.boundary",
])
# Full-video chunking tasks live in chunk_tasks.py alongside each stage's
# tasks.py — a second autodiscover pass with related_name picks them up.
celery_app.autodiscover_tasks(
    ["workers.segmentation", "workers.inpainting"],
    related_name="chunk_tasks",
)

# ── Worker startup signal ──────────────────────────────────────────────────────

from celery.signals import task_failure, worker_process_init

@worker_process_init.connect
def init_worker_redis(**kwargs):
    """Initialize Redis pool in each Celery worker process at startup."""
    from api.core.redis import init_redis_pool
    init_redis_pool()
    log.info("worker_redis_pool_initialized")


@task_failure.connect
def mark_job_failed_in_db(sender=None, args=None, kwargs=None, exception=None, **_):
    """Persist failure to the jobs table for any pipeline task.

    Every VisionErase task takes job_id as its first positional argument, so a
    single global hook covers all workers without touching each except block.
    """
    job_id = (args[0] if args else None) or (kwargs or {}).get("job_id")
    if not job_id:
        return
    from workers.db import mark_job_failed
    mark_job_failed(str(job_id), str(exception))
    log.info(
        "job_marked_failed_in_db",
        job_id=str(job_id),
        task=getattr(sender, "name", None),
    )




def get_celery_app() -> Celery:
    """Return the configured Celery application instance."""
    return celery_app


# ── Placeholder tasks ──────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    queue="segmentation",
    name="workers.celery_app.split_video_into_segments",
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=300,
    time_limit=360,
)
def split_video_into_segments(self, job_id: str, payload: dict) -> list[dict]:
    """Split video into N overlapping segments with 150-frame overlap."""
    pass


@celery_app.task(
    bind=True,
    queue="segmentation",
    name="workers.celery_app.process_segment",
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=600,
    time_limit=660,
)
def process_segment(self, job_id: str, segment_index: int) -> dict:
    """Run SAM2 → XMem++ → ProPainter → chunk stitch for one segment."""
    pass


@celery_app.task(
    bind=True,
    queue="stitching",
    name="workers.celery_app.final_stitch",
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=300,
    time_limit=360,
)
def final_stitch(self, segment_results: list, job_id: str) -> dict:
    """Combine all corrected segments into the final video with overlap blending."""
    pass


@celery_app.task(
    bind=True,
    queue="quality",
    name="workers.celery_app.quality_check",
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=120,
    time_limit=150,
)
def quality_check(self, stitch_result: Any, job_id: str) -> dict:
    """Score final video SSIM/PSNR, flag low-quality regions, publish job_complete."""
    pass


# ── Pipeline canvas ────────────────────────────────────────────────────────────

def build_pipeline_chain(job_id: str, payload: dict) -> chain:
    """Build the full distributed pipeline as a Celery canvas.

    Step 1: split_video_into_segments — splits video into N segments
    Step 2: chord of process_segment tasks — SAM2+XMem+++ProPainter in parallel
    Step 3: apply_boundary_fusion — BoundaryFusion corrects segment boundaries
    Step 4: final_stitch — combines corrected segments into the final video
    Step 5: quality_check — SSIM/PSNR scoring and Redis completion event
    """
    # Late import to avoid circular import at module load time
    from workers.boundary.tasks import apply_boundary_fusion

    cfg = get_settings()
    n = cfg.num_segment_workers

    # Chord: all N segments processed in parallel; callback runs steps 3-5
    segment_chord = chord(
        header=group(process_segment.s(job_id, i) for i in range(n)),
        body=chain(
            # Placeholder: the stitched S3 key is passed at runtime by the
            # preceding dispatcher; si() ignores the chord result.
            apply_boundary_fusion.si(job_id, ""),
            final_stitch.si(job_id),
            quality_check.si(job_id),
        ),
    )

    return chain(
        split_video_into_segments.s(job_id, payload),
        segment_chord,
    )
