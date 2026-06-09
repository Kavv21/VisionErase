"""Unit tests for workers/segmentation/tasks.py.

Redis publish calls are mocked — no real broker or Redis connection needed.
"""
from __future__ import annotations

from unittest import mock

import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    """Replace async Redis helpers with no-ops so no real Redis is needed."""
    async def noop_publish(job_id: str, data: dict) -> None:
        pass

    async def noop_status(job_id: str, status: str, error: str) -> None:
        pass

    monkeypatch.setattr("workers.segmentation.tasks._publish", noop_publish)
    monkeypatch.setattr("workers.segmentation.tasks._update_status", noop_status)


@pytest.fixture()
def app():
    from workers.celery_app import celery_app
    return celery_app


# ── segment_first_frame registration ──────────────────────────────────────────


@pytest.mark.unit
def test_segment_first_frame_is_registered(app):
    """segment_first_frame must be discoverable in the Celery task registry."""
    assert "workers.segmentation.tasks.segment_first_frame" in app.tasks


@pytest.mark.unit
def test_segment_first_frame_queue(app):
    """segment_first_frame must be routed to the segmentation queue."""
    task = app.tasks["workers.segmentation.tasks.segment_first_frame"]
    assert task.queue == "segmentation"


# ── segment_first_frame return value ──────────────────────────────────────────


@pytest.mark.unit
def test_segment_first_frame_returns_correct_dict_shape():
    """Return value must contain exactly the four required keys."""
    from workers.segmentation.tasks import segment_first_frame

    result = segment_first_frame.apply(
        args=["job-abc", "jobs/job-abc/segments/seg_0.mp4", {"points": []}]
    ).get()

    assert isinstance(result, dict)
    assert set(result.keys()) == {"job_id", "segment_s3_key", "mask_s3_key", "status"}


@pytest.mark.unit
def test_segment_first_frame_has_job_id():
    """job_id in the return dict must match the input job_id."""
    from workers.segmentation.tasks import segment_first_frame

    result = segment_first_frame.apply(
        args=["job-xyz", "jobs/job-xyz/segments/seg_0.mp4", {}]
    ).get()

    assert result["job_id"] == "job-xyz"


@pytest.mark.unit
def test_segment_first_frame_has_mask_s3_key():
    """mask_s3_key must follow the expected stub path pattern."""
    from workers.segmentation.tasks import segment_first_frame

    result = segment_first_frame.apply(
        args=["job-123", "jobs/job-123/segments/seg_0.mp4", {}]
    ).get()

    assert result["mask_s3_key"] == "jobs/job-123/masks/mask_stub.npy"


@pytest.mark.unit
def test_segment_first_frame_status_is_stub():
    """status field must be 'stub' until real SAM 2 integration lands."""
    from workers.segmentation.tasks import segment_first_frame

    result = segment_first_frame.apply(
        args=["job-stub", "jobs/job-stub/segments/seg_0.mp4", {}]
    ).get()

    assert result["status"] == "stub"


# ── track_masks registration ───────────────────────────────────────────────────


@pytest.mark.unit
def test_track_masks_is_registered(app):
    """track_masks must be discoverable in the Celery task registry."""
    assert "workers.segmentation.tasks.track_masks" in app.tasks


@pytest.mark.unit
def test_track_masks_queue(app):
    """track_masks must be routed to the segmentation queue."""
    task = app.tasks["workers.segmentation.tasks.track_masks"]
    assert task.queue == "segmentation"


# ── track_masks return value ───────────────────────────────────────────────────


@pytest.mark.unit
def test_track_masks_returns_correct_dict_shape():
    """Return value must contain exactly the four required keys."""
    from workers.segmentation.tasks import track_masks

    result = track_masks.apply(
        args=[
            "job-abc",
            "jobs/job-abc/segments/seg_0.mp4",
            "jobs/job-abc/masks/mask_stub.npy",
        ]
    ).get()

    assert isinstance(result, dict)
    assert set(result.keys()) == {"job_id", "segment_s3_key", "tracked_masks_s3_key", "status"}


@pytest.mark.unit
def test_track_masks_has_tracked_masks_s3_key():
    """tracked_masks_s3_key must follow the expected stub path pattern."""
    from workers.segmentation.tasks import track_masks

    result = track_masks.apply(
        args=[
            "job-789",
            "jobs/job-789/segments/seg_0.mp4",
            "jobs/job-789/masks/mask_stub.npy",
        ]
    ).get()

    assert result["tracked_masks_s3_key"] == "jobs/job-789/masks/tracked_stub.npy"


@pytest.mark.unit
def test_track_masks_has_job_id():
    """job_id in the return dict must match the input job_id."""
    from workers.segmentation.tasks import track_masks

    result = track_masks.apply(
        args=[
            "job-999",
            "jobs/job-999/segments/seg_0.mp4",
            "jobs/job-999/masks/mask_stub.npy",
        ]
    ).get()

    assert result["job_id"] == "job-999"
