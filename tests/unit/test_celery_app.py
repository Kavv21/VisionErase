"""Unit tests for workers/celery_app.py."""
from __future__ import annotations

from unittest import mock

import pytest
from celery import Celery
from celery.canvas import _chain as CeleryChain

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_boundary_tasks(monkeypatch):
    """Prevent workers/boundary/tasks.py from importing heavy CV deps."""
    stub = mock.MagicMock()
    stub.si.return_value = mock.MagicMock()
    stub.s.return_value = mock.MagicMock()
    monkeypatch.setitem(
        __import__("sys").modules,
        "workers.boundary.tasks",
        mock.MagicMock(apply_boundary_fusion=stub),
    )


@pytest.fixture()
def app():
    from workers.celery_app import celery_app
    return celery_app


# ── App configuration ──────────────────────────────────────────────────────────


def test_celery_app_name(app):
    assert app.main == "visionerase"


def test_celery_app_broker_url(app):
    from api.core.config import get_settings
    assert app.conf.broker_url == get_settings().celery_broker_url


def test_task_acks_late(app):
    assert app.conf.task_acks_late is True


def test_task_reject_on_worker_lost(app):
    assert app.conf.task_reject_on_worker_lost is True


def test_worker_max_tasks_per_child(app):
    assert app.conf.worker_max_tasks_per_child == 50


def test_task_serializer(app):
    assert app.conf.task_serializer == "json"


def test_result_expires(app):
    from api.core.config import get_settings
    assert app.conf.result_expires == get_settings().redis_result_ttl


# ── Task routing ───────────────────────────────────────────────────────────────


def test_all_five_queues_in_task_routes(app):
    routes = app.conf.task_routes
    route_keys = list(routes.keys())
    assert any("segmentation" in k for k in route_keys)
    assert any("inpainting" in k for k in route_keys)
    assert any("stitching" in k for k in route_keys)
    assert any("quality" in k for k in route_keys)
    assert any("boundary" in k for k in route_keys)


def test_segmentation_route_queue(app):
    route = app.conf.task_routes.get("workers.segmentation.*", {})
    assert route.get("queue") == "segmentation"


def test_inpainting_route_queue(app):
    route = app.conf.task_routes.get("workers.inpainting.*", {})
    assert route.get("queue") == "inpainting"


def test_stitching_route_queue(app):
    route = app.conf.task_routes.get("workers.stitching.*", {})
    assert route.get("queue") == "stitching"


def test_quality_route_queue(app):
    route = app.conf.task_routes.get("workers.quality.*", {})
    assert route.get("queue") == "quality"


def test_boundary_route_queue(app):
    route = app.conf.task_routes.get("workers.boundary.*", {})
    assert route.get("queue") == "boundary"


# ── get_celery_app ─────────────────────────────────────────────────────────────


def test_get_celery_app_returns_celery_instance():
    from workers.celery_app import get_celery_app
    result = get_celery_app()
    assert isinstance(result, Celery)


def test_get_celery_app_returns_same_instance(app):
    from workers.celery_app import get_celery_app
    assert get_celery_app() is app


# ── Placeholder task stubs ─────────────────────────────────────────────────────


def test_split_video_task_is_registered(app):
    assert "workers.celery_app.split_video_into_segments" in app.tasks


def test_process_segment_task_is_registered(app):
    assert "workers.celery_app.process_segment" in app.tasks


def test_final_stitch_task_is_registered(app):
    assert "workers.celery_app.final_stitch" in app.tasks


def test_quality_check_task_is_registered(app):
    assert "workers.celery_app.quality_check" in app.tasks


def test_split_video_task_queue(app):
    task = app.tasks["workers.celery_app.split_video_into_segments"]
    assert task.queue == "segmentation"


def test_process_segment_task_queue(app):
    task = app.tasks["workers.celery_app.process_segment"]
    assert task.queue == "segmentation"


def test_final_stitch_task_queue(app):
    task = app.tasks["workers.celery_app.final_stitch"]
    assert task.queue == "stitching"


def test_quality_check_task_queue(app):
    task = app.tasks["workers.celery_app.quality_check"]
    assert task.queue == "quality"


# ── build_pipeline_chain ───────────────────────────────────────────────────────


def test_build_pipeline_chain_accepts_job_id_and_payload():
    from workers.celery_app import build_pipeline_chain
    result = build_pipeline_chain("job-abc", {"video_s3_key": "s3://bucket/v.mp4"})
    assert result is not None


def test_build_pipeline_chain_returns_chain_object():
    from workers.celery_app import build_pipeline_chain
    result = build_pipeline_chain("job-xyz", {"video_s3_key": "s3://bucket/v.mp4"})
    assert isinstance(result, CeleryChain)


def test_build_pipeline_chain_different_job_ids():
    from workers.celery_app import build_pipeline_chain
    r1 = build_pipeline_chain("job-1", {})
    r2 = build_pipeline_chain("job-2", {})
    assert isinstance(r1, CeleryChain)
    assert isinstance(r2, CeleryChain)
    assert r1 is not r2
