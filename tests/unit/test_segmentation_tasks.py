"""Unit tests for workers/segmentation/tasks.py.

boto3, cv2, and the SAM2 predictor are mocked — no real S3, video file, or
GPU/model weights are required. Redis publish calls are mocked too.
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import numpy as np
import pytest
from fastapi import HTTPException, status
from starlette.testclient import TestClient

import pipeline.pool.model_pool as pool_module


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    """Replace async Redis helpers with recording no-ops so no real Redis is needed."""
    published: list[dict] = []

    async def noop_publish(job_id: str, data: dict) -> None:
        published.append(data)

    async def noop_status(job_id: str, status_: str, error: str) -> None:
        pass

    monkeypatch.setattr("workers.segmentation.tasks._publish", noop_publish)
    monkeypatch.setattr("workers.segmentation.tasks._update_status", noop_status)
    return published


@pytest.fixture(autouse=True)
def _reset_model_pool():
    """Prevent a mocked SAM2 model registered in one test from leaking to the next."""
    original = pool_module._pool_instance
    pool_module._pool_instance = None
    yield
    pool_module._pool_instance = original


@pytest.fixture()
def app():
    from workers.celery_app import celery_app
    return celery_app


@pytest.fixture()
def mock_s3(monkeypatch):
    """Patch boto3.client so download_file/upload_file never touch real MinIO."""
    s3 = mock.MagicMock()
    monkeypatch.setattr("boto3.client", mock.MagicMock(return_value=s3))
    return s3


@pytest.fixture()
def mock_frame(monkeypatch):
    """Patch cv2 so VideoCapture returns a fixed fake frame instead of reading a real video."""
    frame_bgr = np.zeros((64, 64, 3), dtype=np.uint8)

    cap = mock.MagicMock()
    cap.read.return_value = (True, frame_bgr)
    monkeypatch.setattr("cv2.VideoCapture", mock.MagicMock(return_value=cap))
    monkeypatch.setattr("cv2.cvtColor", lambda img, code: img)
    return frame_bgr


@pytest.fixture()
def mock_sam2(monkeypatch):
    """Inject fake sam2 submodules into sys.modules so no real sam2/GPU/weights are needed.

    The real sam2 package resolves its import path relative to the installed
    package location and refuses to load if it detects certain repo-shadowing
    layouts (unrelated to this project's code). Patching via sys.modules
    sidesteps that entirely and keeps the test hermetic either way.
    """
    fake_model = mock.MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.half.return_value = fake_model

    fake_predictor = mock.MagicMock()
    fake_predictor.predict.return_value = (
        np.ones((1, 64, 64), dtype=bool),
        np.array([0.95]),
        None,
    )

    build_sam_module = types.ModuleType("sam2.build_sam")
    build_sam_module.build_sam2 = mock.MagicMock(return_value=fake_model)

    predictor_module = types.ModuleType("sam2.sam2_image_predictor")
    predictor_module.SAM2ImagePredictor = mock.MagicMock(return_value=fake_predictor)

    sam2_pkg = types.ModuleType("sam2")
    sam2_pkg.__path__ = []

    monkeypatch.setitem(sys.modules, "sam2", sam2_pkg)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_sam_module)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_module)

    return fake_predictor


@pytest.fixture()
def mock_video(monkeypatch):
    """Patch cv2 so VideoCapture yields a finite frame sequence for track_masks.

    Unlike ``mock_frame`` (single seek), this returns three frames then EOF so
    the ``while True: cap.read()`` loop in _extract_all_frames terminates.
    """
    frame_bgr = np.zeros((64, 64, 3), dtype=np.uint8)

    cap = mock.MagicMock()
    cap.read.side_effect = [
        (True, frame_bgr),
        (True, frame_bgr),
        (True, frame_bgr),
        (False, None),
    ]
    monkeypatch.setattr("cv2.VideoCapture", mock.MagicMock(return_value=cap))
    monkeypatch.setattr("cv2.cvtColor", lambda img, code: img)
    return frame_bgr


@pytest.fixture()
def mock_mask_load(monkeypatch):
    """Patch np.load so the (mocked, never-downloaded) mask file is not read from disk."""
    fake_mask = np.full((64, 64), 255, dtype=np.uint8)
    monkeypatch.setattr(np, "load", lambda path: fake_mask)
    return fake_mask


@pytest.fixture()
def mock_xmem(monkeypatch):
    """Inject fake XMem submodules into sys.modules so no real XMem/GPU/weights are needed.

    XMem lives at /home/kavish/XMem and is added to sys.path at call time, not
    installed as a package. Patching sys.modules mirrors how ``mock_sam2`` keeps
    the test hermetic without touching the filesystem or a GPU.
    """
    fake_network = mock.MagicMock()
    fake_network.cuda.return_value = fake_network
    fake_network.eval.return_value = fake_network

    fake_processor = mock.MagicMock()
    # output_prob[0].cpu().numpy() must be a real ndarray so the > 0.5 /
    # astype / np.stack chain in _run_xmem_tracking produces genuine masks.
    out0 = mock.MagicMock()
    out0.cpu.return_value.numpy.return_value = np.ones((64, 64), dtype=np.float32)
    output_prob = mock.MagicMock()
    output_prob.__getitem__.return_value = out0
    fake_processor.step.return_value = output_prob

    network_module = types.ModuleType("model.network")
    network_module.XMem = mock.MagicMock(return_value=fake_network)
    model_pkg = types.ModuleType("model")
    model_pkg.__path__ = []

    core_module = types.ModuleType("inference.inference_core")
    core_module.InferenceCore = mock.MagicMock(return_value=fake_processor)

    utils_module = types.ModuleType("inference.interact.interactive_utils")
    utils_module.image_to_torch = mock.MagicMock(return_value=mock.MagicMock())
    utils_module.index_numpy_to_one_hot_torch = mock.MagicMock(
        return_value=mock.MagicMock()
    )

    inference_pkg = types.ModuleType("inference")
    inference_pkg.__path__ = []
    interact_pkg = types.ModuleType("inference.interact")
    interact_pkg.__path__ = []

    monkeypatch.setitem(sys.modules, "model", model_pkg)
    monkeypatch.setitem(sys.modules, "model.network", network_module)
    monkeypatch.setitem(sys.modules, "inference", inference_pkg)
    monkeypatch.setitem(sys.modules, "inference.inference_core", core_module)
    monkeypatch.setitem(sys.modules, "inference.interact", interact_pkg)
    monkeypatch.setitem(
        sys.modules, "inference.interact.interactive_utils", utils_module
    )

    return fake_processor


VALID_MASK_DATA = {"points": [{"x": 12.0, "y": 34.0}], "frame_index": 0}


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


# ── segment_first_frame real SAM2 path ────────────────────────────────────────


@pytest.mark.unit
class TestSegmentFirstFrameReal:
    def test_returns_status_real(self, mock_s3, mock_frame, mock_sam2):
        from workers.segmentation.tasks import segment_first_frame

        result = segment_first_frame.apply(
            args=["job-real", "jobs/job-real/segments/seg_0.mp4", VALID_MASK_DATA]
        ).get()

        assert result["status"] == "real"

    def test_returns_correct_dict_shape(self, mock_s3, mock_frame, mock_sam2):
        from workers.segmentation.tasks import segment_first_frame

        result = segment_first_frame.apply(
            args=["job-real", "jobs/job-real/segments/seg_0.mp4", VALID_MASK_DATA]
        ).get()

        assert set(result.keys()) == {
            "job_id",
            "segment_s3_key",
            "mask_s3_key",
            "status",
        }

    def test_mask_s3_key_matches_expected_path(self, mock_s3, mock_frame, mock_sam2):
        from workers.segmentation.tasks import segment_first_frame

        result = segment_first_frame.apply(
            args=["job-real", "jobs/job-real/segments/seg_0.mp4", VALID_MASK_DATA]
        ).get()

        assert result["mask_s3_key"] == "jobs/job-real/masks/frame_0_mask.npy"

    def test_downloads_segment_and_uploads_mask(self, mock_s3, mock_frame, mock_sam2):
        from workers.segmentation.tasks import segment_first_frame

        segment_first_frame.apply(
            args=["job-real", "jobs/job-real/segments/seg_0.mp4", VALID_MASK_DATA]
        ).get()

        assert mock_s3.download_file.called
        assert mock_s3.upload_file.called

    def test_publishes_progress_at_0_50_100(
        self, mock_s3, mock_frame, mock_sam2, mock_redis
    ):
        from workers.segmentation.tasks import segment_first_frame

        segment_first_frame.apply(
            args=["job-progress", "jobs/job-progress/segments/seg_0.mp4", VALID_MASK_DATA]
        ).get()

        pct_values = [d["pct"] for d in mock_redis if d.get("stage") == "segmenting"]
        assert pct_values == [0, 50, 100]


# ── segment_first_frame empty-points handling ─────────────────────────────────


@pytest.mark.unit
class TestSegmentFirstFrameEmptyPoints:
    def test_empty_points_list_returns_empty_status(self, mock_s3, mock_frame):
        from workers.segmentation.tasks import segment_first_frame

        result = segment_first_frame.apply(
            args=["job-empty", "jobs/job-empty/segments/seg_0.mp4", {"points": []}]
        ).get()

        assert result["status"] == "empty"

    def test_missing_points_key_does_not_crash(self, mock_s3, mock_frame):
        from workers.segmentation.tasks import segment_first_frame

        result = segment_first_frame.apply(
            args=["job-empty2", "jobs/job-empty2/segments/seg_0.mp4", {}]
        ).get()

        assert result["status"] == "empty"
        assert result["mask_s3_key"] == "jobs/job-empty2/masks/frame_0_mask.npy"

    def test_empty_points_still_publishes_progress_at_0_50_100(
        self, mock_s3, mock_frame, mock_redis
    ):
        from workers.segmentation.tasks import segment_first_frame

        segment_first_frame.apply(
            args=["job-empty3", "jobs/job-empty3/segments/seg_0.mp4", {"points": []}]
        ).get()

        pct_values = [d["pct"] for d in mock_redis if d.get("stage") == "segmenting"]
        assert pct_values == [0, 50, 100]


# ── track_masks (real XMem) ────────────────────────────────────────────────────


@pytest.mark.unit
def test_track_masks_is_registered(app):
    """track_masks must be discoverable in the Celery task registry."""
    assert "workers.segmentation.tasks.track_masks" in app.tasks


@pytest.mark.unit
def test_track_masks_queue(app):
    """track_masks must be routed to the segmentation queue."""
    task = app.tasks["workers.segmentation.tasks.track_masks"]
    assert task.queue == "segmentation"


@pytest.mark.unit
class TestTrackMasksReal:
    ARGS = [
        "job-abc",
        "jobs/job-abc/segments/seg_0.mp4",
        "jobs/job-abc/masks/frame_0_mask.npy",
    ]

    def test_returns_status_real(self, mock_s3, mock_video, mock_mask_load, mock_xmem):
        from workers.segmentation.tasks import track_masks

        result = track_masks.apply(args=self.ARGS).get()
        assert result["status"] == "real"

    def test_returns_correct_dict_shape(
        self, mock_s3, mock_video, mock_mask_load, mock_xmem
    ):
        from workers.segmentation.tasks import track_masks

        result = track_masks.apply(args=self.ARGS).get()
        assert set(result.keys()) == {
            "job_id",
            "segment_s3_key",
            "tracked_masks_s3_key",
            "status",
            "num_frames",
        }

    def test_tracked_masks_s3_key_matches_expected_path(
        self, mock_s3, mock_video, mock_mask_load, mock_xmem
    ):
        from workers.segmentation.tasks import track_masks

        result = track_masks.apply(args=self.ARGS).get()
        assert result["tracked_masks_s3_key"] == "jobs/job-abc/masks/tracked_masks.npy"

    def test_has_job_id_and_frame_count(
        self, mock_s3, mock_video, mock_mask_load, mock_xmem
    ):
        from workers.segmentation.tasks import track_masks

        result = track_masks.apply(args=self.ARGS).get()
        assert result["job_id"] == "job-abc"
        # mock_video yields three frames before EOF
        assert result["num_frames"] == 3

    def test_downloads_video_and_mask_then_uploads_tracked(
        self, mock_s3, mock_video, mock_mask_load, mock_xmem
    ):
        from workers.segmentation.tasks import track_masks

        track_masks.apply(args=self.ARGS).get()
        # both the segment and the SAM2 mask are pulled from MinIO
        assert mock_s3.download_file.call_count == 2
        assert mock_s3.upload_file.called

    def test_publishes_progress_at_0_50_100(
        self, mock_s3, mock_video, mock_mask_load, mock_xmem, mock_redis
    ):
        from workers.segmentation.tasks import track_masks

        track_masks.apply(args=self.ARGS).get()
        pct_values = [d["pct"] for d in mock_redis if d.get("stage") == "tracking"]
        assert pct_values == [0, 50, 100]


# ── Auth regression: job creation (which enqueues segment_first_frame) ────────


@pytest.fixture()
def no_auth_client():
    """TestClient where get_current_user always raises 401 (unauthenticated).

    Mirrors the fixture in tests/unit/test_jobs_router.py — this is a
    regression check that wiring real SAM2 into segment_first_frame did not
    weaken auth on the endpoint that enqueues it.
    """
    from api.core.auth import get_current_user
    from api.core.redis import get_redis
    from api.main import app

    async def fake_get_redis():
        yield mock.AsyncMock()

    async def raise_401():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    app.dependency_overrides[get_redis] = fake_get_redis
    app.dependency_overrides[get_current_user] = raise_401

    with (
        mock.patch("api.core.redis.init_redis_pool"),
        mock.patch("api.core.redis.close_redis_pool", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.init_db", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.close_db", new_callable=mock.AsyncMock),
    ):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.pop(get_redis, None)
    app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.unit
def test_unauthenticated_job_creation_returns_401(no_auth_client: TestClient):
    """A request without a valid Bearer token must still be rejected with 401."""
    resp = no_auth_client.post(
        "/api/v1/jobs/",
        json={
            "video_s3_key": "uploads/abc123/video.mp4",
            "mask": {"points": [{"x": 100.0, "y": 200.0}], "frame_index": 0},
        },
    )
    assert resp.status_code == 401
