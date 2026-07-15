"""Unit tests for POST /api/v1/segment/preview.

All external dependencies (Redis, DB, S3, SAM2/GPU) are mocked — no real
connections, no real model weights.
"""
from __future__ import annotations

import base64
import sys
import types
import uuid
from unittest import mock

import numpy as np
import pytest
from starlette.testclient import TestClient

import pipeline.pool.model_pool as pool_module
from api.core.auth import get_current_user
from api.core.database import get_db
from api.core.redis import get_redis
from api.main import app
from api.models.user import User

_FAKE_USER = User(
    id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
    email="test@example.com",
    display_name="Test User",
)

VALID_PREVIEW_PAYLOAD = {
    "video_s3_key": "uploads/abc123/video.mp4",
    "points": [{"x": 200.0, "y": 150.0, "label": 1}],
    "frame_index": 0,
}

_INFRA_PATCHES = (
    mock.patch("api.core.redis.init_redis_pool"),
    mock.patch("api.core.redis.close_redis_pool", new_callable=mock.AsyncMock),
    mock.patch("api.core.database.init_db", new_callable=mock.AsyncMock),
    mock.patch("api.core.database.close_db", new_callable=mock.AsyncMock),
)


@pytest.fixture
def client():
    """Authenticated TestClient — get_current_user returns _FAKE_USER."""
    async def fake_get_redis():
        yield mock.AsyncMock()

    async def fake_get_current_user():
        return _FAKE_USER

    app.dependency_overrides[get_redis] = fake_get_redis
    app.dependency_overrides[get_current_user] = fake_get_current_user

    with _INFRA_PATCHES[0], _INFRA_PATCHES[1], _INFRA_PATCHES[2], _INFRA_PATCHES[3]:
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.pop(get_redis, None)
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def unauthed_client():
    """TestClient without any auth override — real get_current_user runs."""
    async def fake_get_redis():
        yield mock.AsyncMock()

    async def fake_get_db():
        yield mock.AsyncMock()

    app.dependency_overrides[get_redis] = fake_get_redis
    app.dependency_overrides[get_db] = fake_get_db

    with _INFRA_PATCHES[0], _INFRA_PATCHES[1], _INFRA_PATCHES[2], _INFRA_PATCHES[3]:
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.pop(get_redis, None)
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def _reset_model_pool():
    """Prevent a mocked SAM2 model registered in one test from leaking to the next."""
    original = pool_module._pool_instance
    pool_module._pool_instance = None
    yield
    pool_module._pool_instance = original


@pytest.fixture()
def mock_s3(monkeypatch):
    """Patch boto3.client so download_file never touches real MinIO."""
    s3 = mock.MagicMock()
    monkeypatch.setattr("boto3.client", mock.MagicMock(return_value=s3))
    return s3


@pytest.fixture()
def mock_frame(monkeypatch):
    """Patch cv2 so VideoCapture/imencode never touch a real video file."""
    frame_bgr = np.zeros((64, 64, 3), dtype=np.uint8)

    cap = mock.MagicMock()
    cap.read.return_value = (True, frame_bgr)
    monkeypatch.setattr("cv2.VideoCapture", mock.MagicMock(return_value=cap))
    monkeypatch.setattr("cv2.cvtColor", lambda img, code: img)
    monkeypatch.setattr(
        "cv2.imencode", lambda ext, arr: (True, np.frombuffer(b"fake-png-bytes", dtype=np.uint8))
    )
    return frame_bgr


@pytest.fixture()
def mock_sam2(monkeypatch):
    """Inject fake sam2 submodules into sys.modules so no real sam2/GPU/weights are needed.

    Mirrors the mock_sam2 fixture in test_segmentation_tasks.py — patching via
    sys.modules keeps the test hermetic regardless of whether the real sam2
    package is importable in this environment.
    """
    fake_model = mock.MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.half.return_value = fake_model

    fake_predictor = mock.MagicMock()
    fake_predictor.predict.return_value = (
        np.ones((3, 64, 64), dtype=bool),
        np.array([0.62, 0.94, 0.71]),
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


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSegmentPreviewUnauthenticated:
    def test_no_token_returns_401(self, unauthed_client: TestClient):
        """Requests without a Bearer token must be rejected with 401."""
        resp = unauthed_client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        assert resp.status_code == 401


@pytest.mark.unit
class TestSegmentPreviewAuthenticated:
    def test_valid_points_returns_200(self, client: TestClient, mock_s3, mock_frame, mock_sam2):
        """A valid preview request from an authenticated user must return 200."""
        resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_mask_base64(self, client: TestClient, mock_s3, mock_frame, mock_sam2):
        """Response must include a non-empty base64-encoded mask."""
        resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        body = resp.json()
        assert "mask_base64" in body
        assert len(body["mask_base64"]) > 0
        base64.b64decode(body["mask_base64"])  # must be valid base64

    def test_stub_flag_is_false(self, client: TestClient, mock_s3, mock_frame, mock_sam2):
        """stub=False must always be set now that real SAM2 runs."""
        resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        assert resp.json()["stub"] is False

    def test_score_picks_best_of_multimask_output(
        self, client: TestClient, mock_s3, mock_frame, mock_sam2
    ):
        """The highest-scoring of the three multimask outputs must be returned."""
        resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        assert resp.json()["score"] == pytest.approx(0.94)

    def test_response_has_width_and_height(self, client: TestClient, mock_s3, mock_frame, mock_sam2):
        resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        body = resp.json()
        assert body["width"] == 64
        assert body["height"] == 64

    def test_multiple_points_with_labels_passed_to_sam2(
        self, client: TestClient, mock_s3, mock_frame, mock_sam2
    ):
        """Positive and negative points must both reach predictor.predict()."""
        payload = {
            "video_s3_key": "uploads/abc123/video.mp4",
            "points": [
                {"x": 200.0, "y": 150.0, "label": 1},
                {"x": 10.0, "y": 20.0, "label": -1},
            ],
            "frame_index": 0,
        }
        resp = client.post("/api/v1/segment/preview", json=payload)
        assert resp.status_code == 200

        _, kwargs = mock_sam2.predict.call_args
        np.testing.assert_array_equal(kwargs["point_labels"], np.array([1, -1]))
        assert kwargs["point_coords"].shape == (2, 2)

    def test_missing_video_s3_key_returns_422(self, client: TestClient):
        """Omitting video_s3_key must fail Pydantic validation with 422."""
        payload = {k: v for k, v in VALID_PREVIEW_PAYLOAD.items() if k != "video_s3_key"}
        resp = client.post("/api/v1/segment/preview", json=payload)
        assert resp.status_code == 422

    def test_missing_points_returns_422(self, client: TestClient):
        """Omitting points must fail Pydantic validation with 422."""
        payload = {k: v for k, v in VALID_PREVIEW_PAYLOAD.items() if k != "points"}
        resp = client.post("/api/v1/segment/preview", json=payload)
        assert resp.status_code == 422

    def test_empty_points_returns_422(self, client: TestClient):
        """An empty points list must be rejected before any SAM2 work happens."""
        payload = {**VALID_PREVIEW_PAYLOAD, "points": []}
        resp = client.post("/api/v1/segment/preview", json=payload)
        assert resp.status_code == 422

    def test_sam2_failure_returns_502(self, client: TestClient, mock_s3, mock_frame):
        """If SAM2/model loading raises, the endpoint must return 502, not 500."""
        with mock.patch(
            "workers.segmentation.tasks._load_sam2_model",
            side_effect=RuntimeError("weights missing"),
        ):
            resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        assert resp.status_code == 502
