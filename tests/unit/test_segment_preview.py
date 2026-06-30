"""Unit tests for POST /api/v1/segment/preview.

All external dependencies (Redis, DB) are mocked — no real connections.
"""
from __future__ import annotations

import uuid
from unittest import mock

import pytest
from starlette.testclient import TestClient

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
    "point": {"x": 200.0, "y": 150.0},
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


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSegmentPreviewUnauthenticated:
    def test_no_token_returns_401(self, unauthed_client: TestClient):
        """Requests without a Bearer token must be rejected with 401."""
        resp = unauthed_client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        assert resp.status_code == 401


@pytest.mark.unit
class TestSegmentPreviewAuthenticated:
    def test_valid_point_returns_200(self, client: TestClient):
        """A valid preview request from an authenticated user must return 200."""
        resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_mask_points(self, client: TestClient):
        """Response must include a non-empty mask_points list."""
        resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        body = resp.json()
        assert "mask_points" in body
        assert len(body["mask_points"]) > 0

    def test_stub_flag_is_true(self, client: TestClient):
        """stub=True must always be set so the frontend can show the preview warning."""
        resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        assert resp.json()["stub"] is True

    def test_mask_points_centered_on_input(self, client: TestClient):
        """Placeholder circle centroid must be within 1px of the clicked point.

        A circle sampled at N equal angles sums to exactly (cx, cy) because
        sum(cos(k * 2π/N)) = 0 for all N ≥ 1. This test uses a generous ±2px
        tolerance to guard against floating-point drift.
        """
        resp = client.post("/api/v1/segment/preview", json=VALID_PREVIEW_PAYLOAD)
        points = resp.json()["mask_points"]
        avg_x = sum(p["x"] for p in points) / len(points)
        avg_y = sum(p["y"] for p in points) / len(points)
        assert abs(avg_x - VALID_PREVIEW_PAYLOAD["point"]["x"]) < 2.0
        assert abs(avg_y - VALID_PREVIEW_PAYLOAD["point"]["y"]) < 2.0

    def test_missing_video_s3_key_returns_422(self, client: TestClient):
        """Omitting video_s3_key must fail Pydantic validation with 422."""
        payload = {k: v for k, v in VALID_PREVIEW_PAYLOAD.items() if k != "video_s3_key"}
        resp = client.post("/api/v1/segment/preview", json=payload)
        assert resp.status_code == 422

    def test_missing_point_returns_422(self, client: TestClient):
        """Omitting point must fail Pydantic validation with 422."""
        payload = {k: v for k, v in VALID_PREVIEW_PAYLOAD.items() if k != "point"}
        resp = client.post("/api/v1/segment/preview", json=payload)
        assert resp.status_code == 422
