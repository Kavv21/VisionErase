"""Unit tests for api/routers/jobs.py.

All external dependencies (Redis, database, S3) are mocked.
No real connections are made.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

import pytest
from starlette.testclient import TestClient

import uuid

from fastapi import HTTPException, status

from api.core.auth import get_current_user
from api.core.redis import get_redis
from api.main import app
from api.models.user import User

# ── Shared fixtures ───────────────────────────────────────────────────────────

VALID_JOB_PAYLOAD = {
    "video_s3_key": "uploads/abc123/video.mp4",
    "mask": {
        "points": [{"x": 100.0, "y": 200.0}, {"x": 150.0, "y": 250.0}],
        "frame_index": 0,
    },
}

VALID_UPLOAD_URL_PAYLOAD = {
    "filename": "my-video.mp4",
    "content_type": "video/mp4",
    "size_bytes": 10_000_000,
}

_FAKE_JOB_STATUS = {
    "job_id": "test-job-id-1234",
    "status": "pending",
    "progress_pct": 0.0,
    "created_at": "2024-01-15T10:00:00+00:00",
    "result_s3_key": None,
    "error": None,
}

_FAKE_CACHED_RESULT = {
    "job_id": "cached-job-id-5678",
    "created_at": "2024-01-14T08:00:00+00:00",
    "result_s3_key": "outputs/cached-job-id-5678/result.mp4",
}

_FAKE_UPLOAD_URL_RESULT = {
    "upload_url": "https://minio.example.com/bucket/uploads/uuid/my-video.mp4?sig=abc",
    "s3_key": "uploads/uuid/my-video.mp4",
    "expires_in": 3600,
}


_FAKE_USER = User(
    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
    email="test@example.com",
    display_name="Test User",
)


@pytest.fixture
def client():
    """TestClient with all infrastructure (Redis pool, DB, S3, Auth) mocked out.

    The get_redis and get_current_user dependencies are overridden so no
    real connections are needed.
    """
    async def fake_get_redis():
        yield mock.AsyncMock()

    async def fake_get_current_user():
        return _FAKE_USER

    app.dependency_overrides[get_redis] = fake_get_redis
    app.dependency_overrides[get_current_user] = fake_get_current_user

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


# ── POST /api/v1/jobs/ — job creation ─────────────────────────────────────────

@pytest.mark.unit
class TestCreateJob:
    def test_valid_payload_returns_202(self, client: TestClient):
        """A well-formed job submission must be enqueued and return 202 Accepted."""
        with (
            mock.patch("api.routers.jobs.get_cached_result", new_callable=mock.AsyncMock, return_value=None),
            mock.patch("api.routers.jobs.set_job_status", new_callable=mock.AsyncMock),
            mock.patch("api.routers.jobs.enqueue_job", new_callable=mock.AsyncMock),
        ):
            resp = client.post("/api/v1/jobs/", json=VALID_JOB_PAYLOAD)

        assert resp.status_code == 202

    def test_valid_payload_response_shape(self, client: TestClient):
        """Response for a new job must include job_id, status=pending, cached=False."""
        with (
            mock.patch("api.routers.jobs.get_cached_result", new_callable=mock.AsyncMock, return_value=None),
            mock.patch("api.routers.jobs.set_job_status", new_callable=mock.AsyncMock),
            mock.patch("api.routers.jobs.enqueue_job", new_callable=mock.AsyncMock),
        ):
            resp = client.post("/api/v1/jobs/", json=VALID_JOB_PAYLOAD)

        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "pending"
        assert body["cached"] is False
        assert body["progress_pct"] == 0.0

    def test_missing_video_s3_key_returns_422(self, client: TestClient):
        """Omitting video_s3_key must fail Pydantic validation before the handler runs."""
        payload = {k: v for k, v in VALID_JOB_PAYLOAD.items() if k != "video_s3_key"}
        resp = client.post("/api/v1/jobs/", json=payload)
        assert resp.status_code == 422

    def test_missing_mask_returns_422(self, client: TestClient):
        """Omitting mask must fail Pydantic validation before the handler runs."""
        payload = {k: v for k, v in VALID_JOB_PAYLOAD.items() if k != "mask"}
        resp = client.post("/api/v1/jobs/", json=payload)
        assert resp.status_code == 422

    def test_empty_mask_points_passes_validation(self, client: TestClient):
        """Pydantic v2 does not enforce min-length on list fields by default.

        An empty points list therefore reaches the handler and is enqueued as-is.
        Add a Field(min_length=1) to MaskData.points if this should be rejected.
        """
        payload = {**VALID_JOB_PAYLOAD, "mask": {"points": [], "frame_index": 0}}
        with (
            mock.patch("api.routers.jobs.get_cached_result", new_callable=mock.AsyncMock, return_value=None),
            mock.patch("api.routers.jobs.set_job_status", new_callable=mock.AsyncMock),
            mock.patch("api.routers.jobs.enqueue_job", new_callable=mock.AsyncMock),
        ):
            resp = client.post("/api/v1/jobs/", json=payload)
        assert resp.status_code == 202

    def test_invalid_priority_returns_422(self, client: TestClient):
        """An out-of-range priority value must be rejected with 422."""
        payload = {**VALID_JOB_PAYLOAD, "priority": 99}
        resp = client.post("/api/v1/jobs/", json=payload)
        assert resp.status_code == 422

    def test_duplicate_job_returns_200_with_cached_true(self, client: TestClient):
        """A job submitted with the same video+mask as an existing result must return 200 cached=True."""
        with mock.patch(
            "api.routers.jobs.get_cached_result",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_CACHED_RESULT,
        ):
            resp = client.post("/api/v1/jobs/", json=VALID_JOB_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert body["cached"] is True

    def test_duplicate_job_returns_original_job_id(self, client: TestClient):
        """A deduplicated response must return the original job_id from the cache."""
        with mock.patch(
            "api.routers.jobs.get_cached_result",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_CACHED_RESULT,
        ):
            resp = client.post("/api/v1/jobs/", json=VALID_JOB_PAYLOAD)

        assert resp.json()["job_id"] == _FAKE_CACHED_RESULT["job_id"]

    def test_duplicate_job_returns_completed_status(self, client: TestClient):
        """A cached job response must show status=completed since the work is done."""
        with mock.patch(
            "api.routers.jobs.get_cached_result",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_CACHED_RESULT,
        ):
            resp = client.post("/api/v1/jobs/", json=VALID_JOB_PAYLOAD)

        assert resp.json()["status"] == "completed"

    def test_new_job_enqueues_exactly_once(self, client: TestClient):
        """enqueue_job must be called exactly once per new job submission."""
        with (
            mock.patch("api.routers.jobs.get_cached_result", new_callable=mock.AsyncMock, return_value=None),
            mock.patch("api.routers.jobs.set_job_status", new_callable=mock.AsyncMock),
            mock.patch("api.routers.jobs.enqueue_job", new_callable=mock.AsyncMock) as mock_enqueue,
        ):
            client.post("/api/v1/jobs/", json=VALID_JOB_PAYLOAD)
            mock_enqueue.assert_called_once()

    def test_urgent_priority_accepted(self, client: TestClient):
        """Jobs submitted with priority=URGENT (0) must be accepted normally."""
        payload = {**VALID_JOB_PAYLOAD, "priority": 0}
        with (
            mock.patch("api.routers.jobs.get_cached_result", new_callable=mock.AsyncMock, return_value=None),
            mock.patch("api.routers.jobs.set_job_status", new_callable=mock.AsyncMock),
            mock.patch("api.routers.jobs.enqueue_job", new_callable=mock.AsyncMock),
        ):
            resp = client.post("/api/v1/jobs/", json=payload)
        assert resp.status_code == 202

    def test_optional_webhook_url_accepted(self, client: TestClient):
        """Jobs with an optional webhook_url field must be accepted without error."""
        payload = {**VALID_JOB_PAYLOAD, "webhook_url": "https://example.com/hook"}
        with (
            mock.patch("api.routers.jobs.get_cached_result", new_callable=mock.AsyncMock, return_value=None),
            mock.patch("api.routers.jobs.set_job_status", new_callable=mock.AsyncMock),
            mock.patch("api.routers.jobs.enqueue_job", new_callable=mock.AsyncMock),
        ):
            resp = client.post("/api/v1/jobs/", json=payload)
        assert resp.status_code == 202


# ── GET /api/v1/jobs/{job_id} — status lookup ─────────────────────────────────

@pytest.mark.unit
class TestGetJob:
    def test_existing_job_returns_200(self, client: TestClient):
        """GET /{job_id} must return 200 when the job exists in Redis."""
        with mock.patch(
            "api.routers.jobs.get_job_status",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_JOB_STATUS,
        ):
            resp = client.get("/api/v1/jobs/test-job-id-1234")
        assert resp.status_code == 200

    def test_existing_job_response_fields(self, client: TestClient):
        """The response body must include job_id, status, progress_pct, and created_at."""
        with mock.patch(
            "api.routers.jobs.get_job_status",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_JOB_STATUS,
        ):
            resp = client.get("/api/v1/jobs/test-job-id-1234")

        body = resp.json()
        assert body["job_id"] == _FAKE_JOB_STATUS["job_id"]
        assert body["status"] == _FAKE_JOB_STATUS["status"]
        assert body["progress_pct"] == _FAKE_JOB_STATUS["progress_pct"]
        assert "created_at" in body

    def test_unknown_job_returns_404(self, client: TestClient):
        """GET /{job_id} must return 404 when the job is not found in Redis."""
        with mock.patch(
            "api.routers.jobs.get_job_status",
            new_callable=mock.AsyncMock,
            return_value=None,
        ):
            resp = client.get("/api/v1/jobs/no-such-job")
        assert resp.status_code == 404

    def test_unknown_job_error_detail(self, client: TestClient):
        """The 404 response body must contain a human-readable 'detail' field."""
        with mock.patch(
            "api.routers.jobs.get_job_status",
            new_callable=mock.AsyncMock,
            return_value=None,
        ):
            resp = client.get("/api/v1/jobs/no-such-job")
        assert "detail" in resp.json()

    def test_completed_job_includes_result_key(self, client: TestClient):
        """A completed job with a result must expose result_s3_key in the response."""
        completed_status = {
            **_FAKE_JOB_STATUS,
            "status": "completed",
            "progress_pct": 100.0,
            "result_s3_key": "outputs/test-job-id-1234/result.mp4",
        }
        with mock.patch(
            "api.routers.jobs.get_job_status",
            new_callable=mock.AsyncMock,
            return_value=completed_status,
        ):
            resp = client.get("/api/v1/jobs/test-job-id-1234")

        body = resp.json()
        assert body["result_s3_key"] == "outputs/test-job-id-1234/result.mp4"
        assert body["status"] == "completed"

    def test_failed_job_includes_error_message(self, client: TestClient):
        """A failed job must expose the error field in the response."""
        failed_status = {
            **_FAKE_JOB_STATUS,
            "status": "failed",
            "error": "Segmentation model timed out",
        }
        with mock.patch(
            "api.routers.jobs.get_job_status",
            new_callable=mock.AsyncMock,
            return_value=failed_status,
        ):
            resp = client.get("/api/v1/jobs/test-job-id-1234")

        body = resp.json()
        assert body["status"] == "failed"
        assert body["error"] == "Segmentation model timed out"


# ── GET /api/v1/jobs/{job_id}/download — presigned result download URL ───────

@pytest.mark.unit
class TestDownloadJobResult:
    def test_unauthenticated_returns_401(self, no_auth_client: TestClient):
        """A request without a valid Bearer token must be rejected with 401."""
        resp = no_auth_client.get("/api/v1/jobs/test-job-id-1234/download")
        assert resp.status_code == 401

    def test_unknown_job_returns_404(self, client: TestClient):
        """GET /{job_id}/download must return 404 when the job is not found in Redis."""
        with mock.patch(
            "api.routers.jobs.get_job_status",
            new_callable=mock.AsyncMock,
            return_value=None,
        ):
            resp = client.get("/api/v1/jobs/no-such-job/download")
        assert resp.status_code == 404

    def test_incomplete_job_returns_400(self, client: TestClient):
        """A job that has not reached COMPLETED status must return 400."""
        with mock.patch(
            "api.routers.jobs.get_job_status",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_JOB_STATUS,
        ):
            resp = client.get("/api/v1/jobs/test-job-id-1234/download")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Job not complete yet"


# ── POST /api/v1/jobs/upload-url — presigned URL generation ───────────────────

@pytest.mark.unit
class TestUploadUrl:
    def test_valid_request_returns_200(self, client: TestClient):
        """POST /upload-url with a valid payload must return 200."""
        with mock.patch(
            "api.routers.jobs.generate_upload_url",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_UPLOAD_URL_RESULT,
        ):
            resp = client.post("/api/v1/jobs/upload-url", json=VALID_UPLOAD_URL_PAYLOAD)
        assert resp.status_code == 200

    def test_response_contains_upload_url(self, client: TestClient):
        """The response must include a non-empty upload_url string."""
        with mock.patch(
            "api.routers.jobs.generate_upload_url",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_UPLOAD_URL_RESULT,
        ):
            resp = client.post("/api/v1/jobs/upload-url", json=VALID_UPLOAD_URL_PAYLOAD)

        body = resp.json()
        assert "upload_url" in body
        assert isinstance(body["upload_url"], str)
        assert len(body["upload_url"]) > 0

    def test_response_contains_s3_key(self, client: TestClient):
        """The response must include an s3_key for the client to reference later."""
        with mock.patch(
            "api.routers.jobs.generate_upload_url",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_UPLOAD_URL_RESULT,
        ):
            resp = client.post("/api/v1/jobs/upload-url", json=VALID_UPLOAD_URL_PAYLOAD)

        body = resp.json()
        assert "s3_key" in body
        assert body["s3_key"] == _FAKE_UPLOAD_URL_RESULT["s3_key"]

    def test_response_contains_expires_in(self, client: TestClient):
        """The response must include expires_in so the client knows when the URL lapses."""
        with mock.patch(
            "api.routers.jobs.generate_upload_url",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_UPLOAD_URL_RESULT,
        ):
            resp = client.post("/api/v1/jobs/upload-url", json=VALID_UPLOAD_URL_PAYLOAD)

        assert resp.json()["expires_in"] == 3600

    def test_missing_filename_returns_422(self, client: TestClient):
        """Omitting filename must fail Pydantic validation with 422."""
        payload = {k: v for k, v in VALID_UPLOAD_URL_PAYLOAD.items() if k != "filename"}
        resp = client.post("/api/v1/jobs/upload-url", json=payload)
        assert resp.status_code == 422

    def test_missing_size_bytes_returns_422(self, client: TestClient):
        """Omitting size_bytes must fail Pydantic validation with 422."""
        payload = {k: v for k, v in VALID_UPLOAD_URL_PAYLOAD.items() if k != "size_bytes"}
        resp = client.post("/api/v1/jobs/upload-url", json=payload)
        assert resp.status_code == 422

    def test_default_content_type_is_mp4(self, client: TestClient):
        """When content_type is omitted, the default video/mp4 must be accepted."""
        payload = {"filename": "video.mp4", "size_bytes": 1024}
        with mock.patch(
            "api.routers.jobs.generate_upload_url",
            new_callable=mock.AsyncMock,
            return_value=_FAKE_UPLOAD_URL_RESULT,
        ):
            resp = client.post("/api/v1/jobs/upload-url", json=payload)
        assert resp.status_code == 200


# ── POST /api/v1/jobs/upload — direct multipart upload ───────────────────────

@pytest.fixture
def no_auth_client():
    """TestClient where get_current_user always raises 401 (unauthenticated)."""
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
class TestUploadVideo:
    def test_authenticated_upload_valid_file_returns_200_with_s3_key(
        self, client: TestClient
    ):
        """An authenticated upload of a valid video file must return 200 with s3_key."""
        mock_s3 = mock.MagicMock()
        with mock.patch("api.routers.jobs._s3_client", return_value=mock_s3):
            resp = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "s3_key" in body
        assert body["s3_key"].startswith("uploads/")
        assert body["s3_key"].endswith("/clip.mp4")
        assert "size_bytes" in body
        mock_s3.upload_fileobj.assert_called_once()

    def test_unauthenticated_upload_returns_401(self, no_auth_client: TestClient):
        """A request without a valid Bearer token must be rejected with 401."""
        resp = no_auth_client.post(
            "/api/v1/jobs/upload",
            files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        )
        assert resp.status_code == 401

    def test_wrong_content_type_returns_415(self, client: TestClient):
        """Uploading a non-video file type must be rejected with 415."""
        resp = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("doc.pdf", b"pdf bytes", "application/pdf")},
        )
        assert resp.status_code == 415
