"""Unit tests for auth endpoints and helpers.

All DB/Redis/external dependencies are mocked — no real connections.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
from jose import jwt
from starlette.testclient import TestClient

from api.core.auth import (
    _ALGORITHM,
    create_access_token,
    decode_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from api.core.config import get_settings
from api.core.database import get_db
from api.core.redis import get_redis
from api.main import app
from api.models.user import User

settings = get_settings()

# ── Helpers ───────────────────────────────────────────────────────────────────

_USER_ID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
_USER_EMAIL = "user@example.com"

def _make_user(**kwargs) -> User:
    defaults = dict(
        id=_USER_ID,
        email=_USER_EMAIL,
        hashed_password=hash_password("correctpassword"),
        display_name="Test User",
        google_sub=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return User(**defaults)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """TestClient with no real infra. DB session is mocked per-test via patch."""
    async def fake_get_redis():
        yield mock.AsyncMock()

    app.dependency_overrides[get_redis] = fake_get_redis

    with (
        mock.patch("api.core.redis.init_redis_pool"),
        mock.patch("api.core.redis.close_redis_pool", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.init_db", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.close_db", new_callable=mock.AsyncMock),
    ):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.pop(get_redis, None)
    app.dependency_overrides.clear()


def _mock_db_session(user_returned_by_execute=None):
    """Return a context manager that overrides get_db with a mock session."""
    mock_result = mock.MagicMock()
    mock_result.scalar_one_or_none.return_value = user_returned_by_execute

    mock_session = mock.AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.commit = mock.AsyncMock()
    mock_session.refresh = mock.AsyncMock()
    mock_session.add = mock.MagicMock()

    async def fake_get_db():
        yield mock_session

    return fake_get_db, mock_session


# ── Password helpers ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPasswordHelpers:
    def test_hash_and_verify_roundtrip(self):
        hashed = hash_password("mysecret")
        assert verify_password("mysecret", hashed) is True

    def test_wrong_password_fails(self):
        hashed = hash_password("mysecret")
        assert verify_password("wrongsecret", hashed) is False


# ── JWT helpers ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestJWT:
    def test_create_and_decode(self):
        token = create_access_token(_USER_ID, _USER_EMAIL)
        payload = decode_access_token(token)
        assert payload["sub"] == str(_USER_ID)
        assert payload["email"] == _USER_EMAIL

    def test_expired_token_raises(self):
        expire = datetime.now(timezone.utc) - timedelta(seconds=1)
        payload = {"sub": str(_USER_ID), "email": _USER_EMAIL, "exp": expire}
        token = jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)
        from jose import JWTError
        with pytest.raises(JWTError):
            decode_access_token(token)

    def test_tampered_token_raises(self):
        token = create_access_token(_USER_ID, _USER_EMAIL)
        bad = token[:-4] + "XXXX"
        from jose import JWTError
        with pytest.raises(JWTError):
            decode_access_token(bad)


# ── POST /api/v1/auth/register ────────────────────────────────────────────────

@pytest.mark.unit
class TestRegister:
    def test_new_email_returns_201_with_token(self, client: TestClient):
        fake_db, mock_session = _mock_db_session(user_returned_by_execute=None)

        async def _refresh(obj):
            obj.id = _USER_ID
            obj.email = _USER_EMAIL
            obj.display_name = "Alice"

        mock_session.refresh.side_effect = _refresh
        app.dependency_overrides[get_db] = fake_db

        resp = client.post(
            "/api/v1/auth/register",
            json={"email": "alice@example.com", "password": "securepass", "display_name": "Alice"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_duplicate_email_returns_409(self, client: TestClient):
        existing = _make_user()
        fake_db, _ = _mock_db_session(user_returned_by_execute=existing)
        app.dependency_overrides[get_db] = fake_db

        resp = client.post(
            "/api/v1/auth/register",
            json={"email": _USER_EMAIL, "password": "securepass", "display_name": "Alice"},
        )
        assert resp.status_code == 409

    def test_short_password_returns_422(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/register",
            json={"email": "x@x.com", "password": "short", "display_name": "X"},
        )
        assert resp.status_code == 422


# ── POST /api/v1/auth/login ───────────────────────────────────────────────────

@pytest.mark.unit
class TestLogin:
    def test_correct_credentials_returns_200_with_token(self, client: TestClient):
        user = _make_user()
        fake_db, _ = _mock_db_session(user_returned_by_execute=user)
        app.dependency_overrides[get_db] = fake_db

        resp = client.post(
            "/api/v1/auth/login",
            json={"email": _USER_EMAIL, "password": "correctpassword"},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_wrong_password_returns_401(self, client: TestClient):
        user = _make_user()
        fake_db, _ = _mock_db_session(user_returned_by_execute=user)
        app.dependency_overrides[get_db] = fake_db

        resp = client.post(
            "/api/v1/auth/login",
            json={"email": _USER_EMAIL, "password": "wrongpassword"},
        )
        assert resp.status_code == 401

    def test_nonexistent_email_returns_401(self, client: TestClient):
        fake_db, _ = _mock_db_session(user_returned_by_execute=None)
        app.dependency_overrides[get_db] = fake_db

        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "password123"},
        )
        assert resp.status_code == 401


# ── GET /api/v1/auth/me ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestGetMe:
    def test_valid_token_returns_user(self, client: TestClient):
        user = _make_user()
        fake_db, _ = _mock_db_session(user_returned_by_execute=user)
        app.dependency_overrides[get_db] = fake_db

        token = create_access_token(user.id, user.email)
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == _USER_EMAIL
        assert body["id"] == str(_USER_ID)

    def test_invalid_token_returns_401(self, client: TestClient):
        resp = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer bad.token.here"})
        assert resp.status_code == 401

    def test_expired_token_returns_401(self, client: TestClient):
        expire = datetime.now(timezone.utc) - timedelta(seconds=1)
        payload = {"sub": str(_USER_ID), "email": _USER_EMAIL, "exp": expire}
        token = jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_missing_auth_header_returns_401(self, client: TestClient):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401


# ── GET /api/v1/auth/google ───────────────────────────────────────────────────

@pytest.mark.unit
class TestGoogleOAuth:
    def test_google_oauth_returns_503_when_unconfigured(self, client: TestClient):
        with mock.patch.object(settings.__class__, "google_client_id", new=None, create=True):
            with mock.patch("api.routers.auth.settings") as mock_settings:
                mock_settings.google_client_id = None
                resp = client.get("/api/v1/auth/google", follow_redirects=False)
        assert resp.status_code == 503


# ── Auth gate on POST /api/v1/jobs/ ──────────────────────────────────────────

@pytest.mark.unit
class TestJobsAuthGate:
    VALID_JOB_PAYLOAD = {
        "video_s3_key": "uploads/abc/video.mp4",
        "mask": {"points": [{"x": 10.0, "y": 20.0}], "frame_index": 0},
    }

    def test_no_auth_header_returns_401(self, client: TestClient):
        # Remove any auth override so real dependency runs
        app.dependency_overrides.pop(get_current_user, None)
        resp = client.post("/api/v1/jobs/", json=self.VALID_JOB_PAYLOAD)
        assert resp.status_code == 401

    def test_valid_token_passes_auth_gate(self, client: TestClient):
        user = _make_user()
        fake_db, _ = _mock_db_session(user_returned_by_execute=user)
        app.dependency_overrides[get_db] = fake_db
        # Remove any existing auth override so the real dependency runs
        app.dependency_overrides.pop(get_current_user, None)

        token = create_access_token(user.id, user.email)

        with (
            mock.patch("api.routers.jobs.get_cached_result", new_callable=mock.AsyncMock, return_value=None),
            mock.patch("api.routers.jobs.set_job_status", new_callable=mock.AsyncMock),
            mock.patch("api.routers.jobs.enqueue_job", new_callable=mock.AsyncMock),
        ):
            resp = client.post(
                "/api/v1/jobs/",
                json=self.VALID_JOB_PAYLOAD,
                headers={"Authorization": f"Bearer {token}"},
            )
        # 202 means auth passed and job was accepted
        assert resp.status_code == 202
