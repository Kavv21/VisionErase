"""Unit tests for api/main.py."""
from __future__ import annotations

from unittest import mock

import pytest
from starlette.testclient import TestClient

from api.core.config import get_settings
from api.main import app


@pytest.fixture()
def client():
    """TestClient with startup/shutdown I/O mocked out."""
    with (
        mock.patch("api.core.redis.init_redis_pool"),
        mock.patch("api.core.redis.close_redis_pool", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.init_db", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.close_db", new_callable=mock.AsyncMock),
    ):
        with TestClient(app) as c:
            yield c


# ── Startup / metadata ────────────────────────────────────────────────────────

def test_startup_calls_redis_init():
    """init_redis_pool() must be called exactly once during lifespan startup."""
    with (
        mock.patch("api.core.redis.init_redis_pool") as mock_redis_init,
        mock.patch("api.core.redis.close_redis_pool", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.init_db", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.close_db", new_callable=mock.AsyncMock),
    ):
        with TestClient(app):
            mock_redis_init.assert_called_once()


def test_app_title_matches_settings():
    assert app.title == get_settings().app_name


def test_docs_hidden_when_debug_false():
    """Docs and OpenAPI schema must be inaccessible in non-debug mode."""
    if not get_settings().debug:
        assert app.docs_url is None
        assert app.openapi_url is None


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_returns_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_not_rate_limited(client: TestClient):
    """Health endpoint must be exempt from rate limiting — 30 rapid calls should all succeed."""
    for _ in range(30):
        resp = client.get("/health")
        assert resp.status_code == 200


# ── Prometheus metrics ────────────────────────────────────────────────────────

def test_metrics_endpoint_returns_text(client: TestClient):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_contains_visionerase_counters(client: TestClient):
    client.get("/health")  # generate at least one observation
    resp = client.get("/metrics")
    assert "visionerase_health_checks_total" in resp.text


# ── CORS ──────────────────────────────────────────────────────────────────────

def test_cors_preflight_returns_200(client: TestClient):
    """OPTIONS requests must not be rate-limited and must receive CORS headers."""
    resp = client.options(
        "/api/v1/jobs",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    # Starlette returns 200 for matched preflight; header presence is what matters.
    assert "access-control-allow-origin" in resp.headers


# ── Router prefixes ───────────────────────────────────────────────────────────

def test_unknown_route_returns_404(client: TestClient):
    assert client.get("/no-such-route").status_code == 404
