"""Unit tests for api/routers/websocket.py.

All Redis I/O and pub/sub are mocked — no real connections are made.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest import mock

import pytest
from starlette.testclient import TestClient

from api.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_acquire_redis(redis_client):
    """Return an async context manager factory that always yields redis_client."""
    @asynccontextmanager
    async def _ctx():
        yield redis_client
    return _ctx


def _make_pubsub(messages: list[dict]) -> mock.MagicMock:
    """Return a mock pubsub whose listen() yields a subscribe ack then messages."""
    async def _listen():
        yield {"type": "subscribe", "data": 1}
        for msg in messages:
            yield {"type": "message", "data": json.dumps(msg)}

    pubsub = mock.MagicMock()
    pubsub.subscribe = mock.AsyncMock()
    pubsub.listen = _listen
    return pubsub


def _mock_redis_with_pubsub(messages: list[dict]) -> mock.AsyncMock:
    """Return an AsyncMock redis client whose .pubsub() returns a mock pubsub."""
    mock_redis = mock.AsyncMock()
    mock_redis.pubsub = mock.MagicMock(return_value=_make_pubsub(messages))
    return mock_redis


# ── Client fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """TestClient with all infrastructure (Redis pool, DB) mocked out."""
    with (
        mock.patch("api.core.redis.init_redis_pool"),
        mock.patch("api.core.redis.close_redis_pool", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.init_db", new_callable=mock.AsyncMock),
        mock.patch("api.core.database.close_db", new_callable=mock.AsyncMock),
    ):
        with TestClient(app) as c:
            yield c


# ── TestWebSocketJobNotFound ───────────────────────────────────────────────────

@pytest.mark.unit
class TestWebSocketJobNotFound:
    def test_unknown_job_id_sends_error_message(self, client: TestClient):
        """Connecting with an unknown job_id must receive type=error, message='job not found'."""
        mock_redis = mock.AsyncMock()

        with (
            mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
            mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=None),
        ):
            with client.websocket_connect("/ws/nonexistent-id") as ws:
                msg = ws.receive_json()

        assert msg["type"] == "error"
        assert msg["message"] == "job not found"

    def test_unknown_job_does_not_open_pubsub_channel(self, client: TestClient):
        """No pub/sub subscription must be opened when the job is not found."""
        mock_redis = mock.AsyncMock()
        mock_redis.pubsub = mock.MagicMock()

        with (
            mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
            mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=None),
        ):
            with client.websocket_connect("/ws/nonexistent-id") as ws:
                ws.receive_json()

        mock_redis.pubsub.assert_not_called()


# ── TestWebSocketCompletedJob ─────────────────────────────────────────────────

@pytest.mark.unit
class TestWebSocketCompletedJob:
    def test_completed_job_sends_complete_message_immediately(self, client: TestClient):
        """Connecting to a completed job must receive type=complete with result_s3_key."""
        mock_redis = mock.AsyncMock()
        job_status = {"status": "completed", "result_s3_key": "results/xyz.mp4"}

        with (
            mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
            mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=job_status),
        ):
            with client.websocket_connect("/ws/finished-job") as ws:
                msg = ws.receive_json()

        assert msg["type"] == "complete"
        assert msg["result"]["result_s3_key"] == "results/xyz.mp4"

    def test_completed_job_does_not_subscribe_to_pubsub(self, client: TestClient):
        """A completed-job connect must not open a pub/sub channel."""
        mock_redis = mock.AsyncMock()
        mock_redis.pubsub = mock.MagicMock()
        job_status = {"status": "completed", "result_s3_key": "results/xyz.mp4"}

        with (
            mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
            mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=job_status),
        ):
            with client.websocket_connect("/ws/finished-job") as ws:
                ws.receive_json()

        mock_redis.pubsub.assert_not_called()

    def test_failed_job_sends_error_message(self, client: TestClient):
        """Connecting to a failed job must receive type=error with the stored error message."""
        mock_redis = mock.AsyncMock()
        job_status = {"status": "failed", "error": "inpainting model OOM"}

        with (
            mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
            mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=job_status),
        ):
            with client.websocket_connect("/ws/failed-job") as ws:
                msg = ws.receive_json()

        assert msg["type"] == "error"
        assert msg["message"] == "inpainting model OOM"


# ── TestWebSocketProgressForwarding ───────────────────────────────────────────

@pytest.mark.unit
class TestWebSocketProgressForwarding:
    def test_initial_snapshot_sent_for_in_progress_job(self, client: TestClient):
        """Handler must send a progress snapshot immediately on connect for an in-progress job."""
        mock_redis = _mock_redis_with_pubsub([
            {"type": "complete", "result": {"result_s3_key": "results/done.mp4"}},
        ])
        job_status = {"status": "segmenting", "progress_pct": 15.0}

        with (
            mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
            mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=job_status),
        ):
            with client.websocket_connect("/ws/live-job") as ws:
                snapshot = ws.receive_json()
                ws.receive_json()  # consume the terminal event

        assert snapshot["type"] == "progress"
        assert snapshot["stage"] == "segmenting"
        assert snapshot["pct"] == 15.0

    def test_pubsub_progress_messages_are_forwarded(self, client: TestClient):
        """Progress events published on Redis must be forwarded verbatim to the client."""
        pubsub_events = [
            {"type": "progress", "stage": "inpainting", "pct": 60.0},
            {"type": "complete", "result": {"result_s3_key": "results/done.mp4"}},
        ]
        mock_redis = _mock_redis_with_pubsub(pubsub_events)
        job_status = {"status": "segmenting", "progress_pct": 10.0}

        with (
            mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
            mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=job_status),
        ):
            with client.websocket_connect("/ws/live-job") as ws:
                ws.receive_json()           # initial snapshot
                progress = ws.receive_json()
                complete = ws.receive_json()

        assert progress["type"] == "progress"
        assert progress["stage"] == "inpainting"
        assert progress["pct"] == 60.0
        assert complete["type"] == "complete"

    def test_connection_closes_after_complete_message(self, client: TestClient):
        """Handler must stop forwarding after the type=complete terminal event."""
        pubsub_events = [
            {"type": "complete", "result": {"result_s3_key": "results/done.mp4"}},
            # This message must never reach the client
            {"type": "progress", "stage": "should_not_arrive", "pct": 99.0},
        ]
        mock_redis = _mock_redis_with_pubsub(pubsub_events)
        job_status = {"status": "processing", "progress_pct": 50.0}
        received = []

        with (
            mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
            mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=job_status),
        ):
            with client.websocket_connect("/ws/closing-job") as ws:
                received.append(ws.receive_json())  # snapshot
                received.append(ws.receive_json())  # complete → handler breaks here

        stages = [m.get("stage") for m in received]
        assert "should_not_arrive" not in stages
        types = [m["type"] for m in received]
        assert "complete" in types

    def test_connection_closes_after_error_terminal_event(self, client: TestClient):
        """Handler must stop forwarding after a type=error event from pub/sub."""
        pubsub_events = [
            {"type": "error", "message": "worker crashed"},
            {"type": "progress", "stage": "should_not_arrive", "pct": 99.0},
        ]
        mock_redis = _mock_redis_with_pubsub(pubsub_events)
        job_status = {"status": "processing", "progress_pct": 20.0}
        received = []

        with (
            mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
            mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=job_status),
        ):
            with client.websocket_connect("/ws/error-job") as ws:
                received.append(ws.receive_json())  # snapshot
                received.append(ws.receive_json())  # error event → handler breaks

        stages = [m.get("stage") for m in received]
        assert "should_not_arrive" not in stages


# ── TestWebSocketGaugeTracking ────────────────────────────────────────────────

@pytest.mark.unit
class TestWebSocketGaugeTracking:
    def test_gauge_increments_on_connect(self, client: TestClient):
        """WEBSOCKET_CONNECTIONS_ACTIVE must be incremented when a client connects."""
        mock_redis = mock.AsyncMock()

        with mock.patch("api.routers.websocket.WEBSOCKET_CONNECTIONS_ACTIVE") as mock_gauge:
            with (
                mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
                mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=None),
            ):
                with client.websocket_connect("/ws/any-job") as ws:
                    ws.receive_json()

        mock_gauge.inc.assert_called_once()

    def test_gauge_decrements_on_disconnect(self, client: TestClient):
        """WEBSOCKET_CONNECTIONS_ACTIVE must be decremented after the connection closes."""
        mock_redis = mock.AsyncMock()

        with mock.patch("api.routers.websocket.WEBSOCKET_CONNECTIONS_ACTIVE") as mock_gauge:
            with (
                mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
                mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=None),
            ):
                with client.websocket_connect("/ws/any-job") as ws:
                    ws.receive_json()

        mock_gauge.dec.assert_called_once()

    def test_gauge_decrements_even_when_job_not_found(self, client: TestClient):
        """Gauge cleanup in the finally block must run regardless of early-return paths."""
        mock_redis = mock.AsyncMock()

        with mock.patch("api.routers.websocket.WEBSOCKET_CONNECTIONS_ACTIVE") as mock_gauge:
            with (
                mock.patch("api.routers.websocket.acquire_redis", _make_acquire_redis(mock_redis)),
                mock.patch("api.routers.websocket.get_job_status", new_callable=mock.AsyncMock, return_value=None),
            ):
                with client.websocket_connect("/ws/ghost-job") as ws:
                    ws.receive_json()

        mock_gauge.inc.assert_called_once()
        mock_gauge.dec.assert_called_once()
