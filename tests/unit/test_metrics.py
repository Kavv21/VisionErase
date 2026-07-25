"""Unit tests for api/core/metrics.py.

Verifies that every metric is importable, correctly named, and can be
observed/incremented/set without raising exceptions.
"""
from __future__ import annotations

import pytest

import api.core.metrics as m


# ── Import sanity ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMetricsImport:
    def test_all_metrics_are_importable(self):
        """Every exported metric object must be non-None after module import."""
        assert m.HTTP_REQUESTS_TOTAL is not None
        assert m.HTTP_REQUEST_DURATION_SECONDS is not None
        assert m.JOB_SUBMISSIONS_TOTAL is not None
        assert m.RATE_LIMIT_HITS_TOTAL is not None
        assert m.WEBSOCKET_CONNECTIONS_ACTIVE is not None
        assert m.HEALTH_CHECKS_TOTAL is not None
        assert m.JOB_STATUS_REQUESTS_TOTAL is not None
        assert m.UPLOAD_URL_REQUESTS_TOTAL is not None

    def test_metric_count_matches_expected(self):
        """Catches accidental removal of a metric — there must be exactly 34."""
        public = [v for k, v in vars(m).items() if k.isupper() and not k.startswith("_")]
        assert len(public) == 34, f"Expected 34 metrics, found {len(public)}: {[k for k in vars(m) if k.isupper()]}"


# ── Naming convention ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMetricNames:
    _ALL_METRICS = [
        ("HTTP_REQUESTS_TOTAL", lambda: m.HTTP_REQUESTS_TOTAL),
        ("HTTP_REQUEST_DURATION_SECONDS", lambda: m.HTTP_REQUEST_DURATION_SECONDS),
        ("JOB_SUBMISSIONS_TOTAL", lambda: m.JOB_SUBMISSIONS_TOTAL),
        ("RATE_LIMIT_HITS_TOTAL", lambda: m.RATE_LIMIT_HITS_TOTAL),
        ("WEBSOCKET_CONNECTIONS_ACTIVE", lambda: m.WEBSOCKET_CONNECTIONS_ACTIVE),
        ("HEALTH_CHECKS_TOTAL", lambda: m.HEALTH_CHECKS_TOTAL),
        ("JOB_STATUS_REQUESTS_TOTAL", lambda: m.JOB_STATUS_REQUESTS_TOTAL),
        ("UPLOAD_URL_REQUESTS_TOTAL", lambda: m.UPLOAD_URL_REQUESTS_TOTAL),
    ]

    @pytest.mark.parametrize("attr_name,getter", _ALL_METRICS)
    def test_metric_name_starts_with_visionerase(self, attr_name: str, getter) -> None:
        """Every metric registered with Prometheus must use the visionerase_ namespace."""
        metric = getter()
        assert metric._name.startswith("visionerase_"), (
            f"{attr_name}: _name={metric._name!r} does not start with 'visionerase_'"
        )


# ── Counter operations ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCounterOperations:
    def test_http_requests_total_increments(self):
        """HTTP_REQUESTS_TOTAL can be incremented with all three required labels."""
        m.HTTP_REQUESTS_TOTAL.labels(method="GET", endpoint="/test", status_code="200").inc()

    def test_job_submissions_accepted_increments(self):
        """JOB_SUBMISSIONS_TOTAL increments for the 'accepted' outcome label."""
        m.JOB_SUBMISSIONS_TOTAL.labels(outcome="accepted").inc()

    def test_job_submissions_deduplicated_increments(self):
        """JOB_SUBMISSIONS_TOTAL increments for the 'deduplicated' outcome label."""
        m.JOB_SUBMISSIONS_TOTAL.labels(outcome="deduplicated").inc()

    def test_job_submissions_rejected_increments(self):
        """JOB_SUBMISSIONS_TOTAL increments for the 'rejected' outcome label."""
        m.JOB_SUBMISSIONS_TOTAL.labels(outcome="rejected").inc()

    def test_rate_limit_hits_per_minute_increments(self):
        """RATE_LIMIT_HITS_TOTAL increments for 'per_minute' window."""
        m.RATE_LIMIT_HITS_TOTAL.labels(window="per_minute").inc()

    def test_rate_limit_hits_per_hour_increments(self):
        """RATE_LIMIT_HITS_TOTAL increments for 'per_hour' window."""
        m.RATE_LIMIT_HITS_TOTAL.labels(window="per_hour").inc()

    def test_health_checks_total_increments(self):
        """HEALTH_CHECKS_TOTAL (no labels) increments without error."""
        m.HEALTH_CHECKS_TOTAL.inc()

    def test_job_status_requests_total_increments(self):
        """JOB_STATUS_REQUESTS_TOTAL (no labels) increments without error."""
        m.JOB_STATUS_REQUESTS_TOTAL.inc()

    def test_upload_url_requests_total_increments(self):
        """UPLOAD_URL_REQUESTS_TOTAL (no labels) increments without error."""
        m.UPLOAD_URL_REQUESTS_TOTAL.inc()

    def test_counter_value_increases_on_increment(self):
        """Incrementing a counter must raise its value by exactly 1.0."""
        before = m.HEALTH_CHECKS_TOTAL._value.get()
        m.HEALTH_CHECKS_TOTAL.inc()
        after = m.HEALTH_CHECKS_TOTAL._value.get()
        assert after == before + 1.0


# ── Histogram operations ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestHistogramOperations:
    def test_http_request_duration_observe(self):
        """HTTP_REQUEST_DURATION_SECONDS accepts a float observation without error."""
        m.HTTP_REQUEST_DURATION_SECONDS.labels(method="POST", endpoint="/api/v1/jobs").observe(0.05)

    def test_histogram_observe_zero(self):
        """A zero-duration observation is a valid edge case the histogram must accept."""
        m.HTTP_REQUEST_DURATION_SECONDS.labels(method="GET", endpoint="/health").observe(0.0)

    def test_histogram_observe_large_value(self):
        """A large observation (slow request) must not cause an error."""
        m.HTTP_REQUEST_DURATION_SECONDS.labels(method="POST", endpoint="/api/v1/jobs").observe(9.99)


# ── Gauge operations ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestGaugeOperations:
    def test_websocket_connections_set(self):
        """WEBSOCKET_CONNECTIONS_ACTIVE can be set to a positive integer."""
        m.WEBSOCKET_CONNECTIONS_ACTIVE.set(5)

    def test_websocket_connections_set_zero(self):
        """WEBSOCKET_CONNECTIONS_ACTIVE can be set to zero (all connections closed)."""
        m.WEBSOCKET_CONNECTIONS_ACTIVE.set(0)

    def test_websocket_connections_inc_dec(self):
        """Gauge supports inc() and dec() for tracking connection open/close events."""
        m.WEBSOCKET_CONNECTIONS_ACTIVE.inc()
        m.WEBSOCKET_CONNECTIONS_ACTIVE.dec()
