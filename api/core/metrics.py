from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "visionerase_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "visionerase_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

JOB_SUBMISSIONS_TOTAL = Counter(
    "visionerase_job_submissions_total",
    "Total job submissions by outcome (accepted, deduplicated, rejected)",
    ["outcome"],
)

RATE_LIMIT_HITS_TOTAL = Counter(
    "visionerase_rate_limit_hits_total",
    "Total requests rejected by the sliding-window rate limiter",
    ["window"],
)

WEBSOCKET_CONNECTIONS_ACTIVE = Gauge(
    "visionerase_websocket_connections_active",
    "Number of currently active WebSocket connections",
)

HEALTH_CHECKS_TOTAL = Counter(
    "visionerase_health_checks_total",
    "Total health check requests",
)

JOB_STATUS_REQUESTS_TOTAL = Counter(
    "visionerase_job_status_requests_total",
    "Total GET /jobs/{job_id} requests",
)

UPLOAD_URL_REQUESTS_TOTAL = Counter(
    "visionerase_upload_url_requests_total",
    "Total presigned upload URL requests",
)

SEGMENT_PROCESSING_TIME = Histogram(
    "visionerase_segment_processing_seconds",
    "Time to process one video segment",
    ["worker_id"],
    buckets=[10, 30, 60, 120, 300, 600, 1200],
)

BOUNDARY_FUSION_TIME = Histogram(
    "visionerase_boundary_fusion_seconds",
    "Time for BoundaryFusion to correct one boundary",
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

SEGMENTS_TOTAL = Counter(
    "visionerase_segments_total",
    "Total video segments processed",
    ["status"],
)

BOUNDARY_QUALITY_SCORE = Histogram(
    "visionerase_boundary_quality_score",
    "SSIM quality score at segment boundaries after BoundaryFusion",
    buckets=[0.3, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0],
)
