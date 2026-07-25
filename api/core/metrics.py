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

CHUNKS_CREATED_TOTAL = Counter(
    "visionerase_chunks_created_total",
    "Total video chunks extracted and uploaded to S3 for inpainting",
)

CHUNK_EXTRACTION_SECONDS = Histogram(
    "visionerase_chunk_extraction_seconds",
    "Time to extract one video chunk via ffmpeg",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

MODEL_LOAD_TIME = Histogram(
    "visionerase_model_load_time_seconds",
    "Time to load a CV model into the pool",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

MODEL_POOL_SIZE = Gauge(
    "visionerase_model_pool_size",
    "Number of CV models currently loaded in the pool",
)

VRAM_USAGE = Gauge(
    "visionerase_vram_usage_gb",
    "Current VRAM usage in gigabytes (CUDA only)",
)

AUTH_REGISTRATIONS_TOTAL = Counter(
    "visionerase_auth_registrations_total",
    "Total successful user registrations",
)

AUTH_LOGINS_TOTAL = Counter(
    "visionerase_auth_logins_total",
    "Total successful user logins",
)

SEGMENT_PREVIEW_REQUESTS_TOTAL = Counter(
    "visionerase_segment_preview_requests_total",
    "Total segment preview requests (real-time SAM2 mask preview)",
)

SEGMENT_PREVIEW_ERRORS_TOTAL = Counter(
    "visionerase_segment_preview_errors_total",
    "Total segment preview requests that failed",
)

SEGMENT_PREVIEW_LATENCY_SECONDS = Histogram(
    "visionerase_segment_preview_latency_seconds",
    "Time to run real-time SAM2 segmentation for one preview request",
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
)

UPLOAD_REQUESTS_TOTAL = Counter(
    "visionerase_upload_requests_total",
    "Total direct video upload requests via multipart POST",
)

DOWNLOAD_URL_REQUESTS_TOTAL = Counter(
    "visionerase_download_url_requests_total",
    "Total presigned result download URL requests",
)

ADMIN_JOBS_REQUESTS_TOTAL = Counter(
    "visionerase_admin_jobs_requests_total",
    "Total GET /api/v1/admin/jobs requests",
)

QUALITY_SSIM_SCORE = Histogram(
    "visionerase_quality_ssim_score",
    "Mean SSIM between input and result video frames",
    buckets=[0.3, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0],
)

QUALITY_PSNR_DB = Histogram(
    "visionerase_quality_psnr_db",
    "Mean PSNR (dB) between input and result video frames",
    buckets=[10, 15, 20, 25, 30, 35, 40, 50],
)

QUALITY_TEMPORAL_CONSISTENCY = Histogram(
    "visionerase_quality_temporal_consistency",
    "Mean cosine similarity between consecutive result frames",
    buckets=[0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
)

INPAINT_ROI_MODE_TOTAL = Counter(
    "visionerase_inpaint_roi_mode_total",
    "Chunks inpainted per resolution mode (roi crop vs full frame)",
    ["mode"],
)

INPAINT_ROI_COVERAGE = Histogram(
    "visionerase_inpaint_roi_coverage_ratio",
    "ROI crop area as a fraction of the full frame area",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.6, 0.8, 1.0],
)

CHUNK_DIFFICULTY_SCORE = Histogram(
    "visionerase_chunk_difficulty_score",
    "Heuristic difficulty of a chunk for ProPainter (0-1); drives DiffuEraser",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],
)

DIFFUERASER_REFINEMENTS_TOTAL = Counter(
    "visionerase_diffueraser_refinements_total",
    "DiffuEraser refinement attempts by outcome",
    ["outcome"],
)

DIFFUERASER_SECONDS = Histogram(
    "visionerase_diffueraser_seconds",
    "Wall time of one DiffuEraser refinement pass, including Modal transfer",
    buckets=[30, 60, 120, 180, 300, 450, 600, 900],
)

INPAINT_POSTPROCESS_SECONDS = Histogram(
    "visionerase_inpaint_postprocess_seconds",
    "Time spent on three-zone blending, colour correction and grain per chunk",
    buckets=[0.5, 1, 2.5, 5, 10, 20, 45, 90],
)
