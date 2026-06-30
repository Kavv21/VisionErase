from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Application
    app_env: str = "development"
    app_name: str = "VisionErase"
    secret_key: str
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Redis
    redis_url: str
    redis_result_ttl: int = 86400       # seconds — TTL for job result keys
    redis_job_dedup_ttl: int = 3600     # seconds — TTL for dedup hash keys

    # Celery
    celery_broker_url: str
    celery_result_backend: str

    # Database
    database_url: str

    # S3 / MinIO
    s3_bucket: str
    s3_endpoint_url: str | None = None
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_public_endpoint_url: str | None = None  # browser-facing MinIO URL
    # CV device & model pool
    device: str = "cpu"
    model_cache_dir: str = "/app/model_weights"
    max_model_pool_size: int = 2
    max_vram_gb: float = 4.0
    use_fp16: bool = False

    # Video pipeline
    chunk_duration_sec: int = 5
    chunk_overlap_frames: int = 2
    max_video_duration_sec: int = 3600
    max_video_size_mb: int = 2000

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Google OAuth (optional — app boots fine without these set)
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str | None = None

    # Rate limiting
    rate_limit_per_minute: int = 20
    rate_limit_per_hour: int = 200

    # Quality gates
    min_ssim_score: float = 0.75
    min_psnr_db: float = 25.0

    # Hierarchical pipeline
    num_segment_workers: int = 4
    segment_overlap_frames: int = 150
    boundary_window_frames: int = 10
    boundary_fusion_enabled: bool = True
    boundary_fusion_weights: str = "/app/model_weights/boundary_fusion.pt"


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
