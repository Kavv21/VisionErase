from __future__ import annotations
import uuid
from functools import lru_cache
from typing import Any
import boto3
import structlog
from botocore.config import Config
from api.core.config import get_settings
log = structlog.get_logger(__name__)
_PRESIGNED_EXPIRES_IN = 3600
@lru_cache(maxsize=1)
def _s3_client() -> Any:
    """Return a cached boto3 S3 client built from settings."""
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
    )
async def generate_upload_url(
    filename: str,
    content_type: str,
    size_bytes: int,  # noqa: ARG001 — reserved for future policy enforcement
) -> dict[str, Any]:
    """Return a presigned S3 PUT URL, the target key, and its TTL in seconds."""
    settings = get_settings()
    s3_key = f"uploads/{uuid.uuid4()}/{filename}"
    upload_url: str = _s3_client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.s3_bucket,
            "Key": s3_key,
            "ContentType": content_type,
        },
        ExpiresIn=_PRESIGNED_EXPIRES_IN,
    )
    # Replace internal Docker hostname with public-facing URL
    # so the browser can actually reach MinIO directly
    if settings.s3_public_endpoint_url and settings.s3_endpoint_url:
        upload_url = upload_url.replace(
            settings.s3_endpoint_url,
            settings.s3_public_endpoint_url,
        )
    log.info("presigned_url_generated", s3_key=s3_key, expires_in=_PRESIGNED_EXPIRES_IN)
    return {
        "upload_url": upload_url,
        "s3_key": s3_key,
        "expires_in": _PRESIGNED_EXPIRES_IN,
    }
async def generate_download_url(s3_key: str, expires_in: int = _PRESIGNED_EXPIRES_IN) -> str:
    """Return a presigned S3 GET URL for the given key."""
    settings = get_settings()
    url: str = _s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": s3_key},
        ExpiresIn=expires_in,
    )
    # Replace internal Docker hostname with public-facing URL
    # so the browser can actually reach MinIO directly
    if settings.s3_public_endpoint_url and settings.s3_endpoint_url:
        url = url.replace(
            settings.s3_endpoint_url,
            settings.s3_public_endpoint_url,
        )
    log.info("presigned_download_url_generated", s3_key=s3_key, expires_in=expires_in)
    return url
