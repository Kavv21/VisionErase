"""Unit tests for api/services/storage.py.

All S3 interactions are mocked — no real MinIO connection is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from api.services.storage import generate_download_url, generate_upload_url

# ── Shared fixtures ───────────────────────────────────────────────────────────

FAKE_PRESIGNED_PUT = "https://minio.example.com/test-bucket/uploads/uuid/video.mp4?sig=abc"
FAKE_PRESIGNED_GET = "https://minio.example.com/test-bucket/outputs/result.mp4?sig=xyz"


@pytest.fixture
def mock_s3_client():
    """A MagicMock boto3 S3 client whose presigned URL methods are pre-configured."""
    client = MagicMock()
    client.generate_presigned_url.side_effect = lambda op, **_kw: (
        FAKE_PRESIGNED_PUT if op == "put_object" else FAKE_PRESIGNED_GET
    )
    return client


@pytest.fixture(autouse=True)
def patch_storage_deps(mock_s3_client):
    """Patch _s3_client() and get_settings() for every test in this module."""
    fake_settings = MagicMock()
    fake_settings.s3_bucket = "test-bucket"
    fake_settings.s3_endpoint_url = None
    fake_settings.aws_access_key_id = "test-key"
    fake_settings.aws_secret_access_key = "test-secret"

    with (
        patch("api.services.storage._s3_client", return_value=mock_s3_client),
        patch("api.services.storage.get_settings", return_value=fake_settings),
    ):
        yield


# ── generate_upload_url ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestGenerateUploadUrl:
    @pytest.mark.asyncio
    async def test_returns_a_string_url(self):
        """generate_upload_url() must return a non-empty string for 'upload_url'."""
        result = await generate_upload_url("video.mp4", "video/mp4", 1024)
        assert isinstance(result["upload_url"], str)
        assert len(result["upload_url"]) > 0

    @pytest.mark.asyncio
    async def test_upload_url_is_not_none(self):
        """The presigned upload URL must never be None."""
        result = await generate_upload_url("clip.mp4", "video/mp4", 512)
        assert result["upload_url"] is not None

    @pytest.mark.asyncio
    async def test_s3_key_contains_original_filename(self):
        """The s3_key must embed the original filename so the file is identifiable."""
        result = await generate_upload_url("my-video.mp4", "video/mp4", 2048)
        assert "my-video.mp4" in result["s3_key"]

    @pytest.mark.asyncio
    async def test_s3_key_is_under_uploads_prefix(self):
        """All uploads must land under the 'uploads/' prefix to separate them from outputs."""
        result = await generate_upload_url("test.mp4", "video/mp4", 100)
        assert result["s3_key"].startswith("uploads/")

    @pytest.mark.asyncio
    async def test_expires_in_matches_constant(self):
        """expires_in must equal the module-level _PRESIGNED_EXPIRES_IN constant (3600s)."""
        result = await generate_upload_url("test.mp4", "video/mp4", 100)
        assert result["expires_in"] == 3600

    @pytest.mark.asyncio
    async def test_result_contains_all_required_keys(self):
        """The returned dict must contain exactly: upload_url, s3_key, expires_in."""
        result = await generate_upload_url("test.mp4", "video/mp4", 100)
        assert set(result.keys()) == {"upload_url", "s3_key", "expires_in"}

    @pytest.mark.asyncio
    async def test_each_call_generates_unique_s3_key(self):
        """Two calls with the same filename must produce different s3_keys (UUID isolation)."""
        r1 = await generate_upload_url("same.mp4", "video/mp4", 100)
        r2 = await generate_upload_url("same.mp4", "video/mp4", 100)
        assert r1["s3_key"] != r2["s3_key"]

    @pytest.mark.asyncio
    async def test_calls_put_object_on_s3_client(self, mock_s3_client):
        """generate_upload_url() must call generate_presigned_url with 'put_object'."""
        await generate_upload_url("video.mp4", "video/mp4", 1024)
        call_args = mock_s3_client.generate_presigned_url.call_args
        assert call_args[0][0] == "put_object"

    @pytest.mark.asyncio
    async def test_content_type_passed_to_s3(self, mock_s3_client):
        """The ContentType must be forwarded to the presigned URL params."""
        await generate_upload_url("video.mp4", "video/quicktime", 1024)
        call_kwargs = mock_s3_client.generate_presigned_url.call_args[1]
        assert call_kwargs["Params"]["ContentType"] == "video/quicktime"


# ── generate_download_url ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestGenerateDownloadUrl:
    @pytest.mark.asyncio
    async def test_returns_a_string_url(self):
        """generate_download_url() must return a non-empty string."""
        url = await generate_download_url("outputs/job-id/result.mp4")
        assert isinstance(url, str)
        assert len(url) > 0

    @pytest.mark.asyncio
    async def test_url_is_not_none(self):
        """The presigned download URL must never be None."""
        url = await generate_download_url("outputs/job-id/result.mp4")
        assert url is not None

    @pytest.mark.asyncio
    async def test_calls_get_object_on_s3_client(self, mock_s3_client):
        """generate_download_url() must call generate_presigned_url with 'get_object'."""
        await generate_download_url("outputs/job-id/result.mp4")
        call_args = mock_s3_client.generate_presigned_url.call_args
        assert call_args[0][0] == "get_object"

    @pytest.mark.asyncio
    async def test_s3_key_forwarded_to_params(self, mock_s3_client):
        """The exact s3_key must be passed as 'Key' in the presigned URL params."""
        key = "outputs/specific-job/result.mp4"
        await generate_download_url(key)
        call_kwargs = mock_s3_client.generate_presigned_url.call_args[1]
        assert call_kwargs["Params"]["Key"] == key

    @pytest.mark.asyncio
    async def test_custom_expires_in_forwarded(self, mock_s3_client):
        """A custom expires_in value must be forwarded to the boto3 call."""
        await generate_download_url("outputs/job/result.mp4", expires_in=7200)
        call_kwargs = mock_s3_client.generate_presigned_url.call_args[1]
        assert call_kwargs["ExpiresIn"] == 7200

    @pytest.mark.asyncio
    async def test_default_expires_in_is_3600(self, mock_s3_client):
        """Default expiry for download URLs must be 3600 seconds."""
        await generate_download_url("outputs/job/result.mp4")
        call_kwargs = mock_s3_client.generate_presigned_url.call_args[1]
        assert call_kwargs["ExpiresIn"] == 3600
