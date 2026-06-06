"""Day 13 — end-to-end integration tests using a real video file.

Tests the full platform layer (chunking, segmentation, model pool, API flow)
against real Docker services (Redis, MinIO). No GPU workers required.

Run with:
  python -m pytest tests/integration/test_pipeline_integration.py -v --timeout=30
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from starlette.testclient import TestClient

import pipeline.pool.model_pool as pool_module
from api.core.config import get_settings
from pipeline.chunker.video_chunker import compute_chunks, extract_chunk, probe_video
from pipeline.pool.model_pool import ModelPool, get_model_pool
from pipeline.segmenter.segment_splitter import compute_segments

# ── Constants ──────────────────────────────────────────────────────────────────

FIXTURE_VIDEO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "test_10sec_720p.mp4")
)
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_KEY = "test/fixtures/test_10sec_720p.mp4"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _s3():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _ensure_bucket(s3_client, bucket: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError:
        try:
            s3_client.create_bucket(Bucket=bucket)
        except ClientError:
            pass


# ── Tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_fixture_video_exists() -> None:
    """Verify tests/fixtures/test_10sec_720p.mp4 exists and is non-empty."""
    assert os.path.exists(FIXTURE_VIDEO), f"Fixture not found at: {FIXTURE_VIDEO}"
    assert os.path.getsize(FIXTURE_VIDEO) > 0, "Fixture video file is empty"


@pytest.mark.integration
def test_upload_video_to_minio() -> None:
    """Upload the test fixture to MinIO bucket and verify the object exists."""
    settings = get_settings()
    bucket = settings.s3_bucket
    s3 = _s3()
    _ensure_bucket(s3, bucket)

    try:
        s3.upload_file(FIXTURE_VIDEO, bucket, MINIO_KEY)
        head = s3.head_object(Bucket=bucket, Key=MINIO_KEY)
        assert head["ContentLength"] > 0, "Uploaded object has zero bytes"
    finally:
        try:
            s3.delete_object(Bucket=bucket, Key=MINIO_KEY)
        except ClientError:
            pass


@pytest.mark.integration
def test_probe_video_metadata() -> None:
    """probe_video() must return correct metadata for the 10s 720p 30fps fixture."""
    meta = probe_video(FIXTURE_VIDEO)

    assert meta.fps == 30.0, f"Expected fps=30.0, got {meta.fps}"
    assert meta.width == 1280, f"Expected width=1280, got {meta.width}"
    assert meta.height == 720, f"Expected height=720, got {meta.height}"
    assert meta.total_frames == 300, f"Expected 300 frames, got {meta.total_frames}"
    assert abs(meta.duration_sec - 10.0) < 0.5, (
        f"Expected duration ~10.0s, got {meta.duration_sec}"
    )


@pytest.mark.integration
def test_compute_chunks_for_test_video() -> None:
    """compute_chunks() must return gap-free (start, end) tuples spanning all 300 frames."""
    meta = probe_video(FIXTURE_VIDEO)
    chunks = compute_chunks(meta)

    assert isinstance(chunks, list)
    assert len(chunks) >= 1, "Expected at least one chunk"

    assert chunks[0][0] == 0, f"First chunk must start at frame 0, got {chunks[0][0]}"
    assert chunks[-1][1] == 300, f"Last chunk must end at frame 300, got {chunks[-1][1]}"

    for i in range(len(chunks) - 1):
        assert chunks[i][1] == chunks[i + 1][0], (
            f"Gap between chunk {i} (end={chunks[i][1]}) "
            f"and chunk {i + 1} (start={chunks[i + 1][0]})"
        )


@pytest.mark.integration
def test_extract_single_chunk() -> None:
    """extract_chunk() must write a valid non-empty MP4 that ffprobe can decode."""
    meta = probe_video(FIXTURE_VIDEO)
    chunks = compute_chunks(meta)
    start, end = chunks[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "chunk_0.mp4")
        extract_chunk(FIXTURE_VIDEO, start, end, meta.fps, out_path)

        assert os.path.exists(out_path), "extract_chunk() did not create the output file"
        assert os.path.getsize(out_path) > 0, "Extracted chunk file is empty"

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", out_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"ffprobe rejected chunk file: {result.stderr}"

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        assert any(s.get("codec_type") == "video" for s in streams), (
            "No video stream found in extracted chunk"
        )


@pytest.mark.integration
def test_compute_segments() -> None:
    """compute_segments() must split 300 frames into 2 segments with correct overlap fields."""
    segments = compute_segments(
        total_frames=300, fps=30.0, num_workers=2, overlap_frames=10
    )

    assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"

    assert segments[0].start_frame == 0, (
        f"First segment must start at frame 0, got {segments[0].start_frame}"
    )
    assert segments[-1].end_frame == 300, (
        f"Last segment must end at frame 300, got {segments[-1].end_frame}"
    )

    # No frame gap between the two segments
    assert segments[0].end_frame == segments[1].start_frame, (
        f"Gap between segments: seg0 ends at {segments[0].end_frame}, "
        f"seg1 starts at {segments[1].start_frame}"
    )

    for seg in segments:
        assert seg.overlap_start <= seg.start_frame, (
            f"Segment {seg.segment_index}: overlap_start {seg.overlap_start} "
            f"> start_frame {seg.start_frame}"
        )
        assert seg.overlap_end >= seg.end_frame, (
            f"Segment {seg.segment_index}: overlap_end {seg.overlap_end} "
            f"< end_frame {seg.end_frame}"
        )


@pytest.mark.integration
def test_model_pool_initialises() -> None:
    """get_model_pool() must return an empty pool with all 5 stub loaders registered."""
    # Reset singleton so this test always starts with a fresh pool
    pool_module._pool_instance = None

    try:
        pool = get_model_pool()

        assert isinstance(pool, ModelPool)
        assert len(pool._pool) == 0, "Pool must start empty — no models pre-loaded"

        stats = pool.stats()
        assert stats["pool_size"] == 0, f"Expected pool_size=0, got {stats['pool_size']}"

        expected_names = {"sam2", "xmem", "propainter", "raft", "boundary_fusion"}
        assert set(pool._loaders.keys()) == expected_names, (
            f"Registered loaders {set(pool._loaders.keys())} != {expected_names}"
        )
    finally:
        # Reset so subsequent tests get a clean pool
        pool_module._pool_instance = None


@pytest.mark.integration
def test_job_submission_full_flow(ws_client: TestClient) -> None:
    """Full flow: upload-url → submit job → poll status → WebSocket connection accepted."""
    # Step 1: Request a presigned upload URL
    r_url = ws_client.post(
        "/api/v1/jobs/upload-url",
        json={"filename": "test_10sec_720p.mp4", "size_bytes": os.path.getsize(FIXTURE_VIDEO)},
    )
    assert r_url.status_code == 200, f"upload-url returned {r_url.status_code}: {r_url.text}"
    url_data = r_url.json()
    assert "upload_url" in url_data, "Response missing 'upload_url'"
    assert "s3_key" in url_data, "Response missing 's3_key'"

    s3_key = url_data["s3_key"]

    # Step 2: Submit job with the returned s3_key
    r_job = ws_client.post(
        "/api/v1/jobs",
        json={
            "video_s3_key": s3_key,
            "mask": {"points": [{"x": 640, "y": 360}], "frame_index": 0},
        },
    )
    assert r_job.status_code == 202, f"Job submit returned {r_job.status_code}: {r_job.text}"
    job_data = r_job.json()
    assert "job_id" in job_data, "Job response missing 'job_id'"
    job_id = job_data["job_id"]

    # Step 3: Poll job status — must be "pending"
    r_status = ws_client.get(f"/api/v1/jobs/{job_id}")
    assert r_status.status_code == 200, f"Status poll returned {r_status.status_code}"
    status_data = r_status.json()
    assert status_data["status"] == "pending", (
        f"Expected status=pending, got {status_data['status']}"
    )

    # Step 4: WebSocket must accept connection and send an initial message
    with ws_client.websocket_connect(f"/ws/{job_id}") as ws:
        msg = ws.receive_json()

    assert msg is not None, "WebSocket sent no initial message"
    assert "type" in msg, f"WebSocket message missing 'type' key: {msg}"
