"""Unit tests for pipeline/chunker/video_chunker.py.

All subprocess and S3 calls are mocked — no real ffmpeg, ffprobe, or MinIO needed.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from pipeline.chunker.video_chunker import (
    VideoChunk,
    VideoMeta,
    chunk_video,
    compute_chunks,
    extract_chunk,
    probe_video,
)

# ── Shared test data ──────────────────────────────────────────────────────────

_FFPROBE_JSON = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30000/1001",
            "nb_frames": "900",
            "duration": "30.03",
        }
    ],
    "format": {"duration": "30.03"},
}

_FFPROBE_RESULT = subprocess.CompletedProcess(
    args=[], returncode=0, stdout=json.dumps(_FFPROBE_JSON), stderr=""
)

_FFMPEG_OK = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


@pytest.fixture
def fake_settings():
    s = MagicMock()
    s.chunk_duration_sec = 5
    s.chunk_overlap_frames = 2
    s.s3_bucket = "test-bucket"
    s.s3_endpoint_url = None
    s.aws_access_key_id = "test-key"
    s.aws_secret_access_key = "test-secret"
    return s


# ── probe_video ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestProbeVideo:
    def test_returns_videometa_instance(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run", return_value=_FFPROBE_RESULT):
            meta = probe_video("/fake/video.mp4")
        assert isinstance(meta, VideoMeta)

    def test_fps_parsed_as_fraction(self):
        """r_frame_rate '30000/1001' must parse to ~29.97 fps."""
        with patch("pipeline.chunker.video_chunker.subprocess.run", return_value=_FFPROBE_RESULT):
            meta = probe_video("/fake/video.mp4")
        assert abs(meta.fps - (30000 / 1001)) < 1e-6

    def test_width_height_codec(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run", return_value=_FFPROBE_RESULT):
            meta = probe_video("/fake/video.mp4")
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.codec == "h264"

    def test_total_frames_from_nb_frames(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run", return_value=_FFPROBE_RESULT):
            meta = probe_video("/fake/video.mp4")
        assert meta.total_frames == 900

    def test_duration_populated(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run", return_value=_FFPROBE_RESULT):
            meta = probe_video("/fake/video.mp4")
        assert abs(meta.duration_sec - 30.03) < 1e-4

    def test_total_frames_fallback_from_duration(self):
        """When nb_frames is 'N/A', total_frames = int(duration × fps)."""
        data = {
            "streams": [{
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "25/1",
                "nb_frames": "N/A",
                "duration": "10.0",
            }],
            "format": {"duration": "10.0"},
        }
        result = subprocess.CompletedProcess(args=[], returncode=0,
                                             stdout=json.dumps(data), stderr="")
        with patch("pipeline.chunker.video_chunker.subprocess.run", return_value=result):
            meta = probe_video("/fake/video.mp4")
        assert meta.total_frames == 250  # int(10.0 * 25)

    def test_integer_fps_string(self):
        """r_frame_rate without a slash must be parsed as a plain float."""
        data = {
            "streams": [{
                "codec_type": "video",
                "codec_name": "mpeg4",
                "width": 640,
                "height": 480,
                "r_frame_rate": "24",
                "nb_frames": "480",
                "duration": "20.0",
            }],
            "format": {"duration": "20.0"},
        }
        result = subprocess.CompletedProcess(args=[], returncode=0,
                                             stdout=json.dumps(data), stderr="")
        with patch("pipeline.chunker.video_chunker.subprocess.run", return_value=result):
            meta = probe_video("/fake/video.mp4")
        assert meta.fps == 24.0


# ── compute_chunks ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestComputeChunks:
    """compute_chunks splits total_frames into chunk_duration_sec × fps sized chunks."""

    def _meta(self, total_frames: int, fps: float = 30.0) -> VideoMeta:
        return VideoMeta(fps=fps, width=1920, height=1080,
                         total_frames=total_frames, duration_sec=total_frames / fps,
                         codec="h264")

    def test_returns_correct_number_of_chunks(self, fake_settings):
        """5s × 30fps = 150 frames/chunk; 300 total → 2 chunks."""
        meta = self._meta(300)
        with patch("pipeline.chunker.video_chunker.get_settings", return_value=fake_settings):
            chunks = compute_chunks(meta)
        assert len(chunks) == 2

    def test_three_chunks_for_450_frames(self, fake_settings):
        meta = self._meta(450)
        with patch("pipeline.chunker.video_chunker.get_settings", return_value=fake_settings):
            chunks = compute_chunks(meta)
        assert len(chunks) == 3

    def test_covers_all_frames_no_gaps(self, fake_settings):
        """Consecutive (start, end) pairs must be contiguous: end[i] == start[i+1]."""
        meta = self._meta(300)
        with patch("pipeline.chunker.video_chunker.get_settings", return_value=fake_settings):
            chunks = compute_chunks(meta)
        for a, b in zip(chunks, chunks[1:]):
            assert a[1] == b[0], f"Gap: chunk ends at {a[1]}, next starts at {b[0]}"

    def test_first_chunk_starts_at_zero(self, fake_settings):
        meta = self._meta(300)
        with patch("pipeline.chunker.video_chunker.get_settings", return_value=fake_settings):
            chunks = compute_chunks(meta)
        assert chunks[0][0] == 0

    def test_last_chunk_ends_at_total_frames(self, fake_settings):
        meta = self._meta(300)
        with patch("pipeline.chunker.video_chunker.get_settings", return_value=fake_settings):
            chunks = compute_chunks(meta)
        assert chunks[-1][1] == 300

    def test_last_chunk_ends_at_total_frames_uneven(self, fake_settings):
        """Last chunk must end exactly at total_frames even when not evenly divisible."""
        meta = self._meta(310)
        with patch("pipeline.chunker.video_chunker.get_settings", return_value=fake_settings):
            chunks = compute_chunks(meta)
        assert chunks[-1][1] == 310

    def test_overlap_applied_correctly(self, fake_settings):
        """Verify overlap_start/overlap_end values that chunk_video will compute from chunk ranges.

        chunk_video applies: overlap_start = max(0, start - overlap)
                             overlap_end   = min(total, end + overlap)
        """
        meta = self._meta(300)
        with patch("pipeline.chunker.video_chunker.get_settings", return_value=fake_settings):
            chunks = compute_chunks(meta)

        overlap = fake_settings.chunk_overlap_frames  # 2
        total = meta.total_frames  # 300

        # chunk 0: (0, 150) → overlap_start must clip at 0, overlap_end extends to 152
        s0, e0 = chunks[0]
        assert max(0, s0 - overlap) == 0
        assert min(total, e0 + overlap) == 152

        # chunk 1: (150, 300) → overlap_start = 148, overlap_end clips at 300
        s1, e1 = chunks[1]
        assert max(0, s1 - overlap) == 148
        assert min(total, e1 + overlap) == 300


# ── extract_chunk ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestExtractChunk:
    def test_calls_ffmpeg_subprocess(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run",
                   return_value=_FFMPEG_OK) as mock_run:
            extract_chunk("/src.mp4", 0, 150, 30.0, "/out.mp4")
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][0] == "ffmpeg"

    def test_no_audio_flag_present(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run",
                   return_value=_FFMPEG_OK) as mock_run:
            extract_chunk("/src.mp4", 0, 150, 30.0, "/out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "-an" in cmd

    def test_h264_codec(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run",
                   return_value=_FFMPEG_OK) as mock_run:
            extract_chunk("/src.mp4", 0, 150, 30.0, "/out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "libx264" in cmd

    def test_preset_fast(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run",
                   return_value=_FFMPEG_OK) as mock_run:
            extract_chunk("/src.mp4", 0, 150, 30.0, "/out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "fast" in cmd

    def test_crf_18(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run",
                   return_value=_FFMPEG_OK) as mock_run:
            extract_chunk("/src.mp4", 0, 150, 30.0, "/out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "18" in cmd

    def test_correct_start_time(self):
        """start_sec = start_frame / fps must appear after -ss."""
        with patch("pipeline.chunker.video_chunker.subprocess.run",
                   return_value=_FFMPEG_OK) as mock_run:
            extract_chunk("/src.mp4", 60, 120, 30.0, "/out.mp4")
        cmd = mock_run.call_args[0][0]
        ss_val = cmd[cmd.index("-ss") + 1]
        assert abs(float(ss_val) - 2.0) < 1e-4  # 60/30 = 2.0

    def test_correct_duration(self):
        """duration_sec = (end - start) / fps must appear after -t."""
        with patch("pipeline.chunker.video_chunker.subprocess.run",
                   return_value=_FFMPEG_OK) as mock_run:
            extract_chunk("/src.mp4", 60, 120, 30.0, "/out.mp4")
        cmd = mock_run.call_args[0][0]
        t_val = cmd[cmd.index("-t") + 1]
        assert abs(float(t_val) - 2.0) < 1e-4  # (120-60)/30 = 2.0

    def test_input_and_output_paths_in_cmd(self):
        with patch("pipeline.chunker.video_chunker.subprocess.run",
                   return_value=_FFMPEG_OK) as mock_run:
            extract_chunk("/src.mp4", 0, 30, 30.0, "/out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "/src.mp4" in cmd
        assert "/out.mp4" in cmd

    def test_raises_runtime_error_on_ffmpeg_failure(self):
        fail = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="bad input")
        with patch("pipeline.chunker.video_chunker.subprocess.run", return_value=fail):
            with pytest.raises(RuntimeError, match="ffmpeg failed"):
                extract_chunk("/src.mp4", 0, 30, 30.0, "/out.mp4")


# ── VideoChunk dataclass ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestVideoChunkDataclass:
    _REQUIRED_FIELDS = {
        "chunk_index", "start_frame", "end_frame",
        "overlap_start", "overlap_end", "s3_key",
        "fps", "width", "height",
    }

    def test_has_all_required_fields(self):
        actual = {f.name for f in fields(VideoChunk)}
        assert self._REQUIRED_FIELDS.issubset(actual)

    def test_can_be_instantiated_with_all_fields(self):
        chunk = VideoChunk(
            chunk_index=0,
            start_frame=0,
            end_frame=150,
            overlap_start=0,
            overlap_end=152,
            s3_key="jobs/abc/chunks/chunk_0000.mp4",
            fps=30.0,
            width=1920,
            height=1080,
        )
        assert chunk.chunk_index == 0
        assert chunk.start_frame == 0
        assert chunk.end_frame == 150
        assert chunk.overlap_start == 0
        assert chunk.overlap_end == 152
        assert chunk.s3_key == "jobs/abc/chunks/chunk_0000.mp4"
        assert chunk.fps == 30.0
        assert chunk.width == 1920
        assert chunk.height == 1080

    def test_s3_key_field_type_is_str(self):
        chunk = VideoChunk(
            chunk_index=1, start_frame=150, end_frame=300,
            overlap_start=148, overlap_end=300,
            s3_key="jobs/x/chunks/chunk_0001.mp4",
            fps=25.0, width=1280, height=720,
        )
        assert isinstance(chunk.s3_key, str)


# ── chunk_video (async, overlap integration) ──────────────────────────────────

@pytest.mark.unit
class TestChunkVideoOverlap:
    """Verify that chunk_video applies overlap correctly to VideoChunk objects."""

    @pytest.mark.asyncio
    async def test_overlap_start_end_on_returned_chunks(self, fake_settings):
        """First chunk's overlap_start == 0; last chunk's overlap_end == total_frames."""
        ffprobe_data = {
            "streams": [{
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "30/1",
                "nb_frames": "300",
                "duration": "10.0",
            }],
            "format": {"duration": "10.0"},
        }

        def _mock_subprocess(cmd, **_kw):
            if "ffprobe" in cmd[0]:
                return subprocess.CompletedProcess(args=cmd, returncode=0,
                                                   stdout=json.dumps(ffprobe_data), stderr="")
            return _FFMPEG_OK

        mock_s3 = MagicMock()
        mock_s3.download_file.return_value = None
        mock_s3.upload_file.return_value = None

        with (
            patch("pipeline.chunker.video_chunker.get_settings", return_value=fake_settings),
            patch("pipeline.chunker.video_chunker.boto3.client", return_value=mock_s3),
            patch("pipeline.chunker.video_chunker.subprocess.run", side_effect=_mock_subprocess),
        ):
            chunks = await chunk_video("job123", "uploads/video.mp4")

        # 300 frames / 150 per chunk = 2 chunks
        assert len(chunks) == 2

        # First chunk: (0, 150) → overlap_start clips at 0, overlap_end = 152
        assert chunks[0].overlap_start == 0
        assert chunks[0].overlap_end == 152

        # Second chunk: (150, 300) → overlap_start = 148, overlap_end clips at 300
        assert chunks[1].overlap_start == 148
        assert chunks[1].overlap_end == 300

    @pytest.mark.asyncio
    async def test_s3_keys_follow_expected_pattern(self, fake_settings):
        """Each chunk's s3_key must be jobs/{job_id}/chunks/chunk_{i:04d}.mp4."""
        ffprobe_data = {
            "streams": [{
                "codec_type": "video", "codec_name": "h264",
                "width": 640, "height": 480,
                "r_frame_rate": "30/1", "nb_frames": "150", "duration": "5.0",
            }],
            "format": {"duration": "5.0"},
        }

        def _mock_subprocess(cmd, **_kw):
            if "ffprobe" in cmd[0]:
                return subprocess.CompletedProcess(args=cmd, returncode=0,
                                                   stdout=json.dumps(ffprobe_data), stderr="")
            return _FFMPEG_OK

        mock_s3 = MagicMock()

        with (
            patch("pipeline.chunker.video_chunker.get_settings", return_value=fake_settings),
            patch("pipeline.chunker.video_chunker.boto3.client", return_value=mock_s3),
            patch("pipeline.chunker.video_chunker.subprocess.run", side_effect=_mock_subprocess),
        ):
            chunks = await chunk_video("myjob", "uploads/v.mp4")

        assert chunks[0].s3_key == "jobs/myjob/chunks/chunk_0000.mp4"
