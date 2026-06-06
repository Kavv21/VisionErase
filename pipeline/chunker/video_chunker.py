from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

import boto3
import structlog
from botocore.config import Config

from api.core.config import get_settings
from api.core.metrics import CHUNK_EXTRACTION_SECONDS, CHUNKS_CREATED_TOTAL

log = structlog.get_logger(__name__)


@dataclass
class VideoChunk:
    chunk_index:   int
    start_frame:   int
    end_frame:     int
    overlap_start: int
    overlap_end:   int
    s3_key:        str
    fps:           float
    width:         int
    height:        int


@dataclass
class VideoMeta:
    fps:          float
    width:        int
    height:       int
    total_frames: int
    duration_sec: float
    codec:        str


def probe_video(path: str) -> VideoMeta:
    """Return video metadata via ffprobe — never loads frames into memory."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)

    video_stream = next(
        s for s in data["streams"] if s.get("codec_type") == "video"
    )

    fps_raw = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate", "25/1")
    if "/" in fps_raw:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_raw)

    duration_str = (
        video_stream.get("duration")
        or data.get("format", {}).get("duration", "0")
    )
    duration_sec = float(duration_str or 0.0)

    nb_frames = video_stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        total_frames = int(nb_frames)
    else:
        total_frames = int(duration_sec * fps)

    meta = VideoMeta(
        fps=fps,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        total_frames=total_frames,
        duration_sec=duration_sec,
        codec=video_stream.get("codec_name", "unknown"),
    )
    log.info("video_probed", path=path, fps=fps, width=meta.width,
             height=meta.height, total_frames=total_frames)
    return meta


def compute_chunks(meta: VideoMeta) -> list[tuple[int, int]]:
    """Split total_frames into non-overlapping logical (start, end) tuples.

    chunk_video applies chunk_overlap_frames on top when building VideoChunk
    objects. Reads chunk_duration_sec from settings.
    """
    settings = get_settings()
    chunk_size = max(1, int(settings.chunk_duration_sec * meta.fps))

    chunks: list[tuple[int, int]] = []
    cursor = 0
    while cursor < meta.total_frames:
        end = min(cursor + chunk_size, meta.total_frames)
        chunks.append((cursor, end))
        cursor = end

    log.info(
        "chunks_computed",
        total_frames=meta.total_frames,
        chunk_size=chunk_size,
        num_chunks=len(chunks),
    )
    return chunks


def extract_chunk(
    video_path: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    output_path: str,
) -> None:
    """Extract [start_frame, end_frame) from video_path to output_path.

    Output is H.264 MP4, preset=fast, crf=18, no audio.
    Uses time-based ffmpeg seeking — never loads the full video into RAM.
    """
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame) / fps

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.6f}",
        "-i", video_path,
        "-t", f"{duration_sec:.6f}",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        output_path,
    ]

    log.info("extracting_chunk", start_frame=start_frame, end_frame=end_frame)
    with CHUNK_EXTRACTION_SECONDS.time():
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for frames [{start_frame}, {end_frame}): {result.stderr}"
        )


async def chunk_video(job_id: str, video_s3_key: str) -> list[VideoChunk]:
    """Download video from S3, split into overlapping chunks, upload each chunk.

    Returns list of VideoChunk with all fields populated.
    Uses TemporaryDirectory for scratch space — auto-cleaned on exit.
    Never keeps the full video in RAM after chunking completes.
    """
    settings = get_settings()
    bound_log = log.bind(job_id=job_id, video_s3_key=video_s3_key)

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
    )

    loop = asyncio.get_event_loop()

    with tempfile.TemporaryDirectory() as tmp_dir:
        src_path = os.path.join(tmp_dir, "source.mp4")

        bound_log.info("downloading_source_video")
        await loop.run_in_executor(
            None, lambda: s3.download_file(settings.s3_bucket, video_s3_key, src_path)
        )

        meta = await loop.run_in_executor(None, lambda: probe_video(src_path))
        bound_log.info(
            "video_probed",
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            total_frames=meta.total_frames,
        )

        ranges = compute_chunks(meta)
        overlap = settings.chunk_overlap_frames
        chunks: list[VideoChunk] = []

        for i, (start, end) in enumerate(ranges):
            overlap_start = max(0, start - overlap)
            overlap_end = min(meta.total_frames, end + overlap)

            chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}.mp4")

            bound_log.info(
                "extracting_chunk",
                chunk_index=i,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
            )
            await loop.run_in_executor(
                None,
                lambda s=overlap_start, e=overlap_end, p=chunk_path: extract_chunk(
                    src_path, s, e, meta.fps, p
                ),
            )

            dest_key = f"jobs/{job_id}/chunks/chunk_{i:04d}.mp4"
            await loop.run_in_executor(
                None,
                lambda p=chunk_path, k=dest_key: s3.upload_file(p, settings.s3_bucket, k),
            )

            chunks.append(
                VideoChunk(
                    chunk_index=i,
                    start_frame=start,
                    end_frame=end,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                    s3_key=dest_key,
                    fps=meta.fps,
                    width=meta.width,
                    height=meta.height,
                )
            )
            CHUNKS_CREATED_TOTAL.inc()

        bound_log.info("chunking_complete", num_chunks=len(chunks))
        return chunks
