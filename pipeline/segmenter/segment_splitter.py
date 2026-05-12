from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import structlog

from api.core.config import get_settings

log = structlog.get_logger(__name__)


@dataclass
class VideoSegment:
    segment_index: int
    start_frame: int        # logical start (no overlap)
    end_frame: int          # logical end (no overlap)
    overlap_start: int      # actual start including leading overlap
    overlap_end: int        # actual end including trailing overlap
    s3_key: str             # S3 path of this segment's extracted video
    fps: float
    width: int
    height: int


def compute_segments(
    total_frames: int,
    fps: float,
    num_workers: int,
    overlap_frames: int,
) -> list[VideoSegment]:
    """Split total_frames into num_workers equal segments with overlap at each boundary.

    Each segment's logical range is non-overlapping; overlap_start / overlap_end
    extend into the neighbouring segment's territory so boundary workers have
    enough context to blend seamlessly.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if total_frames < 1:
        raise ValueError(f"total_frames must be >= 1, got {total_frames}")

    # Clamp to avoid creating more workers than frames
    num_workers = min(num_workers, total_frames)

    base = total_frames // num_workers
    remainder = total_frames % num_workers

    segments: list[VideoSegment] = []
    cursor = 0

    for i in range(num_workers):
        # Distribute remainder frames across first `remainder` segments
        segment_frames = base + (1 if i < remainder else 0)
        start = cursor
        end = cursor + segment_frames  # exclusive

        # Overlap clipped to valid frame range
        overlap_start = max(0, start - overlap_frames)
        overlap_end = min(total_frames, end + overlap_frames)

        segments.append(
            VideoSegment(
                segment_index=i,
                start_frame=start,
                end_frame=end,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                s3_key="",          # filled in by extract_segment
                fps=fps,
                width=0,            # filled in by extract_segment
                height=0,           # filled in by extract_segment
            )
        )
        cursor = end

    log.info(
        "segments_computed",
        total_frames=total_frames,
        num_workers=num_workers,
        overlap_frames=overlap_frames,
        num_segments=len(segments),
    )
    return segments


async def extract_segment(
    video_s3_key: str,
    segment: VideoSegment,
    job_id: str,
) -> VideoSegment:
    """Download a video from S3, extract this segment's frames, and re-upload.

    Returns the same VideoSegment with s3_key, width, and height populated.
    Uses ffmpeg for frame extraction. All I/O is async-compatible (subprocess).
    """
    import asyncio
    import os
    import tempfile

    import boto3
    from botocore.config import Config

    settings = get_settings()
    bound_log = log.bind(
        job_id=job_id,
        segment_index=segment.segment_index,
        video_s3_key=video_s3_key,
    )

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        src_path = os.path.join(tmp_dir, "source.mp4")
        seg_path = os.path.join(tmp_dir, f"segment_{segment.segment_index}.mp4")

        # Download source video
        bound_log.info("downloading_source_video")
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: s3.download_file(settings.s3_bucket, video_s3_key, src_path)
        )

        # Extract segment frames via ffmpeg (overlap_start → overlap_end)
        start_sec = segment.overlap_start / segment.fps
        duration_frames = segment.overlap_end - segment.overlap_start
        duration_sec = duration_frames / segment.fps

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", src_path,
            "-t", str(duration_sec),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            seg_path,
        ]

        bound_log.info("extracting_segment_frames", ffmpeg_cmd=ffmpeg_cmd)
        proc = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for segment {segment.segment_index}: {stderr.decode()}"
            )

        # Probe width/height from the extracted segment
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            seg_path,
        ]
        probe = await asyncio.create_subprocess_exec(
            *probe_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        probe_out, _ = await probe.communicate()
        width, height = (int(v) for v in probe_out.decode().strip().split(","))

        # Upload segment to S3
        dest_key = f"jobs/{job_id}/segments/segment_{segment.segment_index}.mp4"
        bound_log.info("uploading_segment", dest_key=dest_key)
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: s3.upload_file(seg_path, settings.s3_bucket, dest_key),
        )

    bound_log.info("segment_extracted", dest_key=dest_key, width=width, height=height)
    return VideoSegment(
        segment_index=segment.segment_index,
        start_frame=segment.start_frame,
        end_frame=segment.end_frame,
        overlap_start=segment.overlap_start,
        overlap_end=segment.overlap_end,
        s3_key=dest_key,
        fps=segment.fps,
        width=width,
        height=height,
    )
