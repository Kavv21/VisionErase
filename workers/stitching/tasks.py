from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from typing import Any

import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.config import get_settings
from api.core.metrics import SEGMENTS_TOTAL
from api.core.redis import acquire_redis, publish_progress, set_job_status
from api.models.job import JobStatus
from workers.celery_app import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(
    bind=True,
    queue="stitching",
    max_retries=2,
    soft_time_limit=300,
    time_limit=360,
    name="workers.stitching.tasks.stitch_segments",
)
def stitch_segments(
    self,
    job_id: str,
    inpainted_s3_keys: list[str],
) -> dict[str, Any]:
    """Stub: combine inpainted segments into final video (Month 2 integration)."""
    bound_log = log.bind(job_id=job_id, task_id=self.request.id)
    bound_log.info("stitching_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "stitching", "pct": 0})
        )
        loop.run_until_complete(
            _update_status(job_id, JobStatus.STITCHING, None)
        )
        SEGMENTS_TOTAL.labels(status="stitching_started").inc()

        # Pass through the real inpainted video from ProPainter
        result_s3_key = inpainted_s3_keys[0] if inpainted_s3_keys else f"jobs/{job_id}/result/output_stub.mp4"

        loop.run_until_complete(
            _publish(job_id, {"stage": "stitching", "pct": 100})
        )
        SEGMENTS_TOTAL.labels(status="stitching_complete").inc()

        from workers.quality.tasks import quality_check_final
        quality_check_final.apply_async(
            args=[job_id, result_s3_key],
            queue="quality",
        )

        return {
            "job_id": job_id,
            "result_s3_key": result_s3_key,
            "status": "stub",
        }

    except SoftTimeLimitExceeded:
        bound_log.error("stitching_soft_timeout")
        SEGMENTS_TOTAL.labels(status="stitching_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "stitch_segments timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("stitching_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="stitching_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    queue="stitching",
    max_retries=2,
    # Blending decodes every chunk and re-encodes the whole video with libx264,
    # which is far slower than the stream-copy concat this replaced.
    soft_time_limit=900,
    time_limit=1020,
    name="workers.stitching.tasks.stitch_all_chunks",
)
def stitch_all_chunks(
    self,
    job_id: str,
    total_chunks: int,
    original_s3_key: str | None = None,
) -> dict[str, Any]:
    """Assemble every inpainted chunk into the final result video.

    Chunks produced by the overlapping chunker share OVERLAP frames with their
    neighbour; those frames are cross-faded (see _blend_overlap) so the seam is
    a gradual transition rather than a hard cut. Jobs chunked without overlap
    (overlap=0 in Redis, or no key at all for jobs predating this) fall back to
    the original stream-copy ffmpeg concat.

    When original_s3_key is given, the source video's audio track is muxed
    back into the (silent) inpainted output; videos without audio pass
    through unchanged.
    """
    settings = get_settings()
    bound_log = log.bind(job_id=job_id, total_chunks=total_chunks, task_id=self.request.id)
    bound_log.info("chunk_stitching_started")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _publish(job_id, {"stage": "stitching", "pct": 0})
        )
        loop.run_until_complete(
            _update_status(job_id, JobStatus.STITCHING, None)
        )
        SEGMENTS_TOTAL.labels(status="chunk_stitching_started").inc()

        s3 = _make_s3_client(settings)
        overlap, expected_frames = loop.run_until_complete(_load_stitch_params(job_id))
        bound_log = bound_log.bind(overlap=overlap)

        with tempfile.TemporaryDirectory() as tmp:
            chunk_paths = []
            for i in range(total_chunks):
                chunk_key = f"jobs/{job_id}/chunks/{i}/inpainted_chunk.mp4"
                chunk_path = os.path.join(tmp, f"chunk_{i:04d}.mp4")
                s3.download_file(settings.s3_bucket, chunk_key, chunk_path)
                chunk_paths.append(chunk_path)

            loop.run_until_complete(
                _publish(job_id, {"stage": "stitching", "pct": 40})
            )

            output_path = os.path.join(tmp, "output.mp4")
            if overlap > 0 and len(chunk_paths) > 1:
                frame_count = _stitch_with_blend(
                    chunk_paths, overlap, output_path, bound_log
                )
                # The blended assembly must reproduce the source frame count
                # exactly — a mismatch means chunk ranges and decode disagree.
                if expected_frames and frame_count != expected_frames:
                    bound_log.error(
                        "stitched_frame_count_mismatch",
                        expected_frames=expected_frames,
                        actual_frames=frame_count,
                    )
                    SEGMENTS_TOTAL.labels(status="stitch_frame_count_mismatch").inc()
                    raise RuntimeError(
                        f"stitched {frame_count} frames but source video has "
                        f"{expected_frames}"
                    )
                SEGMENTS_TOTAL.labels(status="stitch_blended").inc()
                bound_log.info("chunks_blended", frame_count=frame_count)
            else:
                _stitch_with_concat(chunk_paths, tmp, output_path)
                SEGMENTS_TOTAL.labels(status="stitch_concat").inc()
                bound_log.info("chunks_concatenated")

            loop.run_until_complete(
                _publish(job_id, {"stage": "stitching", "pct": 80})
            )

            if original_s3_key:
                output_path = _mux_original_audio(
                    s3, settings, tmp, original_s3_key, output_path, bound_log
                )

            result_s3_key = f"jobs/{job_id}/result/output.mp4"
            s3.upload_file(output_path, settings.s3_bucket, result_s3_key)

        loop.run_until_complete(
            _publish(job_id, {"stage": "stitching", "pct": 100})
        )
        SEGMENTS_TOTAL.labels(status="chunk_stitching_complete").inc()

        # BoundaryFusion temporarily bypassed — needs retraining at higher resolution
        from workers.quality.tasks import quality_check_final
        quality_check_final.apply_async(
            args=[job_id, result_s3_key],
            queue="quality",
        )

        return {
            "job_id": job_id,
            "result_s3_key": result_s3_key,
            "total_chunks": total_chunks,
            "status": "real",
        }

    except SoftTimeLimitExceeded:
        bound_log.error("chunk_stitching_soft_timeout")
        SEGMENTS_TOTAL.labels(status="chunk_stitching_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "stitch_all_chunks timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("chunk_stitching_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="chunk_stitching_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_video_to_frames(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # BGR uint8
    cap.release()
    return frames


def _blend_overlap(frames_a, frames_b, overlap):
    """
    Linear alpha blend between last 'overlap' frames of chunk A
    and first 'overlap' frames of chunk B.

    frames_a: list of numpy arrays (H, W, 3) uint8
    frames_b: list of numpy arrays (H, W, 3) uint8
    overlap: int — number of frames to blend

    Returns: list of blended numpy arrays
    """
    import numpy as np
    result = []
    for i in range(overlap):
        # alpha goes from 1.0 (all A) to 0.0 (all B)
        alpha = 1.0 - (i + 1) / (overlap + 1)
        fa = frames_a[-(overlap - i)].astype(np.float32)
        fb = frames_b[i].astype(np.float32)
        blended = (alpha * fa + (1.0 - alpha) * fb)
        result.append(blended.clip(0, 255).astype(np.uint8))
    return result


def _stitch_with_blend(
    chunk_paths: list[str],
    overlap: int,
    output_path: str,
    bound_log: Any,
) -> int:
    """Decode every chunk, cross-fade the shared frames, and encode the result.

    Chunk i covers source frames [i*STRIDE, i*STRIDE + CHUNK_SIZE), so the last
    `overlap` frames of chunk i and the first `overlap` frames of chunk i+1 are
    two independent inpaintings of the *same* source frames. Emitting a linear
    ramp between them replaces the hard cut with a gradual hand-off.

    Returns the number of frames written.
    """
    import cv2

    final_frames: list[Any] = []
    prev_chunk_frames = None
    last_index = len(chunk_paths) - 1

    for i, chunk_path in enumerate(chunk_paths):
        chunk_frames = decode_video_to_frames(chunk_path)
        if not chunk_frames:
            raise RuntimeError(f"decoded 0 frames from chunk {i} ({chunk_path})")

        # A chunk shorter than the overlap would make the slicing below wrap
        # around and silently drop or duplicate frames.
        if overlap > 0 and i > 0 and len(chunk_frames) <= overlap:
            raise RuntimeError(
                f"chunk {i} has {len(chunk_frames)} frames, which is not more "
                f"than the {overlap}-frame overlap"
            )

        if i == 0:
            # First chunk: hold back the tail — it gets blended with chunk 1.
            if overlap > 0 and len(chunk_frames) > overlap and last_index > 0:
                final_frames.extend(chunk_frames[:-overlap])
            else:
                final_frames.extend(chunk_frames)
            prev_chunk_frames = chunk_frames

        elif i == last_index:
            # Last chunk: blend the shared head, then take everything after it.
            if overlap > 0 and prev_chunk_frames is not None:
                final_frames.extend(
                    _blend_overlap(prev_chunk_frames, chunk_frames, overlap)
                )
                final_frames.extend(chunk_frames[overlap:])
            else:
                final_frames.extend(chunk_frames)

        else:
            # Middle chunks: blend the shared head, add the exclusive middle,
            # and hold back the tail for the next iteration's blend.
            if overlap > 0 and prev_chunk_frames is not None:
                final_frames.extend(
                    _blend_overlap(prev_chunk_frames, chunk_frames, overlap)
                )
                final_frames.extend(chunk_frames[overlap:-overlap])
            else:
                final_frames.extend(chunk_frames)
            prev_chunk_frames = chunk_frames

        bound_log.debug(
            "chunk_decoded",
            chunk_index=i,
            chunk_frames=len(chunk_frames),
            frames_so_far=len(final_frames),
        )

    cap = cv2.VideoCapture(chunk_paths[0])
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    height, width = final_frames[0].shape[:2]
    _write_frames_ffmpeg(final_frames, output_path, width, height, fps)
    return len(final_frames)


def _write_frames_ffmpeg(
    frames: list[Any],
    output_path: str,
    width: int,
    height: int,
    fps: float,
) -> None:
    """Encode BGR frames to H.264 by piping raw video into ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass  # ffmpeg died early; the returncode check below reports why
    stderr = proc.stderr.read().decode("utf-8", "replace")
    proc.stderr.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg rawvideo encode failed: {stderr[-500:]}")


def _stitch_with_concat(chunk_paths: list[str], tmp: str, output_path: str) -> None:
    """Stream-copy concat — the pre-overlap behaviour, kept for overlap=0 jobs."""
    concat_list_path = os.path.join(tmp, "concat_list.txt")
    with open(concat_list_path, "w") as concat_list:
        for chunk_path in chunk_paths:
            concat_list.write(f"file '{chunk_path}'\n")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")


async def _load_stitch_params(job_id: str) -> tuple[int, int]:
    """Return (overlap, total_frames) written by the chunker; (0, 0) if absent."""
    from redis import asyncio as _aioredis
    from api.core.config import get_settings as _get_settings
    _r = _aioredis.Redis.from_url(_get_settings().redis_url, decode_responses=True)
    try:
        overlap_raw = await _r.get(f"job:{job_id}:overlap")
        frames_raw = await _r.get(f"job:{job_id}:total_frames")
        return int(overlap_raw or 0), int(frames_raw or 0)
    finally:
        await _r.aclose()


def _mux_original_audio(
    s3: Any,
    settings: Any,
    tmp: str,
    original_s3_key: str,
    video_path: str,
    bound_log: Any,
) -> str:
    """Copy the original video's audio track onto the inpainted video.

    Best-effort: if the original has no audio (or muxing fails) the silent
    inpainted video is returned unchanged rather than failing the job.
    """
    original_path = os.path.join(tmp, "original.mp4")
    s3.download_file(settings.s3_bucket, original_s3_key, original_path)

    # .mka (Matroska audio) accepts any codec with stream copy, so this works
    # for AAC and MP3 sources alike without a probe step.
    audio_path = os.path.join(tmp, "audio.mka")
    extract = subprocess.run(
        ["ffmpeg", "-y", "-i", original_path, "-vn", "-acodec", "copy", audio_path],
        capture_output=True,
        text=True,
    )
    if extract.returncode != 0:
        bound_log.info("audio_extract_skipped", detail=extract.stderr[-300:])
        SEGMENTS_TOTAL.labels(status="audio_mux_skipped").inc()
        return video_path

    muxed_path = os.path.join(tmp, "output_with_audio.mp4")
    mux = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",
            muxed_path,
        ],
        capture_output=True,
        text=True,
    )
    if mux.returncode != 0:
        bound_log.warning("audio_mux_failed", detail=mux.stderr[-300:])
        SEGMENTS_TOTAL.labels(status="audio_mux_failed").inc()
        return video_path

    bound_log.info("audio_muxed_from_original", original_s3_key=original_s3_key)
    SEGMENTS_TOTAL.labels(status="audio_muxed").inc()
    return muxed_path


def _make_s3_client(settings: Any) -> Any:
    """Return a boto3 S3 client configured for the MinIO endpoint."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
    )


async def _publish(job_id: str, data: dict) -> None:
    from redis import asyncio as _aioredis
    import json as _json
    from api.core.config import get_settings as _get_settings
    _s = _get_settings()
    _r = _aioredis.Redis.from_url(_s.redis_url, decode_responses=True)
    try:
        channel = f"progress:{job_id}"
        message = _json.dumps({"type": "progress", **data})
        # Persist the latest progress event so REST polling and WebSocket
        # snapshots can report it (read back via api.core.redis.get_job_progress)
        await _r.setex(f"job:progress:{job_id}", _s.redis_result_ttl, message)
        await _r.publish(channel, message)
    finally:
        await _r.aclose()


async def _update_status(job_id: str, status: str, error: str | None = None, extra: dict | None = None) -> None:
    import json as _json
    from redis import asyncio as _aioredis
    from api.core.config import get_settings as _get_settings
    _s = _get_settings()
    status_str = status.value if hasattr(status, "value") else str(status)
    _r = _aioredis.Redis.from_url(_s.redis_url, decode_responses=True)
    try:
        key = f"job:status:{job_id}"
        existing_raw = await _r.get(key)
        existing = _json.loads(existing_raw) if existing_raw else {}
        existing["status"] = status_str
        if error is not None:
            existing["error"] = error
        if extra:
            existing.update(extra)
        await _r.setex(key, _s.redis_result_ttl, _json.dumps(existing))
    finally:
        await _r.aclose()
