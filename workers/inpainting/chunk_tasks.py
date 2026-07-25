from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import Any

import numpy as np
import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.config import get_settings
from api.core.metrics import (
    INPAINT_POSTPROCESS_SECONDS,
    INPAINT_ROI_COVERAGE,
    INPAINT_ROI_MODE_TOTAL,
    SEGMENTS_TOTAL,
)
from api.core.redis import acquire_redis, publish_progress, set_job_status
from workers.celery_app import celery_app
from workers.inpainting.tasks import (
    MODAL_MIN_HEIGHT,
    MODAL_MIN_WIDTH,
    _encode_frames_jpeg,
    _encode_masks_png,
    _make_s3_client,
    _read_fps,
    _read_resolution,
    _run_modal_inpainting,
    _run_propainter_inpainting,
)

log = structlog.get_logger(__name__)

COMPLETED_CHUNKS_TTL = 86400  # 24hr

# ROI mode: crop around the masked object and inpaint at native resolution,
# then paste the result back into the full-res frame. Skipped when the object
# is very large (the crop is basically the whole frame anyway).
ROI_PADDING_RATIO = 0.20
ROI_MAX_W = 960
ROI_MAX_H = 720
ROI_COVERAGE_CUTOFF = 0.60  # object > 60% of frame → use full-frame path

# Three-zone mask: ProPainter owns the core, the original frame owns everything
# past the transition band, and the band itself is a smoothstep ramp between the
# two. Replaces the hard binary composite that produced visible mask outlines.
TRANSITION_PX = 16
GRAIN_STRENGTH = 0.6

# ProPainter composites its own result with a mask dilated by only ~4px, so the
# alpha ramp above (core ≤12px + 16px band) would fall outside the generated
# region and blend original against original — leaving ProPainter's hard edge
# fully intact. Widening the mask we hand it puts the whole ramp inside real
# generated content, which is the point of the three-zone blend.
INPAINT_MASK_MARGIN_PX = 28

# How far past the mask the post-processing stages can reach: the widest core
# (12px) plus the transition band (16px) plus the 61px colour/grain ring kernel.
# 96 leaves headroom on all three.
POSTPROCESS_MARGIN_PX = 96


def compute_roi_crop(
    masks: Any,
    frames: Any,
    padding_ratio: float = ROI_PADDING_RATIO,
    max_crop_w: int = ROI_MAX_W,
    max_crop_h: int = ROI_MAX_H,
) -> tuple[int, int, int, int]:
    """Return the (x1, y1, x2, y2) box covering every mask in the chunk, padded.

    The crop is what actually gets sent to ProPainter: inpainting a 640x480 box
    out of a 1080p frame keeps the object near native resolution instead of
    downscaling the whole frame to 854x480. Dimensions are snapped to a multiple
    of 8 because ProPainter's patch embedding requires it.

    Falls back to the full frame when no mask in the chunk has any pixels set.
    """
    H, W = masks[0].shape[:2]

    all_ys, all_xs = [], []
    for mask in masks:
        if mask.max() == 0:
            continue
        ys, xs = np.where(mask > 127)
        if len(ys) > 0:
            all_ys.extend(ys.tolist())
            all_xs.extend(xs.tolist())

    if not all_ys:
        return 0, 0, W, H

    min_y, max_y = min(all_ys), max(all_ys)
    min_x, max_x = min(all_xs), max(all_xs)

    obj_h = max(max_y - min_y, 1)
    obj_w = max(max_x - min_x, 1)
    pad_y = int(obj_h * padding_ratio) + 40
    pad_x = int(obj_w * padding_ratio) + 40

    y1 = max(0, min_y - pad_y)
    y2 = min(H, max_y + pad_y)
    x1 = max(0, min_x - pad_x)
    x2 = min(W, max_x + pad_x)

    cy = (y1 + y2) // 2
    cx = (x1 + x2) // 2
    crop_h = min(y2 - y1, max_crop_h)
    crop_w = min(x2 - x1, max_crop_w)
    crop_h = (crop_h // 8) * 8
    crop_w = (crop_w // 8) * 8

    y1 = max(0, cy - crop_h // 2)
    y2 = min(H, y1 + crop_h)
    y1 = max(0, y2 - crop_h)
    x1 = max(0, cx - crop_w // 2)
    x2 = min(W, x1 + crop_w)
    x1 = max(0, x2 - crop_w)

    return int(x1), int(y1), int(x2), int(y2)


def create_three_zone_mask(mask: Any, flow_speed: float | None = None) -> Any:
    """Turn a binary mask into a soft compositing alpha.

    Zone 1 (core):       dilated mask — ProPainter's output replaces it entirely
    Zone 2 (transition): TRANSITION_PX band with a smoothstep ramp
    Zone 3 (protected):  original pixels, untouched

    flow_speed (px/frame of mask motion) widens the core on fast-moving objects,
    where tracking lags behind the object and a tight mask leaves ghost edges.

    Returns: alpha (H, W) float32 in [0, 1]; 1.0 = fully inpainted.
    """
    import cv2
    from scipy.ndimage import distance_transform_edt

    binary = (mask > 127).astype(np.uint8)

    if flow_speed is not None:
        dil_radius = int(np.clip(3 + 0.20 * flow_speed, 3, 12))
    else:
        dil_radius = 6

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dil_radius * 2 + 1, dil_radius * 2 + 1)
    )
    core = cv2.dilate(binary, kernel, iterations=1)

    trans_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (TRANSITION_PX * 2 + 1, TRANSITION_PX * 2 + 1)
    )
    outer = cv2.dilate(core, trans_kernel, iterations=1)

    transition_band = outer - core
    dist_inside_core = distance_transform_edt(1 - core)

    alpha = np.zeros(mask.shape[:2], dtype=np.float32)
    alpha[core > 0] = 1.0

    # Smoothstep falloff across the band: 1.0 at the core edge → 0.0 at its rim
    t = np.clip(dist_inside_core / TRANSITION_PX, 0, 1)
    smooth = 1.0 - (3 * t**2 - 2 * t**3)
    alpha[transition_band > 0] = smooth[transition_band > 0]

    return alpha


def local_colour_correction(inpainted: Any, original: Any, mask_alpha: Any) -> Any:
    """Match the colour of the inpainted region to the pixels surrounding it.

    ProPainter hallucinates content from other frames, so the patch often sits a
    few Lab units off its neighbourhood and reads as a visible rectangle. This
    compares a ring just outside the mask against a ring just inside it and
    applies the affine correction that reconciles them, strongest at the
    boundary where the mismatch is visible and tapering to nothing at the centre.

    Median/MAD statistics rather than mean/std so a few blown-out pixels in
    either ring cannot drag the whole correction.

    inpainted/original are BGR uint8; returns BGR uint8.
    """
    import cv2

    inp_lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Reference ring: original pixels just outside the mask
    mask_binary = (mask_alpha > 0.1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    outer_ring = (cv2.dilate(mask_binary, kernel) - mask_binary).astype(bool)

    if outer_ring.sum() < 100:
        return inpainted

    # Inner ring: generated pixels just inside the inpainted region
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    inner_ring = (mask_binary - cv2.erode(mask_binary, inner_kernel)).astype(bool)

    # Strong near the boundary (alpha~0.5), weak at the centre (alpha=1.0)
    boundary_weight = (1.0 - (mask_alpha - 0.5).clip(0, 1) * 2.0).clip(0, 1)

    corrected_lab = inp_lab.copy()
    for c in range(3):
        ref_vals = orig_lab[:, :, c][outer_ring]
        gen_vals = inp_lab[:, :, c][inner_ring]

        if len(ref_vals) < 50 or len(gen_vals) < 50:
            continue

        ref_med = np.median(ref_vals)
        gen_med = np.median(gen_vals)
        ref_mad = np.median(np.abs(ref_vals - ref_med)) + 1e-6
        gen_mad = np.median(np.abs(gen_vals - gen_med)) + 1e-6

        scale = float(np.clip(ref_mad / gen_mad, 0.7, 1.4))  # prevent overcorrection
        offset = ref_med - scale * gen_med

        correction = scale * inp_lab[:, :, c] + offset
        corrected_lab[:, :, c] = (
            boundary_weight * correction + (1 - boundary_weight) * inp_lab[:, :, c]
        )

    # Blend back in BGR, not Lab: a BGR→Lab→BGR round-trip shifts 8-bit values
    # by a unit or two, and blending in Lab would spread that error across every
    # untouched pixel in the frame. Compositing here keeps alpha=0 bit-identical.
    corrected_bgr = cv2.cvtColor(
        corrected_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    ).astype(np.float32)
    mask_3ch = mask_alpha[:, :, None]
    result = mask_3ch * corrected_bgr + (1 - mask_3ch) * original.astype(np.float32)
    return result.clip(0, 255).astype(np.uint8)


def restore_texture_and_grain(
    inpainted: Any,
    original: Any,
    mask_alpha: Any,
    sigma: float = 1.5,
    noise: Any | None = None,
) -> Any:
    """Add back the high-frequency grain that ProPainter's smoothing removes.

    The inpainted patch is noticeably cleaner than the footage around it, which
    reads as a soft smear even when the colour matches. Grain strength is
    measured from the residual (original minus its own low-pass) in a ring
    outside the mask, then re-synthesised as spatially-correlated noise inside.

    Pass `noise` to reuse one field across a whole chunk: independently sampled
    noise per frame flickers and costs more temporal consistency than the grain
    buys back in realism.

    inpainted/original are BGR uint8; returns BGR uint8.
    """
    import cv2

    orig_blur = cv2.GaussianBlur(original.astype(np.float32), (0, 0), sigma)
    texture_residual = original.astype(np.float32) - orig_blur

    # Sample the residual from an annulus just outside the mask
    mask_binary = (mask_alpha > 0.1).astype(np.uint8)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    outer_band = (cv2.dilate(mask_binary, dilate_k) - mask_binary).astype(bool)

    if outer_band.sum() < 200:
        return inpainted

    residual_gray = cv2.cvtColor(
        np.abs(texture_residual).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    grain_sigma = float(np.clip(np.std(residual_gray[outer_band]), 1.0, 12.0))

    if noise is None:
        noise = make_grain_noise(original.shape[:2])
    # Chroma noise is far less visible than luma, so damp two of the channels
    noise_3ch = np.stack([noise, noise * 0.6, noise * 0.6], axis=2) * grain_sigma

    texture_weight = mask_alpha[:, :, None] * GRAIN_STRENGTH
    result = inpainted.astype(np.float32) + noise_3ch * texture_weight
    return result.clip(0, 255).astype(np.uint8)


def make_grain_noise(shape: tuple[int, int], seed: int | None = None) -> Any:
    """Unit-variance, slightly spatially-correlated noise field for grain."""
    import cv2

    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 0.8)
    # Blurring costs most of the variance; renormalise so the caller's measured
    # grain sigma is the sigma actually applied.
    return noise / max(float(noise.std()), 1e-6)


@celery_app.task(
    bind=True,
    queue="inpainting",
    max_retries=2,
    soft_time_limit=600,
    time_limit=720,
    name="workers.inpainting.chunk_tasks.inpaint_chunk",
)
def inpaint_chunk(
    self,
    job_id: str,
    chunk_index: int,
    total_chunks: int,
    segment_s3_key: str,
    masks_s3_key: str,
    chunk_start_frame: int,
    chunk_end_frame: int,
) -> dict[str, Any]:
    """Inpaint one chunk of frames with ProPainter, guided by its tracked masks.

    masks_s3_key points at the full-video (T, H, W) mask array produced by
    SAM2 Video Predictor; this task slices out its own frame range.

    The quality pipeline runs in a fixed order — each stage assumes the previous
    one has already happened:
      ROI crop → ProPainter → ROI composite → three-zone alpha →
      Lab colour correction → grain restoration

    Once every chunk's inpainting has completed (tracked via an atomic Redis
    counter), triggers stitch_all_chunks to assemble the final video.
    """
    settings = get_settings()
    bound_log = log.bind(
        job_id=job_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        task_id=self.request.id,
    )
    bound_log.info("chunk_inpainting_started")

    loop = asyncio.new_event_loop()

    def _report_progress(pct: int) -> None:
        # Progress events use 1-based chunk numbers (see api/core/progress.py)
        loop.run_until_complete(
            _publish(job_id, {"stage": "inpainting", "chunk": chunk_index + 1, "total": total_chunks, "pct": pct})
        )

    try:
        _report_progress(0)
        SEGMENTS_TOTAL.labels(status="chunk_inpainting_started").inc()

        s3 = _make_s3_client(settings)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "video.mp4")
            masks_path = os.path.join(tmp, "tracked_masks.npy")
            s3.download_file(settings.s3_bucket, segment_s3_key, video_path)
            s3.download_file(settings.s3_bucket, masks_s3_key, masks_path)

            all_masks = np.load(masks_path)  # (T, H, W) uint8 full-video masks
            masks = all_masks[chunk_start_frame:chunk_end_frame]
            width, height = _read_resolution(video_path)
            fps = _read_fps(video_path)

            frames = _extract_chunk_frames(video_path, chunk_start_frame, chunk_end_frame)
            if len(frames) != len(masks):
                raise RuntimeError(
                    f"frame/mask count mismatch for chunk {chunk_index}: "
                    f"got {len(frames)} frames for {len(masks)} masks from {segment_s3_key}"
                )
            bound_log.info("chunk_frames_extracted", num_frames=len(frames))

            # Tracking runs at ≤1024px and pads to a multiple of 8; every stage
            # below indexes masks with frame coordinates, so undo that first.
            masks = _masks_to_frame_size(masks, width, height)

            use_modal = settings.modal_enabled and (
                width > MODAL_MIN_WIDTH or height > MODAL_MIN_HEIGHT
            )

            output_key = f"jobs/{job_id}/chunks/{chunk_index}/inpainted_chunk.mp4"
            output_path = os.path.join(tmp, "inpainted_chunk.mp4")

            x1, y1, x2, y2 = compute_roi_crop(masks, frames)
            roi_coverage = ((x2 - x1) * (y2 - y1)) / float(width * height)
            use_roi = (
                x2 > x1
                and y2 > y1
                and roi_coverage < ROI_COVERAGE_CUTOFF
                and (x2 - x1, y2 - y1) != (width, height)
            )
            INPAINT_ROI_COVERAGE.observe(roi_coverage)
            INPAINT_ROI_MODE_TOTAL.labels(mode="roi" if use_roi else "full_frame").inc()

            if use_roi:
                # Copied, not sliced: OpenCV rejects the non-contiguous views a
                # bare crop would produce.
                proc_frames = [np.ascontiguousarray(f[y1:y2, x1:x2]) for f in frames]
                proc_masks = np.ascontiguousarray(masks[:, y1:y2, x1:x2])
                bound_log.info(
                    "roi_crop_selected",
                    roi=f"{x2 - x1}x{y2 - y1}+{x1}+{y1}",
                    frame_size=f"{width}x{height}",
                    roi_coverage_pct=round(roi_coverage * 100, 2),
                )
            else:
                proc_frames, proc_masks = frames, masks
                bound_log.info(
                    "roi_crop_skipped",
                    roi_coverage_pct=round(roi_coverage * 100, 2),
                    cutoff_pct=ROI_COVERAGE_CUTOFF * 100,
                )

            # ProPainter regenerates this widened region; the untouched masks
            # stay the reference for the alpha ramp further down.
            gen_masks = _dilate_masks(proc_masks, INPAINT_MASK_MARGIN_PX)

            if use_modal:
                bound_log.info(
                    "using_modal_propainter",
                    resolution=f"{proc_frames[0].shape[1]}x{proc_frames[0].shape[0]}",
                    num_frames=len(proc_frames),
                )
                frames_bytes = _encode_frames_jpeg(proc_frames)
                masks_bytes = _encode_masks_png(gen_masks)
                _report_progress(20)

                video_bytes = _run_modal_inpainting(frames_bytes, masks_bytes, fps)
                _report_progress(75)

                modal_path = os.path.join(tmp, "modal_out.mp4")
                with open(modal_path, "wb") as f:
                    f.write(video_bytes)
                # Post-processing works on pixels, so the returned clip has to be
                # decoded rather than uploaded straight through as it used to be.
                comp_frames = _decode_video_frames(modal_path)
                status = "modal_hires"
            else:
                bound_log.info(
                    "using_local_propainter",
                    resolution=f"{proc_frames[0].shape[1]}x{proc_frames[0].shape[0]}",
                    num_frames=len(proc_frames),
                )
                comp_frames = _run_propainter_inpainting(
                    proc_frames, gen_masks, settings, bound_log, _report_progress
                )
                status = "real"

            comp_frames = _align_to_reference(comp_frames, proc_frames, bound_log)

            if use_roi:
                inpainted_frames = []
                for orig, patch in zip(frames, comp_frames):
                    full = orig.copy()
                    full[y1:y2, x1:x2] = patch
                    inpainted_frames.append(full)
            else:
                inpainted_frames = comp_frames

            _report_progress(85)
            final_frames = _postprocess_chunk(
                inpainted_frames, frames, masks, chunk_index, bound_log
            )

            _write_chunk_video(final_frames, output_path, fps)
            s3.upload_file(output_path, settings.s3_bucket, output_key)

        _report_progress(100)
        SEGMENTS_TOTAL.labels(status="chunk_inpainting_complete").inc()

        completed_count = loop.run_until_complete(
            _increment_completed_chunks(job_id, total_chunks)
        )
        bound_log.info("chunk_inpainting_done", completed_chunks=completed_count)

        if completed_count == total_chunks:
            from workers.stitching.tasks import stitch_all_chunks
            stitch_all_chunks.apply_async(
                # segment_s3_key = the original video, used to restore audio
                args=[job_id, total_chunks, segment_s3_key],
                queue="stitching",
            )

        return {
            "job_id": job_id,
            "chunk_index": chunk_index,
            "inpainted_chunk_s3_key": output_key,
            "status": status,
        }

    except SoftTimeLimitExceeded:
        bound_log.error("chunk_inpainting_soft_timeout")
        SEGMENTS_TOTAL.labels(status="chunk_inpainting_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", f"inpaint_chunk timed out on chunk {chunk_index}")
        )
        raise
    except Exception as exc:
        bound_log.exception("chunk_inpainting_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="chunk_inpainting_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_chunk_frames(video_path: str, start_frame: int, end_frame: int) -> list[np.ndarray]:
    """Read frames [start_frame, end_frame) as (H, W, 3) uint8 RGB arrays."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(end_frame - start_frame):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _decode_video_frames(video_path: str) -> list[np.ndarray]:
    """Decode every frame of a video as (H, W, 3) uint8 RGB arrays."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _masks_to_frame_size(masks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map tracked masks from tracking geometry onto the frame's pixel grid.

    sam2_video_tracker downscales to ≤1024px and pads the bottom/right edges to
    a multiple of 8. Resizing the padded array straight to the frame size would
    squash the mask by up to 7px worth of offset, so the padding is cropped off
    before the resize.
    """
    import cv2

    mask_h, mask_w = masks.shape[1:3]
    if (mask_w, mask_h) == (width, height):
        return masks

    from workers.segmentation.sam2_video_tracker import _processing_geometry

    proc_w, proc_h, pad_w, pad_h, _ = _processing_geometry(width, height)
    if (mask_w, mask_h) == (proc_w + pad_w, proc_h + pad_h):
        masks = masks[:, :proc_h, :proc_w]

    return np.stack(
        [cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST) for m in masks]
    )


def _dilate_masks(masks: np.ndarray, radius_px: int) -> np.ndarray:
    """Grow every mask in the stack by radius_px, keeping 0/255 uint8 values."""
    import cv2

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1)
    )
    return np.stack(
        [cv2.dilate((m > 127).astype(np.uint8) * 255, kernel, iterations=1) for m in masks]
    )


def _align_to_reference(
    frames: list[np.ndarray],
    reference: list[np.ndarray],
    bound_log: Any,
) -> list[np.ndarray]:
    """Force ProPainter's output back onto the reference frames' count and size.

    The Modal path round-trips through an H.264 encode, and a dropped tail frame
    or an encoder-adjusted dimension would otherwise surface much later as a
    stitching frame-count mismatch.
    """
    import cv2

    ref_h, ref_w = reference[0].shape[:2]
    aligned = []
    for i, ref in enumerate(reference):
        frame = frames[i] if i < len(frames) else None
        if frame is None:
            aligned.append(ref)
            continue
        if frame.shape[:2] != (ref_h, ref_w):
            frame = cv2.resize(frame, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
        aligned.append(frame)

    if len(frames) != len(reference):
        bound_log.warning(
            "inpainted_frame_count_mismatch",
            returned=len(frames),
            expected=len(reference),
        )
    return aligned


def _estimate_flow_speed(masks: np.ndarray) -> float | None:
    """Median per-frame mask centroid displacement, in pixels.

    A cheap stand-in for optical flow: it only needs to tell a static object from
    a fast-moving one so create_three_zone_mask can widen the core accordingly.
    Returns None when fewer than two frames carry a mask.
    """
    centroids: list[tuple[float, float] | None] = []
    for mask in masks:
        ys, xs = np.nonzero(mask > 127)
        centroids.append((float(xs.mean()), float(ys.mean())) if xs.size else None)

    deltas = [
        float(np.hypot(b[0] - a[0], b[1] - a[1]))
        for a, b in zip(centroids, centroids[1:])
        if a is not None and b is not None
    ]
    return float(np.median(deltas)) if deltas else None


def _postprocess_chunk(
    inpainted: list[np.ndarray],
    originals: list[np.ndarray],
    masks: np.ndarray,
    chunk_index: int,
    bound_log: Any,
) -> list[np.ndarray]:
    """Blend, colour-match and re-grain every frame of the chunk.

    Runs three-zone alpha compositing, Lab colour correction and grain
    restoration in that order — colour correction reads the blended pixels, and
    grain must land on top of the final colour, not be rescaled by it.

    Frames are RGB in and RGB out; the colour/grain stages work in BGR because
    OpenCV's Lab conversion is defined on BGR input.
    """
    import cv2

    started = time.perf_counter()
    height, width = originals[0].shape[:2]
    flow_speed = _estimate_flow_speed(masks)
    # One noise field for the whole chunk: resampling it per frame makes the
    # grain crawl, which costs more temporal consistency than it buys realism.
    noise_full = make_grain_noise((height, width), seed=chunk_index)

    box = _postprocess_box(masks, width, height)
    if box is None:
        bound_log.info("chunk_postprocess_skipped_empty_masks")
        return [o.copy() for o in originals]

    bx1, by1, bx2, by2 = box
    noise = np.ascontiguousarray(noise_full[by1:by2, bx1:bx2])

    out: list[np.ndarray] = []
    for inp_rgb, orig_rgb, mask in zip(inpainted, originals, masks):
        inp_c = np.ascontiguousarray(inp_rgb[by1:by2, bx1:bx2])
        orig_c = np.ascontiguousarray(orig_rgb[by1:by2, bx1:bx2])

        alpha = create_three_zone_mask(
            np.ascontiguousarray(mask[by1:by2, bx1:bx2]), flow_speed
        )
        alpha_3 = alpha[:, :, None]
        blended = (
            alpha_3 * inp_c.astype(np.float32)
            + (1 - alpha_3) * orig_c.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        orig_bgr = cv2.cvtColor(orig_c, cv2.COLOR_RGB2BGR)
        corrected = local_colour_correction(blended_bgr, orig_bgr, alpha)
        grained = restore_texture_and_grain(corrected, orig_bgr, alpha, noise=noise)

        frame = orig_rgb.copy()
        frame[by1:by2, bx1:bx2] = cv2.cvtColor(grained, cv2.COLOR_BGR2RGB)
        out.append(frame)

    elapsed = time.perf_counter() - started
    INPAINT_POSTPROCESS_SECONDS.observe(elapsed)
    bound_log.info(
        "chunk_postprocessed",
        num_frames=len(out),
        flow_speed_px=round(flow_speed, 2) if flow_speed is not None else None,
        work_box=f"{bx2 - bx1}x{by2 - by1}+{bx1}+{by1}",
        elapsed_sec=round(elapsed, 2),
    )
    return out


def _postprocess_box(
    masks: np.ndarray, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Region the post-processing stages can affect; None if nothing is masked.

    Every stage is the identity outside mask + POSTPROCESS_MARGIN_PX: alpha is
    0 there (so the blend and colour composite return the original untouched)
    and the grain weight is 0. Confining the work to this box makes it ~5x
    cheaper on 1080p without changing a single output pixel — the margin is
    wider than the furthest-reaching kernel, so even the ring statistics that
    drive the correction see exactly the same pixels.
    """
    ys, xs = np.nonzero(masks.max(axis=0) > 127)
    if ys.size == 0:
        return None

    x1 = max(0, int(xs.min()) - POSTPROCESS_MARGIN_PX)
    y1 = max(0, int(ys.min()) - POSTPROCESS_MARGIN_PX)
    x2 = min(width, int(xs.max()) + POSTPROCESS_MARGIN_PX + 1)
    y2 = min(height, int(ys.max()) + POSTPROCESS_MARGIN_PX + 1)
    return x1, y1, x2, y2


def _write_chunk_video(frames: list[np.ndarray], output_path: str, fps: float) -> None:
    """Encode RGB frames to H.264 CRF 18.

    Not cv2.VideoWriter's mp4v: the stitcher decodes these chunks and re-encodes
    them, so a lossy intermediate codec would spend quality that the colour and
    grain work just bought back.
    """
    import cv2

    from workers.stitching.tasks import _write_frames_ffmpeg

    height, width = frames[0].shape[:2]
    bgr_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    _write_frames_ffmpeg(bgr_frames, output_path, width, height, fps)


async def _increment_completed_chunks(job_id: str, total_chunks: int) -> int:
    """Atomically increment the completed-chunk counter and return the new count."""
    from redis import asyncio as _aioredis
    from api.core.config import get_settings as _gs
    _r = _aioredis.Redis.from_url(_gs().redis_url, decode_responses=True)
    try:
        key = f"jobs:{job_id}:completed_chunks"
        count = await _r.incr(key)
        await _r.expire(key, COMPLETED_CHUNKS_TTL)
        return count
    finally:
        await _r.aclose()


async def _publish(job_id: str, data: dict) -> None:
    import json as _json
    from redis import asyncio as _aioredis
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
