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
    BACKGROUND_PLATE_COVERAGE,
    CHUNK_DIFFICULTY_SCORE,
    DIFFUERASER_SECONDS,
    INPAINT_POSTPROCESS_SECONDS,
    INPAINT_ROI_COVERAGE,
    INPAINT_ROI_MODE_TOTAL,
    SEGMENTS_TOTAL,
)
from api.core.redis import acquire_redis, publish_progress, set_job_status
from workers.celery_app import celery_app
from workers.inpainting.background_plate import (
    PLATE_CONFIDENCE_THRESHOLD,
    apply_background_plate,
    build_background_plate,
)
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

# Below this recoverable fraction *of the hole* the background plate is not
# worth compositing: a handful of scattered pixels cannot beat ProPainter's fill
# and only risks seams where the two meet. Measured on Wimbledon, a chunk where
# the plate genuinely helps recovers 18-53% of the hole, so this is a floor and
# not a tuning knob.
MIN_PLATE_COVERAGE = 0.05

# How far past the mask the post-processing stages can reach: the widest core
# (12px) plus the transition band (16px) plus the 61px colour/grain ring kernel.
# 96 leaves headroom on all three.
POSTPROCESS_MARGIN_PX = 96

# Defect-only second ProPainter pass. After post-processing, the finished chunk
# is measured against the originals for the two failure modes ProPainter
# actually produces: a colour step at the hole boundary, and a smeared fill with
# no texture. Only the pixels that fail get re-generated.
DEFECT_COLOUR_THRESHOLD = 15.0    # mean |RGB| delta across the mask boundary ring
DEFECT_VARIANCE_THRESHOLD = 20.0  # mean |Laplacian| inside the mask
SECOND_PASS_MIN_RATIO = 0.05      # below this the defect is not worth a second pass
SECOND_PASS_MAX_RATIO = 0.25      # above this a second pass is slow and likely worse
SECOND_PASS_MASK_MARGIN_PX = 12   # ProPainter needs slack around what it regenerates
SECOND_PASS_FEATHER_SIGMA = 3.0   # soften the defect edge so the repair has no seam


def compute_chunk_difficulty(
    frames: list[np.ndarray],
    masks: Any,
    inpainted_frames: list[np.ndarray],
) -> float:
    """Score how hard this chunk was for ProPainter, in [0, 1].

    Higher means more likely to benefit from DiffuEraser's diffusion
    refinement. Combines three cheap signals: how much of the frame the mask
    covers, how textured the region being replaced is, and how far the
    inpainted pixels ended up from the originals around the mask edge.

    Frames are RGB, matching the rest of this module.
    """
    import cv2

    scores: list[float] = []

    # 1. Mask area ratio — large holes leave ProPainter less to copy from
    mask_coverage = float(np.mean([m.mean() / 255.0 for m in masks]))
    scores.append(min(mask_coverage * 3.0, 1.0))

    # 2. Texture entropy inside the mask — flat regions are easy to fill
    entropies: list[float] = []
    for frame, mask in zip(frames, masks):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        masked_region = gray[mask > 127]
        if len(masked_region) > 100:
            hist = np.histogram(masked_region, bins=32, range=(0, 256))[0]
            hist = hist / (hist.sum() + 1e-8)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            entropies.append(entropy / 5.0)  # 5 bits ≈ max for 32 bins
    if entropies:
        scores.append(min(float(np.mean(entropies)), 1.0))

    # 3. How far the result moved from the original around the mask edge
    discontinuities: list[float] = []
    for orig, inp, mask in zip(frames, inpainted_frames, masks):
        mask_binary = (mask > 127).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        boundary = cv2.dilate(mask_binary, kernel) - cv2.erode(mask_binary, kernel)
        if boundary.sum() > 0:
            orig_boundary = orig[boundary > 0].astype(float)
            inp_boundary = inp[boundary > 0].astype(float)
            diff = np.abs(orig_boundary - inp_boundary).mean() / 255.0
            discontinuities.append(diff)
    if discontinuities:
        scores.append(min(float(np.mean(discontinuities)) * 5.0, 1.0))

    return float(np.mean(scores)) if scores else 0.0


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


def detect_inpainting_defects(
    original_frames: list[np.ndarray],
    inpainted_frames: list[np.ndarray],
    masks: Any,
    threshold_colour: float = DEFECT_COLOUR_THRESHOLD,
    threshold_variance: float = DEFECT_VARIANCE_THRESHOLD,
) -> list[np.ndarray]:
    """Find the regions a finished chunk got visibly wrong.

    Two signals, both cheap and both matching a failure mode that is actually
    visible in the output:

      1. Boundary colour discontinuity — the fill sits at a different level from
         the frame around it, so the hole reads as a patch. Measured on a ring
         straddling the mask edge; when it fails, the band just inside the mask
         is marked.
      2. Low texture variance — the fill is a smear with no detail, which reads
         as a blur even when the colour matches. Measured as mean |Laplacian|
         inside the mask; when it fails, the whole masked region is marked.

    Frames are RGB (as everywhere in this module); the colour test is
    channel-order agnostic and the variance test uses the RGB→grey conversion.

    Returns: list of (H, W) uint8 masks, 0 or 255, always a subset of `masks`.
    """
    import cv2

    defect_masks: list[np.ndarray] = []

    for orig, inp, mask in zip(original_frames, inpainted_frames, masks):
        mask_binary = (mask > 127).astype(np.uint8)
        H, W = mask.shape[:2]
        defect = np.zeros((H, W), dtype=np.uint8)

        if mask_binary.sum() == 0:
            defect_masks.append(defect)
            continue

        # 1. Boundary colour discontinuity
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        boundary = cv2.dilate(mask_binary, kernel) - cv2.erode(mask_binary, kernel)

        if boundary.sum() > 0:
            orig_boundary = orig[boundary > 0].astype(float)
            inp_boundary = inp[boundary > 0].astype(float)
            colour_diff = np.abs(orig_boundary - inp_boundary).mean(axis=1)
            if colour_diff.mean() > threshold_colour:
                # Expand the ring inward; clipped to the mask so the defect
                # region never reaches pixels ProPainter did not generate.
                defect_region = cv2.dilate(boundary, kernel, iterations=2)
                defect = np.maximum(defect, (defect_region * mask_binary) * 255)

        # 2. Low texture variance (smeared/blurry fill)
        inp_gray = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY).astype(float)
        local_var = np.abs(cv2.Laplacian(inp_gray, cv2.CV_64F))

        inside_mask = mask_binary > 0
        if local_var[inside_mask].mean() < threshold_variance:
            defect = np.maximum(defect, mask_binary * 255)

        defect_masks.append(defect)

    return defect_masks


@celery_app.task(
    bind=True,
    queue="inpainting",
    max_retries=2,
    # Doubled from 600s: the defect-only second pass adds a whole extra
    # ProPainter call to the worst-case chunk.
    soft_time_limit=1200,
    time_limit=1320,
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
      ROI crop → ProPainter → ROI composite → background plate recovery →
      three-zone alpha → Lab colour correction → grain restoration →
      defect-only second ProPainter pass

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

            _apply_background_plates(inpainted_frames, frames, masks, bound_log)

            _report_progress(85)
            final_frames = _postprocess_chunk(
                inpainted_frames, frames, masks, chunk_index, bound_log
            )

            final_frames = _maybe_second_pass(
                final_frames, frames, masks, (x1, y1, x2, y2), use_roi,
                use_modal, fps, settings, bound_log,
            )

            difficulty = compute_chunk_difficulty(frames, masks, final_frames)
            CHUNK_DIFFICULTY_SCORE.observe(difficulty)
            bound_log.info(
                "chunk_difficulty_score",
                chunk_index=chunk_index,
                difficulty=round(difficulty, 3),
            )

            refined = _maybe_refine_with_diffueraser(
                final_frames, frames, masks, difficulty, fps, settings, bound_log
            )
            if refined is not None:
                final_frames = refined
                status = f"{status}_diffueraser"

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


def _apply_background_plates(
    inpainted_frames: list[np.ndarray],
    frames: list[np.ndarray],
    masks: np.ndarray,
    bound_log: Any,
) -> None:
    """Replace ProPainter's guess with real background pixels where they exist.

    Runs before the three-zone blend, colour correction and grain restoration so
    those stages treat the recovered pixels exactly like any other fill — the
    plate is composited only inside the mask, so the alpha ramp and the ring
    statistics downstream see the same geometry either way.

    Mutates inpainted_frames in place; frames stay RGB throughout.
    """
    started = time.perf_counter()
    plates_applied = 0

    for i in range(len(inpainted_frames)):
        plate, confidence = build_background_plate(
            frames=frames,
            masks=masks,
            target_idx=i,
        )
        # Coverage is measured against the hole, not the frame. The object is
        # only 1-5% of a 1080p frame here, so a whole-frame denominator can
        # never clear MIN_PLATE_COVERAGE and the plate would never fire.
        hole = masks[i] > 127
        recovered = confidence >= PLATE_CONFIDENCE_THRESHOLD
        coverage = float(recovered[hole].mean()) if hole.any() else 0.0
        BACKGROUND_PLATE_COVERAGE.observe(coverage)
        if coverage > MIN_PLATE_COVERAGE:
            inpainted_frames[i] = apply_background_plate(
                orig=frames[i],
                propainter=inpainted_frames[i],
                mask=masks[i],
                plate=plate,
                confidence=confidence,
            )
            plates_applied += 1

    bound_log.info(
        "background_plate_recovery",
        frames_improved=plates_applied,
        total_frames=len(inpainted_frames),
        elapsed_sec=round(time.perf_counter() - started, 2),
    )


def _maybe_second_pass(
    final_frames: list[np.ndarray],
    originals: list[np.ndarray],
    masks: np.ndarray,
    roi: tuple[int, int, int, int],
    use_roi: bool,
    use_modal: bool,
    fps: float,
    settings: Any,
    bound_log: Any,
) -> list[np.ndarray]:
    """Re-inpaint only the pixels the first pass got wrong, if there are any.

    ProPainter is re-run on the *original* frames with the defect regions as the
    hole, then only those pixels are taken from the result. Feeding it the
    originals rather than the finished chunk keeps the second pass from
    compounding the first pass's mistakes.

    Best-effort like DiffuEraser refinement: any failure leaves the first-pass
    result in place rather than failing the chunk.
    """
    import cv2

    from api.core.metrics import SECOND_PASS_DEFECT_RATIO, SECOND_PASS_TOTAL

    defect_masks = detect_inpainting_defects(originals, final_frames, masks)

    # Areas in pixels, not summed 0/255 values — a defect mask stores 255 per
    # pixel while the reference mask is counted as a boolean, so summing raw
    # values would inflate the ratio 255x and never let a second pass run.
    total_defect_area = sum(int((d > 0).sum()) for d in defect_masks)
    total_mask_area = sum(int((m > 127).sum()) for m in masks)
    defect_ratio = total_defect_area / (total_mask_area + 1)

    SECOND_PASS_DEFECT_RATIO.observe(defect_ratio)
    bound_log.info(
        "defect_detection",
        defect_ratio=round(float(defect_ratio), 3),
        needs_second_pass=defect_ratio > SECOND_PASS_MIN_RATIO,
    )

    if defect_ratio >= SECOND_PASS_MAX_RATIO:
        # Too much of the fill is bad for a targeted repair to help: the pass
        # would cost as much as the first one and is as likely to make it worse.
        SECOND_PASS_TOTAL.labels(outcome="high_defect_ratio").inc()
        bound_log.warning(
            "high_defect_ratio_skipping_second_pass",
            defect_ratio=round(float(defect_ratio), 3),
        )
        return final_frames

    if defect_ratio <= SECOND_PASS_MIN_RATIO:
        SECOND_PASS_TOTAL.labels(outcome="not_needed").inc()
        return final_frames

    started = time.perf_counter()
    try:
        x1, y1, x2, y2 = roi
        defect_stack = np.stack(defect_masks)
        if use_roi:
            proc_frames = [np.ascontiguousarray(f[y1:y2, x1:x2]) for f in originals]
            proc_defects = np.ascontiguousarray(defect_stack[:, y1:y2, x1:x2])
        else:
            proc_frames = originals
            proc_defects = defect_stack

        # Same reason as INPAINT_MASK_MARGIN_PX on the first pass: ProPainter
        # composites against a mask dilated by only ~4px, so the region we
        # actually take pixels from has to sit well inside what it regenerated.
        gen_masks = _dilate_masks(proc_defects, SECOND_PASS_MASK_MARGIN_PX)

        if use_modal:
            video_bytes = _run_modal_inpainting(
                _encode_frames_jpeg(proc_frames), _encode_masks_png(gen_masks), fps
            )
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_file.write(video_bytes)
                second_path = tmp_file.name
            try:
                second = _decode_video_frames(second_path)
            finally:
                os.unlink(second_path)
        else:
            second = _run_propainter_inpainting(
                proc_frames, gen_masks, settings, bound_log, lambda _pct: None
            )

        second = _align_to_reference(second, proc_frames, bound_log)

        out: list[np.ndarray] = []
        for prior, patch, defect in zip(final_frames, second, defect_masks):
            if defect.max() == 0:
                out.append(prior)
                continue
            if use_roi:
                full = prior.copy()
                full[y1:y2, x1:x2] = patch
                patch = full
            # Feathered rather than the hard defect edge: a binary cut would
            # butt second-pass pixels against first-pass ones and reintroduce
            # exactly the seam the three-zone blend exists to avoid.
            alpha = cv2.GaussianBlur(
                (defect > 0).astype(np.float32), (0, 0), SECOND_PASS_FEATHER_SIGMA
            )[:, :, None]
            out.append(
                (alpha * patch.astype(np.float32) + (1 - alpha) * prior.astype(np.float32))
                .clip(0, 255).astype(np.uint8)
            )

        SECOND_PASS_TOTAL.labels(outcome="repaired").inc()
        bound_log.info(
            "second_pass_complete",
            defect_ratio=round(float(defect_ratio), 3),
            elapsed_sec=round(time.perf_counter() - started, 2),
        )
        return out

    except Exception as exc:
        SECOND_PASS_TOTAL.labels(outcome="failed").inc()
        bound_log.warning(
            "second_pass_failed_skipping",
            error=str(exc),
            elapsed_sec=round(time.perf_counter() - started, 2),
        )
        return final_frames


def _maybe_refine_with_diffueraser(
    inpainted: list[np.ndarray],
    originals: list[np.ndarray],
    masks: np.ndarray,
    difficulty: float,
    fps: float,
    settings: Any,
    bound_log: Any,
) -> list[np.ndarray] | None:
    """Run DiffuEraser over a hard chunk, or return None to keep ProPainter's.

    Refinement is strictly best-effort: any failure — disabled flag, too-short
    chunk, Modal error, wrong frame count back — leaves the ProPainter result
    in place rather than failing the chunk. A refinement pass is an
    optimisation, never a dependency.
    """
    import cv2

    from api.core.metrics import DIFFUERASER_REFINEMENTS_TOTAL

    if not getattr(settings, "diffueraser_enabled", False):
        return None
    if difficulty <= settings.diffueraser_threshold:
        DIFFUERASER_REFINEMENTS_TOTAL.labels(outcome="below_threshold").inc()
        return None

    from pipeline.modal_diffueraser import MIN_FRAMES

    if len(inpainted) < MIN_FRAMES:
        # DiffuEraser raises below this; skip rather than pay for a container.
        bound_log.info("diffueraser_skipped_short_chunk", num_frames=len(inpainted))
        DIFFUERASER_REFINEMENTS_TOTAL.labels(outcome="too_short").inc()
        return None

    started = time.perf_counter()
    try:
        bound_log.info("diffueraser_refinement_started", difficulty=round(difficulty, 3))

        def _jpeg(frame_rgb):
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()

        frames_bytes = [_jpeg(f) for f in originals]
        masks_bytes = [cv2.imencode(".png", m)[1].tobytes() for m in masks]
        prior_bytes = [_jpeg(f) for f in inpainted]

        import modal

        fn = modal.Function.from_name("visionerase-diffueraser", "refine_with_diffueraser")
        video_bytes = fn.remote(
            frames_bytes,
            masks_bytes,
            prior_bytes,
            fps=fps,
            max_img_size=settings.diffueraser_max_img_size,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            refined_path = tmp_file.name
        try:
            refined = _decode_video_frames(refined_path)
        finally:
            os.unlink(refined_path)

        if len(refined) != len(inpainted):
            raise RuntimeError(
                f"DiffuEraser returned {len(refined)} frames, expected {len(inpainted)}"
            )

        # Keep the refinement inside the hole. DiffuEraser works at
        # max_img_size and its whole frame comes back through a downscale and
        # back up, so adopting it wholesale would soften every pixel we never
        # asked it to touch.
        composited = []
        for prior, ref, mask in zip(inpainted, refined, masks):
            if ref.shape[:2] != prior.shape[:2]:
                ref = cv2.resize(
                    ref, (prior.shape[1], prior.shape[0]), interpolation=cv2.INTER_LANCZOS4
                )
            alpha = create_three_zone_mask(mask)[:, :, None]
            composited.append(
                (alpha * ref.astype(np.float32) + (1 - alpha) * prior.astype(np.float32))
                .clip(0, 255).astype(np.uint8)
            )
        refined = composited

        elapsed = time.perf_counter() - started
        DIFFUERASER_SECONDS.observe(elapsed)
        DIFFUERASER_REFINEMENTS_TOTAL.labels(outcome="refined").inc()
        bound_log.info(
            "diffueraser_refinement_complete",
            num_frames=len(refined),
            elapsed_sec=round(elapsed, 1),
        )
        return refined

    except Exception as exc:
        DIFFUERASER_REFINEMENTS_TOTAL.labels(outcome="failed").inc()
        bound_log.warning(
            "diffueraser_failed_fallback_propainter",
            error=str(exc),
            elapsed_sec=round(time.perf_counter() - started, 1),
        )
        return None


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
