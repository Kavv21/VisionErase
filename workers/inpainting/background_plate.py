"""Recover real background pixels from other frames instead of hallucinating them.

ProPainter invents the contents of the hole from a learned prior. Whenever the
object moves across a static background, the pixels it is currently covering are
genuinely visible in some other frame of the same chunk — copying those back is
strictly better than any generated guess, because they are the ground truth.

This module builds that "background plate" per frame and composites it over
ProPainter's output wherever enough donor frames agree. Where no donor saw the
background (a stationary object, or a region the object never uncovers),
confidence falls to zero and ProPainter's fill is kept unchanged.

Donors are read at the same pixel coordinates as the target, so this only helps
on a locked-off or slowly-panning camera; fast camera motion drives the donor
pixels out of alignment and the confidence gate is what keeps that from being
composited in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# A pixel needs roughly three nearby donors before its plate value is trusted.
# exp(-10/20) is the weight of a donor 10 frames away, so this is "three frames
# within ten"; the x5 turns it into the weight at which confidence saturates.
MIN_DONOR_WEIGHT = 3 * np.exp(-10 / 20.0)

# Temporal falloff, in frames, for the donor weighting. Nearby frames look more
# like the target's background (lighting, exposure, slow parallax) than distant
# ones, so they dominate the average.
DONOR_HALFLIFE = 20.0

# Below this many fillable pixels a donor contributes nothing worth the pass.
MIN_DONOR_PIXELS = 50

# Confidence a pixel needs before its recovered value is used at all. Shared
# with the caller's coverage gate so "enough of the frame is recoverable" is
# measured against the same bar the compositor actually applies.
PLATE_CONFIDENCE_THRESHOLD = 0.65


def build_background_plate(
    frames: list[np.ndarray],
    masks: Any,
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the real background behind the object in frames[target_idx].

    For every pixel the target frame needs filled, average the pixels at the
    same coordinates from every other frame where the object is *not* present,
    weighted by temporal proximity.

    frames: list of (H, W, 3) uint8
    masks:  list/array of (H, W) uint8
    target_idx: index into both

    Returns: plate (H, W, 3) uint8, confidence (H, W) float32 in [0, 1].
             Both are zero everywhere outside the target mask.
    """
    H, W = frames[0].shape[:2]
    T = len(frames)

    target_mask = masks[target_idx]
    plate = np.zeros((H, W, 3), dtype=np.uint8)
    confidence = np.zeros((H, W), dtype=np.float32)

    # Only pixels the target frame needs filled can ever be non-zero, so confine
    # the accumulation to their bounding box. The result is bit-identical to
    # running over the whole frame and ~10x cheaper on 1080p, where the object
    # is a small fraction of the image (same trick as _postprocess_box).
    ys, xs = np.nonzero(target_mask > 127)
    if ys.size == 0:
        return plate, confidence
    by1, by2 = int(ys.min()), int(ys.max()) + 1
    bx1, bx2 = int(xs.min()), int(xs.max()) + 1

    need_fill = target_mask[by1:by2, bx1:bx2] > 127
    box_h, box_w = need_fill.shape

    plate_sum = np.zeros((box_h, box_w, 3), dtype=np.float64)
    plate_weight = np.zeros((box_h, box_w), dtype=np.float64)

    for t in range(T):
        if t == target_idx:
            continue

        # Valid donor pixel: not masked in the donor frame, and masked in the
        # target frame (otherwise there is nothing to fill).
        can_donate = masks[t][by1:by2, bx1:bx2] < 127
        valid = need_fill & can_donate

        if valid.sum() < MIN_DONOR_PIXELS:
            continue

        w = float(np.exp(-abs(t - target_idx) / DONOR_HALFLIFE))
        donor = frames[t][by1:by2, bx1:bx2]

        plate_sum[valid] += donor[valid].astype(np.float64) * w
        plate_weight[valid] += w

    covered = plate_weight > 0
    if not covered.any():
        return plate, confidence

    plate[by1:by2, bx1:bx2][covered] = (
        plate_sum[covered] / plate_weight[covered, None]
    ).clip(0, 255).astype(np.uint8)

    # Confidence is how much donor weight backs each pixel, saturating once
    # several nearby frames agree.
    box_conf = np.clip(plate_weight / (MIN_DONOR_WEIGHT * 5), 0, 1).astype(np.float32)
    box_conf[~covered] = 0.0
    confidence[by1:by2, bx1:bx2] = box_conf

    return plate, confidence


def apply_background_plate(
    orig: np.ndarray,
    propainter: np.ndarray,
    mask: np.ndarray,
    plate: np.ndarray,
    confidence: np.ndarray,
    threshold: float = PLATE_CONFIDENCE_THRESHOLD,
    blend_px: int = 16,
) -> np.ndarray:
    """Composite the plate over ProPainter's fill, weighted by confidence.

    High confidence → real recovered background wins. Low confidence →
    ProPainter's generated fill is kept. The confidence map is blurred first so
    the handover between the two is a soft ramp rather than a visible patchwork
    of per-pixel decisions.

    All images are (H, W, 3) uint8 in whatever colour order the caller uses
    (RGB in this pipeline); mask is (H, W) uint8.
    """
    import cv2

    mask_f = (mask > 127).astype(np.float32)

    # Drop pixels the plate is not confident about before blurring, so a thinly
    # supported value can only ever reach the frame via the ramp around a
    # well-supported neighbourhood — never on its own strength. This is also the
    # bar the caller's coverage gate measures against.
    gated = np.where(confidence >= threshold, confidence, 0.0).astype(np.float32)

    # Smooth for a soft transition. Multiplying by the mask first pins it to 0
    # at the mask edge, so the plate fades out before it can reach pixels the
    # mask does not own.
    conf_smooth = cv2.GaussianBlur(
        (gated * mask_f),
        (blend_px * 2 + 1, blend_px * 2 + 1),
        blend_px / 3,
    )
    conf_smooth = np.clip(conf_smooth, 0, 1)

    c = conf_smooth[:, :, None]
    inpainted = c * plate.astype(np.float32) + (1 - c) * propainter.astype(np.float32)

    # Everything outside the mask stays exactly as it came in.
    result = (
        mask_f[:, :, None] * inpainted
        + (1 - mask_f[:, :, None]) * orig.astype(np.float32)
    )
    return result.clip(0, 255).astype(np.uint8)
