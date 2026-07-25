"""Unit tests for the inpainting quality pipeline.

Covers the four stages added to workers/inpainting/chunk_tasks.py: ROI cropping,
three-zone alpha compositing, Lab colour correction and grain restoration.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

import workers.inpainting.chunk_tasks as ct

H, W = 540, 960
OBJ_Y, OBJ_X, OBJ_H, OBJ_W = 200, 300, 80, 60
STEP = 5  # px the object moves per frame


class _NullLog:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass


@pytest.fixture
def clip() -> tuple[list[np.ndarray], np.ndarray]:
    """Six RGB frames of a bright square drifting over a noisy background."""
    rng = np.random.default_rng(0)
    frames, masks = [], []
    for t in range(6):
        frame = rng.normal(120, 8, (H, W, 3)).clip(0, 255).astype(np.uint8)
        mask = np.zeros((H, W), np.uint8)
        y, x = OBJ_Y, OBJ_X + t * STEP
        mask[y:y + OBJ_H, x:x + OBJ_W] = 255
        frame[y:y + OBJ_H, x:x + OBJ_W] = 240
        frames.append(frame)
        masks.append(mask)
    return frames, np.stack(masks)


# ── ROI cropping ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestComputeRoiCrop:
    def test_crop_covers_every_frames_mask(self, clip):
        frames, masks = clip
        x1, y1, x2, y2 = ct.compute_roi_crop(masks, frames)
        assert x1 <= OBJ_X and x2 >= OBJ_X + 5 * STEP + OBJ_W
        assert y1 <= OBJ_Y and y2 >= OBJ_Y + OBJ_H

    def test_dimensions_are_multiples_of_eight(self, clip):
        """ProPainter's patch embedding requires it."""
        frames, masks = clip
        x1, y1, x2, y2 = ct.compute_roi_crop(masks, frames)
        assert (x2 - x1) % 8 == 0 and (y2 - y1) % 8 == 0

    def test_crop_stays_inside_the_frame(self, clip):
        frames, masks = clip
        x1, y1, x2, y2 = ct.compute_roi_crop(masks, frames)
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H

    def test_crop_is_much_smaller_than_the_frame(self, clip):
        """The whole point: a small object gets far more effective resolution."""
        frames, masks = clip
        x1, y1, x2, y2 = ct.compute_roi_crop(masks, frames)
        assert (x2 - x1) * (y2 - y1) < 0.25 * W * H

    def test_empty_masks_fall_back_to_the_full_frame(self, clip):
        frames, _ = clip
        assert ct.compute_roi_crop(np.zeros((3, H, W), np.uint8), frames) == (0, 0, W, H)

    def test_large_object_clamps_to_the_max_crop(self, clip):
        frames, _ = clip
        big = np.zeros((2, H, W), np.uint8)
        big[:, 10:530, 10:950] = 255
        x1, y1, x2, y2 = ct.compute_roi_crop(big, frames)
        assert x2 - x1 <= ct.ROI_MAX_W and y2 - y1 <= ct.ROI_MAX_H


# ── Three-zone alpha ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCreateThreeZoneMask:
    def test_returns_float32_alpha_in_unit_range(self, clip):
        _, masks = clip
        alpha = ct.create_three_zone_mask(masks[0])
        assert alpha.dtype == np.float32 and alpha.shape == (H, W)
        assert 0.0 <= alpha.min() and alpha.max() <= 1.0

    def test_object_interior_is_fully_inpainted(self, clip):
        _, masks = clip
        alpha = ct.create_three_zone_mask(masks[0])
        assert alpha[OBJ_Y + 40, OBJ_X + 30] == 1.0

    def test_far_outside_is_fully_original(self, clip):
        _, masks = clip
        alpha = ct.create_three_zone_mask(masks[0])
        assert alpha[OBJ_Y - 40, OBJ_X + 30] == 0.0

    def test_alpha_decreases_monotonically_outward(self, clip):
        """A non-monotonic ramp would read as a ring artifact."""
        _, masks = clip
        alpha = ct.create_three_zone_mask(masks[0])
        ramp = [float(alpha[OBJ_Y - d, OBJ_X + 30]) for d in range(0, 30, 3)]
        assert all(a >= b - 1e-6 for a, b in zip(ramp, ramp[1:]))

    def test_transition_band_is_partial(self, clip):
        _, masks = clip
        alpha = ct.create_three_zone_mask(masks[0])
        assert 0.0 < alpha[OBJ_Y - 14, OBJ_X + 30] < 1.0

    def test_empty_mask_yields_all_zero_alpha(self):
        """Guarantees a frame with no object passes through untouched."""
        assert ct.create_three_zone_mask(np.zeros((H, W), np.uint8)).max() == 0

    def test_faster_motion_widens_the_core(self, clip):
        _, masks = clip
        slow = ct.create_three_zone_mask(masks[0], flow_speed=0.0)
        fast = ct.create_three_zone_mask(masks[0], flow_speed=60.0)
        assert (fast == 1).sum() > (slow == 1).sum()

    def test_propainter_mask_covers_the_whole_blend_band(self, clip):
        """The ramp must land on generated pixels, not on ProPainter's hard edge."""
        _, masks = clip
        widened = ct._dilate_masks(masks, ct.INPAINT_MASK_MARGIN_PX)
        band = ct.create_three_zone_mask(masks[0], flow_speed=60.0) > 0
        assert np.all(widened[0][band] > 0)


# ── Colour correction ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestLocalColourCorrection:
    def test_reduces_a_colour_mismatch_at_the_boundary(self, clip):
        frames, masks = clip
        alpha = ct.create_three_zone_mask(masks[0])
        original = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        inpainted = original.astype(np.float32)
        inpainted[alpha > 0] *= 0.75  # patch comes back 25% too dark
        inpainted = inpainted.clip(0, 255).astype(np.uint8)

        corrected = ct.local_colour_correction(inpainted, original, alpha)

        # Correction is strongest at the boundary and fades to nothing at the
        # centre, so the boundary ramp is where it has to show up.
        band = (alpha > 0.05) & (alpha < 0.6)
        before = abs(inpainted[band].mean() - original[band].mean())
        after = abs(corrected[band].mean() - original[band].mean())
        assert after < before

    def test_pixels_outside_the_mask_are_bit_identical(self, clip):
        """Blending in Lab would shift untouched pixels and cost global SSIM."""
        frames, masks = clip
        alpha = ct.create_three_zone_mask(masks[0])
        original = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        inpainted = cv2.GaussianBlur(original, (0, 0), 3.0)

        corrected = ct.local_colour_correction(inpainted, original, alpha)

        outside = alpha == 0
        assert np.array_equal(corrected[outside], original[outside])

    def test_returns_input_unchanged_when_there_is_no_reference_ring(self, clip):
        frames, _ = clip
        original = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        empty = np.zeros((H, W), np.float32)
        assert ct.local_colour_correction(original, original, empty) is original


# ── Grain restoration ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRestoreTextureAndGrain:
    def test_noise_field_has_unit_variance(self):
        """Grain is scaled by a measured sigma, so the field must be normalised."""
        noise = ct.make_grain_noise((H, W), seed=1)
        assert abs(noise.std() - 1.0) < 0.05

    def test_noise_field_is_deterministic_for_a_seed(self):
        assert np.array_equal(
            ct.make_grain_noise((64, 64), seed=7), ct.make_grain_noise((64, 64), seed=7)
        )

    def test_adds_high_frequency_energy_to_a_smoothed_patch(self, clip):
        frames, masks = clip
        alpha = ct.create_three_zone_mask(masks[0])
        original = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        smoothed = cv2.GaussianBlur(original, (0, 0), 3.0)

        grained = ct.restore_texture_and_grain(
            smoothed, original, alpha, noise=ct.make_grain_noise((H, W), seed=1)
        )

        core = alpha > 0.9
        before = cv2.Laplacian(smoothed[core], cv2.CV_32F).std()
        after = cv2.Laplacian(grained[core], cv2.CV_32F).std()
        assert after > before

    def test_pixels_outside_the_mask_are_bit_identical(self, clip):
        frames, masks = clip
        alpha = ct.create_three_zone_mask(masks[0])
        original = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        smoothed = cv2.GaussianBlur(original, (0, 0), 3.0)

        grained = ct.restore_texture_and_grain(smoothed, original, alpha)

        outside = alpha == 0
        assert np.array_equal(grained[outside], smoothed[outside])

    def test_returns_input_unchanged_without_a_sampling_band(self, clip):
        frames, _ = clip
        original = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        empty = np.zeros((H, W), np.float32)
        assert ct.restore_texture_and_grain(original, original, empty) is original


# ── Chunk-level orchestration ─────────────────────────────────────────────────

@pytest.mark.unit
class TestPostprocessChunk:
    def _blurred(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return [cv2.GaussianBlur(f, (0, 0), 3.0) for f in frames]

    def test_matches_the_unrestricted_full_frame_computation(self, clip):
        """The work-box optimisation must not change a single output pixel."""
        frames, masks = clip
        inpainted = self._blurred(frames)

        got = ct._postprocess_chunk(inpainted, frames, masks, 0, _NullLog())

        noise = ct.make_grain_noise((H, W), seed=0)
        speed = ct._estimate_flow_speed(masks)
        for i, (inp, orig, mask) in enumerate(zip(inpainted, frames, masks)):
            alpha = ct.create_three_zone_mask(mask, speed)[:, :, None]
            blended = (
                alpha * inp.astype(np.float32) + (1 - alpha) * orig.astype(np.float32)
            ).clip(0, 255).astype(np.uint8)
            orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
            corrected = ct.local_colour_correction(
                cv2.cvtColor(blended, cv2.COLOR_RGB2BGR), orig_bgr, alpha[:, :, 0]
            )
            expected = cv2.cvtColor(
                ct.restore_texture_and_grain(
                    corrected, orig_bgr, alpha[:, :, 0], noise=noise
                ),
                cv2.COLOR_BGR2RGB,
            )
            assert np.array_equal(expected, got[i]), f"frame {i} differs"

    def test_empty_masks_pass_frames_through_untouched(self, clip):
        frames, masks = clip
        out = ct._postprocess_chunk(
            self._blurred(frames), frames, np.zeros_like(masks), 0, _NullLog()
        )
        assert all(np.array_equal(a, b) for a, b in zip(out, frames))

    def test_grain_does_not_flicker_between_frames(self, clip):
        """One noise field per chunk — resampling per frame costs consistency."""
        frames, masks = clip
        static = [frames[0]] * len(frames)
        out = ct._postprocess_chunk(
            self._blurred(static), static, np.stack([masks[0]] * len(masks)),
            0, _NullLog(),
        )
        assert np.array_equal(out[0], out[1])


@pytest.mark.unit
class TestEstimateFlowSpeed:
    def test_measures_centroid_displacement(self, clip):
        _, masks = clip
        assert abs(ct._estimate_flow_speed(masks) - STEP) < 0.5

    def test_static_object_reports_zero(self, clip):
        _, masks = clip
        assert ct._estimate_flow_speed(np.stack([masks[0]] * 4)) == 0.0

    def test_returns_none_without_masks(self):
        assert ct._estimate_flow_speed(np.zeros((3, H, W), np.uint8)) is None


@pytest.mark.unit
class TestMasksToFrameSize:
    def test_undoes_tracking_downscale_and_padding(self):
        from workers.segmentation.sam2_video_tracker import _processing_geometry

        proc_w, proc_h, pad_w, pad_h, _ = _processing_geometry(1920, 1080)
        tracked = np.zeros((3, proc_h + pad_h, proc_w + pad_w), np.uint8)
        tracked[:, 100:200, 100:200] = 255

        resized = ct._masks_to_frame_size(tracked, 1920, 1080)

        assert resized.shape == (3, 1080, 1920)
        assert (resized > 127).any()

    def test_masks_already_on_the_frame_grid_are_returned_as_is(self, clip):
        _, masks = clip
        assert ct._masks_to_frame_size(masks, W, H) is masks

    def test_padding_is_cropped_rather_than_squashed(self):
        """Resizing the padded array would shift the mask down by a few pixels."""
        from workers.segmentation.sam2_video_tracker import _processing_geometry

        # 1920x1000 tracks at 1024x533 and is padded by 3 rows; 1080p happens to
        # land on a multiple of 8 and so exercises nothing.
        src_w, src_h = 1920, 1000
        proc_w, proc_h, pad_w, pad_h, _ = _processing_geometry(src_w, src_h)
        assert pad_h > 0, "fixture assumes this resolution needs padding"
        tracked = np.zeros((1, proc_h + pad_h, proc_w + pad_w), np.uint8)
        tracked[0, proc_h - 10:proc_h, :] = 255  # last band of real rows

        resized = ct._masks_to_frame_size(tracked, src_w, src_h)

        assert resized[0, -1, 0] == 255, "bottom of the frame must stay masked"


@pytest.mark.unit
class TestAlignToReference:
    def test_pads_a_short_result_with_original_frames(self, clip):
        frames, _ = clip
        aligned = ct._align_to_reference(frames[:4], frames, _NullLog())
        assert len(aligned) == len(frames)
        assert np.array_equal(aligned[5], frames[5])

    def test_resizes_frames_that_come_back_at_the_wrong_size(self, clip):
        frames, _ = clip
        shrunk = [cv2.resize(f, (W // 2, H // 2)) for f in frames]
        aligned = ct._align_to_reference(shrunk, frames, _NullLog())
        assert all(f.shape == frames[0].shape for f in aligned)


# ── Chunk difficulty scoring (drives DiffuEraser refinement) ──────────────────

@pytest.mark.unit
class TestComputeChunkDifficulty:
    def test_score_is_in_unit_range(self, clip):
        frames, masks = clip
        inpainted = [cv2.GaussianBlur(f, (0, 0), 3.0) for f in frames]
        score = ct.compute_chunk_difficulty(frames, masks, inpainted)
        assert isinstance(score, float) and 0.0 <= score <= 1.0

    def test_perfect_reconstruction_scores_lower_than_a_bad_one(self, clip):
        """The boundary term must reward a result that matches its surroundings."""
        frames, masks = clip
        identical = [f.copy() for f in frames]
        garbage = [np.zeros_like(f) for f in frames]
        assert ct.compute_chunk_difficulty(
            frames, masks, identical
        ) < ct.compute_chunk_difficulty(frames, masks, garbage)

    def test_larger_masks_score_higher(self, clip):
        frames, masks = clip
        big = np.zeros_like(masks)
        big[:, 100:400, 100:600] = 255
        inpainted = [f.copy() for f in frames]
        assert ct.compute_chunk_difficulty(
            frames, big, inpainted
        ) > ct.compute_chunk_difficulty(frames, masks, inpainted)

    def test_empty_masks_do_not_crash(self, clip):
        """A chunk the object has left must score without dividing by zero."""
        frames, masks = clip
        score = ct.compute_chunk_difficulty(
            frames, np.zeros_like(masks), [f.copy() for f in frames]
        )
        assert 0.0 <= score <= 1.0

    def test_textured_region_scores_above_flat_region(self):
        """Entropy term: a flat wall is easier to fill than dense texture."""
        rng = np.random.default_rng(1)
        mask = np.zeros((H, W), np.uint8)
        mask[200:300, 300:400] = 255
        masks = np.stack([mask] * 3)

        flat = [np.full((H, W, 3), 128, np.uint8) for _ in range(3)]
        textured = [
            rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(3)
        ]
        assert ct.compute_chunk_difficulty(
            textured, masks, [f.copy() for f in textured]
        ) > ct.compute_chunk_difficulty(flat, masks, [f.copy() for f in flat])
