"""Unit tests for workers/segmentation/oiv_refiner.py.

The real OIV checkpoint/GPU is never touched — _load_oiv_model is patched
with a scripted fake that returns embeddings in a fixed order, one per
frame with a non-empty mask (mirroring the real embedding loop's call
order). This lets each scenario (confirmed / recovered / no-embedding)
be driven deterministically without a trained model.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch

from workers.segmentation import oiv_refiner


def _frame(size: int = 64) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _mask_with_square(value: int, size: int = 64) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[10:20, 10:20] = value
    return mask


def _empty_mask(size: int = 64) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)


class _ScriptedModel:
    """Returns one vector per call, in the order embeddings are requested."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        vec = self._vectors[self.calls]
        self.calls += 1
        return torch.tensor([vec], dtype=torch.float32)


@pytest.mark.unit
class TestSkipConditions:
    def test_all_empty_masks_returns_unchanged_without_loading_model(self, monkeypatch):
        monkeypatch.setattr(
            oiv_refiner, "_load_oiv_model", mock.Mock(side_effect=AssertionError)
        )
        frames = [_frame() for _ in range(3)]
        tracked_masks = np.zeros((3, 64, 64), dtype=np.uint8)

        result = oiv_refiner.refine_tracked_masks(frames, tracked_masks, "/fake/dir")

        assert result is tracked_masks
        assert not np.any(result)

    def test_empty_reference_frame_returns_unchanged_without_loading_model(self, monkeypatch):
        monkeypatch.setattr(
            oiv_refiner, "_load_oiv_model", mock.Mock(side_effect=AssertionError)
        )
        frames = [_frame() for _ in range(2)]
        tracked_masks = np.stack([_empty_mask(), _mask_with_square(10)])

        result = oiv_refiner.refine_tracked_masks(frames, tracked_masks, "/fake/dir")

        assert result is tracked_masks
        assert not np.any(result[0])
        assert np.any(result[1])


@pytest.mark.unit
class TestRefinement:
    """5-frame scenario: confirmed, confirmed, lost+recovered, no-mask, lost+recovered."""

    def _run(self, monkeypatch):
        vectors = [
            [1.0, 0.0, 0.0, 0.0],  # frame 0 (reference)
            [1.0, 0.0, 0.0, 0.0],  # frame 1: same direction -> confirmed
            [-1.0, 0.0, 0.0, 0.0],  # frame 2: opposite -> lost, recovers from frame 1
            # frame 3 has an empty mask: no embedding call
            [-1.0, 0.0, 0.0, 0.0],  # frame 4: opposite -> lost, skips frame 2, recovers from frame 1
        ]
        monkeypatch.setattr(
            oiv_refiner, "_load_oiv_model", lambda *a, **k: _ScriptedModel(vectors)
        )

        frames = [_frame() for _ in range(5)]
        tracked_masks = np.stack(
            [
                _mask_with_square(10),
                _mask_with_square(20),
                _mask_with_square(30),
                _empty_mask(),
                _mask_with_square(50),
            ]
        )
        original_frame1 = tracked_masks[1].copy()

        result = oiv_refiner.refine_tracked_masks(frames, tracked_masks, "/fake/dir")
        return result, original_frame1

    def test_confirmed_frame_is_untouched(self, monkeypatch):
        result, original_frame1 = self._run(monkeypatch)
        np.testing.assert_array_equal(result[1], original_frame1)

    def test_lost_frame_recovers_nearest_confirmed_mask(self, monkeypatch):
        result, original_frame1 = self._run(monkeypatch)
        np.testing.assert_array_equal(result[2], original_frame1)

    def test_empty_mask_frame_is_left_as_is(self, monkeypatch):
        result, _ = self._run(monkeypatch)
        assert not np.any(result[3])

    def test_lost_frame_skips_disqualified_source_and_recovers_from_frame_1(self, monkeypatch):
        result, original_frame1 = self._run(monkeypatch)
        np.testing.assert_array_equal(result[4], original_frame1)


@pytest.mark.unit
def test_bbox_from_mask_pads_by_20_percent_and_clips_to_bounds():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255  # 10x10 box -> 20% pad = 2px each side

    bbox = oiv_refiner._bbox_from_mask(mask)

    assert bbox == (8, 22, 8, 22)


@pytest.mark.unit
def test_bbox_from_mask_returns_none_for_empty_mask():
    assert oiv_refiner._bbox_from_mask(np.zeros((64, 64), dtype=np.uint8)) is None
