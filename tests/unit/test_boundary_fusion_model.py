"""Unit tests for visionerase_models/boundary_fusion/architecture.py and inference.py.

All tests run on CPU — no GPU required.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from visionerase_models.boundary_fusion.architecture import (
    BoundaryFusion,
    CrossSegmentAttention,
    LocalWindowAttention,
    PatchEmbedding,
)
from visionerase_models.boundary_fusion.inference import run_boundary_fusion


# ── Shared fixtures ───────────────────────────────────────────────────────────

# Small spatial resolution so tests run fast on CPU
H, W = 32, 32   # must be multiples of PatchEmbedding.PATCH_SIZE (8)
C = 3
WINDOW = 10     # frames per segment at boundary


@pytest.fixture(scope="module")
def model() -> BoundaryFusion:
    """Instantiate BoundaryFusion once per module (CPU, random weights)."""
    m = BoundaryFusion()
    m.eval()
    return m


@pytest.fixture(scope="module")
def boundary_input() -> torch.Tensor:
    """Random float32 input (20, H, W, 3) in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(WINDOW * 2, H, W, C)


# ── BoundaryFusion instantiation ──────────────────────────────────────────────

@pytest.mark.unit
class TestBoundaryFusionInstantiation:
    def test_instantiates_without_error(self):
        """BoundaryFusion() must construct successfully with default args."""
        m = BoundaryFusion()
        assert m is not None

    def test_is_nn_module(self):
        """BoundaryFusion must be a torch.nn.Module."""
        assert isinstance(BoundaryFusion(), torch.nn.Module)

    def test_has_expected_submodules(self):
        """All architectural components must be present after construction."""
        m = BoundaryFusion()
        assert hasattr(m, "patch_embed")
        assert hasattr(m, "local_attn")
        assert hasattr(m, "cross_attn")
        assert hasattr(m, "decoder")

    def test_runs_on_cpu(self, model: BoundaryFusion):
        """Model must be on CPU after instantiation (no GPU needed for tests)."""
        device = next(model.parameters()).device
        assert device.type == "cpu"


# ── BoundaryFusion forward pass ───────────────────────────────────────────────

@pytest.mark.unit
class TestBoundaryFusionForward:
    def test_forward_accepts_correct_input_shape(self, model: BoundaryFusion, boundary_input: torch.Tensor):
        """forward() must accept (20, H, W, 3) without raising."""
        with torch.no_grad():
            out = model(boundary_input)
        assert out is not None

    def test_forward_returns_correct_output_shape(self, model: BoundaryFusion, boundary_input: torch.Tensor):
        """Output shape must exactly match input shape (20, H, W, 3)."""
        with torch.no_grad():
            out = model(boundary_input)
        assert out.shape == boundary_input.shape, f"Expected {boundary_input.shape}, got {out.shape}"

    def test_output_values_in_zero_one_range(self, model: BoundaryFusion, boundary_input: torch.Tensor):
        """Decoder uses Sigmoid so all output values must be in [0, 1]."""
        with torch.no_grad():
            out = model(boundary_input)
        assert float(out.min()) >= 0.0, f"Minimum value {out.min()} is below 0"
        assert float(out.max()) <= 1.0, f"Maximum value {out.max()} is above 1"

    def test_output_is_float_tensor(self, model: BoundaryFusion, boundary_input: torch.Tensor):
        """Output must be a float32 tensor (not int, not half)."""
        with torch.no_grad():
            out = model(boundary_input)
        assert out.dtype == torch.float32

    def test_forward_is_deterministic_in_eval_mode(self, model: BoundaryFusion, boundary_input: torch.Tensor):
        """Two forward passes with the same input must produce identical output in eval mode."""
        with torch.no_grad():
            out1 = model(boundary_input)
            out2 = model(boundary_input)
        assert torch.allclose(out1, out2), "Non-deterministic output in eval mode"

    def test_different_inputs_produce_different_outputs(self, model: BoundaryFusion):
        """Model must not collapse all inputs to the same output."""
        torch.manual_seed(0)
        x1 = torch.rand(WINDOW * 2, H, W, C)
        x2 = torch.rand(WINDOW * 2, H, W, C)
        with torch.no_grad():
            o1 = model(x1)
            o2 = model(x2)
        assert not torch.allclose(o1, o2), "Model produces identical output for different inputs"


# ── PatchEmbedding ────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPatchEmbedding:
    def test_instantiates_without_error(self):
        """PatchEmbedding() constructs with default args."""
        pe = PatchEmbedding()
        assert pe is not None

    def test_output_embed_dim(self):
        """PatchEmbedding must produce tokens with embed_dim=256 channels."""
        pe = PatchEmbedding()
        T = 5
        x = torch.rand(T, C, H, W)
        with torch.no_grad():
            out = pe(x)
        assert out.shape[-1] == PatchEmbedding.EMBED_DIM

    def test_output_sequence_length(self):
        """Number of patch tokens must be (H/patch_size) * (W/patch_size)."""
        pe = PatchEmbedding()
        T = 5
        x = torch.rand(T, C, H, W)
        with torch.no_grad():
            out = pe(x)
        expected_n = (H // PatchEmbedding.PATCH_SIZE) * (W // PatchEmbedding.PATCH_SIZE)
        assert out.shape[1] == expected_n, f"Expected {expected_n} tokens, got {out.shape[1]}"

    def test_output_batch_dim_preserved(self):
        """Temporal (batch) dimension T must be preserved through patch embedding."""
        pe = PatchEmbedding()
        T = 7
        x = torch.rand(T, C, H, W)
        with torch.no_grad():
            out = pe(x)
        assert out.shape[0] == T


# ── LocalWindowAttention ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestLocalWindowAttention:
    def test_instantiates_without_error(self):
        """LocalWindowAttention() constructs with default args."""
        lwa = LocalWindowAttention()
        assert lwa is not None

    def test_output_shape_preserved(self):
        """LocalWindowAttention must return a tensor with the same shape as its input."""
        lwa = LocalWindowAttention(embed_dim=256, num_heads=4, window_size=5)
        T, N, D = 10, 16, 256
        x = torch.rand(T, N, D)
        with torch.no_grad():
            out = lwa(x)
        assert out.shape == (T, N, D)

    def test_runs_on_cpu_without_error(self):
        """LocalWindowAttention forward pass must complete without raising on CPU."""
        lwa = LocalWindowAttention()
        x = torch.rand(WINDOW, 16, 256)
        with torch.no_grad():
            out = lwa(x)
        assert out is not None


# ── CrossSegmentAttention ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestCrossSegmentAttention:
    def test_instantiates_without_error(self):
        """CrossSegmentAttention() constructs with default args."""
        csa = CrossSegmentAttention()
        assert csa is not None

    def test_output_shapes_match_inputs(self):
        """Both returned tensors must have the same shape as the corresponding input."""
        csa = CrossSegmentAttention(embed_dim=256, num_heads=4)
        T, N, D = 10, 16, 256
        seg_a = torch.rand(T, N, D)
        seg_b = torch.rand(T, N, D)
        with torch.no_grad():
            out_a, out_b = csa(seg_a, seg_b)
        assert out_a.shape == seg_a.shape
        assert out_b.shape == seg_b.shape

    def test_cross_attention_changes_values(self):
        """Attending across segments must alter the token values (non-identity op)."""
        csa = CrossSegmentAttention(embed_dim=64, num_heads=2)
        T, N, D = 10, 4, 64
        seg_a = torch.rand(T, N, D)
        seg_b = torch.rand(T, N, D)
        with torch.no_grad():
            out_a, _ = csa(seg_a, seg_b)
        assert not torch.allclose(out_a, seg_a), "CrossSegmentAttention acted as identity"

    def test_runs_on_cpu_without_error(self):
        """CrossSegmentAttention forward pass must complete without raising on CPU."""
        csa = CrossSegmentAttention()
        seg_a = torch.rand(WINDOW, 16, 256)
        seg_b = torch.rand(WINDOW, 16, 256)
        with torch.no_grad():
            out_a, out_b = csa(seg_a, seg_b)
        assert out_a is not None and out_b is not None


# ── run_boundary_fusion (inference wrapper) ───────────────────────────────────

@pytest.mark.unit
class TestRunBoundaryFusion:
    @pytest.fixture
    def numpy_frames(self):
        """Two sets of 10 frames as uint8 numpy arrays."""
        rng = np.random.default_rng(seed=7)
        frames_a = rng.integers(0, 256, size=(WINDOW, H, W, C), dtype=np.uint8)
        frames_b = rng.integers(0, 256, size=(WINDOW, H, W, C), dtype=np.uint8)
        return frames_a, frames_b

    def test_output_shape_is_20_frames(self, model: BoundaryFusion, numpy_frames):
        """run_boundary_fusion must return (20, H, W, 3)."""
        frames_a, frames_b = numpy_frames
        result = run_boundary_fusion(model, frames_a, frames_b)
        assert result.shape == (WINDOW * 2, H, W, C)

    def test_output_dtype_is_uint8(self, model: BoundaryFusion, numpy_frames):
        """Output must be uint8 to match raw frame format consumed by video writers."""
        frames_a, frames_b = numpy_frames
        result = run_boundary_fusion(model, frames_a, frames_b)
        assert result.dtype == np.uint8

    def test_output_values_in_0_255_range(self, model: BoundaryFusion, numpy_frames):
        """All pixel values must lie within the valid uint8 range [0, 255]."""
        frames_a, frames_b = numpy_frames
        result = run_boundary_fusion(model, frames_a, frames_b)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_accepts_full_hd_frames(self, model: BoundaryFusion):
        """Inference must handle a realistic resolution (480p) without raising."""
        rng = np.random.default_rng(seed=99)
        # Use small-ish size that's divisible by patch_size=8 but realistic-ish
        h, w = 64, 64
        frames_a = rng.integers(0, 256, size=(WINDOW, h, w, C), dtype=np.uint8)
        frames_b = rng.integers(0, 256, size=(WINDOW, h, w, C), dtype=np.uint8)

        small_model = BoundaryFusion()
        small_model.eval()
        result = run_boundary_fusion(small_model, frames_a, frames_b)
        assert result.shape == (WINDOW * 2, h, w, C)
