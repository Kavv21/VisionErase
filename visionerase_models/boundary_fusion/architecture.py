from __future__ import annotations

"""BoundaryFusion — proprietary model for segment boundary correction.

STATUS: Architecture defined. Weights not yet trained.
        Train this model in Month 2 Week 5-6 on Kaggle.
        See docs/boundary_fusion_training.md for training guide.

Proprietary to VisionErase. Do NOT open-source.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split (T, H, W, C) frames into 8×8 patches and project to embed_dim."""

    PATCH_SIZE: int = 8
    EMBED_DIM: int = 256

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            self.EMBED_DIM,
            kernel_size=self.PATCH_SIZE,
            stride=self.PATCH_SIZE,
        )
        self.norm = nn.LayerNorm(self.EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, C, H, W) → (T, N_patches, embed_dim)."""
        T = x.shape[0]
        # Project each frame independently
        x = self.proj(x)                    # (T, embed_dim, H/8, W/8)
        x = x.flatten(2).transpose(1, 2)   # (T, N_patches, embed_dim)
        return self.norm(x)


class LocalWindowAttention(nn.Module):
    """Self-attention restricted to a sliding temporal window of window_size frames.

    Prevents distant frames from influencing each other, reducing memory cost
    and keeping temporal attention local and stable.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, window_size: int = 5) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, N_patches, embed_dim) → same shape after local temporal attention."""
        T, N, D = x.shape
        out = x.clone()

        for t in range(T):
            w_start = max(0, t - self.window_size // 2)
            w_end = min(T, w_start + self.window_size)
            window = x[w_start:w_end]           # (W, N, D)

            # MultiheadAttention (batch_first=True) expects (batch, seq, embed).
            # Frame t is query (batch=1, seq=N); window is key/value flattened to
            # (batch=1, seq=W*N) so both share the same batch dimension.
            query = x[t:t+1]                    # (1, N, D)
            kv = window.reshape(1, -1, D)       # (1, W*N, D)
            attended, _ = self.attn(query, kv, kv)
            out[t] = self.norm(x[t] + attended.squeeze(0))

        return out


class CrossSegmentAttention(nn.Module):
    """Cross-attention between frames from segment A and segment B.

    Frames in A attend to B and vice versa, allowing the model to resolve
    inconsistencies introduced by parallel processing of adjacent segments.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

    def forward(
        self, seg_a: torch.Tensor, seg_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """seg_a, seg_b: (T, N_patches, embed_dim) → corrected (T, N, D) for each.

        Frames in A act as queries attending to B (and vice versa), then each
        result is residual-added and normalised.
        """
        T, N, D = seg_a.shape

        # Flatten temporal dimension to attend over all patch tokens jointly
        a_flat = seg_a.reshape(1, T * N, D)
        b_flat = seg_b.reshape(1, T * N, D)

        a_corrected, _ = self.attn_a2b(a_flat, b_flat, b_flat)
        b_corrected, _ = self.attn_b2a(b_flat, a_flat, a_flat)

        a_out = self.norm_a(seg_a + a_corrected.reshape(T, N, D))
        b_out = self.norm_b(seg_b + b_corrected.reshape(T, N, D))
        return a_out, b_out


class BoundaryFusion(nn.Module):
    """Full BoundaryFusion model.

    Input:  (20, H, W, 3) — uint8 frames (10 from each adjacent segment)
    Output: (20, H, W, 3) — corrected boundary frames (float32 in [0, 1])

    Pipeline:
        1. Patch embed all 20 frames
        2. Local window attention within each segment's 10 frames
        3. Cross-segment attention between the two halves
        4. Decode back to pixel space via transposed convolution
        5. Return corrected frames
    """

    WINDOW_FRAMES: int = 10   # frames per segment at the boundary
    EMBED_DIM: int = 256
    PATCH_SIZE: int = 8

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels)
        self.local_attn = LocalWindowAttention(self.EMBED_DIM, num_heads=4, window_size=5)
        self.cross_attn = CrossSegmentAttention(self.EMBED_DIM, num_heads=4)

        # Pixel decoder: project embed_dim back to patch pixels
        self.decoder = nn.Sequential(
            nn.Linear(self.EMBED_DIM, self.PATCH_SIZE * self.PATCH_SIZE * in_channels),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (20, H, W, C) float32 in [0, 1] → (20, H, W, C) corrected."""
        T, H, W, C = x.shape

        # Rearrange to (T, C, H, W) for Conv2d
        x_chw = x.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

        # 1. Patch embedding
        tokens = self.patch_embed(x_chw)             # (T, N, D)
        n_h = H // self.PATCH_SIZE
        n_w = W // self.PATCH_SIZE
        N = n_h * n_w

        # 2. Local window attention per segment
        seg_a_tokens = self.local_attn(tokens[:self.WINDOW_FRAMES])
        seg_b_tokens = self.local_attn(tokens[self.WINDOW_FRAMES:])

        # 3. Cross-segment attention
        seg_a_tokens, seg_b_tokens = self.cross_attn(seg_a_tokens, seg_b_tokens)
        all_tokens = torch.cat([seg_a_tokens, seg_b_tokens], dim=0)  # (20, N, D)

        # 4. Decode to pixels
        pixels = self.decoder(all_tokens)            # (20, N, patch_size^2 * C)
        pixels = pixels.reshape(T, n_h, n_w, self.PATCH_SIZE, self.PATCH_SIZE, C)
        # Rearrange patches back to image: (T, H, W, C)
        pixels = pixels.permute(0, 1, 3, 2, 4, 5).contiguous()
        pixels = pixels.reshape(T, H, W, C)

        return pixels
