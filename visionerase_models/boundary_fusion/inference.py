"""Inference helpers for the BoundaryFusion model.

Proprietary to VisionErase. Do not open-source.
"""
from __future__ import annotations

import numpy as np
import structlog
import torch

from visionerase_models.boundary_fusion.architecture import BoundaryFusion

log = structlog.get_logger(__name__)


def load_boundary_fusion_model(weights_path: str) -> BoundaryFusion:
    """Load BoundaryFusion weights and return the model in eval mode.

    Uses FP16 precision when CUDA is available to reduce VRAM consumption.
    Falls back to FP32 on CPU.
    """
    model = BoundaryFusion()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if device.type == "cuda":
        model.half()
        log.info("boundary_fusion_loaded_fp16", weights_path=weights_path, device=str(device))
    else:
        log.info("boundary_fusion_loaded_fp32", weights_path=weights_path, device=str(device))

    return model


def run_boundary_fusion(
    model: BoundaryFusion,
    frames_a: np.ndarray,   # (10, H, W, 3) uint8
    frames_b: np.ndarray,   # (10, H, W, 3) uint8
) -> np.ndarray:            # (20, H, W, 3) uint8
    """Run BoundaryFusion inference on boundary frames from two adjacent segments.

    Preprocessing:  uint8 [0, 255] → float32 [0, 1]
    Postprocessing: float32 [0, 1] → uint8 [0, 255]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    combined = np.concatenate([frames_a, frames_b], axis=0)  # (20, H, W, 3)
    tensor = torch.from_numpy(combined.astype(np.float32) / 255.0).to(device=device, dtype=dtype)

    log.debug(
        "boundary_fusion_inference",
        input_shape=list(tensor.shape),
        device=str(device),
        dtype=str(dtype),
    )

    with torch.no_grad():
        output = model(tensor)                      # (20, H, W, 3) float32 in [0, 1]

    result = (output.float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    log.debug("boundary_fusion_inference_done", output_shape=list(result.shape))
    return result
