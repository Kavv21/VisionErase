"""Inference helpers for the BoundaryFusion model.

Proprietary to VisionErase. Do not open-source.
"""
from __future__ import annotations

import os

import numpy as np
import structlog
import torch

from visionerase_models.boundary_fusion.architecture import BoundaryFusion

log = structlog.get_logger(__name__)

WEIGHTS_FILENAME = "boundary_fusion.pth"


def load_boundary_fusion_model(
    weights_path: str,
    device: str | None = None,
) -> BoundaryFusion:
    """Load BoundaryFusion weights and return the model in eval mode.

    Accepts both raw state dicts and training checkpoints that nest the
    weights under "model_state_dict".

    device=None auto-selects (CUDA if available); pass "cpu" to force CPU.
    Uses FP16 precision on CUDA to reduce VRAM consumption, FP32 on CPU.
    """
    model = BoundaryFusion()
    resolved = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    state = torch.load(weights_path, map_location=resolved)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(resolved)
    model.eval()

    if resolved.type == "cuda":
        model.half()
        log.info("boundary_fusion_loaded_fp16", weights_path=weights_path, device=str(resolved))
    else:
        log.info("boundary_fusion_loaded_fp32", weights_path=weights_path, device=str(resolved))

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


def run_boundary_fusion_on_chunks(
    chunk_paths: list[str],   # ordered list of chunk MP4 paths
    output_path: str,         # where to write the corrected video
    model_cache_dir: str | None = None,   # path to model_weights/
    device: str = "cpu",      # small model (890K params) — CPU is fast enough
    model: BoundaryFusion | None = None,  # pass a pool-acquired model to skip loading
) -> str:
    """Correct every chunk boundary and write the full fused video.

    For each adjacent chunk pair (N, N+1):
      1. Extract the last WINDOW_FRAMES frames of chunk N
      2. Extract the first WINDOW_FRAMES frames of chunk N+1
      3. Run BoundaryFusion on the combined 20 frames
      4. Replace those frames in the output video

    Callers inside the pipeline should acquire the model from the model pool
    and pass it via `model`; otherwise it is loaded from model_cache_dir.
    Returns output_path.
    """
    import cv2

    if not chunk_paths:
        raise ValueError("chunk_paths must not be empty")

    if model is None:
        if model_cache_dir is None:
            raise ValueError("either model or model_cache_dir must be provided")
        weights_path = os.path.join(model_cache_dir, WEIGHTS_FILENAME)
        model = load_boundary_fusion_model(weights_path, device=device)

    window = BoundaryFusion.WINDOW_FRAMES

    prev_frames, fps = _read_video_frames(chunk_paths[0])
    if not prev_frames:
        raise RuntimeError(f"no frames decoded from chunk {chunk_paths[0]}")

    height, width = prev_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for boundary_index, path in enumerate(chunk_paths[1:]):
            cur_frames, _ = _read_video_frames(path)
            if len(prev_frames) >= window and len(cur_frames) >= window:
                corrected = _fuse_boundary(
                    model,
                    np.stack(prev_frames[-window:]),
                    np.stack(cur_frames[:window]),
                )
                prev_frames[-window:] = list(corrected[:window])
                cur_frames[:window] = list(corrected[window:])
                log.info("boundary_fused", boundary_index=boundary_index)
            else:
                log.warning(
                    "boundary_skipped_short_chunk",
                    boundary_index=boundary_index,
                    prev_len=len(prev_frames),
                    cur_len=len(cur_frames),
                    window=window,
                )
            for frame in prev_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            prev_frames = cur_frames

        for frame in prev_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    log.info(
        "boundary_fusion_video_written",
        output_path=output_path,
        num_chunks=len(chunk_paths),
        num_boundaries=len(chunk_paths) - 1,
    )
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fuse_boundary(
    model: BoundaryFusion,
    frames_a: np.ndarray,
    frames_b: np.ndarray,
) -> np.ndarray:
    """Fuse one boundary, padding H/W to a multiple of PATCH_SIZE if needed."""
    patch = BoundaryFusion.PATCH_SIZE
    h, w = frames_a.shape[1:3]
    pad_h = (-h) % patch
    pad_w = (-w) % patch

    if pad_h or pad_w:
        padding = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
        frames_a = np.pad(frames_a, padding, mode="edge")
        frames_b = np.pad(frames_b, padding, mode="edge")

    corrected = run_boundary_fusion(model, frames_a, frames_b)
    return corrected[:, :h, :w, :]


def _read_video_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """Return all frames of a video as RGB uint8 arrays, plus its FPS."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps
