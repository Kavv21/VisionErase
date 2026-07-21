"""SAM2 Video Predictor mask tracking — replaces XMem++ for full-video propagation.

Tracks the first-frame SAM2 mask across every frame of the video in one pass on
the local GPU (~30s for 360 frames vs ~2min per 30-frame chunk on Modal XMem++).
Long videos are processed in 500-frame segments, seeding each segment with the
last mask of the previous one to keep temporal continuity.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any, Callable

import numpy as np
import structlog

log = structlog.get_logger(__name__)

SAM2_REPO_PATH = "/home/kavish/sam2"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT_FILE = "sam2_hiera_small.pt"

MAX_FRAME_SIDE = 1024        # SAM2-recommended max resolution
MAX_SEGMENT_FRAMES = 500     # longer videos are tracked in seeded segments
MAX_POSITIVE_POINTS = 5
LARGE_MASK_COVERAGE_PCT = 50.0
TINY_MASK_COVERAGE_PCT = 0.05
JPEG_QUALITY = 95


def track_with_sam2_video_predictor(
    video_path: str,
    first_frame_mask: np.ndarray,
    model_cache_dir: str,
    device: str = "cuda",
    on_progress: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Track first_frame_mask across every frame of video_path with SAM2 VP.

    Returns (T, H, W) uint8 masks (0/255) at the tracking resolution
    (≤1024px longest side, padded to a multiple of 8). Falls back to CPU
    tracking on CUDA OOM; any other failure propagates and fails the job.
    """
    import torch

    try:
        return _track(video_path, first_frame_mask, model_cache_dir, device, on_progress)
    except torch.cuda.OutOfMemoryError:
        if device == "cpu":
            raise
        log.warning("sam2_vp_fallback_cpu", video_path=video_path)
        torch.cuda.empty_cache()
        return _track(video_path, first_frame_mask, model_cache_dir, "cpu", on_progress)


def _track(
    video_path: str,
    first_frame_mask: np.ndarray,
    model_cache_dir: str,
    device: str,
    on_progress: Callable[[int], None] | None,
) -> np.ndarray:
    import cv2
    import sys
    import torch

    if SAM2_REPO_PATH not in sys.path:
        sys.path.insert(0, SAM2_REPO_PATH)
    from sam2.build_sam import build_sam2_video_predictor

    started = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Container metadata can over/under-report; used only as an upper bound on
    # reads so a broken stream can't loop forever — EOF is the real terminator.
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError(f"could not read video dimensions from {video_path}")

    proc_w, proc_h, pad_w, pad_h, scale = _processing_geometry(src_w, src_h)
    out_h, out_w = proc_h + pad_h, proc_w + pad_w

    seed_mask = _resize_and_pad_mask(first_frame_mask, proc_w, proc_h, pad_w, pad_h)
    coverage_pct = float((seed_mask > 0).mean() * 100)

    if coverage_pct == 0.0:
        total = _count_frames(cap, meta_frames)
        cap.release()
        log.warning("empty_mask_no_tracking", num_frames=total)
        return np.zeros((total, out_h, out_w), dtype=np.uint8)
    if coverage_pct > LARGE_MASK_COVERAGE_PCT:
        log.warning("large_mask_coverage", coverage_pct=coverage_pct)
    if coverage_pct < TINY_MASK_COVERAGE_PCT:
        log.warning("tiny_mask_coverage", coverage_pct=coverage_pct)

    # Release VRAM held by the SAM2 image predictor before loading SAM2 VP —
    # they must never be resident simultaneously on the 4GB RTX 3050.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Loaded directly (not via the model pool): segmentation and tracking run
    # strictly sequentially in this worker and VP is freed right after use, so
    # the pool's LRU/VRAM management would only get in the way here.
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        os.path.join(model_cache_dir, SAM2_CHECKPOINT_FILE),
        device=device,
    )
    log.info(
        "sam2_vp_model_loaded",
        device=device,
        proc_size=f"{out_w}x{out_h}",
        seed_coverage_pct=round(coverage_pct, 2),
    )

    from workers.segmentation.oiv_refiner import refine_tracked_masks

    segments: list[np.ndarray] = []
    frames_done = 0
    parent_tmp = tempfile.mkdtemp(prefix="sam2vp_")
    try:
        segment_index = 0
        max_reads = meta_frames if meta_frames > 0 else None
        while True:
            seg_dir = os.path.join(parent_tmp, f"seg_{segment_index}")
            os.makedirs(seg_dir)
            budget = MAX_SEGMENT_FRAMES
            if max_reads is not None:
                budget = min(budget, max_reads - frames_done)
            num_frames = _extract_segment_jpegs(
                cap, seg_dir, budget, proc_w, proc_h, pad_w, pad_h
            )
            if num_frames == 0:
                shutil.rmtree(seg_dir, ignore_errors=True)
                break

            if segment_index > 0 and not np.any(seed_mask):
                # Object left the frame at a segment boundary — nothing to
                # seed propagation with, so the remainder is object-free.
                log.warning(
                    "segment_seed_mask_empty",
                    segment_index=segment_index,
                    frames_remaining=num_frames,
                )
                segments.append(np.zeros((num_frames, out_h, out_w), dtype=np.uint8))
            else:
                seg_masks = _track_segment(
                    predictor,
                    seg_dir,
                    seed_mask,
                    use_points=(segment_index == 0),
                    total_hint=(max_reads or num_frames),
                    frames_before=frames_done,
                    on_progress=on_progress,
                )
                # Keep OIV drift verification, exactly as the XMem++ path did.
                seg_masks = refine_tracked_masks(
                    frames=_JpegFrameDir(seg_dir, num_frames),
                    tracked_masks=seg_masks,
                    model_cache_dir=model_cache_dir,
                    device="cpu",  # CPU so it never contends with SAM2 VP VRAM
                )
                segments.append(seg_masks)
                seed_mask = seg_masks[-1]

            frames_done += num_frames
            shutil.rmtree(seg_dir, ignore_errors=True)
            segment_index += 1
            if num_frames < budget or (max_reads is not None and frames_done >= max_reads):
                break
    finally:
        cap.release()
        shutil.rmtree(parent_tmp, ignore_errors=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not segments:
        raise RuntimeError(f"no frames decoded from {video_path}")

    result = np.concatenate(segments, axis=0)
    elapsed = time.perf_counter() - started
    log.info(
        "sam2_vp_tracking_complete",
        num_frames=len(result),
        coverage_mean=float(result.mean() / 255 * 100),
        elapsed_sec=round(elapsed, 2),
        num_segments=len(segments),
        device=device,
    )
    return result


def _track_segment(
    predictor: Any,
    frames_dir: str,
    seed_mask: np.ndarray,
    use_points: bool,
    total_hint: int,
    frames_before: int,
    on_progress: Callable[[int], None] | None,
) -> np.ndarray:
    """Propagate the seed mask across one ≤500-frame segment of JPEG frames."""
    import contextlib

    import torch

    on_cuda = predictor.device.type == "cuda"
    # bf16 autocast matches the official SAM2 VP examples and roughly halves
    # inference VRAM/time; RTX 3050 (Ampere) supports bf16 natively.
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if on_cuda and torch.cuda.is_bf16_supported()
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), autocast:
        # Keeping frames + per-frame state in CPU RAM is what holds peak VRAM
        # at ~0.9GB — the default loads every frame onto the GPU (~3.7GB for
        # 300 frames at 1024px), an instant OOM on the 4GB RTX 3050.
        state = predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=on_cuda,
            offload_state_to_cpu=on_cuda,
        )
        try:
            if use_points:
                points = _points_from_mask(seed_mask)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=np.array(points, dtype=np.float32),
                    labels=np.ones(len(points), dtype=np.int32),
                )
            else:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    mask=(seed_mask > 0),
                )

            num_frames = state["num_frames"]
            tracked: dict[int, np.ndarray] = {}
            for frame_idx, _obj_ids, masks in predictor.propagate_in_video(state):
                mask = np.squeeze((masks[0] > 0.5).cpu().numpy()).astype(np.uint8) * 255
                tracked[frame_idx] = mask
                if on_progress and frame_idx % 30 == 0 and total_hint > 0:
                    pct = int((frames_before + frame_idx + 1) / total_hint * 100)
                    on_progress(min(pct, 99))
        finally:
            predictor.reset_state(state)

    template = next(iter(tracked.values()))
    return np.stack(
        [tracked.get(i, np.zeros_like(template)) for i in range(num_frames)]
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


class _JpegFrameDir:
    """Index-addressable view over extracted JPEG frames, decoded on access.

    Lets OIV refinement crop frames one at a time without holding a full
    500-frame RGB segment (~880MB at 1024px) in RAM.
    """

    def __init__(self, frames_dir: str, count: int) -> None:
        self._dir = frames_dir
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx: int) -> np.ndarray:
        import cv2

        path = os.path.join(self._dir, f"{idx:05d}.jpg")
        frame_bgr = cv2.imread(path)
        if frame_bgr is None:
            raise RuntimeError(f"could not re-read extracted frame {path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _processing_geometry(src_w: int, src_h: int) -> tuple[int, int, int, int, float]:
    """Return (proc_w, proc_h, pad_w, pad_h, scale) for ≤1024px, %8-padded frames."""
    scale = min(1.0, MAX_FRAME_SIDE / max(src_w, src_h))
    proc_w = max(8, int(round(src_w * scale)))
    proc_h = max(8, int(round(src_h * scale)))
    pad_w = (8 - proc_w % 8) % 8
    pad_h = (8 - proc_h % 8) % 8
    return proc_w, proc_h, pad_w, pad_h, scale


def _resize_and_pad_mask(
    mask: np.ndarray, proc_w: int, proc_h: int, pad_w: int, pad_h: int
) -> np.ndarray:
    """Bring the first-frame mask into tracking geometry (0/255 uint8)."""
    import cv2

    binary = (mask > 127).astype(np.uint8) * 255
    if binary.shape[:2] != (proc_h, proc_w):
        binary = cv2.resize(binary, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
    if pad_w or pad_h:
        binary = cv2.copyMakeBorder(
            binary, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )
    return binary


def _extract_segment_jpegs(
    cap: Any,
    seg_dir: str,
    max_frames: int,
    proc_w: int,
    proc_h: int,
    pad_w: int,
    pad_h: int,
) -> int:
    """Decode up to max_frames frames into seg_dir as 00000.jpg… at tracking size."""
    import cv2

    count = 0
    while count < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_bgr.shape[1] != proc_w or frame_bgr.shape[0] != proc_h:
            frame_bgr = cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        if pad_w or pad_h:
            frame_bgr = cv2.copyMakeBorder(
                frame_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )
        cv2.imwrite(
            os.path.join(seg_dir, f"{count:05d}.jpg"),
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        count += 1
    return count


def _count_frames(cap: Any, meta_frames: int) -> int:
    """Count decodable frames (bounded by metadata when available)."""
    count = 0
    limit = meta_frames if meta_frames > 0 else None
    while limit is None or count < limit:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    return count


def _points_from_mask(mask: np.ndarray) -> list[list[int]]:
    """Pick up to 5 positive point prompts from inside the mask.

    Centroid (snapped onto the object for concave shapes) plus the four
    extreme mask pixels nudged 20% toward the centroid, so every prompt is a
    confident interior click rather than a boundary pixel.
    """
    ys, xs = np.nonzero(mask)
    cy, cx = int(round(float(ys.mean()))), int(round(float(xs.mean())))
    if not mask[cy, cx]:
        nearest = int(((ys - cy) ** 2 + (xs - cx) ** 2).argmin())
        cy, cx = int(ys[nearest]), int(xs[nearest])

    points: list[list[int]] = [[cx, cy]]
    extremes = [
        (int(xs[ys.argmin()]), int(ys.min())),  # topmost
        (int(xs[ys.argmax()]), int(ys.max())),  # bottommost
        (int(xs.min()), int(ys[xs.argmin()])),  # leftmost
        (int(xs.max()), int(ys[xs.argmax()])),  # rightmost
    ]
    for ex, ey in extremes:
        nx = int(round(ex + 0.2 * (cx - ex)))
        ny = int(round(ey + 0.2 * (cy - ey)))
        px, py = (nx, ny) if mask[ny, nx] else (ex, ey)
        if [px, py] not in points:
            points.append([px, py])
    return points[:MAX_POSITIVE_POINTS]
