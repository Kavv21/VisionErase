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

SAM2_PATH = "/home/kavish/sam2"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT_FILE = "sam2_hiera_small.pt"

MAX_FRAME_SIDE = 1024        # SAM2-recommended max resolution
MAX_SEGMENT_FRAMES = 500     # longer videos are tracked in seeded segments
MAX_POSITIVE_POINTS = 5
LARGE_MASK_COVERAGE_PCT = 50.0
TINY_MASK_COVERAGE_PCT = 0.05
JPEG_QUALITY = 95

# Bidirectional tracking: a second SAM2 VP pass over the reversed video, seeded
# with the forward pass's last mask. Where the two passes agree the mask is
# trustworthy; where they diverge the object was probably lost or drifted.
# It doubles tracking wall time and ffmpeg's `reverse` filter buffers the whole
# decoded video in RAM, so it only runs on short clips.
BACKWARD_TRACKING_MAX_FRAMES = 500
HIGH_CONFIDENCE_IOU = 0.85   # >= this: intersection (both passes agree)
MEDIUM_CONFIDENCE_IOU = 0.65  # >= this: union (OIV validates later)
FFMPEG_REVERSE_TIMEOUT_SEC = 900


# Executed in a fresh interpreter (sys.argv[1] = input pickle, sys.argv[2] =
# output .npy). Celery's prefork workers cannot re-initialize CUDA ("Cannot
# re-initialize CUDA in forked subprocess"), which silently drops SAM2 VP to
# CPU at ~25x the per-frame cost — a clean process initializes CUDA normally.
# The CUDA availability check must happen here, in the fresh process, not in
# the forked parent where it can misreport.
_RUNNER_SCRIPT = """\
import sys

sys.path.insert(0, "/home/kavish/visionerase")
sys.path.insert(0, "/home/kavish/sam2")
sys.path.insert(0, "/home/kavish/XMem")
sys.path.insert(0, "/home/kavish/ProPainter")

import pickle

import numpy as np
import torch

with open(sys.argv[1], "rb") as f:
    kwargs = pickle.load(f)

# Authoritative device decision: this is a fresh process, so the check is
# trustworthy here (unlike in the forked Celery parent, which can misreport
# in either direction). The requested value is ignored on purpose — always
# use CUDA when this process can see it.
device = "cuda" if torch.cuda.is_available() else "cpu"

from workers.segmentation.sam2_video_tracker import _track

result = _track(
    video_path=kwargs["video_path"],
    first_frame_mask=kwargs["first_frame_mask"],
    model_cache_dir=kwargs["model_cache_dir"],
    device=device,
    on_progress=None,
)
np.save(sys.argv[2], result)
"""

SUBPROCESS_TIMEOUT_SEC = 7200  # 2h — long enough for a 60-min video (~86k frames × 0.15s/frame)


def track_with_sam2_video_predictor(
    video_path: str,
    first_frame_mask,
    model_cache_dir: str,
    device: str = "cuda",
    on_progress: Callable[[int], None] | None = None,
) -> "np.ndarray":
    """Track the first-frame mask across the video, forward then (short clips) back.

    The forward pass is authoritative. On clips of at most
    BACKWARD_TRACKING_MAX_FRAMES a second pass runs over the reversed video and
    the two are reconciled per frame by IoU (see compute_mask_consensus), which
    catches drift the forward pass alone cannot see. Backward tracking is
    strictly best-effort: anything that goes wrong falls back to the forward
    masks rather than failing the job.
    """
    from api.core.metrics import TRACKING_CONSENSUS_IOU, TRACKING_DIRECTION_TOTAL

    forward_masks = _run_track_subprocess(
        video_path,
        np.asarray(first_frame_mask),
        model_cache_dir,
        requested_device=device,
        direction="forward",
    )

    total_frames = len(forward_masks)
    if total_frames > BACKWARD_TRACKING_MAX_FRAMES:
        # Backward tracking doubles tracking time; not worth it on long videos.
        log.info(
            "skipping_backward_tracking_long_video",
            total_frames=total_frames,
            max_frames=BACKWARD_TRACKING_MAX_FRAMES,
        )
        TRACKING_DIRECTION_TOTAL.labels(direction="forward_only_long_video").inc()
        return forward_masks

    try:
        last_mask = forward_masks[-1]
        if not np.any(last_mask > 127):
            # Nothing to seed the reverse pass with — the object left the frame
            # (or was never found) by the last frame.
            log.info("skipping_backward_tracking_empty_last_mask", total_frames=total_frames)
            TRACKING_DIRECTION_TOTAL.labels(direction="forward_only_empty_seed").inc()
            return forward_masks

        backward_masks = run_backward_tracking(video_path, last_mask, model_cache_dir)

        if backward_masks.shape != forward_masks.shape:
            # A re-encode that drops or duplicates a frame would silently
            # misalign every mask, so refuse the consensus rather than guess.
            raise RuntimeError(
                f"backward masks {backward_masks.shape} do not match "
                f"forward masks {forward_masks.shape}"
            )

        consensus_masks, reliability = compute_mask_consensus(forward_masks, backward_masks)

        unreliable = int((reliability < MEDIUM_CONFIDENCE_IOU).sum())
        TRACKING_CONSENSUS_IOU.observe(float(reliability.mean()))
        TRACKING_DIRECTION_TOTAL.labels(direction="bidirectional").inc()
        log.info(
            "bidirectional_tracking_complete",
            unreliable_frames=unreliable,
            mean_iou=round(float(reliability.mean()), 4),
            min_iou=round(float(reliability.min()), 4),
            total_frames=total_frames,
        )
        return consensus_masks

    except Exception as exc:
        log.warning("backward_tracking_failed_using_forward", error=str(exc))
        TRACKING_DIRECTION_TOTAL.labels(direction="forward_only_failed").inc()
        return forward_masks


def run_backward_tracking(
    video_path: str,
    last_frame_mask: "np.ndarray",
    model_cache_dir: str,
) -> "np.ndarray":
    """Track the object from the last frame back to the first.

    The video is reversed with ffmpeg and tracked exactly like the forward
    pass, seeded with the forward pass's final mask; the resulting masks are
    flipped back into original frame order before returning.

    Returns (T, H, W) uint8 masks in the same tracking geometry as the
    forward pass.
    """
    import subprocess

    work_dir = tempfile.mkdtemp(prefix="sam2vp_rev_")
    try:
        reversed_path = os.path.join(work_dir, "reversed.mp4")
        started = time.perf_counter()
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "reverse",
                "-an",                      # audio is irrelevant and areverse is costly
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                reversed_path,
            ],
            capture_output=True,
            text=True,
            timeout=FFMPEG_REVERSE_TIMEOUT_SEC,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg reverse failed (rc={proc.returncode}):\n{proc.stderr[-1000:]}"
            )
        log.info(
            "video_reversed",
            elapsed_sec=round(time.perf_counter() - started, 2),
            size_mb=round(os.path.getsize(reversed_path) / 1e6, 2),
        )

        backward = _run_track_subprocess(
            reversed_path,
            _strip_tracking_padding(last_frame_mask, reversed_path),
            model_cache_dir,
            requested_device="cuda",
            direction="backward",
        )
        # Masks come back in reversed-video order; put them back on the
        # original timeline. .copy() because the negative stride view would
        # otherwise be passed on to cv2/np.save.
        return backward[::-1].copy()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def compute_mask_consensus(
    forward_masks: "np.ndarray",
    backward_masks: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Reconcile the forward and backward masks frame by frame.

    IoU >= 0.85: accept the intersection (high confidence — both passes agree,
                 so the intersection trims tracking bleed without losing object)
    IoU 0.65-0.85: accept the union (medium confidence, OIV validates later)
    IoU < 0.65: unreliable — keep the forward mask and flag it via the score

    Returns: consensus_masks (T, H, W) uint8 0/255
             reliability_scores (T,) float32 — the per-frame IoU
    """
    T = len(forward_masks)
    consensus = np.zeros_like(forward_masks)
    reliability = np.zeros(T, dtype=np.float32)

    for t in range(T):
        fwd = (forward_masks[t] > 127).astype(np.uint8)
        bwd = (backward_masks[t] > 127).astype(np.uint8)

        intersection = int((fwd & bwd).sum())
        union = int((fwd | bwd).sum())

        # Both passes agree the object is absent — perfect agreement, not a
        # divide-by-zero.
        iou = 1.0 if union == 0 else intersection / union
        reliability[t] = float(iou)

        if iou >= HIGH_CONFIDENCE_IOU:
            consensus[t] = (fwd & bwd) * 255
        elif iou >= MEDIUM_CONFIDENCE_IOU:
            consensus[t] = (fwd | bwd) * 255
        else:
            consensus[t] = forward_masks[t]

    return consensus, reliability


def _strip_tracking_padding(mask: "np.ndarray", video_path: str) -> "np.ndarray":
    """Undo _track's bottom/right %8 padding so the mask can re-seed a new pass.

    Tracked masks come back at (proc_h + pad_h, proc_w + pad_w). Handing that
    straight back to _resize_and_pad_mask would resize it down to
    (proc_h, proc_w) and then pad again, squashing the mask by up to 7px.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if src_w <= 0 or src_h <= 0:
        return mask

    proc_w, proc_h, pad_w, pad_h, _ = _processing_geometry(src_w, src_h)
    if (pad_w or pad_h) and mask.shape[:2] == (proc_h + pad_h, proc_w + pad_w):
        return np.ascontiguousarray(mask[:proc_h, :proc_w])
    return mask


def _run_track_subprocess(
    video_path: str,
    seed_mask: "np.ndarray",
    model_cache_dir: str,
    requested_device: str = "cuda",
    direction: str = "forward",
) -> "np.ndarray":
    """Run _track() in a fresh subprocess so CUDA can initialize.

    The caller's device argument (settings.device, "cpu" in .env) is
    overridden: CUDA is always requested when available, and the subprocess
    re-checks in its own fresh process, which is the only trustworthy place.
    on_progress is not forwarded — callables can't cross the process boundary.
    Per-frame progress appears only in the relayed subprocess logs.
    """
    import pickle
    import subprocess
    import sys

    import torch as _torch

    _forced_device = "cuda" if _torch.cuda.is_available() else "cpu"

    started = time.perf_counter()
    # Private per-call dir so concurrent workers never race on a shared
    # script/pickle path in /tmp.
    work_dir = tempfile.mkdtemp(prefix="sam2vp_sub_")
    runner_path = os.path.join(work_dir, "runner.py")
    input_path = os.path.join(work_dir, "input.pkl")
    output_path = os.path.join(work_dir, "tracked.npy")
    try:
        with open(runner_path, "w") as f:
            f.write(_RUNNER_SCRIPT)
        with open(input_path, "wb") as f:
            pickle.dump(
                {
                    "video_path": video_path,
                    "first_frame_mask": np.asarray(seed_mask),
                    "model_cache_dir": model_cache_dir,
                    "device": _forced_device,
                },
                f,
            )

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            "/home/kavish/visionerase:"
            "/home/kavish/sam2:"
            "/home/kavish/XMem:"
            "/home/kavish/ProPainter:"
            + env.get("PYTHONPATH", "")
        )

        log.info(
            "sam2_vp_subprocess_started",
            video_path=video_path,
            device=_forced_device,
            requested_device=requested_device,
            direction=direction,
        )
        result = subprocess.run(
            [sys.executable, runner_path, input_path, output_path],
            env=env,
            timeout=SUBPROCESS_TIMEOUT_SEC,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SAM2 VP subprocess failed (rc={result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        tracked = np.load(output_path)
        # Relay the subprocess's own structlog output (device, sec/frame,
        # VRAM) — capture_output would otherwise swallow it entirely. tqdm
        # progress bars are dropped so they don't crowd out the real lines.
        sub_logs = "\n".join(
            line
            for line in (result.stdout + result.stderr).splitlines()
            if line.strip() and "it/s" not in line
        )
        # Extract and re-log OIV stats from subprocess output
        import re as _re
        oiv_match = _re.search(
            r'oiv_refinement_stats\s+absent=(\d+)\s+confirmed=(\d+)\s+lost_no_embedding=(\d+)\s+recovered=(\d+)\s+total_frames=(\d+)',
            sub_logs
        )
        if oiv_match:
            log.info("oiv_refinement_stats",
                absent=int(oiv_match.group(1)),
                confirmed=int(oiv_match.group(2)),
                lost_no_embedding=int(oiv_match.group(3)),
                recovered=int(oiv_match.group(4)),
                total_frames=int(oiv_match.group(5)),
            )
        log.info(
            "sam2_vp_subprocess_complete",
            num_frames=len(tracked),
            direction=direction,
            elapsed_sec=round(time.perf_counter() - started, 2),
            subprocess_logs=sub_logs[-2000:],
        )
        return tracked
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _track(
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
    except torch.cuda.OutOfMemoryError as exc:
        if device == "cpu":
            raise
        # CPU tracking is ~25x slower (3-4s/frame vs 0.15s/frame) — if this
        # fires, tracking still succeeds but the speed win is gone. Loud log
        # so a "0% GPU utilization" run is diagnosable from the worker logs.
        log.warning(
            "sam2_vp_fallback_cpu",
            video_path=video_path,
            oom_error=str(exc),
            vram_allocated_gb=round(torch.cuda.memory_allocated() / 1e9, 2),
        )
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

    if SAM2_PATH not in sys.path:
        sys.path.insert(0, SAM2_PATH)
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
    # Must run from sam2 directory for hydra config resolution
    _orig_dir = os.getcwd()
    os.chdir(SAM2_PATH)
    try:
        predictor = build_sam2_video_predictor(
            SAM2_CONFIG,
            os.path.join(model_cache_dir, SAM2_CHECKPOINT_FILE),
            device=device,
        )
    finally:
        os.chdir(_orig_dir)  # restore working directory

    # Verify the weights actually landed on the requested device — a CPU
    # predictor here means every propagation step silently runs ~25x slower.
    for name, param in list(predictor.named_parameters())[:2]:
        log.info("param_device", name=name, device=str(param.device))
    device_check = next(predictor.parameters()).device
    if device == "cuda" and device_check.type != "cuda":
        log.warning("predictor_on_cpu_moving_to_cuda")
        predictor = predictor.to("cuda")
        device_check = next(predictor.parameters()).device

    log.info(
        "sam2_vp_model_loaded",
        device=device,
        param_device=str(device_check),
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
            total_estimate = max_reads if max_reads is not None else frames_done
            log.info(
                "long_video_progress",
                segments_done=segment_index,
                segments_total=(
                    (total_estimate + MAX_SEGMENT_FRAMES - 1) // MAX_SEGMENT_FRAMES
                    if total_estimate
                    else segment_index
                ),
                frames_done=frames_done,
                total_frames=total_estimate,
                pct_complete=(
                    round(frames_done / total_estimate * 100, 1)
                    if total_estimate
                    else None
                ),
            )
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
        peak_vram_gb=(
            round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available()
            else None
        ),
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
            offload_video_to_cpu=True,   # keep frames in CPU RAM
            offload_state_to_cpu=False,  # keep state on GPU for fast compute
            async_loading_frames=False,
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
            if on_cuda:
                torch.cuda.synchronize()
            prop_started = time.perf_counter()
            for frame_idx, _obj_ids, masks in predictor.propagate_in_video(state):
                mask = np.squeeze((masks[0] > 0.5).cpu().numpy()).astype(np.uint8) * 255
                tracked[frame_idx] = mask
                if len(tracked) == 5:
                    # Early GPU-activity probe: on CUDA this should read
                    # ~0.15s/frame; ~3-4s/frame means compute fell to CPU.
                    if on_cuda:
                        torch.cuda.synchronize()
                    log.info(
                        "sam2_vp_propagation_speed",
                        sec_per_frame=round((time.perf_counter() - prop_started) / 5, 3),
                        device="cuda" if on_cuda else "cpu",
                    )
                if on_progress and frame_idx % 30 == 0 and total_hint > 0:
                    pct = int((frames_before + frame_idx + 1) / total_hint * 100)
                    on_progress(min(pct, 99))
        finally:
            predictor.reset_state(state)

    elapsed = time.perf_counter() - prop_started
    log.info(
        "sam2_vp_segment_tracked",
        num_frames=num_frames,
        sec_per_frame=round(elapsed / max(num_frames, 1), 3),
        elapsed_sec=round(elapsed, 2),
        device="cuda" if on_cuda else "cpu",
    )

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
