from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any, Callable

import numpy as np
import structlog
from celery.exceptions import SoftTimeLimitExceeded

from api.core.config import get_settings
from api.core.metrics import SEGMENTS_TOTAL
from api.core.redis import acquire_redis, publish_progress, set_job_status
from api.models.job import JobStatus
from workers.celery_app import celery_app

log = structlog.get_logger(__name__)

PROPAINTER_ROOT = "/home/kavish/ProPainter"
RAFT_ITERS = 10  # reduced for 4GB VRAM
REF_STRIDE = 10
NEIGHBOR_LENGTH = 10

MODAL_APP_NAME = "visionerase-propainter"
MODAL_FUNCTION_NAME = "inpaint_frames"
# Segments above this resolution are routed to Modal's T4 (16GB) instead of
# running locally, where ProPainter is capped to fit 4GB VRAM.
MODAL_MIN_WIDTH = 854
MODAL_MIN_HEIGHT = 480


@celery_app.task(
    bind=True,
    queue="inpainting",
    max_retries=2,
    soft_time_limit=600,
    time_limit=720,
    name="workers.inpainting.tasks.inpaint_segment",
)
def inpaint_segment(
    self,
    job_id: str,
    segment_s3_key: str,
    tracked_masks_s3_key: str,
) -> dict[str, Any]:
    """Inpaint a segment with ProPainter, guided by XMem++ tracked masks."""
    settings = get_settings()
    bound_log = log.bind(job_id=job_id, task_id=self.request.id)
    bound_log.info("inpainting_started")

    loop = asyncio.new_event_loop()

    def _report_progress(pct: int) -> None:
        loop.run_until_complete(_publish(job_id, {"stage": "inpainting", "pct": pct}))

    try:
        _report_progress(0)
        loop.run_until_complete(
            _update_status(job_id, JobStatus.INPAINTING, None)
        )
        SEGMENTS_TOTAL.labels(status="inpainting_started").inc()

        s3 = _make_s3_client(settings)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "segment.mp4")
            masks_path = os.path.join(tmp, "tracked_masks.npy")
            s3.download_file(settings.s3_bucket, segment_s3_key, video_path)
            s3.download_file(settings.s3_bucket, tracked_masks_s3_key, masks_path)

            masks = np.load(masks_path)  # (T, H, W) uint8, values 0 or 255
            width, height = _read_resolution(video_path)
            fps = _read_fps(video_path)

            use_modal = settings.modal_enabled and (
                width > MODAL_MIN_WIDTH or height > MODAL_MIN_HEIGHT
            )

            output_key = f"jobs/{job_id}/inpainted/output.mp4"
            output_path = os.path.join(tmp, "output.mp4")

            if use_modal:
                # Extract only as many frames as we have masks
                # Cap at 10 frames for Modal T4 VRAM constraints
                # Process all frames — Modal A10G handles full video
                n_frames = len(masks)
                frames_rgb = _extract_n_frames(video_path, n_frames)
                num_frames = len(frames_rgb)
                bound_log.info(
                    "using_modal_propainter",
                    resolution=f"{width}x{height}",
                    num_frames=num_frames,
                )

                frames_bytes = _encode_frames_jpeg(frames_rgb)
                masks_bytes = _encode_masks_png(masks)
                _report_progress(20)

                video_bytes = _run_modal_inpainting(frames_bytes, masks_bytes, fps)
                _report_progress(80)

                with open(output_path, "wb") as f:
                    f.write(video_bytes)
                s3.upload_file(output_path, settings.s3_bucket, output_key)

                status = "modal_hires"
            else:
                num_frames = len(masks)
                bound_log.info(
                    "using_local_propainter",
                    resolution=f"{width}x{height}",
                    num_frames=num_frames,
                )

                frames = _extract_n_frames(video_path, num_frames)
                if len(frames) != num_frames:
                    raise RuntimeError(
                        f"frame/mask count mismatch: got {len(frames)} frames "
                        f"for {num_frames} masks from {segment_s3_key}"
                    )
                bound_log.info("frames_extracted", num_frames=num_frames)

                comp_frames = _run_propainter_inpainting(
                    frames, masks, settings, bound_log, _report_progress
                )
                _write_video(comp_frames, output_path, fps)
                s3.upload_file(output_path, settings.s3_bucket, output_key)

                status = "real"

        _report_progress(100)
        SEGMENTS_TOTAL.labels(status="inpainting_complete").inc()

        from workers.stitching.tasks import stitch_segments
        stitch_segments.apply_async(
            args=[job_id, [output_key]],
            queue="stitching",
        )

        return {
            "job_id": job_id,
            "inpainted_s3_key": output_key,
            "status": status,
            "num_frames": num_frames,
        }

    except SoftTimeLimitExceeded:
        bound_log.error("inpainting_soft_timeout")
        SEGMENTS_TOTAL.labels(status="inpainting_timeout").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", "inpaint_segment timed out")
        )
        raise
    except Exception as exc:
        bound_log.exception("inpainting_error", error=str(exc))
        SEGMENTS_TOTAL.labels(status="inpainting_error").inc()
        loop.run_until_complete(
            _update_status(job_id, "failed", str(exc))
        )
        raise
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_s3_client(settings: Any) -> Any:
    """Return a boto3 S3 client configured for the MinIO endpoint."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
    )


def _extract_n_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    """Read exactly num_frames frames as (H, W, 3) uint8 RGB arrays."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while len(frames) < num_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _read_fps(video_path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def _read_resolution(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the video without decoding any frames."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def _extract_all_frames(video_path: str) -> list[np.ndarray]:
    """Read every frame as a (H, W, 3) uint8 BGR array."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(frame_bgr)
    cap.release()
    return frames


def _encode_frames_jpeg(frames_rgb: list[np.ndarray]) -> list[bytes]:
    """JPEG-encode RGB frames for transport to the Modal function."""
    import cv2

    frames_bytes = []
    for frame in frames_rgb:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames_bytes.append(buf.tobytes())
    return frames_bytes


def _encode_masks_png(masks: np.ndarray) -> list[bytes]:
    """PNG-encode tracked masks for transport to the Modal function."""
    import cv2

    masks_bytes = []
    for mask in masks:
        _, buf = cv2.imencode(".png", mask)
        masks_bytes.append(buf.tobytes())
    return masks_bytes


def _run_modal_inpainting(
    frames_bytes: list[bytes],
    masks_bytes: list[bytes],
    fps: float,
) -> bytes:
    """Call the deployed Modal function to inpaint at full resolution on a T4."""
    import modal

    f = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
    return f.remote(frames_bytes, masks_bytes, fps=fps)


def _write_video(frames: list[np.ndarray], output_path: str, fps: float) -> None:
    """Write a list of (H, W, 3) uint8 RGB frames to an MP4 file."""
    import cv2

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    try:
        for frame_rgb in frames:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _get_ref_index(
    mid_neighbor_id: int,
    neighbor_ids: list[int],
    length: int,
    ref_stride: int = 10,
    ref_num: int = -1,
) -> list[int]:
    """Pick global reference frame indices outside the local neighbor window."""
    ref_index: list[int] = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


def _load_propainter_models(model_dir: str, device: Any, bound_log: Any) -> tuple:
    """Load RAFT, flow-completion and inpainting models directly onto CUDA.

    Loaded directly (not via the model pool): XMem++ runs on CPU during
    tracking, so the GPU is free for ProPainter and the pool's VRAM
    management would only get in the way here.
    """
    import torch
    from propainter_model.modules.flow_comp_raft import RAFT_bi
    from propainter_model.propainter import InpaintGenerator
    from propainter_model.recurrent_flow_completion import RecurrentFlowCompleteNet

    torch.cuda.empty_cache()

    fix_raft = RAFT_bi(
        model_path=os.path.join(model_dir, "raft-things.pth"),
        device=device,
    )

    fix_flow_complete = RecurrentFlowCompleteNet()
    flow_ckpt = torch.load(
        os.path.join(model_dir, "recurrent_flow_completion.pth"),
        map_location="cpu",
    )
    fix_flow_complete.load_state_dict(flow_ckpt)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    model = InpaintGenerator(
        init_weights=False,
        model_path=os.path.join(model_dir, "ProPainter.pth"),
    ).to(device)
    model.eval()

    bound_log.info("propainter_models_loaded")
    return fix_raft, fix_flow_complete, model


def _extract_n_frames(video_path: str, n: int) -> list:
    """Extract exactly n frames from video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _run_propainter_inpainting(
    frames: list[np.ndarray],
    masks: np.ndarray,
    settings: Any,
    bound_log: Any,
    on_progress: Callable[[int], None],
) -> list[np.ndarray]:
    """Inpaint frames guided by tracked masks using ProPainter.

    Mirrors the preprocessing/inference pipeline in
    ProPainter/inference_propainter.py (resize to a multiple of 8, RAFT
    flow, flow completion, image propagation, then the feature
    propagation + transformer pass over neighbor/reference frames).
    """
    # Insert ProPainter at front and ensure visionerase model/ doesn't shadow it
    import sys as _sys
    if PROPAINTER_ROOT not in _sys.path:
        _sys.path.insert(0, PROPAINTER_ROOT)
    # Remove cwd from path to avoid model/ conflict with ProPainter
    for _p in list(_sys.path):
        if _p in ('', '.', '/home/kavish/visionerase'):
            _sys.path.remove(_p)

    import cv2
    import scipy.ndimage
    import torch
    from PIL import Image

    from core.utils import to_tensors

    device = torch.device("cuda")

    pil_frames = [Image.fromarray(f) for f in frames]
    torch.cuda.empty_cache()  # free VRAM from SAM2/XMem before ProPainter
    out_size = pil_frames[0].size  # (W, H)
    # Cap resolution for 4GB VRAM — ProPainter is VRAM hungry
    MAX_W, MAX_H = 432, 240
    scale = min(MAX_W / out_size[0], MAX_H / out_size[1], 1.0)
    proc_w = int(out_size[0] * scale) - int(out_size[0] * scale) % 8
    proc_h = int(out_size[1] * scale) - int(out_size[1] * scale) % 8
    proc_size = (proc_w, proc_h)
    if proc_size != out_size:
        pil_frames = [f.resize(proc_size) for f in pil_frames]

    flow_masks: list[Any] = []
    masks_dilated: list[Any] = []
    for mask in masks:
        mask_img = np.array(Image.fromarray(mask).resize(proc_size, Image.NEAREST).convert("L"))
        flow_mask = scipy.ndimage.binary_dilation(mask_img, iterations=8).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask * 255))
        dilated = scipy.ndimage.binary_dilation(mask_img, iterations=4).astype(np.uint8)
        masks_dilated.append(Image.fromarray(dilated * 255))

    ori_frames = [np.array(f).astype(np.uint8) for f in pil_frames]

    frames_t = (to_tensors()(pil_frames).unsqueeze(0) * 2 - 1).to(device)
    flow_masks_t = to_tensors()(flow_masks).unsqueeze(0).to(device)
    masks_dilated_t = to_tensors()(masks_dilated).unsqueeze(0).to(device)

    model_dir = os.path.join(settings.model_cache_dir, "propainter")
    fix_raft, fix_flow_complete, model = _load_propainter_models(model_dir, device, bound_log)
    on_progress(30)

    w, h = proc_size
    video_length = frames_t.size(1)

    with torch.no_grad():
        if w <= 640:
            short_clip_len = 12
        elif w <= 720:
            short_clip_len = 8
        elif w <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        if video_length > short_clip_len:
            flows_f_list, flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames_t[:, f:end_f], iters=RAFT_ITERS)
                else:
                    flows_f, flows_b = fix_raft(frames_t[:, f - 1:end_f], iters=RAFT_ITERS)
                flows_f_list.append(flows_f)
                flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
            gt_flows_bi = (torch.cat(flows_f_list, dim=1), torch.cat(flows_b_list, dim=1))
        else:
            gt_flows_bi = fix_raft(frames_t, iters=RAFT_ITERS)
            torch.cuda.empty_cache()

        bound_log.info("propainter_flow_computed")
        on_progress(60)

        pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_t)
        pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_t)
        torch.cuda.empty_cache()

        masked_frames = frames_t * (1 - masks_dilated_t)
        b, t, _, _, _ = masks_dilated_t.size()
        prop_imgs, updated_local_masks = model.img_propagation(
            masked_frames, pred_flows_bi, masks_dilated_t, "nearest"
        )
        updated_frames = (
            frames_t * (1 - masks_dilated_t) + prop_imgs.view(b, t, 3, h, w) * masks_dilated_t
        )
        updated_masks = updated_local_masks.view(b, t, 1, h, w)
        torch.cuda.empty_cache()

    comp_frames: list[Any] = [None] * video_length
    neighbor_stride = NEIGHBOR_LENGTH // 2

    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = list(
            range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))
        )
        ref_ids = _get_ref_index(f, neighbor_ids, video_length, REF_STRIDE, ref_num=-1)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids]
        selected_masks = masks_dilated_t[:, neighbor_ids + ref_ids]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1]],
            pred_flows_bi[1][:, neighbor_ids[:-1]],
        )

        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(
                selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t
            )
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = (
                masks_dilated_t[0, neighbor_ids].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            )

            for i, idx in enumerate(neighbor_ids):
                img = (
                    pred_img[i].astype(np.uint8) * binary_masks[i]
                    + ori_frames[idx] * (1 - binary_masks[i])
                )
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (
                        comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    ).astype(np.uint8)
        torch.cuda.empty_cache()

    bound_log.info("propainter_inference_complete", num_frames=video_length)

    return [cv2.resize(f, out_size) for f in comp_frames]


async def _publish(job_id: str, data: dict) -> None:
    from redis import asyncio as _aioredis
    import json as _json
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
