from __future__ import annotations

import os

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

log = structlog.get_logger(__name__)

OIV_THRESHOLD = 0.2  # lowered from 0.3 — less aggressive rejection
CROP_PADDING_FRAC = 0.2
INPUT_SIZE = (256, 256)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class OIVProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class OIVModel(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = OIVProjectionHead(2048, 512, 256)

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        embedding = self.projection(features)
        return F.normalize(embedding, dim=1)


def _load_oiv_model(model_cache_dir: str, device: str):
    model = OIVModel()
    ckpt = torch.load(
        os.path.join(model_cache_dir, "oiv_final.pth"), map_location=device
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


def _build_transform():
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Resize(INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )


def refine_tracked_masks(
    frames: list,
    tracked_masks: np.ndarray,
    model_cache_dir: str,
    device: str = "cpu",
) -> np.ndarray:
    """Verify XMem++ tracked masks against an OIV reference embedding and
    recover frames where tracking drifted off the original object.

    OIV is loaded directly (not via the model pool): it runs on CPU while
    SAM2/XMem hold the GPU, so the pool's VRAM management would only get in
    the way here.
    """
    if not np.any(tracked_masks):
        log.warning("oiv_refinement_skipped_no_masks", num_frames=len(tracked_masks))
        return tracked_masks

    if not np.any(tracked_masks[0]):
        log.warning("oiv_refinement_skipped_empty_reference_frame")
        return tracked_masks

    from PIL import Image

    model = _load_oiv_model(model_cache_dir, device)
    log.info("oiv_model_loaded")
    transform = _build_transform()

    num_frames = len(tracked_masks)
    embeddings: list = [None] * num_frames

    with torch.no_grad():
        for t in range(num_frames):
            bbox = _bbox_from_mask(tracked_masks[t])
            if bbox is None:
                continue
            y0, y1, x0, x1 = bbox
            crop = Image.fromarray(frames[t][y0:y1, x0:x1])
            tensor = transform(crop).unsqueeze(0).to(device)
            embeddings[t] = model(tensor)[0]

    ref_emb = embeddings[0]

    confirmed = 1
    recovered = 0
    absent = 0
    lost_no_embedding = 0

    for t in range(1, num_frames):
        emb = embeddings[t]
        if emb is None:
            lost_no_embedding += 1
            continue

        sim = F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
        if sim >= OIV_THRESHOLD:
            confirmed += 1
            continue

        log.info("oiv_tracking_lost", frame=t, similarity=sim)
        recovery_j = None
        for j in range(t - 1, -1, -1):
            emb_j = embeddings[j]
            if emb_j is None:
                continue
            sim_j = F.cosine_similarity(emb_j.unsqueeze(0), ref_emb.unsqueeze(0)).item()
            if sim_j >= OIV_THRESHOLD:
                recovery_j = j
                break

        if recovery_j is not None:
            tracked_masks[t] = tracked_masks[recovery_j]
            recovered += 1
            log.info("oiv_tracking_recovered", frame=t, recovered_from=recovery_j)
        else:
            tracked_masks[t] = np.zeros_like(tracked_masks[t])
            absent += 1
            log.info("oiv_object_marked_absent", frame=t)

    log.info(
        "oiv_refinement_stats",
        total_frames=num_frames,
        confirmed=confirmed,
        recovered=recovered,
        absent=absent,
        lost_no_embedding=lost_no_embedding,
    )

    return tracked_masks


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (y0, y1, x0, x1) bounding box with 20% padding, or None if mask is empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad_y = int((y1 - y0 + 1) * CROP_PADDING_FRAC)
    pad_x = int((x1 - x0 + 1) * CROP_PADDING_FRAC)

    h, w = mask.shape[:2]
    return (
        max(0, y0 - pad_y),
        min(h, y1 + pad_y + 1),
        max(0, x0 - pad_x),
        min(w, x1 + pad_x + 1),
    )
