"""Overall job progress computation from worker progress events.

Stage weights mirror computeOverallPct in frontend/src/pages/JobProgress.tsx —
keep the two implementations in sync.
"""
from __future__ import annotations

from typing import Any


def compute_overall_pct(msg: dict[str, Any]) -> float:
    """Map a worker progress event to an overall 0-100 job percentage.

    Chunked stages (tracking, inpainting) publish 1-based chunk/total fields;
    events without chunk info are treated as a single chunk spanning the stage.

    Stage budget: segmenting 0-5, tracking 5-45, inpainting 45-80,
    stitching 83, boundary_fusion 90, quality_check 96, completed 100.
    """
    stage = msg.get("stage") or ""
    chunk = msg.get("chunk") or 1
    total = msg.get("total") or 1
    pct = msg.get("pct") or 0

    if stage in ("segmenting", "segmentation"):
        return 5.0

    if stage == "tracking":
        fraction = (chunk - 1 + pct / 100) / total
        return float(round(min(max(5 + fraction * 40, 5.0), 45.0)))

    if stage == "inpainting":
        fraction = (chunk - 1 + pct / 100) / total
        return float(round(min(max(45 + fraction * 35, 45.0), 80.0)))

    if stage == "stitching":
        return 83.0
    if stage == "boundary_fusion":
        return 90.0
    if stage in ("quality", "quality_check"):
        return 96.0
    if stage == "completed":
        return 100.0

    # Unknown stage: fall back to the raw stage pct
    return float(pct or 0)
