# VisionErase — Project Bible

## What this is
A distributed AI video object removal platform. Users upload a video,
paint over an unwanted object on frame 1, and the platform removes it
across every frame with temporal consistency using SAM 2 + ProPainter.

## Architecture (one paragraph)
FastAPI gateway → validates request → checks Redis dedup cache →
enqueues to Redis priority queue → video split into N segments →
N parallel Celery workers each process their segment
(SAM2 segmentation → XMem++ tracking → ProPainter inpainting →
chunk stitching) → BoundaryFusion worker corrects segment boundaries →
final stitcher combines all segments → result in S3 + Redis →
frontend receives completion via WebSocket.

## Folder map
api/              FastAPI gateway, routers, middleware, models, services
api/core/         config, redis, database, metrics
api/middleware/   rate_limiter, logging
api/routers/      jobs, websocket, webhooks, health
api/models/       pydantic schemas
api/services/     storage (S3), auth
models/                   proprietary model definitions
models/boundary_fusion/   BoundaryFusion architecture + inference
workers/          Celery tasks
workers/celery_app.py   pipeline chain definition
workers/segmentation/   SAM 2 tasks
workers/inpainting/     ProPainter tasks
workers/stitching/      chunk merge tasks
workers/quality/        SSIM/PSNR tasks
workers/boundary/       BoundaryFusion Celery tasks
pipeline/         CV model layer
pipeline/pool/    model memory pool (LRU + VRAM management)
pipeline/tracker/ XMem++ mask propagation
pipeline/inpainter/ ProPainter inpainting
pipeline/chunker/ video chunking + seam blending
pipeline/segmenter/       hierarchical segment splitting
infra/            Docker, Prometheus, Grafana, Nginx configs
tests/            unit/, integration/, load/
frontend/         React + TypeScript + Tailwind + Zustand

## Non-negotiable rules (apply to ALL agents)
1. NEVER load a CV model directly — always use pipeline/pool/model_pool.py
2. NEVER hardcode secrets — always read from api/core/config.py settings
3. ALWAYS add a Prometheus metric when adding a new endpoint or task
4. ALWAYS use structlog for logging, never print()
5. ALWAYS use async/await for any I/O in FastAPI endpoints
6. ALWAYS set TTL on every Redis key that is written
7. NEVER put blocking code inside an async FastAPI endpoint

## Key files to understand first
api/core/config.py        — all settings, read this before anything
api/core/redis.py         — rate limiter, dedup, pub/sub, priority queue
pipeline/pool/model_pool.py — LRU model pool, always use this
workers/celery_app.py     — pipeline chain, task routing
docker-compose.yml        — all 10 services wired together

## Proprietary Models

### BoundaryFusion
Our first proprietary model. A lightweight transformer that corrects
temporal inconsistencies at segment boundaries in the hierarchical
parallel processing pipeline.

Input:
  Last 10 frames from Worker N     (10 × H × W × 3)
  First 10 frames from Worker N+1  (10 × H × W × 3)
  Corresponding inpainted masks    (20 × H × W × 1)

Architecture:
  Patch embedding
  Local window attention (5-frame window)
  Cross-segment attention between two workers
  Decoder → 20 corrected boundary frames

Output:
  20 temporally consistent boundary frames

Training:
  Dataset: DAVIS + YouTube-VOS + synthetic broken boundaries
  Hardware: Kaggle T4 GPU
  Target: Month 2 Week 5-6

This model is proprietary to VisionErase. Never use it as
open source. It is our key differentiator vs competitors.
