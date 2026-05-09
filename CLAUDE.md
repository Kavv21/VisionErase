# VisionErase — Project Bible

## What this is
A distributed AI video object removal platform. Users upload a video,
paint over an unwanted object on frame 1, and the platform removes it
across every frame with temporal consistency using SAM 2 + ProPainter.

## Architecture (one paragraph)
FastAPI gateway → validates request → checks Redis dedup cache →
enqueues to Redis priority queue → Celery workers (segmentation runs
SAM 2, tracking runs XMem++, inpainting runs ProPainter, stitching
merges chunks, quality scores SSIM/PSNR) → results in S3 + Redis →
frontend polls via WebSocket for live progress.

## Folder map
api/              FastAPI gateway, routers, middleware, models, services
api/core/         config, redis, database, metrics
api/middleware/   rate_limiter, logging
api/routers/      jobs, websocket, webhooks, health
api/models/       pydantic schemas
api/services/     storage (S3), auth
workers/          Celery tasks
workers/celery_app.py   pipeline chain definition
workers/segmentation/   SAM 2 tasks
workers/inpainting/     ProPainter tasks
workers/stitching/      chunk merge tasks
workers/quality/        SSIM/PSNR tasks
pipeline/         CV model layer
pipeline/pool/    model memory pool (LRU + VRAM management)
pipeline/tracker/ XMem++ mask propagation
pipeline/inpainter/ ProPainter inpainting
pipeline/chunker/ video chunking + seam blending
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
