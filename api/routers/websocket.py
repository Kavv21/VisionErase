from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.core.metrics import WEBSOCKET_CONNECTIONS_ACTIVE
from api.core.redis import acquire_redis, get_job_progress, get_job_status

log = structlog.get_logger(__name__)
router = APIRouter(tags=["websocket"])


@router.websocket("/{job_id}")
async def job_progress(websocket: WebSocket, job_id: str) -> None:
    """Stream real-time job progress to a WebSocket client.

    Sends a status snapshot immediately on connect, then subscribes to Redis
    pub/sub and forwards events until the job completes, fails, or the client
    disconnects.
    """
    await websocket.accept()
    WEBSOCKET_CONNECTIONS_ACTIVE.inc()
    bound_log = log.bind(job_id=job_id)
    bound_log.info("ws_connected")

    try:
        # ── Snapshot current status immediately on connect ─────────────────────
        try:
            async with acquire_redis() as redis:
                status = await get_job_status(redis, job_id)
                latest_progress = await get_job_progress(redis, job_id)
        except RuntimeError:
            await websocket.send_json({"type": "error", "message": "service unavailable"})
            return

        if status is None:
            await websocket.send_json({"type": "error", "message": "job not found"})
            bound_log.info("ws_job_not_found")
            return

        current_status = status.get("status")

        if current_status == "completed":
            await websocket.send_json({
                "type": "complete",
                "result": {"result_s3_key": status.get("result_s3_key")},
            })
            bound_log.info("ws_job_already_completed")
            return

        if current_status == "failed":
            await websocket.send_json({
                "type": "error",
                "message": status.get("error") or "job failed",
            })
            bound_log.info("ws_job_already_failed")
            return

        # Send in-progress snapshot so the client has something immediately.
        # Prefer the last stored worker progress event — it carries chunk-level
        # detail (stage, chunk, total, pct) that the status key lacks.
        if latest_progress is not None:
            await websocket.send_json(latest_progress)
        else:
            await websocket.send_json({
                "type": "progress",
                "stage": current_status,
                "pct": status.get("progress_pct", 0.0),
            })

        # ── Subscribe to Redis pub/sub and stream events ───────────────────────
        try:
            async with acquire_redis() as redis:
                pubsub = redis.pubsub()
                await pubsub.subscribe(f"progress:{job_id}")
                bound_log.info("ws_subscribed")

                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue

                    data = json.loads(message["data"])
                    await websocket.send_json(data)
                    bound_log.debug("ws_event_forwarded", event_type=data.get("type"))

                    if data.get("type") in ("complete", "error"):
                        bound_log.info("ws_terminal_event", event_type=data.get("type"))
                        break

        except RuntimeError:
            await websocket.send_json({"type": "error", "message": "service unavailable"})

    except WebSocketDisconnect:
        bound_log.info("ws_client_disconnected")
    except Exception as exc:
        bound_log.exception("ws_unexpected_error", error=str(exc))
        try:
            await websocket.send_json({"type": "error", "message": "internal server error"})
        except Exception:
            pass
    finally:
        WEBSOCKET_CONNECTIONS_ACTIVE.dec()
        bound_log.info("ws_closed")
