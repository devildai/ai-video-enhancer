"""Real-time progress streaming via Server-Sent Events (SSE) and WebSocket."""

import asyncio
import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import StreamingResponse

from app.pipeline.task_store import task_store

logger = logging.getLogger("app.api.sse_ws")
router = APIRouter()


@router.get(
    "/progress/{task_id}",
    summary="Server-Sent Events real-time progress stream",
    response_class=StreamingResponse,
)
async def sse_progress_stream(task_id: str) -> StreamingResponse:
    """Delivers real-time execution events formatted as text/event-stream."""
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        task_store.subscribe_sse(task_id),
        media_type="text/event-stream",
        headers=headers,
    )


@router.websocket("/ws/progress/{task_id}")
async def websocket_progress_stream(websocket: WebSocket, task_id: str) -> None:
    """Full-duplex WebSocket connection delivering live progress and receiving control messages."""
    task = task_store.get_task(task_id)
    if not task:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Task not found")
        return

    await websocket.accept()

    # Send initial state snapshot immediately
    initial_event: Dict[str, Any] = {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "stage": task.stage,
        "fps": task.fps,
        "eta": task.eta,
        "download_url": task.download_url,
        "error": task.error,
    }
    await websocket.send_json(initial_event)

    queue = task_store.add_subscriber(task_id)
    if not queue:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    async def sender():
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                if event.get("status") in ("completed", "failed", "cancelled"):
                    break
        except Exception:
            pass

    async def receiver():
        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action", "").lower().strip()
                if action == "cancel":
                    task_store.cancel_task(task_id)
                    await websocket.send_json({"task_id": task_id, "status": "cancelled"})
                elif action == "ping":
                    await websocket.send_json({"action": "pong"})
        except (WebSocketDisconnect, Exception):
            pass

    sender_task = asyncio.create_task(sender())
    receiver_task = asyncio.create_task(receiver())

    done, pending = await asyncio.wait(
        [sender_task, receiver_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for p in pending:
        p.cancel()

    task_store.remove_subscriber(task_id, queue)
    try:
        await websocket.close()
    except Exception:
        pass
