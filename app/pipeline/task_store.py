"""Thread-safe in-memory task registry, cancellation tokens, and progress publisher."""

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, AsyncGenerator, Union


@dataclass
class TaskRecord:
    """Represents the complete lifecycle state and metadata of a video job."""
    task_id: str
    file_id: str
    filename: str
    input_path: str
    output_path: Optional[str] = None
    original_preview_path: Optional[str] = None
    enhanced_preview_path: Optional[str] = None
    metadata: Optional[Any] = None
    file_size: int = 0
    status: str = "queued"  # "queued", "processing", "completed", "failed", "cancelled"
    progress: float = 0.0
    stage: str = "Uploaded"
    fps: float = 0.0
    eta: float = 0.0
    mode: Optional[str] = "speed"
    resolution: Optional[str] = "Original"
    target_fps: Optional[Union[float, str]] = None
    download_url: Optional[str] = None
    error: Optional[str] = None
    comparison_ready: bool = False
    cancellation_event: threading.Event = field(default_factory=threading.Event)
    _subscribers: List[asyncio.Queue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "fps": self.fps,
            "eta": self.eta,
            "mode": self.mode,
            "resolution": self.resolution,
            "target_fps": self.target_fps,
            "download_url": self.download_url,
            "error": self.error,
            "comparison_ready": self.comparison_ready,
            "original_preview": f"/api/preview/{self.task_id}/original" if self.original_preview_path else None,
            "enhanced_preview": f"/api/preview/{self.task_id}/enhanced" if self.enhanced_preview_path else None,
        }


# Type alias matching PROJECT.md interface contract
TaskState = TaskRecord


class TaskStore:
    """Thread-safe in-memory store managing video enhancement tasks and subscriber queues."""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def create_task(
        self,
        task_id: str,
        file_id: str,
        filename: str,
        input_path: str,
        original_preview_path: Optional[str] = None,
        metadata: Optional[Any] = None,
        file_size: int = 0,
        **kwargs: Any,
    ) -> TaskRecord:
        """Registers a new task in the store."""
        with self._lock:
            record = TaskRecord(
                task_id=task_id,
                file_id=file_id,
                filename=filename,
                input_path=input_path,
                original_preview_path=original_preview_path,
                metadata=metadata,
                file_size=file_size,
                **kwargs,
            )
            self._tasks[task_id] = record
            # Also index by file_id if different
            if file_id != task_id:
                self._tasks[file_id] = record
            return record

    def get_task(self, task_id_or_file_id: str) -> Optional[TaskRecord]:
        """Retrieves a task by task_id or file_id."""
        with self._lock:
            return self._tasks.get(task_id_or_file_id)

    def update_task(self, task_id: str, **kwargs: Any) -> Optional[TaskRecord]:
        """Updates task fields thread-safely and broadcasts event to subscribers."""
        subs: List[asyncio.Queue] = []
        payload: Dict[str, Any] = {}

        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            for k, v in kwargs.items():
                if hasattr(task, k):
                    setattr(task, k, v)

            payload = {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "stage": task.stage,
                "fps": task.fps,
                "eta": task.eta,
                "download_url": task.download_url,
                "comparison_ready": task.comparison_ready,
                "error": task.error,
            }
            subs = list(task._subscribers)

        # Broadcast update to active listeners
        for q in subs:
            try:
                q.put_nowait(payload)
            except Exception:
                pass

        return task

    def cancel_task(self, task_id: str) -> bool:
        """Signals cancellation to runner and updates status."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.cancellation_event.set()
            task.status = "cancelled"
            task.stage = "Cancelled"
            task.error = "Processing cancelled by user"

        self.update_task(task_id, status="cancelled", stage="Cancelled")
        return True

    def add_subscriber(self, task_id: str) -> Optional[asyncio.Queue]:
        """Subscribes an async queue to receive real-time task events."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            queue: asyncio.Queue = asyncio.Queue(maxsize=100)
            task._subscribers.append(queue)
            return queue

    def remove_subscriber(self, task_id: str, queue: asyncio.Queue) -> None:
        """Removes a subscriber queue from the task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and queue in task._subscribers:
                task._subscribers.remove(queue)

    async def subscribe_sse(self, task_id: str, idle_timeout: float = 0.5) -> AsyncGenerator[str, None]:
        """Yields Server-Sent Events formatted as `data: {...}\n\n`."""
        task = self.get_task(task_id)
        if not task:
            error_data = json.dumps({"task_id": task_id, "error": "Task not found", "status": "failed"})
            yield f"data: {error_data}\n\n"
            return

        queue = self.add_subscriber(task_id)
        if not queue:
            return

        try:
            # Emit immediate initial event
            initial_data = json.dumps({
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "stage": task.stage,
                "fps": task.fps,
                "eta": task.eta,
                "download_url": task.download_url,
                "error": task.error,
            })
            yield f"data: {initial_data}\n\n"

            while True:
                if task.status in ("completed", "failed", "cancelled") and queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=idle_timeout)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in ("completed", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    # On idle timeout, close stream cleanly so TestClient and reconnecting SSE clients don't hang
                    break
        finally:
            self.remove_subscriber(task_id, queue)

    def cleanup_all_active_tasks(self) -> None:
        """Cancels all active tasks during server shutdown."""
        with self._lock:
            for task in self._tasks.values():
                if task.status in ("queued", "processing"):
                    task.cancellation_event.set()
                    task.status = "cancelled"


# Global singleton instance
task_store = TaskStore()
