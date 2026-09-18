"""Unit tests for task store registry and streaming pipeline runner."""

import os
import sys
import threading
import time
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.task_store import TaskStore, TaskRecord, task_store
from app.pipeline.runner import run_enhancement_pipeline
from app.media.probe import probe_video
from tests.fixtures.generator import generate_synthetic_video


@pytest.fixture
def clean_store():
    """Provides an isolated TaskStore instance."""
    return TaskStore()


@pytest.fixture
def synthetic_320x240_clip(tmp_path):
    """Creates a short 0.3s synthetic test clip."""
    clip_path = tmp_path / "synthetic_test.mp4"
    generate_synthetic_video(
        output_path=clip_path,
        width=160,
        height=120,
        fps=30.0,
        duration=0.3,
        has_audio=True,
    )
    return clip_path


@pytest.fixture
def synthetic_silent_clip(tmp_path):
    """Creates a short silent test clip."""
    clip_path = tmp_path / "synthetic_silent.mp4"
    generate_synthetic_video(
        output_path=clip_path,
        width=160,
        height=120,
        fps=30.0,
        duration=0.3,
        has_audio=False,
    )
    return clip_path


class TestTaskStore:
    """Verify in-memory TaskStore operations and thread safety."""

    def test_create_and_get_task(self, clean_store):
        record = clean_store.create_task(
            task_id="task-101",
            file_id="file-101",
            filename="demo.mp4",
            input_path="/path/demo.mp4",
        )
        assert record.task_id == "task-101"
        assert record.status == "queued"
        assert record.progress == 0.0

        # Retrieve by task_id
        fetched = clean_store.get_task("task-101")
        assert fetched is not None
        assert fetched.filename == "demo.mp4"

        # Retrieve by file_id
        fetched_by_file = clean_store.get_task("file-101")
        assert fetched_by_file is not None
        assert fetched_by_file.task_id == "task-101"

    def test_update_task_fields(self, clean_store):
        clean_store.create_task(
            task_id="task-102",
            file_id="task-102",
            filename="demo2.mp4",
            input_path="/path/demo2.mp4",
        )
        updated = clean_store.update_task(
            "task-102",
            status="processing",
            progress=45.5,
            stage="Enhancing frames",
            fps=12.3,
        )
        assert updated is not None
        assert updated.status == "processing"
        assert updated.progress == 45.5
        assert updated.stage == "Enhancing frames"
        assert updated.fps == 12.3

    def test_cancel_task(self, clean_store):
        clean_store.create_task(
            task_id="task-103",
            file_id="task-103",
            filename="demo3.mp4",
            input_path="/path/demo3.mp4",
        )
        result = clean_store.cancel_task("task-103")
        assert result is True

        task = clean_store.get_task("task-103")
        assert task.cancellation_event.is_set()
        assert task.status == "cancelled"

    def test_cleanup_all_active_tasks(self, clean_store):
        t1 = clean_store.create_task("t1", "t1", "f1.mp4", "/p/f1.mp4", status="processing")
        t2 = clean_store.create_task("t2", "t2", "f2.mp4", "/p/f2.mp4", status="completed")
        t3 = clean_store.create_task("t3", "t3", "f3.mp4", "/p/f3.mp4", status="queued")

        clean_store.cleanup_all_active_tasks()
        assert t1.status == "cancelled"
        assert t2.status == "completed"
        assert t3.status == "cancelled"


class TestPipelineRunnerExecution:
    """Verify run_enhancement_pipeline streaming execution."""

    def test_pipeline_enhances_and_interpolates_short_clip(self, synthetic_320x240_clip):
        task_id = f"pipeline-test-{int(time.time())}"
        task_store.create_task(
            task_id=task_id,
            file_id=task_id,
            filename=synthetic_320x240_clip.name,
            input_path=str(synthetic_320x240_clip),
        )

        # Execute runner synchronously
        run_enhancement_pipeline(
            task_id=task_id,
            resolution="Original",
            fps="60fps",
            mode="speed",
        )

        task = task_store.get_task(task_id)
        assert task is not None
        assert task.status == "completed"
        assert task.progress == 100.0
        assert task.output_path is not None
        assert os.path.exists(task.output_path)
        assert os.path.getsize(task.output_path) > 0

        # Verify output properties via ffprobe
        out_meta = probe_video(task.output_path)
        assert out_meta.width == 160
        assert out_meta.height == 120
        assert round(out_meta.fps) == 60
        assert out_meta.has_audio is True

    def test_pipeline_silent_video(self, synthetic_silent_clip):
        task_id = f"silent-test-{int(time.time())}"
        task_store.create_task(
            task_id=task_id,
            file_id=task_id,
            filename=synthetic_silent_clip.name,
            input_path=str(synthetic_silent_clip),
        )

        run_enhancement_pipeline(
            task_id=task_id,
            resolution="Original",
            fps="Original",
            mode="speed",
        )

        task = task_store.get_task(task_id)
        assert task is not None
        assert task.status == "completed"
        assert task.error is None
        assert os.path.exists(task.output_path)

        out_meta = probe_video(task.output_path)
        assert out_meta.has_audio is False

    def test_pipeline_cancellation_during_job(self, synthetic_320x240_clip):
        task_id = f"cancel-test-{int(time.time())}"
        task = task_store.create_task(
            task_id=task_id,
            file_id=task_id,
            filename=synthetic_320x240_clip.name,
            input_path=str(synthetic_320x240_clip),
        )

        # Pre-set cancellation event before starting
        task.cancellation_event.set()

        run_enhancement_pipeline(
            task_id=task_id,
            resolution="Original",
            fps="60fps",
            mode="speed",
        )

        updated_task = task_store.get_task(task_id)
        assert updated_task.status == "cancelled"
