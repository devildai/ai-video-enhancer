"""Adversarial test suite: Subprocesses, Concurrency, and Mid-Processing Abort."""

import io
import os
import sys
import time
import uuid
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.config import UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR, TEMP_DIR
from app.media.stream import VideoFrameDecoder, VideoFrameEncoder
from app.pipeline.task_store import task_store
from app.pipeline.runner import run_enhancement_pipeline
from tests.fixtures.generator import generate_synthetic_video


def get_ffmpeg_pids() -> set[int]:
    """Inspects running FFmpeg process IDs via Windows tasklist."""
    try:
        res = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq ffmpeg.exe"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        pids = []
        for line in res.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2 and "ffmpeg" in parts[0].lower():
                pid_str = parts[1].strip(' "')
                if pid_str.isdigit():
                    pids.append(int(pid_str))
        return set(pids)
    except Exception:
        return set()


@pytest.fixture(scope="module")
def shared_client():
    """Module-scoped TestClient to avoid repeated lifespan shutdown wipes."""
    app.state.no_browser = True
    with TestClient(app) as test_client:
        yield test_client


class TestSubprocessLifecycleAndAbort:
    """Stress tests for FFmpeg subprocess cleanup on mid-stream abort."""

    def test_decoder_subprocess_terminates_on_close(self, tmp_path):
        """Verify VideoFrameDecoder closes cleanly and terminates FFmpeg without orphans."""
        test_vid = tmp_path / "dec_abort.mp4"
        generate_synthetic_video(test_vid, width=320, height=240, fps=30, duration=2.0, has_audio=False)

        decoder = VideoFrameDecoder(str(test_vid))
        decoder._start_process()
        pid = decoder._process.pid
        assert pid in get_ffmpeg_pids(), "FFmpeg decoder process must be active in OS"

        # Read a few frames then abort
        read_count = 0
        for _ in decoder:
            read_count += 1
            if read_count >= 5:
                break

        decoder.close()
        time.sleep(0.5)

        active_pids = get_ffmpeg_pids()
        assert pid not in active_pids, f"Orphaned decoder FFmpeg process {pid} detected!"
        assert decoder._closed is True
        assert decoder._process is None

    def test_encoder_subprocess_terminates_on_close(self, tmp_path):
        """Verify VideoFrameEncoder closes cleanly on abort without orphaned processes."""
        out_vid = tmp_path / "enc_abort.mp4"
        encoder = VideoFrameEncoder(str(out_vid), width=320, height=240, fps=30.0)
        encoder.start()
        pid = encoder._process.pid
        assert pid in get_ffmpeg_pids(), "FFmpeg encoder process must be active in OS"

        import numpy as np
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(5):
            encoder.write_frame(frame)

        # Abort encoding early
        encoder.close(raise_on_error=False)
        time.sleep(0.5)

        active_pids = get_ffmpeg_pids()
        assert pid not in active_pids, f"Orphaned encoder FFmpeg process {pid} detected!"
        assert encoder._closed is True
        assert encoder._process is None

    def test_pipeline_abort_mid_processing_cleans_subprocesses(self, shared_client, tmp_path):
        """Verify cancelling an active video enhancement job stops FFmpeg processes."""
        test_vid = tmp_path / "long_abort.mp4"
        generate_synthetic_video(test_vid, width=320, height=240, fps=30, duration=4.0, has_audio=True)

        with open(test_vid, "rb") as f:
            up_resp = shared_client.post("/api/upload", files={"file": ("abort_test.mp4", f, "video/mp4")})
        assert up_resp.status_code == 200
        task_id = up_resp.json()["task_id"]

        initial_pids = get_ffmpeg_pids()

        proc_resp = shared_client.post(
            "/api/process",
            json={"task_id": task_id, "resolution": "Original", "fps": "60fps", "mode": "speed"}
        )
        assert proc_resp.status_code == 200

        # Wait briefly for worker thread to begin processing
        time.sleep(0.6)

        # Send cancellation
        cancel_resp = shared_client.post(f"/api/cancel/{task_id}")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

        # Wait for worker thread to exit
        time.sleep(2.0)

        # Check for leaked FFmpeg processes
        current_pids = get_ffmpeg_pids()
        leaked_pids = current_pids - initial_pids
        assert not leaked_pids, f"Leaked FFmpeg processes detected after cancellation: {leaked_pids}"

        # Status must remain cancelled
        st_resp = shared_client.get(f"/api/status/{task_id}")
        assert st_resp.json()["status"] == "cancelled"

    def test_websocket_cancellation_aborts_active_task(self, shared_client, tmp_path):
        """Verify WebSocket cancellation action cleanly cancels running task."""
        test_vid = tmp_path / "ws_cancel.mp4"
        generate_synthetic_video(test_vid, width=160, height=120, fps=15, duration=2.0, has_audio=False)

        with open(test_vid, "rb") as f:
            up_resp = shared_client.post("/api/upload", files={"file": ("ws_cancel.mp4", f, "video/mp4")})
        assert up_resp.status_code == 200
        task_id = up_resp.json()["task_id"]

        shared_client.post("/api/process", json={"task_id": task_id, "resolution": "Original", "fps": "Original", "mode": "speed"})

        with shared_client.websocket_connect(f"/ws/progress/{task_id}") as ws:
            msg = ws.receive_json()
            assert "status" in msg
            ws.send_json({"action": "cancel"})
            resp = ws.receive_json()
            assert resp.get("status") == "cancelled"

        time.sleep(1.0)
        task = task_store.get_task(task_id)
        assert task.status == "cancelled"


class TestConcurrencyStress:
    """Stress tests for concurrent uploads and simultaneous processing tasks."""

    def test_concurrent_uploads_10_parallel(self, shared_client, tmp_path):
        """Verify 10 parallel uploads succeed with distinct task IDs and no race conditions."""
        test_vid = tmp_path / "conc_seed.mp4"
        generate_synthetic_video(test_vid, width=160, height=120, fps=10, duration=0.3, has_audio=False)
        with open(test_vid, "rb") as f:
            vid_bytes = f.read()

        def do_upload(i: int):
            resp = shared_client.post(
                "/api/upload",
                files={"file": (f"parallel_upload_{i}.mp4", io.BytesIO(vid_bytes), "video/mp4")}
            )
            return resp.status_code, resp.json()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(do_upload, i) for i in range(10)]
            results = [f.result() for f in futures]

        statuses = [r[0] for r in results]
        task_ids = [r[1]["task_id"] for r in results if r[0] == 200]

        assert all(s == 200 for s in statuses), f"Some concurrent uploads failed: {statuses}"
        assert len(set(task_ids)) == 10, "Task IDs collided under concurrent upload!"

    def test_concurrent_pipeline_execution(self, tmp_path):
        """Verify multiple enhancement tasks execute concurrently without thread contention."""
        vid1 = tmp_path / "conc_run1.mp4"
        vid2 = tmp_path / "conc_run2.mp4"
        generate_synthetic_video(vid1, width=160, height=120, fps=10, duration=0.3, has_audio=True)
        generate_synthetic_video(vid2, width=160, height=120, fps=10, duration=0.3, has_audio=True)

        id1 = f"conc-job-{uuid.uuid4().hex[:6]}-1"
        id2 = f"conc-job-{uuid.uuid4().hex[:6]}-2"

        t1 = task_store.create_task(id1, id1, "v1.mp4", str(vid1))
        t2 = task_store.create_task(id2, id2, "v2.mp4", str(vid2))

        th1 = threading.Thread(target=run_enhancement_pipeline, args=(id1, "Original", "Original", "speed"))
        th2 = threading.Thread(target=run_enhancement_pipeline, args=(id2, "Original", "Original", "speed"))

        th1.start()
        th2.start()

        th1.join(timeout=60.0)
        th2.join(timeout=60.0)

        assert t1.status == "completed", f"Task 1 failed: {t1.error}"
        assert t2.status == "completed", f"Task 2 failed: {t2.error}"
        assert t1.output_path != t2.output_path, "Output paths collided!"
        assert os.path.exists(t1.output_path) and os.path.getsize(t1.output_path) > 0
        assert os.path.exists(t2.output_path) and os.path.getsize(t2.output_path) > 0
