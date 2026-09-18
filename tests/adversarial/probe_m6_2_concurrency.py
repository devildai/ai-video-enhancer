"""Probe 2: Subprocess lifecycle, WebSocket cancel/disconnect, and Concurrency stress testing."""

import asyncio
import os
import sys
import time
import uuid
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.config import UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR, TEMP_DIR
from app.media.stream import VideoFrameDecoder, VideoFrameEncoder
from app.pipeline.task_store import task_store
from app.pipeline.runner import run_enhancement_pipeline
from tests.fixtures.generator import generate_synthetic_video

app.state.no_browser = True
client = TestClient(app)

def get_ffmpeg_pids():
    """Get set of running ffmpeg process PIDs on Windows."""
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
    except Exception as e:
        print(f"Error checking tasklist: {e}")
        return set()

def test_stream_subprocesses_mid_abort():
    print("\n--- TEST: Direct Streaming Subprocesses Mid-Abort ---")
    temp_dir = Path("tests/fixtures/cache")
    test_vid = temp_dir / "stream_abort_test.mp4"
    generate_synthetic_video(test_vid, width=320, height=240, fps=30, duration=3.0, has_audio=True)

    # 1. Test VideoFrameDecoder abort
    pids_before = get_ffmpeg_pids()
    decoder = VideoFrameDecoder(str(test_vid))
    decoder._start_process()
    dec_pid = decoder._process.pid
    print(f"Decoder spawned with PID {dec_pid}")
    assert dec_pid in get_ffmpeg_pids(), "Decoder process must be running in OS tasklist"

    # Read 10 frames
    frames = []
    for i, frame in enumerate(decoder):
        frames.append(frame)
        if i >= 10:
            break
    print(f"Read {len(frames)} frames, closing decoder...")
    decoder.close()

    time.sleep(0.5)
    poll_res = decoder._process # Should be None after close()
    assert decoder._closed is True
    pids_after_dec = get_ffmpeg_pids()
    print(f"Decoder closed. PID {dec_pid} in pids: {dec_pid in pids_after_dec}")
    assert dec_pid not in pids_after_dec, f"Orphaned decoder FFmpeg process {dec_pid} detected!"

    # 2. Test VideoFrameEncoder abort
    out_path = temp_dir / "stream_abort_out.mp4"
    encoder = VideoFrameEncoder(str(out_path), width=320, height=240, fps=30.0)
    encoder.start()
    enc_pid = encoder._process.pid
    print(f"Encoder spawned with PID {enc_pid}")
    assert enc_pid in get_ffmpeg_pids(), "Encoder process must be running in OS tasklist"

    # Write 5 frames
    for f in frames[:5]:
        encoder.write_frame(f)
    print("Wrote 5 frames, closing encoder with abort (raise_on_error=False)...")
    encoder.close(raise_on_error=False)

    time.sleep(0.5)
    pids_after_enc = get_ffmpeg_pids()
    print(f"Encoder closed. PID {enc_pid} in pids: {enc_pid in pids_after_enc}")
    assert enc_pid not in pids_after_enc, f"Orphaned encoder FFmpeg process {enc_pid} detected!"
    out_path.unlink(missing_ok=True)
    print("Direct streaming subprocess abort test PASSED!")

def test_websocket_cancellation():
    print("\n--- TEST: WebSocket Live Cancellation ---")
    temp_dir = Path("tests/fixtures/cache")
    test_vid = temp_dir / "ws_cancel_test.mp4"
    generate_synthetic_video(test_vid, width=160, height=120, fps=15, duration=2.0, has_audio=False)

    with open(test_vid, "rb") as f:
        up_resp = client.post("/api/upload", files={"file": ("ws_test.mp4", f, "video/mp4")})
    assert up_resp.status_code == 200
    task_id = up_resp.json()["task_id"]

    # Launch process
    proc_resp = client.post("/api/process", json={"task_id": task_id, "resolution": "Original", "fps": "Original", "mode": "speed"})
    assert proc_resp.status_code == 200

    # Connect to WebSocket
    with client.websocket_connect(f"/ws/progress/{task_id}") as ws:
        init_data = ws.receive_json()
        print(f"WebSocket connected, initial message: {init_data.get('status')}")
        # Send cancel action over WebSocket
        ws.send_json({"action": "cancel"})
        resp = ws.receive_json()
        print(f"WebSocket response to cancel: {resp}")
        assert resp.get("status") == "cancelled"

    time.sleep(1.0)
    task = task_store.get_task(task_id)
    print(f"Final task status after WS cancel: {task.status}")
    assert task.status == "cancelled"
    print("WebSocket cancellation test PASSED!")

def test_concurrent_uploads():
    print("\n--- TEST: Concurrent Uploads (10 parallel clients) ---")
    temp_dir = Path("tests/fixtures/cache")
    test_vid = temp_dir / "conc_upload.mp4"
    generate_synthetic_video(test_vid, width=160, height=120, fps=10, duration=0.5, has_audio=False)
    with open(test_vid, "rb") as f:
        vid_bytes = f.read()

    def do_upload(i):
        # Create a new TestClient or use shared client
        with TestClient(app) as local_client:
            resp = local_client.post(
                "/api/upload",
                files={"file": (f"parallel_{i}.mp4", io_bytes(vid_bytes), "video/mp4")}
            )
            return resp.status_code, resp.json() if resp.status_code == 200 else resp.text

    import io
    def io_bytes(b):
        return io.BytesIO(b)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(do_upload, i) for i in range(10)]
        results = [f.result() for f in futures]

    statuses = [r[0] for r in results]
    task_ids = [r[1]["task_id"] for r in results if r[0] == 200]
    print(f"Concurrent upload statuses: {statuses}")
    print(f"Unique task_ids: {len(set(task_ids))} / {len(task_ids)}")
    assert all(s == 200 for s in statuses), f"Some uploads failed: {statuses}"
    assert len(set(task_ids)) == 10, "Task IDs collided!"
    print("Concurrent uploads test PASSED!")

def test_concurrent_processing_tasks():
    print("\n--- TEST: Concurrent Multiple Task Execution (3 simultaneous jobs) ---")
    temp_dir = Path("tests/fixtures/cache")
    test_vid = temp_dir / "conc_proc.mp4"
    generate_synthetic_video(test_vid, width=160, height=120, fps=10, duration=0.4, has_audio=True)
    with open(test_vid, "rb") as f:
        vid_bytes = f.read()

    import io
    task_ids = []
    for i in range(3):
        resp = client.post(
            "/api/upload",
            files={"file": (f"conc_job_{i}.mp4", io.BytesIO(vid_bytes), "video/mp4")}
        )
        assert resp.status_code == 200
        task_ids.append(resp.json()["task_id"])

    print(f"Uploaded 3 videos for concurrent execution: {task_ids}")

    # Launch all 3 concurrently
    for tid in task_ids:
        resp = client.post("/api/process", json={"task_id": tid, "resolution": "Original", "fps": "Original", "mode": "speed"})
        assert resp.status_code == 200

    # Monitor until all complete or fail
    deadline = time.time() + 60.0
    while time.time() < deadline:
        statuses = [client.get(f"/api/status/{tid}").json()["status"] for tid in task_ids]
        if all(s in ("completed", "failed") for s in statuses):
            break
        time.sleep(1.0)

    final_statuses = [client.get(f"/api/status/{tid}").json() for tid in task_ids]
    print(f"Final task results: {[(f['task_id'][:8], f['status'], f.get('error')) for f in final_statuses]}")
    assert all(f["status"] == "completed" for f in final_statuses), "Not all concurrent tasks completed successfully!"

    # Verify all outputs exist, are non-empty, and are distinct
    out_files = [task_store.get_task(tid).output_path for tid in task_ids]
    print(f"Output files: {out_files}")
    assert len(set(out_files)) == 3, "Output files collided!"
    for out in out_files:
        assert os.path.exists(out) and os.path.getsize(out) > 0, f"Output file missing or empty: {out}"

    print("Concurrent processing tasks test PASSED!")

if __name__ == "__main__":
    test_stream_subprocesses_mid_abort()
    test_websocket_cancellation()
    test_concurrent_uploads()
    test_concurrent_processing_tasks()
    print("\n--- ALL PROBE 2 TESTS PASSED ---")
