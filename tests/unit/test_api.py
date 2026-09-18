"""Unit tests for FastAPI backend REST and streaming endpoints."""

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.pipeline.task_store import task_store
from tests.fixtures.generator import generate_synthetic_video


@pytest.fixture(scope="module")
def client():
    """FastAPI test client running against app instance."""
    app.state.no_browser = True
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def temp_sample_video(tmp_path):
    """Generates a small valid MP4 video fixture."""
    vid_path = tmp_path / "test_sample.mp4"
    generate_synthetic_video(
        output_path=vid_path,
        width=320,
        height=240,
        fps=30.0,
        duration=0.5,
        has_audio=True,
    )
    return vid_path


class TestRootAndStaticEndpoints:
    """Verify root / and static file handling."""

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


class TestUploadEndpoint:
    """Verify POST /api/upload behavior and validations."""

    def test_upload_valid_mp4(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert "file_id" in data
        assert data["filename"] == "sample.mp4"
        assert "metadata" in data
        meta = data["metadata"]
        assert meta["width"] == 320
        assert meta["height"] == 240
        assert meta["has_audio"] is True
        assert "preview_url" in data

    def test_upload_zero_byte_rejected(self, client):
        resp = client.post("/api/upload", files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")})
        assert resp.status_code == 400
        assert "empty" in resp.json().get("detail", "").lower()

    def test_upload_unsupported_extension_rejected(self, client):
        resp = client.post("/api/upload", files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")})
        assert resp.status_code == 400
        assert "unsupported" in resp.json().get("detail", "").lower()

    def test_upload_corrupt_video_rejected(self, client):
        fake_data = b"\x00\x00\x00\x20ftypmp42corruptcontenthere"
        resp = client.post("/api/upload", files={"file": ("corrupt.mp4", io.BytesIO(fake_data), "video/mp4")})
        assert resp.status_code == 400
        assert "valid video" in resp.json().get("detail", "").lower() or "missing" in resp.json().get("detail", "").lower()


class TestProcessEndpoint:
    """Verify POST /api/process parameter validation and launching."""

    def test_process_valid_request(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        assert u_resp.status_code == 200
        task_id = u_resp.json()["task_id"]

        p_resp = client.post("/api/process", json={
            "task_id": task_id,
            "resolution": "Original",
            "fps": "Original",
            "mode": "speed",
        })
        assert p_resp.status_code == 200
        data = p_resp.json()
        assert data["task_id"] == task_id
        assert data["status"] in ("processing", "queued")

    def test_process_nonexistent_task_returns_404(self, client):
        resp = client.post("/api/process", json={
            "task_id": "nonexistent-uuid-12345",
            "resolution": "1080p",
            "fps": "60fps",
            "mode": "speed",
        })
        assert resp.status_code == 404

    def test_process_invalid_mode_rejected(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        resp = client.post("/api/process", json={
            "task_id": task_id,
            "resolution": "1080p",
            "fps": "60fps",
            "mode": "unsupported_super_mode",
        })
        assert resp.status_code in (400, 422)

    def test_process_invalid_fps_rejected(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        resp = client.post("/api/process", json={
            "task_id": task_id,
            "fps": "-5fps",
            "mode": "speed",
        })
        assert resp.status_code in (400, 422)


class TestStatusAndMetricsEndpoint:
    """Verify GET /api/status/{task_id} polling."""

    def test_status_for_active_task(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        resp = client.get(f"/api/status/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "progress" in data
        assert 0.0 <= data["progress"] <= 100.0
        assert "stage" in data

    def test_status_nonexistent_returns_404(self, client):
        resp = client.get("/api/status/nonexistent-task-999")
        assert resp.status_code == 404


class TestCancellationEndpoint:
    """Verify POST /api/cancel/{task_id}."""

    def test_cancel_active_task(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        cancel_resp = client.post(f"/api/cancel/{task_id}")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

        status_resp = client.get(f"/api/status/{task_id}")
        assert status_resp.json()["status"] == "cancelled"

    def test_cancel_nonexistent_returns_404(self, client):
        resp = client.post("/api/cancel/nonexistent-task-888")
        assert resp.status_code == 404


class TestCompareFrameAndPreviewEndpoints:
    """Verify comparison slider preview and frame serving."""

    def test_compare_frame_success(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        resp = client.get(f"/api/compare-frame/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert "original_url" in data
        assert "enhanced_url" in data

    def test_compare_frame_image_header(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        # First trigger generation
        client.get(f"/api/compare-frame/{task_id}")

        resp = client.get(f"/api/compare-frame/{task_id}", headers={"Accept": "image/jpeg"})
        assert resp.status_code == 200
        if "image" in resp.headers.get("content-type", ""):
            assert resp.content.startswith(b"\xff\xd8\xff")  # JPEG header magic bytes

    def test_compare_frame_nonexistent_returns_404(self, client):
        resp = client.get("/api/compare-frame/nonexistent-task-777")
        assert resp.status_code == 404

    def test_serve_original_preview(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        resp = client.get(f"/api/preview/{task_id}/original")
        assert resp.status_code == 200
        assert "image/jpeg" in resp.headers.get("content-type", "")


class TestDownloadEndpoint:
    """Verify download endpoint and Range streaming."""

    def test_download_nonexistent_returns_404(self, client):
        resp = client.get("/api/download/nonexistent-123")
        assert resp.status_code == 404

    def test_download_cancelled_task_returns_400(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]
        client.post(f"/api/cancel/{task_id}")

        resp = client.get(f"/api/download/{task_id}")
        assert resp.status_code in (400, 404, 409)

    def test_download_head_request(self, client, temp_sample_video, tmp_path):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        # Fake completion with temp file
        fake_out = tmp_path / "fake_out.mp4"
        fake_out.write_bytes(b"\x00" * 1024)
        task_store.update_task(task_id, status="completed", output_path=str(fake_out))

        head_resp = client.head(f"/api/download/{task_id}")
        assert head_resp.status_code == 200
        assert head_resp.headers.get("content-type") == "video/mp4"
        assert head_resp.headers.get("content-length") == "1024"

    def test_download_range_partial_content(self, client, temp_sample_video, tmp_path):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        fake_out = tmp_path / "fake_stream.mp4"
        fake_out.write_bytes(b"A" * 500 + b"B" * 500)
        task_store.update_task(task_id, status="completed", output_path=str(fake_out))

        range_resp = client.get(f"/api/download/{task_id}", headers={"Range": "bytes=100-199"})
        assert range_resp.status_code == 206
        assert len(range_resp.content) == 100
        assert "bytes 100-199/1000" in range_resp.headers.get("content-range", "")


class TestSSEAndWebSocketProgress:
    """Verify SSE and WebSocket progress streams."""

    def test_sse_progress_connects(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        resp = client.get(f"/api/progress/{task_id}")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert "data:" in resp.text

    def test_sse_nonexistent_returns_404(self, client):
        resp = client.get("/api/progress/nonexistent-9999")
        assert resp.status_code == 404

    def test_websocket_progress_interaction(self, client, temp_sample_video):
        with open(temp_sample_video, "rb") as f:
            u_resp = client.post("/api/upload", files={"file": ("sample.mp4", f, "video/mp4")})
        task_id = u_resp.json()["task_id"]

        with client.websocket_connect(f"/ws/progress/{task_id}") as ws:
            # First message should be initial state
            initial_msg = ws.receive_json()
            assert "task_id" in initial_msg
            assert "progress" in initial_msg

            # Test ping/pong
            ws.send_json({"action": "ping"})
            pong = ws.receive_json()
            assert pong.get("action") == "pong"
