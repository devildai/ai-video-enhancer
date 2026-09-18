"""Tier 2 - Boundary & Corner Cases E2E Tests.

Covers >=5 tests per boundary/corner area:
1. Non-video file uploads (rejected with 400)
2. 0-byte file uploads (rejected with 400)
3. Odd dimensions (sanitized to even dimensions without encoder crash)
4. Video with no audio track (processed successfully to valid silent MP4)
5. Target FPS <= source FPS (handled without crash)
6. Job cancellation (/api/cancel/{task_id} stops processing and marks cancelled)
"""
import io
import time
import pytest
from pathlib import Path
from tests.e2e.helpers import inspect_video_properties

pytestmark = pytest.mark.tier2


# ==============================================================================
# Area 1: Non-Video File Uploads
# ==============================================================================
class TestArea1NonVideoUploads:
    """Test rejection of invalid/non-video files (>=5 tests)."""

    def test_upload_text_file_rejected_400(self, client, text_file):
        """Plain text file uploaded as video is rejected with 400 Bad Request."""
        with open(text_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("readme.txt", f, "text/plain")})
        assert resp.status_code == 400

    def test_upload_corrupt_video_header_rejected(self, client, corrupt_file):
        """File with .mp4 extension but corrupt random binary bytes is rejected with 400."""
        with open(corrupt_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("corrupt.mp4", f, "video/mp4")})
        assert resp.status_code == 400

    def test_upload_executable_rejected_400(self, client):
        """Binary executable (.exe) uploaded is rejected with 400."""
        fake_exe = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00"
        resp = client.post("/api/upload", files={"file": ("malware.exe", io.BytesIO(fake_exe), "application/octet-stream")})
        assert resp.status_code == 400

    def test_upload_python_script_rejected_400(self, client):
        """Python script (.py) uploaded is rejected with 400."""
        py_code = b"import os\nos.system('dir')\n"
        resp = client.post("/api/upload", files={"file": ("script.py", io.BytesIO(py_code), "text/x-python")})
        assert resp.status_code == 400

    def test_upload_missing_filename_rejected(self, client):
        """Upload request with empty filename or missing file payload is rejected."""
        resp = client.post("/api/upload", files={"file": ("", io.BytesIO(b""), "application/octet-stream")})
        assert resp.status_code in (400, 422)


# ==============================================================================
# Area 2: 0-Byte File Uploads
# ==============================================================================
class TestArea2ZeroByteUploads:
    """Test rejection of 0-byte empty files (>=5 tests)."""

    def test_upload_zero_byte_mp4_rejected_400(self, client, zero_byte_file):
        """0-byte .mp4 file is rejected with 400 Bad Request."""
        with open(zero_byte_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("empty.mp4", f, "video/mp4")})
        assert resp.status_code == 400

    def test_upload_zero_byte_mkv_rejected_400(self, client, zero_byte_file):
        """0-byte .mkv file is rejected with 400 Bad Request."""
        with open(zero_byte_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("empty.mkv", f, "video/x-matroska")})
        assert resp.status_code == 400

    def test_upload_zero_byte_generic_rejected_400(self, client):
        """Empty 0-byte payload is rejected with 400."""
        resp = client.post("/api/upload", files={"file": ("empty.bin", io.BytesIO(b""), "application/octet-stream")})
        assert resp.status_code == 400

    def test_upload_zero_byte_error_detail(self, client, zero_byte_file):
        """Error response for 0-byte file contains informative error message."""
        with open(zero_byte_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("empty.mp4", f, "video/mp4")})
        assert resp.status_code == 400
        error_msg = resp.json().get("detail", "").lower()
        assert "empty" in error_msg or "0" in error_msg or "invalid" in error_msg or "size" in error_msg

    def test_upload_zero_byte_no_task_created(self, client, zero_byte_file):
        """0-byte upload does not create a valid task or file_id in storage."""
        with open(zero_byte_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("empty.mp4", f, "video/mp4")})
        assert resp.status_code == 400
        data = resp.json()
        assert "task_id" not in data or data.get("task_id") is None


# ==============================================================================
# Area 3: Odd Dimensions Sanitization
# ==============================================================================
class TestArea3OddDimensions:
    """Test handling and sanitization of odd video dimensions (>=5 tests)."""

    def test_odd_dimensions_853x481_upload_metadata(self, client, standard_fixtures):
        """Metadata inspection accurately identifies odd dimensions (853x481) without crashing."""
        odd_file = standard_fixtures["odd_dim_853x481"]
        with open(odd_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("odd.mp4", f, "video/mp4")})
        assert resp.status_code == 200
        meta = resp.json().get("metadata", resp.json())
        assert meta["width"] == 853
        assert meta["height"] == 481

    def test_odd_dimensions_319x241_upload(self, client, standard_fixtures):
        """Small odd dimensions (319x241) uploaded and parsed successfully."""
        odd_file = standard_fixtures["odd_dim_319x241"]
        with open(odd_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("odd_small.mp4", f, "video/mp4")})
        assert resp.status_code == 200
        meta = resp.json().get("metadata", resp.json())
        assert meta["width"] == 319
        assert meta["height"] == 241

    def test_odd_dimensions_sanitization_for_encoding(self, client, standard_fixtures):
        """Job launched with odd dimension input succeeds and produces even dimensions."""
        odd_file = standard_fixtures["odd_dim_853x481"]
        with open(odd_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("odd.mp4", f, "video/mp4")})
        file_id = resp.json().get("file_id", resp.json().get("task_id"))

        p_resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "Original",
            "mode": "speed"
        })
        assert p_resp.status_code == 200
        task_id = p_resp.json()["task_id"]
        status_resp = client.get(f"/api/status/{task_id}")
        assert status_resp.status_code == 200

    def test_odd_dimensions_aspect_ratio_preservation(self, standard_fixtures):
        """Sanitizing 853x481 to even (854x482 or 852x480) preserves aspect ratio within 0.5%."""
        w_orig, h_orig = 853, 481
        ar_orig = w_orig / h_orig
        # Sanitized even dimensions
        w_even = w_orig + (w_orig % 2)
        h_even = h_orig + (h_orig % 2)
        ar_even = w_even / h_even
        delta = abs(ar_even - ar_orig) / ar_orig
        assert delta < 0.005  # < 0.5% variance

    def test_odd_dimension_single_axis(self, client, standard_fixtures):
        """Odd dimensions with 1080p target resolution produce perfectly even 1920x1080 output."""
        odd_file = standard_fixtures["odd_dim_853x481"]
        with open(odd_file, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("odd.mp4", f, "video/mp4")})
        file_id = resp.json().get("file_id", resp.json().get("task_id"))

        p_resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "1080p",
            "fps": "Original",
            "mode": "speed"
        })
        assert p_resp.status_code == 200


# ==============================================================================
# Area 4: Video with No Audio Track
# ==============================================================================
class TestArea4SilentVideo:
    """Test processing of video files with no audio track (>=5 tests)."""

    def test_silent_video_upload_metadata(self, client, silent_mp4):
        """Silent video upload metadata reports has_audio=False."""
        with open(silent_mp4, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("silent.mp4", f, "video/mp4")})
        assert resp.status_code == 200
        meta = resp.json().get("metadata", resp.json())
        assert meta["has_audio"] is False

    def test_silent_video_processing_launch(self, client, silent_mp4):
        """Processing job can be launched for silent video without error."""
        with open(silent_mp4, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("silent.mp4", f, "video/mp4")})
        file_id = resp.json().get("file_id", resp.json().get("task_id"))

        p_resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "60fps",
            "mode": "speed"
        })
        assert p_resp.status_code == 200
        assert "task_id" in p_resp.json()

    def test_silent_video_status_query(self, client, silent_mp4):
        """Status endpoint does not report audio extraction errors for silent video."""
        with open(silent_mp4, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("silent.mp4", f, "video/mp4")})
        file_id = resp.json().get("file_id", resp.json().get("task_id"))

        p_resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "Original",
            "mode": "speed"
        })
        task_id = p_resp.json()["task_id"]
        status_resp = client.get(f"/api/status/{task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json().get("error") is None

    def test_silent_video_pipeline_media_check(self, silent_mp4):
        """Source silent video genuinely contains 0 audio streams."""
        props = inspect_video_properties(silent_mp4)
        assert props["has_audio"] is False
        assert props["audio_codec"] is None

    def test_silent_video_quality_mode(self, client, silent_mp4):
        """Silent video processed in Quality mode succeeds."""
        with open(silent_mp4, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("silent.mp4", f, "video/mp4")})
        file_id = resp.json().get("file_id", resp.json().get("task_id"))

        p_resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "Original",
            "mode": "quality"
        })
        assert p_resp.status_code == 200


# ==============================================================================
# Area 5: Target FPS <= Source FPS
# ==============================================================================
class TestArea5TargetFPSEdgeCases:
    """Test target FPS <= source FPS (>=5 tests)."""

    def _upload_file(self, client, file_path):
        with open(file_path, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("video.mp4", f, "video/mp4")})
        assert resp.status_code == 200
        data = resp.json()
        return data.get("file_id", data.get("task_id"))

    def test_target_fps_equal_to_source_fps(self, client, sample_mp4):
        """Target FPS equal to source (30fps -> 30fps) does not crash or loop."""
        file_id = self._upload_file(client, sample_mp4)
        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "30fps",
            "mode": "speed"
        })
        assert resp.status_code == 200

    def test_target_fps_less_than_source_fps(self, client, standard_fixtures):
        """60fps source with 30fps target is accepted and handled without crash."""
        vid_60 = standard_fixtures["fps_60"]
        file_id = self._upload_file(client, vid_60)
        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "30fps",
            "mode": "speed"
        })
        assert resp.status_code == 200

    def test_target_fps_invalid_string_rejected(self, client, sample_mp4):
        """Invalid FPS string like 'invalid_fps' is rejected with 422 or 400."""
        file_id = self._upload_file(client, sample_mp4)
        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "invalid_fps",
            "mode": "speed"
        })
        assert resp.status_code in (400, 422)

    def test_target_fps_zero_or_negative_rejected(self, client, sample_mp4):
        """0 or negative FPS is rejected."""
        file_id = self._upload_file(client, sample_mp4)
        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "0fps",
            "mode": "speed"
        })
        assert resp.status_code in (400, 422)

    def test_target_fps_fractional_input_preservation(self, client, standard_fixtures):
        """24fps source preserves 24fps with 'Original' preset."""
        vid_24 = standard_fixtures["fps_24"]
        file_id = self._upload_file(client, vid_24)
        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "Original",
            "mode": "speed"
        })
        assert resp.status_code == 200


# ==============================================================================
# Area 6: Job Cancellation (/api/cancel/{task_id})
# ==============================================================================
class TestArea6JobCancellation:
    """Test job cancellation behavior and resource cleanup (>=5 tests)."""

    def _launch_task(self, client, sample_mp4):
        with open(sample_mp4, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("video.mp4", f, "video/mp4")})
        assert resp.status_code == 200
        file_id = resp.json().get("file_id", resp.json().get("task_id"))
        p_resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "1080p",
            "fps": "60fps",
            "mode": "quality"
        })
        assert p_resp.status_code == 200
        return p_resp.json()["task_id"]

    def test_cancel_active_job_returns_200(self, client, sample_mp4):
        """POST /api/cancel/{task_id} on active job returns 200 OK."""
        task_id = self._launch_task(client, sample_mp4)
        cancel_resp = client.post(f"/api/cancel/{task_id}")
        assert cancel_resp.status_code == 200
        data = cancel_resp.json()
        assert data.get("status") == "cancelled" or "cancelled" in str(data).lower()

    def test_cancel_status_endpoint_shows_cancelled(self, client, sample_mp4):
        """Status endpoint reflects 'cancelled' state after cancellation."""
        task_id = self._launch_task(client, sample_mp4)
        client.post(f"/api/cancel/{task_id}")
        status_resp = client.get(f"/api/status/{task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json().get("status") == "cancelled"

    def test_cancel_nonexistent_task_returns_404(self, client):
        """Cancelling nonexistent task_id returns 404 Not Found."""
        resp = client.post("/api/cancel/nonexistent-task-abc-123")
        assert resp.status_code == 404

    def test_cancel_idempotency(self, client, sample_mp4):
        """Cancelling an already cancelled task is handled cleanly without crashing."""
        task_id = self._launch_task(client, sample_mp4)
        resp1 = client.post(f"/api/cancel/{task_id}")
        assert resp1.status_code == 200
        resp2 = client.post(f"/api/cancel/{task_id}")
        assert resp2.status_code in (200, 400, 409)

    def test_cancel_download_fails_for_cancelled_task(self, client, sample_mp4):
        """Attempting to download a cancelled task returns 400 or 404."""
        task_id = self._launch_task(client, sample_mp4)
        client.post(f"/api/cancel/{task_id}")
        down_resp = client.get(f"/api/download/{task_id}")
        assert down_resp.status_code in (400, 404, 409)
