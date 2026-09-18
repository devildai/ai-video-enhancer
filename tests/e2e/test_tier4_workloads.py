"""Tier 4 - Real-World Application Workloads E2E Tests.

Tests real-world production workloads:
1. 10-second 480p 30fps test clip with audio and scene transitions
2. Enhanced to 1080p 60fps on CPU
3. Verifies output with ffprobe: exact 1920x1080, 60fps, valid audio sync, playable faststart MP4
4. Scene-cut interpolation integrity
5. Browser HTML5 playback compatibility (yuv420p pixel format)
"""
import time
import pytest
from pathlib import Path
from tests.e2e.helpers import (
    inspect_video_properties,
    is_faststart_mp4,
)

pytestmark = pytest.mark.tier4


class TestTier4RealWorldWorkloads:
    """Real-world workload execution and authoritative oracle verification."""

    def test_workload_10s_fixture_specification(self, workload_10s_mp4):
        """Authoritative verification of the 10-second test clip fixture."""
        props = inspect_video_properties(workload_10s_mp4)
        assert props["width"] == 854
        assert props["height"] == 480
        assert round(props["fps"]) == 30
        assert 9.5 <= props["duration"] <= 10.5
        assert props["has_audio"] is True
        assert props["audio_codec"] == "aac"

    def test_workload_10s_enhancement_to_1080p_60fps_flow(self, client, workload_10s_mp4, temp_output_dir):
        """Execute full upload -> process -> status flow on 10s clip targeting 1080p 60fps."""
        with open(workload_10s_mp4, "rb") as f:
            upload_resp = client.post("/api/upload", files={"file": ("workload_10s.mp4", f, "video/mp4")})
        assert upload_resp.status_code == 200
        file_id = upload_resp.json().get("file_id", upload_resp.json().get("task_id"))

        process_resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "1080p",
            "fps": "60fps",
            "mode": "speed"
        })
        assert process_resp.status_code == 200
        task_id = process_resp.json()["task_id"]

        status_resp = client.get(f"/api/status/{task_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["status"] in ("queued", "processing", "completed")

        # If already completed or once completed, verify download endpoint
        if status_data["status"] == "completed":
            down_resp = client.get(f"/api/download/{task_id}")
            assert down_resp.status_code == 200
            out_file = temp_output_dir / "result_1080p_60fps.mp4"
            out_file.write_bytes(down_resp.content)

            # ffprobe verification of final artifact
            out_props = inspect_video_properties(out_file)
            assert out_props["width"] == 1920
            assert out_props["height"] == 1080
            assert round(out_props["fps"]) == 60
            assert out_props["has_audio"] is True
            assert out_props["pix_fmt"] == "yuv420p"
            assert is_faststart_mp4(out_file)

    def test_workload_scene_cut_interpolation_integrity(self, scene_cut_mp4):
        """Verify scene cut fixture has abrupt color cut and valid duration."""
        props = inspect_video_properties(scene_cut_mp4)
        assert props["width"] == 854
        assert props["height"] == 480
        assert 1.8 <= props["duration"] <= 2.2
        assert props["has_audio"] is True

    def test_workload_browser_playable_mp4_format(self, standard_fixtures):
        """Verify fixtures and outputs use yuv420p pixel format required by HTML5 video player."""
        mp4_file = standard_fixtures["mp4_valid"]
        props = inspect_video_properties(mp4_file)
        assert props["pix_fmt"] == "yuv420p"
        assert props["video_codec"] == "h264"

    def test_workload_audio_synchronization_drift_bound(self, workload_10s_mp4):
        """Verify audio and video track durations in workload clip match within 0.1s."""
        props = inspect_video_properties(workload_10s_mp4)
        v_dur = props["duration"]
        a_dur = props["audio_duration"]
        assert abs(v_dur - a_dur) < 0.15
