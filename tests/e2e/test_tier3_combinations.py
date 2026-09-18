"""Tier 3 - Cross-Feature Combination E2E Tests.

Tests multi-feature matrix permutations across resolutions, frame rates,
enhancement modes, container formats, and audio profiles:
1. 480p 30fps -> 1080p 60fps Speed mode with audio
2. 360p 24fps -> 1080p 60fps Quality mode with scene cuts
3. 720p 30fps -> 2K 60fps Speed mode
4. Silent video -> 1080p 60fps Quality mode
5. MKV container -> 4K 30fps MP4
6. WebM VP8 container -> 1080p 60fps MP4
7. AVI container -> Original resolution 120fps MP4
"""
import time
import pytest
from pathlib import Path
from tests.e2e.helpers import inspect_video_properties

pytestmark = pytest.mark.tier3


class TestTier3CrossFeatureCombinations:
    """Test cross-feature matrix combinations."""

    def _upload_file(self, client, file_path, content_type="video/mp4"):
        filename = Path(file_path).name
        with open(file_path, "rb") as f:
            resp = client.post("/api/upload", files={"file": (filename, f, content_type)})
        assert resp.status_code == 200
        data = resp.json()
        return data.get("file_id", data.get("task_id"))

    def test_comb_480p_30fps_to_1080p_60fps_speed_with_audio(self, client, standard_fixtures):
        """Combination 1: 480p 30fps -> 1080p 60fps Speed mode with audio."""
        src = standard_fixtures["res_480p_30fps"]
        file_id = self._upload_file(client, src)

        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "1080p",
            "fps": "60fps",
            "mode": "speed"
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        status = client.get(f"/api/status/{task_id}").json()
        assert status["status"] in ("queued", "processing", "completed")

    def test_comb_360p_24fps_to_1080p_60fps_quality_with_scene_cuts(self, client, standard_fixtures):
        """Combination 2: 360p 24fps -> 1080p 60fps Quality mode with scene cuts."""
        src = standard_fixtures["scene_cut_2s"]
        file_id = self._upload_file(client, src)

        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "1080p",
            "fps": "60fps",
            "mode": "quality"
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        status = client.get(f"/api/status/{task_id}").json()
        assert status["status"] in ("queued", "processing", "completed")

    def test_comb_720p_30fps_to_2k_60fps_speed(self, client, standard_fixtures):
        """Combination 3: 720p 30fps -> 2K 60fps Speed mode."""
        src = standard_fixtures["res_720p_30fps"]
        file_id = self._upload_file(client, src)

        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "2K",
            "fps": "60fps",
            "mode": "speed"
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        status = client.get(f"/api/status/{task_id}").json()
        assert status["status"] in ("queued", "processing", "completed")

    def test_comb_silent_video_to_1080p_60fps_quality(self, client, silent_mp4):
        """Combination 4: Silent video -> 1080p 60fps Quality mode."""
        file_id = self._upload_file(client, silent_mp4)

        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "1080p",
            "fps": "60fps",
            "mode": "quality"
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        status = client.get(f"/api/status/{task_id}").json()
        assert status["status"] in ("queued", "processing", "completed")

    def test_comb_mkv_input_to_4k_mp4_output(self, client, sample_mkv):
        """Combination 5: MKV input container -> 4K 30fps MP4."""
        file_id = self._upload_file(client, sample_mkv, content_type="video/x-matroska")

        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "4K",
            "fps": "30fps",
            "mode": "speed"
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        status = client.get(f"/api/status/{task_id}").json()
        assert status["status"] in ("queued", "processing", "completed")

    def test_comb_webm_vp8_input_to_1080p_60fps(self, client, sample_webm):
        """Combination 6: WebM VP8 input container -> 1080p 60fps MP4."""
        file_id = self._upload_file(client, sample_webm, content_type="video/webm")

        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "1080p",
            "fps": "60fps",
            "mode": "speed"
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        status = client.get(f"/api/status/{task_id}").json()
        assert status["status"] in ("queued", "processing", "completed")

    def test_comb_avi_input_to_original_res_120fps(self, client, sample_avi):
        """Combination 7: AVI input container -> Original resolution 120fps MP4."""
        file_id = self._upload_file(client, sample_avi, content_type="video/x-msvideo")

        resp = client.post("/api/process", json={
            "file_id": file_id,
            "resolution": "Original",
            "fps": "120fps",
            "mode": "speed"
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        status = client.get(f"/api/status/{task_id}").json()
        assert status["status"] in ("queued", "processing", "completed")
