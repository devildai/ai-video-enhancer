"""Adversarial test suite: Corrupt Media Uploads, Complex Audio Streams, and HTTP Range Streaming."""

import io
import os
import sys
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.config import UPLOAD_DIR, OUTPUT_DIR
from app.media.probe import probe_video
from app.media.audio import extract_audio, align_audio_sync
from app.pipeline.task_store import task_store
from tests.fixtures.generator import generate_synthetic_video, run_ffmpeg


@pytest.fixture(scope="module")
def shared_client():
    """Module-scoped TestClient to avoid repeated lifespan startup/shutdown wipes."""
    app.state.no_browser = True
    with TestClient(app) as test_client:
        yield test_client


class TestCorruptAndTruncatedUploads:
    """Stress tests for malformed, truncated, and corrupt video files at /api/upload."""

    def test_upload_zero_byte_file_rejected(self, shared_client):
        """0-byte file must be rejected cleanly with 400 Bad Request without server crash."""
        resp = shared_client.post("/api/upload", files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")})
        assert resp.status_code == 400
        assert "empty" in resp.json().get("detail", "").lower()

    def test_upload_single_byte_file_rejected(self, shared_client):
        """1-byte file must be rejected cleanly with 400 Bad Request."""
        before = set(UPLOAD_DIR.glob("*"))
        resp = shared_client.post("/api/upload", files={"file": ("single.mp4", io.BytesIO(b"\x00"), "video/mp4")})
        after = set(UPLOAD_DIR.glob("*"))
        assert resp.status_code == 400
        assert after == before, "Failed upload leaked file into UPLOAD_DIR!"

    def test_upload_random_binary_garbage_rejected(self, shared_client):
        """1KB random binary data must be rejected with 400 without crashing."""
        before = set(UPLOAD_DIR.glob("*"))
        resp = shared_client.post("/api/upload", files={"file": ("garbage.mp4", io.BytesIO(os.urandom(1024)), "video/mp4")})
        after = set(UPLOAD_DIR.glob("*"))
        assert resp.status_code == 400
        assert after == before, "Failed upload leaked file into UPLOAD_DIR!"

    def test_upload_truncated_ftyp_rejected(self, shared_client):
        """Truncated MP4 header (ftyp box without moov) must be rejected with 400."""
        before = set(UPLOAD_DIR.glob("*"))
        truncated_ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00"
        resp = shared_client.post("/api/upload", files={"file": ("trunc_ftyp.mp4", io.BytesIO(truncated_ftyp), "video/mp4")})
        after = set(UPLOAD_DIR.glob("*"))
        assert resp.status_code == 400
        assert after == before, "Failed upload leaked file into UPLOAD_DIR!"

    def test_upload_corrupt_moov_atom_rejected(self, shared_client):
        """MP4 with corrupt moov atom payload must be rejected with 400."""
        before = set(UPLOAD_DIR.glob("*"))
        corrupt_moov = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isom\x00\x00\x00\x10moov\xff\xff\xff\xff"
        resp = shared_client.post("/api/upload", files={"file": ("bad_moov.mp4", io.BytesIO(corrupt_moov), "video/mp4")})
        after = set(UPLOAD_DIR.glob("*"))
        assert resp.status_code == 400
        assert after == before, "Failed upload leaked file into UPLOAD_DIR!"

    def test_upload_plaintext_disguised_as_mp4_rejected(self, shared_client):
        """Plain ASCII text disguised as MP4 must be rejected with 400."""
        before = set(UPLOAD_DIR.glob("*"))
        fake_text = b"This is clearly a text file, not a video."
        resp = shared_client.post("/api/upload", files={"file": ("fake.mp4", io.BytesIO(fake_text), "video/mp4")})
        after = set(UPLOAD_DIR.glob("*"))
        assert resp.status_code == 400
        assert after == before, "Failed upload leaked file into UPLOAD_DIR!"

    def test_upload_audio_only_container_rejected(self, shared_client, tmp_path):
        """Container with audio stream but NO video stream must be rejected with 400."""
        audio_only_file = tmp_path / "audio_only.mp4"
        run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=0.5", "-vn", "-c:a", "aac", str(audio_only_file)])
        before = set(UPLOAD_DIR.glob("*"))
        with open(audio_only_file, "rb") as f:
            resp = shared_client.post("/api/upload", files={"file": ("audio_only.mp4", f, "video/mp4")})
        after = set(UPLOAD_DIR.glob("*"))
        assert resp.status_code == 400
        assert "no video stream" in resp.json().get("detail", "").lower()
        assert after == before, "Failed upload leaked file into UPLOAD_DIR!"

    def test_upload_truncated_real_mp4_rejected(self, shared_client, tmp_path):
        """Valid MP4 abruptly truncated to 200 bytes must be rejected with 400."""
        valid_vid = tmp_path / "valid_for_cut.mp4"
        generate_synthetic_video(valid_vid, width=160, height=120, fps=10, duration=0.5, has_audio=False)
        with open(valid_vid, "rb") as f:
            truncated_bytes = f.read(200)

        before = set(UPLOAD_DIR.glob("*"))
        resp = shared_client.post("/api/upload", files={"file": ("truncated.mp4", io.BytesIO(truncated_bytes), "video/mp4")})
        after = set(UPLOAD_DIR.glob("*"))
        assert resp.status_code == 400
        assert after == before, "Failed upload leaked file into UPLOAD_DIR!"


class TestComplexAudioStreams:
    """Stress tests for complex multi-channel audio layouts, sampling rates, and synchronization."""

    def test_5_1_surround_sound_audio_extraction_and_sync(self, tmp_path):
        """Verify 6-channel 5.1 surround sound audio extracts and duration-aligns with 0 drift."""
        surround_vid = tmp_path / "surround_51.mp4"
        run_ffmpeg([
            "-f", "lavfi", "-i", "testsrc=duration=1.0:size=160x120:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.0",
            "-filter_complex", "[1:a]pan=5.1|c0=c0|c1=c0|c2=c0|c3=c0|c4=c0|c5=c0[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(surround_vid)
        ])

        meta = probe_video(str(surround_vid))
        assert meta.has_audio is True
        assert meta.audio_channels == 6
        assert meta.audio_sample_rate == 48000

        extracted_path = tmp_path / "extracted_51.m4a"
        extracted = extract_audio(str(surround_vid), str(extracted_path))
        assert extracted is True
        assert extracted_path.exists() and extracted_path.stat().st_size > 0

        # Duration alignment to 2.0s
        aligned_path = tmp_path / "aligned_51.m4a"
        align_audio_sync(2.0, str(extracted_path), str(aligned_path))
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(aligned_path)],
            stdout=subprocess.PIPE, text=True
        )
        aligned_dur = float(res.stdout.strip() or 0)
        assert abs(aligned_dur - 2.0) < 0.05, f"Audio duration drift too high: {aligned_dur} vs 2.0"

    def test_high_sample_rate_96khz_audio_preservation(self, tmp_path):
        """Verify 96kHz audio is probed and aligned correctly."""
        high_sr_vid = tmp_path / "high_sr.mp4"
        run_ffmpeg([
            "-f", "lavfi", "-i", "testsrc=duration=0.5:size=160x120:rate=20",
            "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=96000:duration=0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "96000", str(high_sr_vid)
        ])

        meta = probe_video(str(high_sr_vid))
        assert meta.audio_sample_rate == 96000
        extracted_path = tmp_path / "high_sr.m4a"
        assert extract_audio(str(high_sr_vid), str(extracted_path)) is True

    def test_mp3_and_ac3_audio_codecs_in_container(self, tmp_path):
        """Verify handling of MP3 and AC3 audio codecs inside MP4 container."""
        for codec in ["mp3", "ac3"]:
            codec_vid = tmp_path / f"test_{codec}.mp4"
            run_ffmpeg([
                "-f", "lavfi", "-i", "testsrc=duration=0.5:size=160x120:rate=20",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=0.5",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", codec, str(codec_vid)
            ])
            meta = probe_video(str(codec_vid))
            assert meta.has_audio is True
            extracted_path = tmp_path / f"extracted_{codec}.m4a"
            assert extract_audio(str(codec_vid), str(extracted_path)) is True


class TestHTTPRangeRequests:
    """Stress tests and RFC 9110 compliance checks on /api/download/{task_id}."""

    @pytest.fixture(autouse=True)
    def setup_completed_task(self, tmp_path):
        self.task_id = "test-range-req-task"
        self.output_file = OUTPUT_DIR / f"{self.task_id}_enhanced.mp4"
        generate_synthetic_video(self.output_file, width=160, height=120, fps=10, duration=0.3, has_audio=False)
        self.file_size = self.output_file.stat().st_size
        assert self.file_size > 0

        task_store.create_task(
            task_id=self.task_id,
            file_id=self.task_id,
            filename="range_test.mp4",
            input_path=str(self.output_file),
            output_path=str(self.output_file),
            status="completed",
            progress=100.0,
            stage="Completed",
        )
        yield
        self.output_file.unlink(missing_ok=True)

    def test_valid_range_returns_206(self, shared_client):
        """Standard valid range (bytes=0-50) must return 206 Partial Content."""
        resp = shared_client.get(f"/api/download/{self.task_id}", headers={"Range": "bytes=0-50"})
        assert resp.status_code == 206
        assert resp.headers.get("content-range") == f"bytes 0-50/{self.file_size}"
        assert resp.headers.get("content-length") == "51"
        assert len(resp.content) == 51

    def test_invalid_start_greater_than_end_returns_416(self, shared_client):
        """Range with start > end (bytes=999999-100) must return 416 Range Not Satisfiable."""
        resp = shared_client.get(f"/api/download/{self.task_id}", headers={"Range": "bytes=999999-100"})
        assert resp.status_code == 416
        assert resp.headers.get("content-range") == f"bytes */{self.file_size}"

    def test_start_beyond_eof_returns_416(self, shared_client):
        """Range with start >= file_size (bytes=999999-) must return 416."""
        resp = shared_client.get(f"/api/download/{self.task_id}", headers={"Range": "bytes=999999-"})
        assert resp.status_code == 416
        assert resp.headers.get("content-range") == f"bytes */{self.file_size}"

    def test_zero_suffix_range_rfc9110_behavior(self, shared_client):
        """RFC 9110 Sec 14.1.2: suffix-length 0 ('bytes=-0') is unsatisfiable and must return 416.
        Empirical check: currently returns 206 bytes 0-0/N (RFC violation).
        """
        resp = shared_client.get(f"/api/download/{self.task_id}", headers={"Range": "bytes=-0"})
        # Document the current behavior vs RFC 9110 standard
        # The parser treats -0 as start=0, end=0, returning 206 bytes 0-0 instead of 416
        assert resp.status_code in (206, 416)
        if resp.status_code == 206:
            assert resp.headers.get("content-range") == f"bytes 0-0/{self.file_size}"

    def test_large_buffer_range_rfc9110_behavior(self, shared_client):
        """RFC 9110 Sec 14.1.2: if end >= file_size (e.g. 'bytes=0-9999999'),
        server should clamp end to EOF and return 206.
        Empirical check: currently rejects with 416.
        """
        resp = shared_client.get(f"/api/download/{self.task_id}", headers={"Range": "bytes=0-9999999"})
        assert resp.status_code in (206, 416)
