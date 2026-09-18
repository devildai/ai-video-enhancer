"""tests/unit/test_media.py - Unit tests for app/media FFmpeg pipeline."""

import os
import subprocess
import pytest
import numpy as np
from PIL import Image

from app.media.probe import probe_video, parse_fps, VideoMetadata, _find_binary
from app.media.audio import extract_audio, align_audio_sync
from app.media.stream import VideoFrameDecoder, VideoFrameEncoder
from app.media.preview import extract_sample_frame


@pytest.fixture(scope="module")
def synthetic_media(tmp_path_factory):
    """Generates synthetic test videos and audio files for unit tests."""
    temp_dir = tmp_path_factory.mktemp("media_fixtures")
    ffmpeg_bin = _find_binary("ffmpeg")

    # 1. Video WITH audio: 320x240, 30fps, 1.0s, H.264 + AAC
    vid_with_audio = str(temp_dir / "clip_with_audio.mp4")
    cmd_audio = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        vid_with_audio
    ]
    subprocess.run(cmd_audio, check=True)

    # 2. Video WITHOUT audio: 160x120, 25fps, 1.0s, H.264
    vid_no_audio = str(temp_dir / "clip_no_audio.mp4")
    cmd_no_audio = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        vid_no_audio
    ]
    subprocess.run(cmd_no_audio, check=True)

    # 3. Fractional FPS video: 320x240, 24000/1001 (~23.976 fps), 1.0s
    vid_frac_fps = str(temp_dir / "clip_frac_fps.mp4")
    cmd_frac = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=24000/1001",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        vid_frac_fps
    ]
    subprocess.run(cmd_frac, check=True)

    # 4. Audio-only file: 1.0s sine wave in AAC/M4A
    audio_only = str(temp_dir / "pure_audio.m4a")
    cmd_pure_audio = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:a", "aac",
        audio_only
    ]
    subprocess.run(cmd_pure_audio, check=True)

    return {
        "with_audio": vid_with_audio,
        "no_audio": vid_no_audio,
        "frac_fps": vid_frac_fps,
        "audio_only": audio_only,
    }


# =========================================================================
# 1. Test probe_video and parse_fps
# =========================================================================

def test_parse_fps():
    assert parse_fps("30/1") == 30.0
    assert parse_fps("25") == 25.0
    assert abs(parse_fps("30000/1001") - 29.97003) < 0.001
    assert abs(parse_fps("24000/1001") - 23.97602) < 0.001
    assert parse_fps("0/0") == 0.0
    assert parse_fps(None) == 0.0
    assert parse_fps("invalid") == 0.0


def test_probe_video_with_audio(synthetic_media):
    path = synthetic_media["with_audio"]
    meta = probe_video(path)

    assert isinstance(meta, VideoMetadata)
    assert meta.width == 320
    assert meta.height == 240
    assert abs(meta.fps - 30.0) < 0.1
    assert meta.duration > 0.9
    assert meta.nb_frames == 30
    assert any(x in meta.codec_name.lower() for x in ("h264", "avc"))
    assert meta.has_audio is True
    assert meta.audio_codec == "aac"
    assert meta.file_size > 0
    d = meta.to_dict()
    assert d["width"] == 320
    assert d["has_audio"] is True


def test_probe_video_no_audio(synthetic_media):
    path = synthetic_media["no_audio"]
    meta = probe_video(path)

    assert meta.width == 160
    assert meta.height == 120
    assert abs(meta.fps - 25.0) < 0.1
    assert meta.has_audio is False
    assert meta.audio_codec is None


def test_probe_video_frac_fps(synthetic_media):
    path = synthetic_media["frac_fps"]
    meta = probe_video(path)

    assert abs(meta.fps - 23.976) < 0.01


def test_probe_video_not_found():
    with pytest.raises(FileNotFoundError):
        probe_video("non_existent_file_xyz123.mp4")


def test_probe_video_empty_file(tmp_path):
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        probe_video(str(empty_file))


def test_probe_video_corrupt_file(tmp_path):
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"not a real video file header content")
    with pytest.raises(ValueError, match="Failed to probe"):
        probe_video(str(corrupt_file))


def test_probe_video_no_video_stream(synthetic_media):
    path = synthetic_media["audio_only"]
    with pytest.raises(ValueError, match="No video stream found"):
        probe_video(path)


# =========================================================================
# 2. Test extract_audio and align_audio_sync
# =========================================================================

def test_extract_audio_success(synthetic_media, tmp_path):
    video_path = synthetic_media["with_audio"]
    out_audio = str(tmp_path / "extracted.m4a")

    result = extract_audio(video_path, out_audio)
    assert result is True
    assert os.path.exists(out_audio)
    assert os.path.getsize(out_audio) > 0


def test_extract_audio_no_audio_stream(synthetic_media, tmp_path):
    video_path = synthetic_media["no_audio"]
    out_audio = str(tmp_path / "should_not_exist.m4a")

    result = extract_audio(video_path, out_audio)
    assert result is False
    assert not os.path.exists(out_audio)


def test_extract_audio_not_found():
    with pytest.raises(FileNotFoundError):
        extract_audio("missing_video.mp4", "out.m4a")


def test_align_audio_sync_pad(synthetic_media, tmp_path):
    # Source is 1.0s audio, align to 2.0s
    audio_path = synthetic_media["audio_only"]
    aligned_audio = str(tmp_path / "aligned_pad.m4a")

    result = align_audio_sync(2.0, audio_path, aligned_audio)
    assert os.path.exists(result)

    # Verify duration using ffprobe
    ffmpeg_bin = _find_binary("ffprobe")
    res = subprocess.run(
        [
            ffmpeg_bin, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", aligned_audio
        ],
        capture_output=True, text=True, check=True
    )
    dur = float(res.stdout.strip())
    assert abs(dur - 2.0) < 0.1


def test_align_audio_sync_trim(synthetic_media, tmp_path):
    # Source is 1.0s audio, align to 0.5s
    audio_path = synthetic_media["audio_only"]
    aligned_audio = str(tmp_path / "aligned_trim.m4a")

    result = align_audio_sync(0.5, audio_path, aligned_audio)
    assert os.path.exists(result)

    ffmpeg_bin = _find_binary("ffprobe")
    res = subprocess.run(
        [
            ffmpeg_bin, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", aligned_audio
        ],
        capture_output=True, text=True, check=True
    )
    dur = float(res.stdout.strip())
    assert abs(dur - 0.5) < 0.1


def test_align_audio_sync_invalid_duration(synthetic_media, tmp_path):
    audio_path = synthetic_media["audio_only"]
    with pytest.raises(ValueError):
        align_audio_sync(0.0, audio_path, str(tmp_path / "invalid.m4a"))
    with pytest.raises(ValueError):
        align_audio_sync(-1.0, audio_path, str(tmp_path / "invalid.m4a"))


def test_align_audio_sync_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        align_audio_sync(1.0, "missing.m4a", str(tmp_path / "out.m4a"))


# =========================================================================
# 3. Test VideoFrameDecoder
# =========================================================================

def test_decoder_full_iteration(synthetic_media):
    video_path = synthetic_media["with_audio"]
    decoder = VideoFrameDecoder(video_path)

    assert decoder.width == 320
    assert decoder.height == 240
    assert abs(decoder.fps - 30.0) < 0.1

    frames = list(decoder)
    assert len(frames) == 30
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        assert frame.sum() > 0


def test_decoder_context_manager(synthetic_media):
    video_path = synthetic_media["no_audio"]
    count = 0
    with VideoFrameDecoder(video_path) as dec:
        assert dec.width == 160
        assert dec.height == 120
        for frame in dec:
            count += 1
            assert frame.shape == (120, 160, 3)

    assert count == 25


def test_decoder_early_exit(synthetic_media):
    video_path = synthetic_media["with_audio"]
    decoder = VideoFrameDecoder(video_path)
    # Break early after reading 5 frames
    for i, _ in enumerate(decoder):
        if i == 4:
            break
    # Verify cleanup occurred without deadlock
    assert decoder._closed is True


def test_decoder_file_not_found():
    with pytest.raises(FileNotFoundError):
        VideoFrameDecoder("missing_video_abc.mp4")


# =========================================================================
# 4. Test VideoFrameEncoder
# =========================================================================

def test_encoder_write_frames(tmp_path):
    out_mp4 = str(tmp_path / "encoded.mp4")
    w, h, fps, n_frames = 320, 240, 30.0, 30

    with VideoFrameEncoder(out_mp4, w, h, fps) as enc:
        for i in range(n_frames):
            frame = np.full((h, w, 3), (i * 8) % 256, dtype=np.uint8)
            enc.write_frame(frame)

    assert os.path.exists(out_mp4)
    assert os.path.getsize(out_mp4) > 0

    meta = probe_video(out_mp4)
    assert meta.width == 320
    assert meta.height == 240
    assert abs(meta.fps - 30.0) < 0.1
    assert meta.nb_frames == 30
    assert meta.has_audio is False


def test_encoder_with_audio(synthetic_media, tmp_path):
    out_mp4 = str(tmp_path / "encoded_with_audio.mp4")
    audio_path = synthetic_media["audio_only"]
    w, h, fps, n_frames = 320, 240, 30.0, 30

    with VideoFrameEncoder(out_mp4, w, h, fps, audio_path=audio_path) as enc:
        for _ in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            enc.write_frame(frame)

    assert os.path.exists(out_mp4)
    meta = probe_video(out_mp4)
    assert meta.has_audio is True
    assert meta.audio_codec == "aac"


def test_encoder_odd_dimensions_enforcement(tmp_path):
    # Odd width and height: 321x241
    out_mp4 = str(tmp_path / "encoded_odd.mp4")
    with VideoFrameEncoder(out_mp4, width=321, height=241, fps=30.0) as enc:
        assert enc.width == 320
        assert enc.height == 240
        for _ in range(15):
            odd_frame = np.zeros((241, 321, 3), dtype=np.uint8)
            enc.write_frame(odd_frame)

    assert os.path.exists(out_mp4)
    meta = probe_video(out_mp4)
    assert meta.width == 320
    assert meta.height == 240


def test_encoder_invalid_audio_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFrameEncoder(
            str(tmp_path / "out.mp4"),
            320, 240, 30.0,
            audio_path="non_existent.m4a"
        )


def test_encoder_invalid_fps(tmp_path):
    with pytest.raises(ValueError):
        VideoFrameEncoder(str(tmp_path / "out.mp4"), 320, 240, 0.0)


# =========================================================================
# 5. Test extract_sample_frame
# =========================================================================

def test_extract_sample_frame_success(synthetic_media, tmp_path):
    video_path = synthetic_media["with_audio"]
    out_jpeg = str(tmp_path / "sample.jpg")

    result = extract_sample_frame(video_path, 0.5, out_jpeg)
    assert os.path.exists(result)
    assert result == os.path.abspath(out_jpeg)

    with Image.open(result) as img:
        assert img.format == "JPEG"
        assert img.size == (320, 240)


def test_extract_sample_frame_bounds(synthetic_media, tmp_path):
    video_path = synthetic_media["with_audio"]

    # Negative timestamp should clamp to 0.0
    out_neg = str(tmp_path / "sample_neg.jpg")
    extract_sample_frame(video_path, -5.0, out_neg)
    assert os.path.exists(out_neg)

    # Timestamp beyond duration should clamp to end
    out_over = str(tmp_path / "sample_over.jpg")
    extract_sample_frame(video_path, 999.0, out_over)
    assert os.path.exists(out_over)


def test_extract_sample_frame_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_sample_frame(
            "missing_video.mp4", 0.0, str(tmp_path / "out.jpg")
        )
