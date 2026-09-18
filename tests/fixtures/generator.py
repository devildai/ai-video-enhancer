"""Synthetic video fixture generator for AI Video Enhancement Tool E2E tests."""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

FIXTURES_DIR = Path(__file__).parent.resolve()
CACHE_DIR = FIXTURES_DIR / "cache"


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    """Run ffmpeg command safely with full error capture."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg command failed ({result.returncode}): {' '.join(cmd)}\nStderr: {result.stderr}")
    return result


def ensure_fixture_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def generate_synthetic_video(
    output_path: Path | str,
    width: int = 640,
    height: int = 360,
    fps: int | float = 30,
    duration: float = 1.0,
    has_audio: bool = True,
    pattern: str = "testsrc",
    vcodec: str = "libx264",
    pix_fmt: str = "yuv420p",
    acodec: str = "aac",
    audio_freq: int = 1000,
) -> Path:
    """Generate a synthetic video using ffmpeg lavfi filters."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If already exists and is non-empty, avoid re-generating
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    # Construct video filter
    if pattern == "testsrc":
        v_filter = f"testsrc=duration={duration}:size={width}x{height}:rate={fps}"
    elif pattern == "smptebars":
        v_filter = f"smptebars=duration={duration}:size={width}x{height}:rate={fps}"
    elif pattern == "color":
        v_filter = f"color=c=navy:duration={duration}:size={width}x{height}:rate={fps}"
    else:
        v_filter = f"testsrc=duration={duration}:size={width}x{height}:rate={fps}"

    args = ["-f", "lavfi", "-i", v_filter]

    if has_audio:
        a_filter = f"sine=frequency={audio_freq}:duration={duration}"
        args += ["-f", "lavfi", "-i", a_filter]
        args += ["-c:v", vcodec, "-pix_fmt", pix_fmt, "-c:a", acodec]
    else:
        args += ["-an", "-c:v", vcodec, "-pix_fmt", pix_fmt]

    args.append(str(out_path))
    run_ffmpeg(args)
    return out_path


def generate_scene_cut_video(
    output_path: Path | str,
    width: int = 640,
    height: int = 360,
    fps: int | float = 30,
    duration: float = 2.0,
    has_audio: bool = True,
) -> Path:
    """Generate a video with an abrupt hard scene cut in the middle (Red -> Blue)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    half_duration = duration / 2.0
    # Create two color sources and concatenate
    filter_complex = (
        f"color=c=red:d={half_duration}:s={width}x{height}:r={fps}[v1];"
        f"color=c=blue:d={half_duration}:s={width}x{height}:r={fps}[v2];"
        f"[v1][v2]concat=n=2:v=1:a=0[vout]"
    )

    args = ["-filter_complex", filter_complex]

    if has_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=880:duration={duration}"]
        args += ["-map", "[vout]", "-map", "0:a", "-c:a", "aac"]
    else:
        args += ["-map", "[vout]"]

    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
    run_ffmpeg(args)
    return out_path


def generate_workload_10s_clip(output_path: Path | str) -> Path:
    """Generate a realistic 10-second 480p 30fps clip with audio and scene transitions."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    # 10s: 0-5s testsrc, 5-10s smptebars with continuous 440Hz audio
    filter_complex = (
        "testsrc=duration=5:size=854x480:rate=30[v1];"
        "smptebars=duration=5:size=854x480:rate=30[v2];"
        "[v1][v2]concat=n=2:v=1:a=0[vout]"
    )
    args = [
        "-filter_complex", filter_complex,
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(out_path)
    ]
    run_ffmpeg(args)
    return out_path


def generate_corrupt_file(output_path: Path | str, size_bytes: int = 1024) -> Path:
    """Generate a file with random binary bytes that is not a valid video."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(os.urandom(size_bytes))
    return out_path


def generate_zero_byte_file(output_path: Path | str) -> Path:
    """Generate an empty 0-byte file."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pass
    return out_path


def create_all_standard_fixtures() -> dict[str, Path]:
    """Pre-generate the canonical set of test fixtures used across all E2E tiers."""
    cache = ensure_fixture_cache_dir()
    fixtures = {}

    # Containers
    fixtures["mp4_valid"] = generate_synthetic_video(cache / "valid_360p_30fps.mp4", width=640, height=360, fps=30, duration=1.0)
    fixtures["mkv_valid"] = generate_synthetic_video(cache / "valid_360p_30fps.mkv", width=640, height=360, fps=30, duration=1.0, vcodec="libx264")
    fixtures["avi_valid"] = generate_synthetic_video(cache / "valid_360p_30fps.avi", width=640, height=360, fps=30, duration=1.0, vcodec="mjpeg", acodec="pcm_s16le", pix_fmt="yuvj420p")
    fixtures["mov_valid"] = generate_synthetic_video(cache / "valid_360p_30fps.mov", width=640, height=360, fps=30, duration=1.0, vcodec="libx264")
    fixtures["webm_valid"] = generate_synthetic_video(cache / "valid_360p_30fps.webm", width=640, height=360, fps=30, duration=1.0, vcodec="libvpx", acodec="libvorbis")

    # Audio variations
    fixtures["silent_mp4"] = generate_synthetic_video(cache / "silent_360p_30fps.mp4", width=640, height=360, fps=30, duration=1.0, has_audio=False)

    # Dimension edge cases
    # Note: for odd dimensions in H.264, standard libx264 with yuv420p fails without padding, so we use yuv444p or mjpeg for raw odd input container
    fixtures["odd_dim_853x481"] = generate_synthetic_video(cache / "odd_853x481_30fps.mp4", width=853, height=481, fps=30, duration=1.0, pix_fmt="yuv444p")
    fixtures["odd_dim_319x241"] = generate_synthetic_video(cache / "odd_319x241_30fps.mp4", width=319, height=241, fps=30, duration=1.0, pix_fmt="yuv444p")

    # FPS variations
    fixtures["fps_24"] = generate_synthetic_video(cache / "sample_360p_24fps.mp4", width=640, height=360, fps=24, duration=1.0)
    fixtures["fps_60"] = generate_synthetic_video(cache / "sample_360p_60fps.mp4", width=640, height=360, fps=60, duration=1.0)
    fixtures["res_480p_30fps"] = generate_synthetic_video(cache / "sample_480p_30fps.mp4", width=854, height=480, fps=30, duration=1.0)
    fixtures["res_720p_30fps"] = generate_synthetic_video(cache / "sample_720p_30fps.mp4", width=1280, height=720, fps=30, duration=1.0)

    # Scene cut video
    fixtures["scene_cut_2s"] = generate_scene_cut_video(cache / "scene_cut_480p_30fps.mp4", width=854, height=480, fps=30, duration=2.0)

    # Workload 10-second clip
    fixtures["workload_10s"] = generate_workload_10s_clip(cache / "workload_10s_480p_30fps.mp4")

    # Invalid / Boundary files
    fixtures["corrupt_header"] = generate_corrupt_file(cache / "corrupt_header.mp4")
    fixtures["text_file"] = cache / "fake_video.txt"
    if not fixtures["text_file"].exists():
        fixtures["text_file"].write_text("This is a plain text file pretending to be video.")
    fixtures["zero_byte"] = generate_zero_byte_file(cache / "empty_zero_byte.mp4")

    return fixtures


if __name__ == "__main__":
    print("Generating standard test fixtures...")
    created = create_all_standard_fixtures()
    print(f"Generated {len(created)} test fixtures in {CACHE_DIR}:")
    for k, v in created.items():
        print(f"  - {k}: {v.name} ({v.stat().st_size} bytes)")
