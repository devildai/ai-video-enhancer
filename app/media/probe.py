"""app/media/probe.py - Video probing and metadata extraction via ffprobe."""

from dataclasses import dataclass, asdict
import json
import os
import shutil
import subprocess
from typing import Optional, Dict, Any


@dataclass
class VideoMetadata:
    """Container for video container and stream metadata."""
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int
    codec_name: str
    has_audio: bool
    file_size: int
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    bitrate: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)


def _find_binary(binary_name: str) -> str:
    """Finds binary on system PATH or standard Windows locations."""
    path = shutil.which(binary_name)
    if path:
        return path
    for fallback in [
        rf"C:\ffmpeg\bin\{binary_name}.exe",
        rf"C:\Program Files\ffmpeg\bin\{binary_name}.exe",
    ]:
        if os.path.exists(fallback):
            return fallback
    return binary_name


def parse_fps(rate_str: Optional[str]) -> float:
    """Parses FFmpeg fractional or decimal frame rate string."""
    if not rate_str or rate_str == "0/0":
        return 0.0
    try:
        if "/" in rate_str:
            num_str, den_str = rate_str.split("/", 1)
            num = float(num_str)
            den = float(den_str)
            if den == 0.0:
                return 0.0
            return num / den
        return float(rate_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(video_path: str) -> VideoMetadata:
    """Probes a media file with ffprobe and extracts stream metadata.

    Args:
        video_path: Path to the media file to probe.

    Returns:
        VideoMetadata containing dimensions, fps, duration, etc.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If file is empty, probe fails, or no video stream exists.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if os.path.getsize(video_path) == 0:
        raise ValueError(f"Video file is empty (0 bytes): {video_path}")

    ffprobe_bin = _find_binary("ffprobe")
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-print_format", "json",
        str(os.path.abspath(video_path))
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace"
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe binary not found on system PATH") from exc

    if result.returncode != 0:
        err_msg = (
            result.stderr.strip()
            or f"ffprobe exited with code {result.returncode}"
        )
        raise ValueError(f"Failed to probe video file: {err_msg}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        preview = result.stdout[:200]
        raise ValueError(f"ffprobe returned invalid JSON: {preview}") from exc

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_stream: Optional[Dict[str, Any]] = None
    audio_stream: Optional[Dict[str, Any]] = None

    for stream in streams:
        c_type = stream.get("codec_type")
        if c_type == "video" and video_stream is None:
            video_stream = stream
        elif c_type == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise ValueError(f"No video stream found in media file: {video_path}")

    # Extract dimensions
    try:
        width = int(video_stream["width"])
        height = int(video_stream["height"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Invalid video stream dimensions") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions: {width}x{height}")

    # Extract fps
    fps = parse_fps(video_stream.get("avg_frame_rate"))
    if fps <= 0.0:
        fps = parse_fps(video_stream.get("r_frame_rate"))
    if fps <= 0.0:
        fps = 30.0

    # Extract duration
    duration = 0.0
    for dur_src in [
        video_stream.get("duration"),
        format_info.get("duration")
    ]:
        if dur_src and dur_src != "N/A":
            try:
                val = float(dur_src)
                if val > 0.0:
                    duration = val
                    break
            except ValueError:
                pass

    # Extract nb_frames
    nb_frames = 0
    nb_frames_str = video_stream.get("nb_frames")
    if nb_frames_str and nb_frames_str != "N/A":
        try:
            nb_frames = int(nb_frames_str)
        except ValueError:
            pass

    if nb_frames <= 0 and duration > 0.0 and fps > 0.0:
        nb_frames = int(round(duration * fps))

    # Video codec
    codec_name = str(video_stream.get("codec_name", "unknown"))

    # Audio details
    has_audio = audio_stream is not None
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None

    if audio_stream is not None:
        audio_codec = audio_stream.get("codec_name")
        sr = audio_stream.get("sample_rate")
        if sr:
            try:
                audio_sample_rate = int(sr)
            except ValueError:
                pass
        ch = audio_stream.get("channels")
        if ch:
            try:
                audio_channels = int(ch)
            except ValueError:
                pass

    # File size
    file_size = 0
    fmt_size = format_info.get("size")
    if fmt_size:
        try:
            file_size = int(fmt_size)
        except ValueError:
            pass
    if file_size <= 0:
        try:
            file_size = os.path.getsize(video_path)
        except OSError:
            file_size = 0

    # Bitrate
    bitrate = None
    bitrate_str = format_info.get("bit_rate") or video_stream.get("bit_rate")
    if bitrate_str:
        try:
            bitrate = int(bitrate_str)
        except ValueError:
            pass

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        nb_frames=nb_frames,
        codec_name=codec_name,
        has_audio=has_audio,
        file_size=file_size,
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        bitrate=bitrate,
    )
