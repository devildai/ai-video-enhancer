"""E2E Test Helpers and Observability Oracles for AI Video Enhancement Tool."""
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np


def run_ffprobe(file_path: Path | str) -> dict[str, Any]:
    """Run ffprobe on a file and return parsed stream and format JSON."""
    path = str(Path(file_path).resolve())
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({result.returncode}) for {path}: {result.stderr}")
    return json.loads(result.stdout)


def get_video_stream(probe_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Retrieve first video stream info from probe data."""
    streams = probe_data.get("streams", [])
    for s in streams:
        if s.get("codec_type") == "video":
            return s
    return None


def get_audio_stream(probe_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Retrieve first audio stream info from probe data."""
    streams = probe_data.get("streams", [])
    for s in streams:
        if s.get("codec_type") == "audio":
            return s
    return None


def parse_fractional_fps(rate_str: str) -> float:
    """Parse '60/1' or '30000/1001' into float FPS."""
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f != 0 else 0.0
    return float(rate_str)


def inspect_video_properties(file_path: Path | str) -> dict[str, Any]:
    """Extract standard media properties used for test assertions."""
    probe = run_ffprobe(file_path)
    v_stream = get_video_stream(probe)
    a_stream = get_audio_stream(probe)
    fmt = probe.get("format", {})

    width = int(v_stream.get("width", 0)) if v_stream else 0
    height = int(v_stream.get("height", 0)) if v_stream else 0
    fps = parse_fractional_fps(v_stream.get("r_frame_rate", "0/1")) if v_stream else 0.0
    duration = float(fmt.get("duration", 0.0))
    video_codec = v_stream.get("codec_name", "") if v_stream else ""
    pix_fmt = v_stream.get("pix_fmt", "") if v_stream else ""

    has_audio = a_stream is not None
    audio_codec = a_stream.get("codec_name", "") if a_stream else None
    audio_duration = float(a_stream.get("duration", 0.0)) if (a_stream and "duration" in a_stream) else (duration if has_audio else 0.0)

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "video_codec": video_codec,
        "pix_fmt": pix_fmt,
        "has_audio": has_audio,
        "audio_codec": audio_codec,
        "audio_duration": audio_duration,
        "format_name": fmt.get("format_name", ""),
        "file_size": int(fmt.get("size", 0)),
    }


def parse_sse_events(raw_sse_text: str) -> list[dict[str, Any]]:
    """Parse raw Server-Sent Events stream text into list of JSON event dictionaries."""
    events = []
    lines = raw_sse_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            payload_str = line[len("data:"):].strip()
            if payload_str:
                try:
                    events.append(json.loads(payload_str))
                except json.JSONDecodeError:
                    events.append({"raw": payload_str})
    return events


def extract_frame_cv2(video_path: Path | str, frame_index: int = 0) -> np.ndarray:
    """Read specific frame as uint8 BGR numpy array using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path} via OpenCV")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame at index {frame_index} from {video_path}")
    return frame


def calculate_image_sharpness(img: np.ndarray) -> float:
    """Calculate variance of the Laplacian as an authoritative sharpness metric."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def is_faststart_mp4(file_path: Path | str) -> bool:
    """Verify that an MP4 file has moov atom before mdat (FastStart / web-optimized)."""
    path = Path(file_path)
    with open(path, "rb") as f:
        header = f.read(4096)
    # Search for moov atom in early header bytes
    moov_pos = header.find(b"moov")
    mdat_pos = header.find(b"mdat")
    if moov_pos != -1 and mdat_pos != -1:
        return moov_pos < mdat_pos
    elif moov_pos != -1:
        return True
    return False
