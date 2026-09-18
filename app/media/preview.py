"""app/media/preview.py - Sample frame extraction and preview."""

import os
import subprocess
from app.media.probe import probe_video, _find_binary


def extract_sample_frame(
    video_path: str,
    timestamp_s: float,
    output_path: str
) -> str:
    """Extracts a high quality JPEG frame from the video at timestamp_s.

    Uses FFmpeg with high JPEG quality (-q:v 2) and fast frame seeking.
    Clamps timestamp to ensure it falls within valid media duration bounds.

    Args:
        video_path: Path to the source video file.
        timestamp_s: Timestamp in seconds.
        output_path: Path where the extracted JPEG should be saved.

    Returns:
        Absolute path to the extracted JPEG image file.

    Raises:
        FileNotFoundError: If source video file does not exist.
        ValueError: If video file has invalid duration or format.
        RuntimeError: If FFmpeg fails to extract the frame.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video file not found: {video_path}")

    meta = probe_video(video_path)

    # Bound timestamp within [0.0, duration]
    ts = float(timestamp_s)
    if ts < 0.0:
        ts = 0.0
    if meta.duration > 0.0 and ts >= meta.duration:
        ts = max(0.0, meta.duration - 0.1)

    abs_output = os.path.abspath(output_path)
    out_dir = os.path.dirname(abs_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ffmpeg_bin = _find_binary("ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v", "error",
        "-ss", f"{ts:.4f}",
        "-i", str(os.path.abspath(video_path)),
        "-frames:v", "1",
        "-q:v", "2",
        abs_output
    ]

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )

    if (
        res.returncode != 0
        or not os.path.exists(abs_output)
        or os.path.getsize(abs_output) == 0
    ):
        err_msg = (
            res.stderr.strip() or f"FFmpeg exited with code {res.returncode}"
        )
        raise RuntimeError(
            f"Failed to extract sample frame at {ts:.2f}s: {err_msg}"
        )

    return abs_output
