"""app/media/audio.py - Audio track extraction and synchronization."""

import os
import subprocess
from app.media.probe import probe_video, _find_binary


def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """Extracts the source audio track from a video container.

    First checks if the media contains an audio stream. If none exists,
    returns False. If an audio stream exists, attempts lossless bitstream
    copying (-vn -c:a copy). If copy fails, falls back to high-quality
    AAC re-encoding (-vn -c:a aac -b:a 192k).

    Args:
        video_path: Path to the input video file.
        output_audio_path: Destination path for the extracted audio.

    Returns:
        True if audio was extracted, False if the video has no audio.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        RuntimeError: If FFmpeg fails on both copy and AAC fallback.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    meta = probe_video(video_path)
    if not meta.has_audio:
        return False

    out_dir = os.path.dirname(os.path.abspath(output_audio_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ffmpeg_bin = _find_binary("ffmpeg")

    # 1. Attempt bitstream copy (-c:a copy)
    cmd_copy = [
        ffmpeg_bin,
        "-y",
        "-v", "error",
        "-i", str(os.path.abspath(video_path)),
        "-vn",
        "-c:a", "copy",
        str(os.path.abspath(output_audio_path))
    ]
    res_copy = subprocess.run(
        cmd_copy,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    if (
        res_copy.returncode == 0
        and os.path.exists(output_audio_path)
        and os.path.getsize(output_audio_path) > 0
    ):
        return True

    # 2. Fallback to AAC encoding (-c:a aac -b:a 192k)
    cmd_aac = [
        ffmpeg_bin,
        "-y",
        "-v", "error",
        "-i", str(os.path.abspath(video_path)),
        "-vn",
        "-c:a", "aac",
        "-b:a", "192k",
        str(os.path.abspath(output_audio_path))
    ]
    res_aac = subprocess.run(
        cmd_aac,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    if (
        res_aac.returncode == 0
        and os.path.exists(output_audio_path)
        and os.path.getsize(output_audio_path) > 0
    ):
        return True

    err_msg = (
        res_aac.stderr.strip()
        or res_copy.stderr.strip()
        or "FFmpeg audio extraction failed"
    )
    raise RuntimeError(
        f"Failed to extract audio from '{video_path}': {err_msg}"
    )


def align_audio_sync(
    video_duration: float,
    audio_path: str,
    output_audio_path: str
) -> str:
    """Synchronizes audio duration precisely with target video duration.

    Pads silence if audio is shorter than target video duration,
    and trims excess audio if longer, preventing duration drift.

    Args:
        video_duration: Target video duration in seconds.
        audio_path: Source audio file path.
        output_audio_path: Destination path for the synced audio file.

    Returns:
        Absolute path to the aligned audio file.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ValueError: If video_duration <= 0.
        RuntimeError: If FFmpeg alignment fails.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Source audio file not found: {audio_path}")
    if video_duration <= 0.0:
        raise ValueError(
            f"Target video duration must be positive, got {video_duration}"
        )

    out_dir = os.path.dirname(os.path.abspath(output_audio_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ffmpeg_bin = _find_binary("ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v", "error",
        "-i", str(os.path.abspath(audio_path)),
        "-af", "apad",
        "-t", f"{video_duration:.6f}",
        "-c:a", "aac",
        "-b:a", "192k",
        str(os.path.abspath(output_audio_path))
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
        or not os.path.exists(output_audio_path)
        or os.path.getsize(output_audio_path) == 0
    ):
        err_msg = (
            res.stderr.strip() or f"FFmpeg exited with code {res.returncode}"
        )
        raise RuntimeError(
            f"Failed to align audio sync for '{audio_path}': {err_msg}"
        )

    return os.path.abspath(output_audio_path)
