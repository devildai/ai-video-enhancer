"""app/media/stream.py - Streaming rawvideo decoder and encoder."""

import os
import subprocess
from typing import Optional, Iterator
import numpy as np

from app.media.probe import probe_video, _find_binary, VideoMetadata


class VideoFrameDecoder:
    """Streaming video decoder piping rawvideo rgb24 frames via FFmpeg.

    Can be used as a Python iterator/generator or context manager.
    Yields frames as np.ndarray with shape (height, width, 3) and dtype uint8.
    """

    def __init__(self, video_path: str):
        self.video_path = str(os.path.abspath(video_path))
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.meta: VideoMetadata = probe_video(self.video_path)
        self.width: int = self.meta.width
        self.height: int = self.meta.height
        self.fps: float = self.meta.fps
        self.duration: float = self.meta.duration
        self.nb_frames: int = self.meta.nb_frames

        self._frame_bytes: int = self.width * self.height * 3
        self._process: Optional[subprocess.Popen] = None
        self._closed: bool = False
        self._generator: Optional[Iterator[np.ndarray]] = None

    def __enter__(self) -> "VideoFrameDecoder":
        self._start_process()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _start_process(self) -> None:
        if self._process is not None:
            return

        ffmpeg_bin = _find_binary("ffmpeg")
        cmd = [
            ffmpeg_bin,
            "-v", "error",
            "-i", self.video_path,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-"
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=10 ** 7
        )

    def _read_exact(self, n_bytes: int) -> bytes:
        if self._process is None or self._process.stdout is None:
            return b""
        chunks = []
        bytes_read = 0
        while bytes_read < n_bytes:
            chunk = self._process.stdout.read(n_bytes - bytes_read)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
        return b"".join(chunks)

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._closed:
            raise RuntimeError("VideoFrameDecoder has already been closed")
        if self._process is None:
            self._start_process()

        try:
            while True:
                buf = self._read_exact(self._frame_bytes)
                if not buf or len(buf) < self._frame_bytes:
                    break
                frame = np.frombuffer(
                    buf, dtype=np.uint8
                ).reshape((self.height, self.width, 3))
                yield frame
        finally:
            self.close()

    def close(self) -> None:
        """Safely closes stdout and terminates the decoder subprocess."""
        if self._closed:
            return
        self._closed = True

        if self._process is not None:
            if self._process.stdout:
                try:
                    self._process.stdout.close()
                except Exception:
                    pass
            if self._process.stderr:
                try:
                    self._process.stderr.close()
                except Exception:
                    pass

            if self._process.poll() is None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=1.0)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        self._process.kill()
                        self._process.wait(timeout=1.0)
                    except Exception:
                        pass
            self._process = None


class VideoFrameEncoder:
    """Streaming video encoder piping rawvideo rgb24 frames to H.264 MP4.

    Enforces even width and height (w - w%2, h - h%2). Automatically muxes
    audio if provided. Encodes using libx264 with yuv420p, crf 19, medium
    preset, and faststart.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_path: Optional[str] = None
    ):
        # Enforce even dimensions required for H.264 yuv420p encoding
        self.width: int = width - (width % 2)
        self.height: int = height - (height % 2)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Invalid dimensions after even-alignment: "
                f"{self.width}x{self.height}"
            )

        self.output_path: str = str(os.path.abspath(output_path))
        self.fps: float = float(fps)
        if self.fps <= 0.0:
            raise ValueError(f"Invalid FPS: {fps}")

        self.audio_path: Optional[str] = (
            str(os.path.abspath(audio_path)) if audio_path else None
        )
        if self.audio_path and not os.path.exists(self.audio_path):
            raise FileNotFoundError(
                f"Specified audio file does not exist: {audio_path}"
            )

        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self._process: Optional[subprocess.Popen] = None
        self._frames_written: int = 0
        self._closed: bool = False

    def __enter__(self) -> "VideoFrameEncoder":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close(raise_on_error=(exc_type is None))

    def start(self) -> None:
        """Spawns the FFmpeg encoder subprocess."""
        if self._process is not None:
            return

        ffmpeg_bin = _find_binary("ffmpeg")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.6f}",
            "-i", "-"
        ]

        if self.audio_path:
            cmd.extend(["-i", self.audio_path])

        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "19",
            "-preset", "medium"
        ])

        if self.audio_path:
            if self.audio_path.lower().endswith((".m4a", ".aac")):
                cmd.extend(["-c:a", "copy"])
            else:
                cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            cmd.append("-shortest")

        cmd.extend(["-movflags", "+faststart", self.output_path])

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 ** 7
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """Writes an RGB24 frame to the encoder pipe.

        Args:
            frame: np.ndarray of shape (H, W, 3) in RGB order.
        """
        if self._closed:
            raise RuntimeError(
                "Cannot write frame: VideoFrameEncoder is closed"
            )
        if self._process is None:
            self.start()

        h, w = frame.shape[:2]
        # Crop or adjust if dimensions exceed even target dimensions
        if h != self.height or w != self.width:
            if h >= self.height and w >= self.width:
                frame = frame[:self.height, :self.width, :]
            else:
                pad_h = max(0, self.height - h)
                pad_w = max(0, self.width - w)
                frame = np.pad(
                    frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"
                )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        raw_bytes = frame.tobytes()
        try:
            assert self._process is not None
            assert self._process.stdin is not None
            self._process.stdin.write(raw_bytes)
            self._frames_written += 1
        except (BrokenPipeError, OSError) as exc:
            stderr_msg = ""
            if self._process and self._process.stderr:
                try:
                    stderr_msg = self._process.stderr.read().decode(
                        "utf-8", errors="replace"
                    )
                except Exception:
                    pass
            raise RuntimeError(
                f"FFmpeg encoder pipe broken: {stderr_msg or exc}"
            ) from exc

    def close(self, raise_on_error: bool = True) -> None:
        """Finalizes the video container and verifies exit status."""
        if self._closed:
            return
        self._closed = True

        if self._process is None:
            return

        stdout_data, stderr_data = b"", b""
        try:
            if self._process.stdin:
                self._process.stdin.close()
            stdout_data, stderr_data = self._process.communicate(timeout=60.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            stdout_data, stderr_data = self._process.communicate()
        except Exception:
            pass

        retcode = self._process.returncode
        self._process = None

        if raise_on_error and retcode != 0:
            err_msg = (
                stderr_data.decode("utf-8", errors="replace").strip()
                or f"Exit code {retcode}"
            )
            raise RuntimeError(f"FFmpeg encoder failed: {err_msg}")
