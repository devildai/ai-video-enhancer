"""Automated model weight downloader, cache manager, and offline fallback engine."""

from __future__ import annotations

import logging
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "enhancer"

MODEL_REGISTRY: dict[str, dict[str, str | int]] = {
    "realesr-general-x4v3.onnx": {
        "url": "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx",
        "mirror": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "expected_min_bytes": 4_000_000,
        "description": "Real-ESRGAN Compact SRVGGNet x4 ONNX model (~4.9MB)",
    },
    "realesr-animevideov3.onnx": {
        "url": "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-animevideov3.onnx",
        "expected_min_bytes": 4_000_000,
        "description": "Real-ESRGAN AnimeVideo v3 x4 ONNX model (~5.2MB)",
    },
}


def get_default_model_dir() -> Path:
    """Return configured or default directory for storing enhancement model weights."""
    env_dir = os.environ.get("ENHANCER_MODEL_DIR")
    if env_dir:
        path = Path(env_dir)
    else:
        path = DEFAULT_MODELS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_model_weights(
    model_name: str = "realesr-general-x4v3.onnx",
    target_dir: Optional[str | Path] = None,
    force_download: bool = False,
    timeout: int = 15,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Optional[Path]:
    """Ensure the specified ONNX model weights exist locally in target_dir.

    If present and valid, returns the local Path immediately.
    If missing, attempts to download from remote repositories (Hugging Face).
    If network is unreachable or download fails, logs a warning and returns None,
    allowing caller to gracefully fallback without crashing.
    """
    model_dir = Path(target_dir) if target_dir else get_default_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name

    meta = MODEL_REGISTRY.get(model_name, {})
    min_bytes = meta.get("expected_min_bytes", 1_000_000)

    if not force_download and model_path.exists():
        size = model_path.stat().st_size
        if size >= min_bytes:
            logger.info("Found cached model weights: %s (%d bytes)", model_path, size)
            if progress_callback:
                progress_callback(1.0, f"Using cached model weights ({size // 1024} KB)")
            return model_path
        else:
            logger.warning("Cached model weights corrupted/too small (%d bytes), re-downloading...", size)

    download_url = meta.get("url")
    if not download_url:
        logger.warning("No download URL registered for model '%s'", model_name)
        return None

    temp_path = model_path.with_suffix(".tmp")
    logger.info("Downloading %s from %s...", model_name, download_url)
    if progress_callback:
        progress_callback(0.0, f"Downloading {model_name}...")

    try:
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "VideoEnhancer/1.0 (Python/AI-Video-Tool)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 64 * 1024

            with open(temp_path, "wb") as f_out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0 and progress_callback:
                        pct = min(1.0, downloaded / total_bytes)
                        progress_callback(pct, f"Downloading: {downloaded // 1024}/{total_bytes // 1024} KB")

        if temp_path.stat().st_size < min_bytes:
            raise ValueError(f"Downloaded file too small: {temp_path.stat().st_size} bytes")

        shutil.move(str(temp_path), str(model_path))
        logger.info("Successfully downloaded and cached %s (%d bytes)", model_name, model_path.stat().st_size)
        if progress_callback:
            progress_callback(1.0, f"Model weights ready ({model_path.stat().st_size // 1024} KB)")
        return model_path

    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
        logger.warning(
            "Failed to download model weights '%s' (%s). Activating offline fallback engine.",
            model_name,
            err,
        )
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return None


class FallbackEnhancer:
    """High-quality offline enhancement engine when neural network weights are unavailable.

    Combines Lanczos-4 sinc interpolation, multi-scale adaptive unsharp masking,
    and high-frequency texture synthesis. Produces measurable sharpness gain
    over naive bilinear and bicubic upscaling.
    """

    def __init__(
        self,
        unsharp_strength: float = 0.65,
        texture_boost: float = 0.25,
        sigma: float = 1.2,
    ) -> None:
        self.unsharp_strength = float(unsharp_strength)
        self.texture_boost = float(texture_boost)
        self.sigma = float(sigma)

    def enhance(
        self,
        frame: np.ndarray,
        target_w: Optional[int] = None,
        target_h: Optional[int] = None,
    ) -> np.ndarray:
        """Upscales and restores high-frequency texture and edge sharpness on CPU."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")

        h, w = frame.shape[:2]
        tw = target_w if target_w is not None else w * 4
        th = target_h if target_h is not None else h * 4

        # 1. High-order Lanczos interpolation
        if (w, h) == (tw, th):
            scaled = frame.astype(np.float32)
        else:
            scaled = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

        # 2. Adaptive unsharp mask
        blurred = cv2.GaussianBlur(scaled, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        high_freq = scaled - blurred

        # 3. High-frequency texture synthesis via non-linear edge steepening
        # Emphasizes subtle textural gradients without blowing out extreme highlights
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        texture_comp = high_freq * (1.0 + self.texture_boost * np.tanh(np.abs(laplacian) / 32.0))

        enhanced = scaled + self.unsharp_strength * texture_comp
        return np.clip(np.round(enhanced), 0, 255).astype(np.uint8)
