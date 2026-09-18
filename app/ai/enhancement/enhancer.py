"""Unified Video Quality Enhancement facade and resolution scaler.

Supports selectable modes ("speed" vs "quality") and target resolution presets
("Original", "1080p", "2K", "4K"), preserving aspect ratio and ensuring even dimensions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from app.ai.enhancement.quality_mode import TemporalVideoEnhancer
from app.ai.enhancement.speed_mode import SpeedEnhancer

logger = logging.getLogger(__name__)

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
    "2160p": (3840, 2160),
    "720p": (1280, 720),
}


class VideoEnhancer:
    """Unified AI video quality enhancement processor.

    Coordinates resolution scaling, model inference, dynamic tiling,
    and temporal stabilization.
    """

    def __init__(
        self,
        mode: str = "speed",
        target_resolution: str = "1080p",
        model_path: Optional[str | Path] = None,
        tile_size: int = 256,
        overlap: int = 32,
    ) -> None:
        self.mode = mode.lower().strip()
        if self.mode not in ("speed", "quality"):
            raise ValueError(f"Unknown enhancement mode '{mode}'. Expected 'speed' or 'quality'.")

        self.target_resolution = target_resolution
        self.model_path = model_path
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)

        # Core spatial engine shared across modes
        self._speed_engine = SpeedEnhancer(
            model_path=model_path,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )

        # Temporal engine instantiated once target resolution is determined
        self._temporal_engine: Optional[TemporalVideoEnhancer] = None
        self._cached_target_dims: Optional[tuple[int, int]] = None

    def calculate_target_dimensions(self, input_w: int, input_h: int) -> tuple[int, int]:
        """Calculate output dimensions preserving aspect ratio and ensuring even numbers.

        Args:
            input_w: Input width in pixels.
            input_h: Input height in pixels.

        Returns:
            Tuple of (target_w, target_h), both guaranteed to be even integers.
        """
        preset_key = self.target_resolution.lower().strip()

        # Original resolution: preserve dimensions, ensure even
        if preset_key in ("original", "source", "none", ""):
            tw = max(2, input_w - (input_w % 2))
            th = max(2, input_h - (input_h % 2))
            return (tw, th)

        if preset_key not in RESOLUTION_PRESETS:
            logger.warning("Unrecognized resolution preset '%s', falling back to 1080p", preset_key)
            preset_key = "1080p"

        ref_w, ref_h = RESOLUTION_PRESETS[preset_key]

        # Handle portrait orientation
        if input_h > input_w:
            ref_w, ref_h = ref_h, ref_w

        # If source is already within ~5% of standard 16:9, map directly to preset
        aspect_in = float(input_w) / float(input_h)
        aspect_ref = float(ref_w) / float(ref_h)
        if abs(aspect_in - aspect_ref) < 0.05:
            return (ref_w, ref_h)

        # Proportional bounding-box fit
        scale = min(float(ref_w) / float(input_w), float(ref_h) / float(input_h))
        tw = int(round(input_w * scale))
        th = int(round(input_h * scale))

        # Enforce strictly even dimensions for video codecs (H.264/H.265 yuv420p)
        tw = max(2, tw - (tw % 2))
        th = max(2, th - (th % 2))
        return (tw, th)

    def _ensure_engines(self, input_w: int, input_h: int) -> tuple[int, int]:
        """Ensure engines are configured for current frame dimensions."""
        if self._cached_target_dims is None:
            self._cached_target_dims = self.calculate_target_dimensions(input_w, input_h)

        target_w, target_h = self._cached_target_dims
        if self.mode == "quality" and self._temporal_engine is None:
            self._temporal_engine = TemporalVideoEnhancer(
                target_w=target_w,
                target_h=target_h,
                speed_enhancer=self._speed_engine,
            )

        return target_w, target_h

    def enhance_frame(self, frame: np.ndarray, is_cut: bool = False) -> np.ndarray:
        """Enhance a single video frame to target resolution using selected mode.

        Args:
            frame: Input uint8 RGB frame.
            is_cut: True if an abrupt scene cut occurred prior to this frame.

        Returns:
            Enhanced frame of shape (target_h, target_w, 3).
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")

        h, w = frame.shape[:2]
        target_w, target_h = self._ensure_engines(w, h)

        if self.mode == "quality":
            assert self._temporal_engine is not None
            return self._temporal_engine.enhance_frame(frame, is_cut=is_cut)
        else:
            return self._speed_engine.enhance(frame, target_w=target_w, target_h=target_h)

    def enhance_sequence(
        self,
        frames: Iterable[np.ndarray],
        scene_cuts: Optional[Iterable[bool]] = None,
    ) -> Iterator[np.ndarray]:
        """Streaming generator yielding enhanced frames sequentially.

        Args:
            frames: Iterable of input uint8 RGB frames.
            scene_cuts: Optional iterable of boolean scene cut indicators.

        Yields:
            Enhanced uint8 RGB frames at target resolution.
        """
        cuts_iter = iter(scene_cuts) if scene_cuts is not None else None

        for frame in frames:
            is_cut = next(cuts_iter, False) if cuts_iter is not None else False
            yield self.enhance_frame(frame, is_cut=is_cut)

    def reset(self) -> None:
        """Reset temporal state buffer."""
        if self._temporal_engine is not None:
            self._temporal_engine.reset()
        self._cached_target_dims = None
