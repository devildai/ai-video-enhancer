"""Real-ESRGAN Compact SRVGGNet ONNX engine for fast per-frame quality enhancement.

Restores genuine high-frequency edges and textures, removes compression artifacts,
and scales video frames using ONNX Runtime with CPUExecutionProvider and dynamic tiling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from app.ai.tiling import tile_process
from app.ai.weights import FallbackEnhancer, ensure_model_weights

logger = logging.getLogger(__name__)

_GLOBAL_SPEED_ENHANCER: Optional[SpeedEnhancer] = None


class SpeedEnhancer:
    """Real-ESRGAN Compact SRVGGNet ONNX inference engine running on CPU."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        tile_size: int = 256,
        overlap: int = 32,
        num_threads: Optional[int] = None,
    ) -> None:
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.session: Optional[ort.InferenceSession] = None
        self.fallback: Optional[FallbackEnhancer] = None
        self.input_name: str = "input"
        self.output_name: str = "output"
        self.scale: int = 4

        # Locate model weights
        actual_path: Optional[Path] = None
        if model_path is not None:
            p = Path(model_path)
            if p.exists() and p.stat().st_size > 1_000_000:
                actual_path = p
            else:
                logger.warning("Specified model path %s not found or invalid. Using fallback.", model_path)
                actual_path = None
        else:
            actual_path = ensure_model_weights("realesr-general-x4v3.onnx")

        if actual_path and actual_path.exists():
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = num_threads or min(4, os.cpu_count() or 4)
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                self.session = ort.InferenceSession(
                    str(actual_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                logger.info(
                    "Initialized SpeedEnhancer with ONNX model %s (input=%s, output=%s)",
                    actual_path.name,
                    self.input_name,
                    self.output_name,
                )
            except Exception as err:
                logger.warning("Failed to initialize ONNX Runtime session: %s. Using fallback.", err)
                self.session = None

        if self.session is None:
            logger.info("Initializing fallback Lanczos + unsharp mask texture synthesis engine")
            self.fallback = FallbackEnhancer()

    def _forward_tile(self, tile: np.ndarray) -> np.ndarray:
        """Run single tile tensor through ONNX model (scale 4x)."""
        if self.session is None:
            raise RuntimeError("InferenceSession is not initialized")

        # Tile shape: (H, W, 3) in uint8
        tensor = (tile.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        out = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        out_img = (np.clip(out.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
        return out_img

    def enhance(
        self,
        frame: np.ndarray,
        target_w: Optional[int] = None,
        target_h: Optional[int] = None,
    ) -> np.ndarray:
        """Enhance single video frame and fit to target dimensions.

        Args:
            frame: Input uint8 RGB (or BGR) frame of shape (H, W, 3).
            target_w: Optional target width in pixels.
            target_h: Optional target height in pixels.

        Returns:
            Enhanced frame of shape (target_h, target_w, 3) or (H * 4, W * 4, 3).
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")

        h, w = frame.shape[:2]

        if self.session is not None:
            # Neural enhancement at 4x scale
            if h <= self.tile_size and w <= self.tile_size:
                enhanced_4x = self._forward_tile(frame)
            else:
                enhanced_4x = tile_process(
                    frame,
                    self._forward_tile,
                    tile_size=self.tile_size,
                    overlap=self.overlap,
                    scale=self.scale,
                )

            # Fit to target dimensions if requested
            if target_w is not None and target_h is not None:
                cur_h, cur_w = enhanced_4x.shape[:2]
                if (cur_w, cur_h) == (target_w, target_h):
                    return enhanced_4x
                elif target_w <= cur_w and target_h <= cur_h:
                    # High quality downsampling from 4x super-resolution
                    return cv2.resize(enhanced_4x, (target_w, target_h), interpolation=cv2.INTER_AREA)
                else:
                    return cv2.resize(enhanced_4x, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            return enhanced_4x

        # Offline fallback path
        assert self.fallback is not None
        return self.fallback.enhance(frame, target_w=target_w, target_h=target_h)


def enhance_frame_speed(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Convenience function matching PROJECT.md interface contract for Speed mode.

    Applies Real-ESRGAN Compact SRVGGNet with tiling if required,
    outputting sharp detail at target dimensions.
    """
    global _GLOBAL_SPEED_ENHANCER
    if _GLOBAL_SPEED_ENHANCER is None:
        _GLOBAL_SPEED_ENHANCER = SpeedEnhancer()
    return _GLOBAL_SPEED_ENHANCER.enhance(frame, target_w=target_w, target_h=target_h)
