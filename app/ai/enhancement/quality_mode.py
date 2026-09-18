"""Temporal Video Super-Resolution (T-VSR) engine for flicker-free enhancement.

Maintains rolling temporal state, computes dense backward optical flow via OpenCV DIS,
evaluates photometric occlusion masks, and aggregates temporal detail to eliminate
high-frequency GAN boiling and flickering across consecutive video frames.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from app.ai.enhancement.speed_mode import SpeedEnhancer

logger = logging.getLogger(__name__)


class TemporalVideoEnhancer:
    """Stateful Temporal Video Super-Resolution engine.

    Conforms to the PROJECT.md interface contract:
    - Maintains rolling feature/frame buffer.
    - Enhances frame F_t using spatial features and backward optical flow from F_{t-1}
      with occlusion masking to eliminate frame-to-frame flicker.
    """

    def __init__(
        self,
        target_w: int,
        target_h: int,
        speed_enhancer: Optional[SpeedEnhancer] = None,
        alpha: float = 0.35,
        sigma: float = 0.08,
        flow_preset: int = cv2.DISOPTICAL_FLOW_PRESET_FAST,
    ) -> None:
        self.target_w = int(target_w)
        self.target_h = int(target_h)
        self.alpha = float(alpha)
        self.sigma = float(sigma)

        self.speed_enhancer = speed_enhancer or SpeedEnhancer()
        self.dis = cv2.DISOpticalFlow_create(flow_preset)

        self.prev_lr: Optional[np.ndarray] = None
        self.prev_hr: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset temporal rolling frame buffers (e.g. on scene transitions or new video)."""
        self.prev_lr = None
        self.prev_hr = None

    def enhance_frame(self, frame: np.ndarray, is_cut: bool = False) -> np.ndarray:
        """Enhance frame F_t incorporating aligned temporal detail from F_{t-1}.

        Args:
            frame: Input uint8 RGB frame of shape (H, W, 3).
            is_cut: True if an abrupt scene transition was detected before this frame.

        Returns:
            Temporally stabilized enhanced frame of shape (target_h, target_w, 3).
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")

        h_lr, w_lr = frame.shape[:2]

        # Scene cut or first frame: Initialize new temporal sequence
        if is_cut or self.prev_lr is None or self.prev_hr is None:
            spatial_hr = self.speed_enhancer.enhance(frame, self.target_w, self.target_h)
            self.prev_lr = frame.copy()
            self.prev_hr = spatial_hr.copy()
            return spatial_hr

        # 1. Spatial restoration of current frame F_t
        spatial_hr = self.speed_enhancer.enhance(frame, self.target_w, self.target_h)

        # 2. Dense backward optical flow F_{t -> t-1}
        gray_cur = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        gray_prev = cv2.cvtColor(self.prev_lr, cv2.COLOR_RGB2GRAY) if self.prev_lr.ndim == 3 else self.prev_lr
        flow = self.dis.calc(gray_cur, gray_prev, None)

        # 3. Backward warping in LR space to compute photometric error
        grid_x, grid_y = np.meshgrid(
            np.arange(w_lr, dtype=np.float32),
            np.arange(h_lr, dtype=np.float32),
        )
        map_lr_x = grid_x + flow[..., 0]
        map_lr_y = grid_y + flow[..., 1]

        warped_lr = cv2.remap(
            self.prev_lr,
            map_lr_x,
            map_lr_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # 4. Photometric occlusion error & confidence gate G_t in [0, 1]
        diff = np.abs(frame.astype(np.float32) - warped_lr.astype(np.float32))
        err = np.mean(diff, axis=-1) / 255.0 if frame.ndim == 3 else diff / 255.0
        gate = np.exp(-(err**2) / (2.0 * (self.sigma**2)))

        # 5. Scale flow to high-resolution domain and backward warp prev_hr
        scale_x = float(self.target_w) / float(w_lr)
        scale_y = float(self.target_h) / float(h_lr)

        flow_hr_x = cv2.resize(flow[..., 0] * scale_x, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        flow_hr_y = cv2.resize(flow[..., 1] * scale_y, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        grid_hr_x, grid_hr_y = np.meshgrid(
            np.arange(self.target_w, dtype=np.float32),
            np.arange(self.target_h, dtype=np.float32),
        )
        map_hr_x = grid_hr_x + flow_hr_x
        map_hr_y = grid_hr_y + flow_hr_y

        warped_hr = cv2.remap(
            self.prev_hr,
            map_hr_x,
            map_hr_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # 6. High-resolution confidence gate
        gate_hr = cv2.resize(gate, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        if frame.ndim == 3:
            gate_hr = gate_hr[..., np.newaxis]

        # 7. Temporal consistency aggregation
        weight = self.alpha * gate_hr
        stabilized = (1.0 - weight) * spatial_hr.astype(np.float32) + weight * warped_hr.astype(np.float32)
        stabilized_uint8 = np.clip(np.round(stabilized), 0, 255).astype(np.uint8)

        # 8. Update rolling frame buffer
        self.prev_lr = frame.copy()
        self.prev_hr = stabilized_uint8.copy()

        return stabilized_uint8
