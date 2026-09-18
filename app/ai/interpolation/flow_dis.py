"""Ultra-fast OpenCV DIS Optical Flow interpolation engine.

Computes bidirectional optical flow using cv2.DISOpticalFlow (PRESET_FAST)
and warps intermediate frames for continuous timestep t in (0, 1) with
forward-backward consistency blending.
"""

from typing import Optional
import numpy as np
import cv2


class DISFlowInterpolator:
    """Optical flow frame interpolator using OpenCV DIS Optical Flow.

    Provides fast (~20-35ms per 480p frame on CPU) frame synthesis with
    forward-backward consistency checking and occlusion-aware blending.
    """

    def __init__(self, preset: int = cv2.DISOPTICAL_FLOW_PRESET_FAST) -> None:
        """Initialize DIS optical flow instance.

        Args:
            preset: OpenCV DIS preset (FAST, MEDIUM, or ULTRAFAST).
        """
        self.dis = cv2.DISOpticalFlow_create(preset)

    def interpolate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        """Interpolate an intermediate frame between frame1 and frame2 at timestep t.

        Args:
            frame1: First frame (uint8 or float32, H x W x C or H x W).
            frame2: Second frame (uint8 or float32, H x W x C or H x W).
            timestep: Normalized temporal position t in [0.0, 1.0].
                t=0.0 corresponds to frame1, t=1.0 to frame2.

        Returns:
            Synthesized intermediate frame with same shape and dtype as input.
        """
        if timestep <= 0.001:
            return frame1.copy()
        if timestep >= 0.999:
            return frame2.copy()

        h, w = frame1.shape[:2]
        is_color = (len(frame1.shape) == 3 and frame1.shape[2] >= 3)

        # Convert to grayscale uint8 for optical flow estimation
        f1_uint8 = _to_uint8(frame1)
        f2_uint8 = _to_uint8(frame2)

        if is_color:
            g0 = cv2.cvtColor(f1_uint8, cv2.COLOR_RGB2GRAY)
            g1 = cv2.cvtColor(f2_uint8, cv2.COLOR_RGB2GRAY)
        else:
            g0 = f1_uint8 if len(f1_uint8.shape) == 2 else f1_uint8.squeeze()
            g1 = f2_uint8 if len(f2_uint8.shape) == 2 else f2_uint8.squeeze()

        # Bidirectional flow calculation
        # flow_01: motion vector from frame 0 to frame 1
        # flow_10: motion vector from frame 1 to frame 0
        flow_01 = self.dis.calc(g0, g1, None)
        flow_10 = self.dis.calc(g1, g0, None)

        # Coordinate grid
        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )

        # Warping map coordinates for intermediate frame at timestep t
        # Frame 0 is mapped forward towards t using scaled flow_10
        # Frame 1 is mapped backward towards t using scaled flow_01
        map_x0 = grid_x + timestep * flow_10[..., 0]
        map_y0 = grid_y + timestep * flow_10[..., 1]
        map_x1 = grid_x + (1.0 - timestep) * flow_01[..., 0]
        map_y1 = grid_y + (1.0 - timestep) * flow_01[..., 1]

        warp0 = cv2.remap(frame1, map_x0, map_y0, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warp1 = cv2.remap(frame2, map_x1, map_y1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Forward-backward consistency checking
        # Sample backward flow at (grid + flow_01)
        f10_at_f01_x = cv2.remap(flow_10[..., 0], grid_x + flow_01[..., 0], grid_y + flow_01[..., 1], cv2.INTER_LINEAR)
        f10_at_f01_y = cv2.remap(flow_10[..., 1], grid_x + flow_01[..., 0], grid_y + flow_01[..., 1], cv2.INTER_LINEAR)
        fb_err0 = np.sqrt((flow_01[..., 0] + f10_at_f01_x) ** 2 + (flow_01[..., 1] + f10_at_f01_y) ** 2)

        # Sample forward flow at (grid + flow_10)
        f01_at_f10_x = cv2.remap(flow_01[..., 0], grid_x + flow_10[..., 0], grid_y + flow_10[..., 1], cv2.INTER_LINEAR)
        f01_at_f10_y = cv2.remap(flow_01[..., 1], grid_x + flow_10[..., 0], grid_y + flow_10[..., 1], cv2.INTER_LINEAR)
        fb_err1 = np.sqrt((flow_10[..., 0] + f01_at_f10_x) ** 2 + (flow_10[..., 1] + f01_at_f10_y) ** 2)

        # Occlusion-aware soft confidence weights
        # High consistency error indicates occlusion or flow discontinuity -> lower weight
        conf0 = np.exp(-fb_err0 / 5.0) * (1.0 - timestep)
        conf1 = np.exp(-fb_err1 / 5.0) * timestep
        total_conf = conf0 + conf1 + 1e-6

        if is_color:
            weight0 = (conf0 / total_conf)[..., np.newaxis]
            weight1 = (conf1 / total_conf)[..., np.newaxis]
        else:
            weight0 = conf0 / total_conf
            weight1 = conf1 / total_conf

        blended = warp0.astype(np.float32) * weight0 + warp1.astype(np.float32) * weight1

        if frame1.dtype == np.uint8:
            return np.clip(blended, 0, 255).astype(np.uint8)
        return blended.astype(frame1.dtype)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert input image to uint8 if necessary."""
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        max_val = float(img.max()) if img.size > 0 else 1.0
        if max_val <= 1.0:
            return (img * 255.0).clip(0, 255).astype(np.uint8)
        return img.clip(0, 255).astype(np.uint8)
    return img.astype(np.uint8)


# Module-level default interpolator instance
_default_dis_interpolator: Optional[DISFlowInterpolator] = None


def interpolate_flow_dis(
    frame1: np.ndarray,
    frame2: np.ndarray,
    timestep: float,
) -> np.ndarray:
    """Convenience function for DIS optical flow frame interpolation.

    Args:
        frame1: First frame (uint8 or float32).
        frame2: Second frame (uint8 or float32).
        timestep: Temporal position t in [0.0, 1.0].

    Returns:
        Interpolated intermediate frame.
    """
    global _default_dis_interpolator
    if _default_dis_interpolator is None:
        _default_dis_interpolator = DISFlowInterpolator()
    return _default_dis_interpolator.interpolate(frame1, frame2, timestep)
