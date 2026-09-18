"""AI Frame Interpolation Module.

Provides:
- detect_scene_cut: L1 difference and HSV histogram scene cut detector.
- interpolate_flow_dis / DISFlowInterpolator: Ultra-fast OpenCV DIS Optical Flow engine.
- RIFEInterpolator: RIFE v4 Timestep ONNX neural interpolation engine.
- FrameInterpolator: Arbitrary FPS multiplier with in-flight scene cut protection.
- interpolate_pair: Point-to-point pair interpolation.
"""

from app.ai.interpolation.scene_cut import detect_scene_cut, compute_scene_metrics
from app.ai.interpolation.flow_dis import interpolate_flow_dis, DISFlowInterpolator
from app.ai.interpolation.rife import (
    RIFEInterpolator,
    download_rife_model,
    get_rife_model_path,
)
from app.ai.interpolation.interpolator import FrameInterpolator, interpolate_pair

__all__ = [
    "detect_scene_cut",
    "compute_scene_metrics",
    "interpolate_flow_dis",
    "DISFlowInterpolator",
    "RIFEInterpolator",
    "download_rife_model",
    "get_rife_model_path",
    "FrameInterpolator",
    "interpolate_pair",
]
