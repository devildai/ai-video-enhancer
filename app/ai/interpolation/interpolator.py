"""Unified Frame Interpolation pipeline with arbitrary FPS multiplication and scene-cut snapping.

Provides FrameInterpolator to convert any source frame rate (e.g. 24fps, 30fps)
to arbitrary target frame rates (e.g. 60fps, 120fps) via RIFE neural interpolation
or OpenCV DIS optical flow, with in-flight scene-cut protection.
"""

from typing import Iterable, Sequence, Generator, List, Tuple, Optional
import logging
import numpy as np

from app.ai.interpolation.scene_cut import detect_scene_cut
from app.ai.interpolation.flow_dis import DISFlowInterpolator, interpolate_flow_dis
from app.ai.interpolation.rife import RIFEInterpolator

logger = logging.getLogger("ai.interpolation.interpolator")


class FrameInterpolator:
    """Processes video frames to achieve arbitrary target FPS using AI interpolation."""

    def __init__(
        self,
        target_fps: float,
        source_fps: float,
        engine: str = "auto",
        scene_cut_threshold: float = 30.0,
    ) -> None:
        """Initialize FrameInterpolator.

        Args:
            target_fps: Desired output frame rate (e.g. 60.0, 120.0).
            source_fps: Source video frame rate (e.g. 24.0, 30.0).
            engine: Interpolation backend ('auto', 'rife', or 'flow_dis'/'dis').
                'auto': Uses RIFE ONNX if available, falling back to DIS flow.
                'rife': Uses RIFE v4 ONNX model (falls back to DIS if unavailable).
                'flow_dis': Uses OpenCV DIS Optical Flow directly.
            scene_cut_threshold: Sensitivity threshold for scene cut detection.
        """
        if source_fps <= 0 or target_fps <= 0:
            raise ValueError(f"FPS values must be positive. Got source_fps={source_fps}, target_fps={target_fps}")

        self.source_fps = float(source_fps)
        self.target_fps = float(target_fps)
        self.factor = self.target_fps / self.source_fps
        self.engine_name = engine.lower()
        self.scene_cut_threshold = float(scene_cut_threshold)

        # Initialize chosen interpolation backend
        if self.engine_name in ("flow_dis", "dis"):
            self.engine = DISFlowInterpolator()
        elif self.engine_name == "rife":
            self.engine = RIFEInterpolator(auto_download=True)
        else:  # "auto"
            self.engine = RIFEInterpolator(auto_download=True)

    def interpolate_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        timestep: float,
        is_scene_cut: Optional[bool] = None,
    ) -> np.ndarray:
        """Interpolate intermediate frame between frame1 and frame2 at timestep t in [0, 1].

        If is_scene_cut is True (or detected as True), snaps to nearest keyframe
        to eliminate morphing/stretching artifacts.
        """
        if timestep <= 0.001:
            return frame1.copy()
        if timestep >= 0.999:
            return frame2.copy()

        # Detect scene cut if not explicitly specified
        if is_scene_cut is None:
            is_scene_cut = detect_scene_cut(frame1, frame2, threshold=self.scene_cut_threshold)

        if is_scene_cut:
            # Snap to nearest keyframe
            return frame1.copy() if timestep < 0.5 else frame2.copy()

        return self.engine.interpolate(frame1, frame2, timestep)

    def calculate_timesteps(self, num_src_frames: int) -> List[Tuple[int, int, float]]:
        """Calculate mapping from target frame indices to source frame intervals and timesteps.

        Args:
            num_src_frames: Number of source frames.

        Returns:
            List of (target_frame_idx, source_interval_idx, fractional_timestep_t).
        """
        if num_src_frames <= 0:
            return []

        total_out_frames = int(round(num_src_frames * self.factor))
        results: List[Tuple[int, int, float]] = []

        for k in range(total_out_frames):
            pos = k / self.factor
            src_idx = int(pos)
            t = pos - src_idx

            if src_idx >= num_src_frames - 1:
                src_idx = num_src_frames - 1
                t = 0.0

            results.append((k, src_idx, float(t)))

        return results

    def interpolate_sequence(
        self,
        frames: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        """Interpolate a list of frames to the target frame rate.

        Args:
            frames: Sequence of uint8 RGB frames.

        Returns:
            List of interpolated frames at exact target_fps count.
        """
        return list(self.interpolate_stream(frames, total_frames=len(frames)))

    def interpolate_stream(
        self,
        frame_stream: Iterable[np.ndarray],
        total_frames: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """Streamingly interpolate frames to the target frame rate.

        Maintains an in-flight pair buffer and checks scene cuts on each transition.
        Yields frames at the target frame rate.
        """
        # If frame rate is unchanged (1.0x factor), pass frames straight through
        if abs(self.factor - 1.0) < 1e-4:
            for frame in frame_stream:
                yield frame
            return

        k = 0
        i = 0
        prev_frame: Optional[np.ndarray] = None
        count = 0

        for frame in frame_stream:
            count += 1
            if prev_frame is None:
                prev_frame = frame
                i = 0
                continue

            # We have consecutive pair (prev_frame, frame) representing source interval [i, i+1]
            is_cut = detect_scene_cut(prev_frame, frame, threshold=self.scene_cut_threshold)

            while True:
                tau_k = k / self.target_fps
                src_pos = tau_k * self.source_fps
                if src_pos >= i + 1:
                    break

                t = float(src_pos - i)
                out_frame = self.interpolate_pair(prev_frame, frame, t, is_scene_cut=is_cut)
                yield out_frame
                k += 1

            prev_frame = frame
            i += 1

        # Handle final frame(s) at stream termination
        if prev_frame is not None:
            expected_total = (
                int(round(total_frames * self.factor))
                if total_frames is not None
                else int(round(count * self.factor))
            )
            while k < expected_total:
                yield prev_frame.copy()
                k += 1

    def __call__(
        self,
        frames: Iterable[np.ndarray],
        total_frames: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """Allow calling interpolator instance directly on a frame iterable."""
        return self.interpolate_stream(frames, total_frames=total_frames)


def interpolate_pair(
    frame1: np.ndarray,
    frame2: np.ndarray,
    timestep: float,
    is_scene_cut: bool = False,
    engine: str = "auto",
) -> np.ndarray:
    """Convenience function to interpolate between two frames at timestep t.

    Args:
        frame1: First frame (uint8 RGB).
        frame2: Second frame (uint8 RGB).
        timestep: Normalized timestep t in [0.0, 1.0].
        is_scene_cut: If True, snaps to nearest frame instead of interpolating.
        engine: Backend engine ('auto', 'rife', or 'flow_dis').

    Returns:
        Synthesized frame.
    """
    if is_scene_cut:
        return frame1.copy() if timestep < 0.5 else frame2.copy()
    if timestep <= 0.001:
        return frame1.copy()
    if timestep >= 0.999:
        return frame2.copy()

    if engine in ("flow_dis", "dis"):
        return interpolate_flow_dis(frame1, frame2, timestep)
    else:
        # Default / auto RIFE with fallback
        interpolator = RIFEInterpolator(auto_download=True)
        return interpolator.interpolate(frame1, frame2, timestep)
