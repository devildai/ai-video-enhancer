"""Unit tests for AI frame interpolation pipeline.

Covers:
- Scene cut detection (hard cuts vs smooth camera pans)
- DIS optical flow bidirectional interpolation and consistency blending
- RIFE v4 Timestep ONNX neural interpolation and fallback handling
- FrameInterpolator continuous timestep mapping (24->60, 30->60, 30->120)
- In-flight scene cut protection and morphing elimination
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

from app.ai.interpolation.scene_cut import detect_scene_cut, compute_scene_metrics  # noqa: E402
from app.ai.interpolation.flow_dis import DISFlowInterpolator, interpolate_flow_dis  # noqa: E402
from app.ai.interpolation.rife import (  # noqa: E402
    RIFEInterpolator,
    get_rife_model_path,
    download_rife_model,
)
from app.ai.interpolation.interpolator import FrameInterpolator, interpolate_pair  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for test fixtures
# ---------------------------------------------------------------------------

def create_solid_frame(h: int, w: int, color_rgb: tuple) -> np.ndarray:
    """Create a uniform solid color frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = color_rgb
    return frame


def create_gradient_frame(h: int, w: int, offset_x: int = 0) -> np.ndarray:
    """Create a smooth gradient frame with an offset for pan simulation."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = np.mgrid[0:h, 0:w]
    frame[..., 0] = ((x + offset_x) * 255 / (w + 100)) % 256
    frame[..., 1] = (y * 255 / h).astype(np.uint8)
    frame[..., 2] = 128
    cv2.circle(frame, (80 + offset_x, 60), 25, (220, 200, 50), -1)
    return frame


def create_moving_dot_frame(h: int, w: int, dot_x: int, dot_y: int, radius: int = 15) -> np.ndarray:
    """Create a dark frame with a single moving white circular dot."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(frame, (dot_x, dot_y), radius, (255, 255, 255), -1)
    return frame


# ---------------------------------------------------------------------------
# 1. Scene Cut Detection Tests
# ---------------------------------------------------------------------------

class TestSceneCutDetection:
    """Tests for detect_scene_cut and compute_scene_metrics."""

    def test_identical_frames_not_cut(self):
        """Identical frames should yield zero difference and not be flagged as cuts."""
        red = create_solid_frame(100, 100, (255, 0, 0))
        assert not detect_scene_cut(red, red)

        black = np.zeros((80, 80, 3), dtype=np.uint8)
        assert not detect_scene_cut(black, black)

        l1, corr = compute_scene_metrics(red, red)
        assert l1 == 0.0
        assert corr == 1.0

    def test_hard_color_cut(self):
        """Abrupt transition from solid red to solid blue must be detected as a cut."""
        red = create_solid_frame(100, 100, (255, 0, 0))
        blue = create_solid_frame(100, 100, (0, 0, 255))
        assert detect_scene_cut(red, blue)

        l1, corr = compute_scene_metrics(red, blue)
        assert l1 > 30.0
        assert corr < 0.2

    def test_black_to_white_cut(self):
        """Extreme luminance shift (black to white) must be detected as a cut."""
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert detect_scene_cut(black, white)

    def test_green_to_orange_cut(self):
        """Abrupt scene change with different color distributions must be detected."""
        green = create_solid_frame(120, 120, (0, 255, 0))
        orange = create_solid_frame(120, 120, (255, 128, 0))
        assert detect_scene_cut(green, orange)

    def test_smooth_camera_pan_not_cut(self):
        """A smooth camera pan on natural gradient footage must NOT trigger a scene cut."""
        frame_a = create_gradient_frame(120, 160, offset_x=0)
        frame_b = create_gradient_frame(120, 160, offset_x=4)

        assert not detect_scene_cut(frame_a, frame_b)
        l1, corr = compute_scene_metrics(frame_a, frame_b)
        assert corr > 0.90  # Color histogram remains consistent across pans

    def test_detailed_texture_pan_not_cut(self):
        """Camera motion across a textured image should preserve correlation and avoid cut false-positives."""
        rng = np.random.RandomState(42)
        base = rng.randint(40, 220, (140, 180, 3), dtype=np.uint8)
        # Apply slight Gaussian blur to simulate real camera footage
        base = cv2.GaussianBlur(base, (5, 5), 0)

        pan1 = base[:, :-6]
        pan2 = base[:, 6:]
        assert not detect_scene_cut(pan1, pan2)

    def test_dimension_mismatch_or_none(self):
        """Dimension mismatch or None inputs should safely return True."""
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        f2 = np.zeros((100, 120, 3), dtype=np.uint8)
        assert detect_scene_cut(f1, f2)
        assert detect_scene_cut(None, f1)
        assert detect_scene_cut(f1, None)

    def test_grayscale_frames(self):
        """Grayscale (2D) images are correctly supported."""
        gray_dark = np.full((100, 100), 30, dtype=np.uint8)
        gray_bright = np.full((100, 100), 220, dtype=np.uint8)
        assert detect_scene_cut(gray_dark, gray_bright)
        assert not detect_scene_cut(gray_dark, gray_dark)


# ---------------------------------------------------------------------------
# 2. DIS Optical Flow Interpolation Tests
# ---------------------------------------------------------------------------

class TestDISOpticalFlow:
    """Tests for DIS optical flow bidirectional interpolation."""

    def test_boundary_timesteps(self):
        """Timesteps at 0.0 and 1.0 must return exact copies of source frames."""
        f0 = create_solid_frame(64, 64, (100, 150, 200))
        f1 = create_solid_frame(64, 64, (200, 50, 100))

        dis = DISFlowInterpolator()
        out_0 = dis.interpolate(f0, f1, 0.0)
        out_1 = dis.interpolate(f0, f1, 1.0)

        np.testing.assert_array_equal(out_0, f0)
        np.testing.assert_array_equal(out_1, f1)

    def test_intermediate_motion_position(self):
        """A moving object should be interpolated halfway between start and end at t=0.5."""
        h, w = 100, 160
        # Dot moves horizontally from x=40 to x=80
        f0 = create_moving_dot_frame(h, w, dot_x=40, dot_y=50, radius=12)
        f1 = create_moving_dot_frame(h, w, dot_x=80, dot_y=50, radius=12)

        dis = DISFlowInterpolator()
        mid = dis.interpolate(f0, f1, 0.5)

        assert mid.shape == (h, w, 3)
        assert mid.dtype == np.uint8

        # Locate center of mass of the bright object in interpolated frame
        coords = np.where(mid[..., 0] > 60)
        assert len(coords[1]) > 0
        center_x = float(np.mean(coords[1]))

        # Halfway between 40 and 80 is 60; allow margin of 8 pixels
        assert 52.0 <= center_x <= 68.0

    def test_convenience_function(self):
        """Module-level interpolate_flow_dis function works equivalently."""
        f0 = np.full((64, 64, 3), 100, dtype=np.uint8)
        f1 = np.full((64, 64, 3), 150, dtype=np.uint8)
        out = interpolate_flow_dis(f0, f1, 0.5)
        assert out.shape == (64, 64, 3)
        assert 115 <= int(np.mean(out)) <= 135


# ---------------------------------------------------------------------------
# 3. RIFE Neural Interpolation Tests
# ---------------------------------------------------------------------------

class TestRIFEInterpolation:
    """Tests for RIFE v4 Timestep ONNX engine and fallback."""

    def test_rife_initialization_and_fallback(self):
        """RIFE should initialize or fallback gracefully without raising uncaught exceptions."""
        rife = RIFEInterpolator(auto_download=False)
        assert isinstance(rife, RIFEInterpolator)
        # Even if neural weights are not present, interpolate() works via fallback
        f0 = create_solid_frame(64, 64, (80, 90, 100))
        f1 = create_solid_frame(64, 64, (120, 130, 140))
        out = rife.interpolate(f0, f1, 0.5)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_rife_odd_dimensions_padding(self):
        """RIFE must handle odd resolutions by padding to multiples of 32 and cropping back."""
        # Check if model exists or is available
        rife = RIFEInterpolator(auto_download=True)
        # Dimensions not divisible by 32 (e.g. 71 x 113)
        f0 = create_gradient_frame(71, 113, offset_x=0)
        f1 = create_gradient_frame(71, 113, offset_x=2)

        out = rife.interpolate(f0, f1, 0.5)
        assert out.shape == (71, 113, 3)
        assert out.dtype == np.uint8

    def test_rife_interpolate_pair_scene_cut_snapping(self):
        """interpolate_pair with is_scene_cut=True must snap without blending."""
        red = create_solid_frame(64, 64, (255, 0, 0))
        blue = create_solid_frame(64, 64, (0, 0, 255))

        rife = RIFEInterpolator(auto_download=True)
        # t=0.25 (< 0.5) should snap to red
        snap_f0 = rife.interpolate_pair(red, blue, 0.25, is_scene_cut=True)
        np.testing.assert_array_equal(snap_f0, red)

        # t=0.75 (>= 0.5) should snap to blue
        snap_f1 = rife.interpolate_pair(red, blue, 0.75, is_scene_cut=True)
        np.testing.assert_array_equal(snap_f1, blue)

    def test_rife_helpers(self):
        """Test model path resolution and download caching."""
        path = get_rife_model_path()
        assert path.endswith("RIFE_fp32_timestep.onnx")
        dest = download_rife_model()
        assert dest is not None
        assert os.path.exists(dest)


# ---------------------------------------------------------------------------
# 4. FrameInterpolator Mapping & Multiplier Tests
# ---------------------------------------------------------------------------

class TestFrameInterpolatorMappings:
    """Tests for continuous timestep math and FPS multiplier factor calculations."""

    def test_timesteps_30_to_60(self):
        """30fps to 60fps is 2x factor; 30 source frames produce exactly 60 output frames."""
        interpolator = FrameInterpolator(target_fps=60.0, source_fps=30.0)
        assert interpolator.factor == 2.0

        timesteps = interpolator.calculate_timesteps(num_src_frames=30)
        assert len(timesteps) == 60
        # Check alternating integer frames and t=0.5 intermediates
        assert timesteps[0] == (0, 0, 0.0)
        assert timesteps[1] == (1, 0, 0.5)
        assert timesteps[2] == (2, 1, 0.0)
        assert timesteps[3] == (3, 1, 0.5)

    def test_timesteps_24_to_60(self):
        """24fps to 60fps is 2.5x factor; 24 source frames produce exactly 60 output frames."""
        interpolator = FrameInterpolator(target_fps=60.0, source_fps=24.0)
        assert interpolator.factor == 2.5

        timesteps = interpolator.calculate_timesteps(num_src_frames=24)
        assert len(timesteps) == 60
        # First 5 frames correspond to 2 source intervals: [0, 0.4, 0.8, 0.2, 0.6]
        ts_vals = [round(m[2], 2) for m in timesteps[:5]]
        assert ts_vals == [0.0, 0.4, 0.8, 0.2, 0.6]

    def test_timesteps_30_to_120(self):
        """30fps to 120fps is 4.0x factor; 30 source frames produce exactly 120 output frames."""
        interpolator = FrameInterpolator(target_fps=120.0, source_fps=30.0)
        assert interpolator.factor == 4.0

        timesteps = interpolator.calculate_timesteps(num_src_frames=30)
        assert len(timesteps) == 120

    def test_same_fps_passthrough(self):
        """When source_fps == target_fps, factor is 1.0 and frames pass through."""
        interpolator = FrameInterpolator(target_fps=30.0, source_fps=30.0)
        assert interpolator.factor == 1.0

        dummy_frames = [create_solid_frame(32, 32, (i * 8, 0, 0)) for i in range(10)]
        out_frames = interpolator.interpolate_sequence(dummy_frames)
        assert len(out_frames) == 10
        for i in range(10):
            np.testing.assert_array_equal(out_frames[i], dummy_frames[i])

    def test_invalid_fps_raises(self):
        """Non-positive FPS values must raise ValueError."""
        with pytest.raises(ValueError):
            FrameInterpolator(target_fps=-60.0, source_fps=30.0)
        with pytest.raises(ValueError):
            FrameInterpolator(target_fps=60.0, source_fps=0.0)


# ---------------------------------------------------------------------------
# 5. Full Interpolation Pipeline & Scene Cut Protection Tests
# ---------------------------------------------------------------------------

class TestFullInterpolationPipeline:
    """Tests end-to-end frame sequence and stream processing with scene cut protection."""

    def test_sequence_30_to_60_output_count_and_smoothness(self):
        """30fps -> 60fps interpolation yields exact 2x frames with continuous motion."""
        num_src = 15
        h, w = 80, 120
        # Create a sequence of a dot moving smoothly across the screen
        src_frames = [
            create_moving_dot_frame(h, w, dot_x=20 + i * 4, dot_y=40, radius=10)
            for i in range(num_src)
        ]

        interpolator = FrameInterpolator(target_fps=60.0, source_fps=30.0, engine="flow_dis")
        out_frames = interpolator.interpolate_sequence(src_frames)

        # 15 frames * 2.0 = 30 output frames
        assert len(out_frames) == 30
        for f in out_frames:
            assert f.shape == (h, w, 3)
            assert f.dtype == np.uint8

    def test_sequence_24_to_60_output_count(self):
        """24fps -> 60fps interpolation yields exact 2.5x frames (24 in -> 60 out)."""
        num_src = 24
        src_frames = [create_gradient_frame(60, 80, offset_x=i * 2) for i in range(num_src)]

        interpolator = FrameInterpolator(target_fps=60.0, source_fps=24.0, engine="flow_dis")
        out_frames = interpolator.interpolate_sequence(src_frames)

        assert len(out_frames) == 60

    def test_streaming_generator_execution(self):
        """Streaming generator execution works identically without loading all frames in memory."""
        num_src = 10
        src_stream = (create_gradient_frame(60, 80, offset_x=i) for i in range(num_src))

        interpolator = FrameInterpolator(target_fps=60.0, source_fps=30.0, engine="flow_dis")
        out_stream = interpolator(src_stream, total_frames=num_src)

        out_count = 0
        for f in out_stream:
            assert f.shape == (60, 80, 3)
            out_count += 1

        assert out_count == 20

    def test_scene_cut_protection_eliminates_morphing(self):
        """Intermediate frames across a hard scene cut must snap to keyframes without morphing."""
        red = create_solid_frame(64, 64, (255, 0, 0))
        blue = create_solid_frame(64, 64, (0, 0, 255))

        # Sequence of 4 frames with a hard cut between frame 1 (red) and frame 2 (blue)
        src_frames = [red.copy(), red.copy(), blue.copy(), blue.copy()]

        interpolator = FrameInterpolator(target_fps=60.0, source_fps=30.0, engine="flow_dis")
        out_frames = interpolator.interpolate_sequence(src_frames)

        # 4 frames * 2.0 = 8 output frames
        assert len(out_frames) == 8

        # Examine the intermediate frame between red (index 2) and blue (index 4), which is index 3
        inter_frame = out_frames[3]

        # The intermediate frame must be either pure red or pure blue, NOT purple/blended
        is_pure_red = np.all(inter_frame[:, :, 0] > 200) and np.all(inter_frame[:, :, 2] < 50)
        is_pure_blue = np.all(inter_frame[:, :, 2] > 200) and np.all(inter_frame[:, :, 0] < 50)

        assert is_pure_red or is_pure_blue, "Scene cut produced mixed/morphing artifact frame!"

    def test_interpolate_pair_module_function(self):
        """Module-level interpolate_pair function works correctly."""
        f0 = create_solid_frame(64, 64, (10, 20, 30))
        f1 = create_solid_frame(64, 64, (100, 110, 120))

        # Non-cut interpolation
        res = interpolate_pair(f0, f1, timestep=0.5, is_scene_cut=False, engine="flow_dis")
        assert res.shape == (64, 64, 3)

        # Cut snapping
        snap_0 = interpolate_pair(f0, f1, timestep=0.3, is_scene_cut=True)
        np.testing.assert_array_equal(snap_0, f0)
        snap_1 = interpolate_pair(f0, f1, timestep=0.7, is_scene_cut=True)
        np.testing.assert_array_equal(snap_1, f1)

    def test_downsampling_fps_sequence(self):
        """Downsampling from 60fps to 30fps yields half the frames."""
        src_frames = [create_solid_frame(32, 32, (i * 4, 0, 0)) for i in range(20)]
        interpolator = FrameInterpolator(target_fps=30.0, source_fps=60.0, engine="flow_dis")
        out_frames = interpolator.interpolate_sequence(src_frames)
        assert len(out_frames) == 10

    def test_single_frame_sequence(self):
        """Single-frame input must not crash and should yield proportional frames."""
        single_frame = [create_solid_frame(40, 40, (50, 60, 70))]
        interpolator = FrameInterpolator(target_fps=60.0, source_fps=30.0)
        out_frames = interpolator.interpolate_sequence(single_frame)
        assert len(out_frames) == 2
        np.testing.assert_array_equal(out_frames[0], single_frame[0])

    def test_empty_sequence_returns_empty(self):
        """Empty input stream or sequence returns empty list."""
        interpolator = FrameInterpolator(target_fps=60.0, source_fps=30.0)
        assert interpolator.interpolate_sequence([]) == []

    def test_visual_smoothness_metric(self):
        """Interpolated frame at t=0.5 must be closer to both endpoints than endpoints are to each other."""
        h, w = 100, 160
        f0 = create_moving_dot_frame(h, w, dot_x=40, dot_y=50, radius=12)
        f1 = create_moving_dot_frame(h, w, dot_x=60, dot_y=50, radius=12)

        interpolator = FrameInterpolator(target_fps=60.0, source_fps=30.0, engine="flow_dis")
        out_frames = interpolator.interpolate_sequence([f0, f1])

        # 2 frames * 2.0 = 4 output frames
        assert len(out_frames) == 4
        mid = out_frames[1]

        # Difference between f0 and f1
        diff_f0_f1 = float(np.mean(np.abs(f0.astype(float) - f1.astype(float))))
        # Difference between f0 and mid
        diff_f0_mid = float(np.mean(np.abs(f0.astype(float) - mid.astype(float))))
        # Difference between mid and f1
        diff_mid_f1 = float(np.mean(np.abs(mid.astype(float) - f1.astype(float))))

        assert diff_f0_mid < diff_f0_f1
        assert diff_mid_f1 < diff_f0_f1

    def test_rife_engine_interpolator(self):
        """FrameInterpolator with engine='rife' executes properly."""
        interpolator = FrameInterpolator(target_fps=60.0, source_fps=30.0, engine="rife")
        f0 = create_gradient_frame(64, 64, offset_x=0)
        f1 = create_gradient_frame(64, 64, offset_x=2)

        out_frames = interpolator.interpolate_sequence([f0, f1])
        # 2 frames * 2.0 = 4 output frames
        assert len(out_frames) == 4
        assert out_frames[0].shape == (64, 64, 3)
        assert out_frames[1].shape == (64, 64, 3)
