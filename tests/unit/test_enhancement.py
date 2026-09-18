"""Comprehensive unit tests for the AI Video Quality Enhancement pipeline.

Verifies:
1. Automated model weight management, caching, and fallback mechanism.
2. Memory-safe dynamic tiling with seamless cosine feathering and <500MB RAM footprint.
3. Speed mode Real-ESRGAN Compact ONNX sharpness gain over naive bicubic.
4. Quality mode Temporal Video Super-Resolution (T-VSR) flicker reduction and motion alignment.
5. Unified VideoEnhancer resolution presets and dimension calculations.
"""

from __future__ import annotations

import sys
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

# Ensure project root is in sys.path when running directly via python
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.ai.enhancement.enhancer import VideoEnhancer  # noqa: E402
from app.ai.enhancement.quality_mode import TemporalVideoEnhancer  # noqa: E402
from app.ai.enhancement.speed_mode import SpeedEnhancer, enhance_frame_speed  # noqa: E402
from app.ai.tiling import _get_tile_intervals, _make_1d_feather, tile_process  # noqa: E402
from app.ai.weights import FallbackEnhancer, ensure_model_weights  # noqa: E402


def _create_textured_test_pattern(height: int = 128, width: int = 128) -> np.ndarray:
    """Generate a synthetic test image with crisp edges, text, and gradient textures."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Background gradient
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    img[:, :, 0] = ((x_coords / width) * 200).astype(np.uint8)
    img[:, :, 1] = ((y_coords / height) * 200).astype(np.uint8)
    img[:, :, 2] = 120

    # High frequency features & sharp step edges
    cv2.rectangle(img, (width // 8, height // 8), (width // 2, height // 2), (255, 255, 255), -1)
    cv2.circle(img, (3 * width // 4, 3 * height // 4), min(width, height) // 5, (0, 220, 255), -1)
    cv2.putText(
        img,
        "AI-SR",
        (width // 6, 3 * height // 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )
    return img


class TestModelWeightsAndFallback(unittest.TestCase):
    """Test model downloader, cache inspection, and offline fallback engine."""

    def test_ensure_model_weights_cached_returns_path(self) -> None:
        """Cached model weights are detected and returned as a valid Path."""
        path = ensure_model_weights()
        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 1_000_000)

    def test_ensure_model_weights_unreachable_network_returns_none(self) -> None:
        """When download fails due to network outage, returns None without crashing."""
        with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
            result = ensure_model_weights(
                model_name="nonexistent-model.onnx",
                target_dir="models/enhancer/test_tmp",
            )
            self.assertIsNone(result)

    def test_fallback_enhancer_sharpness_gain_over_bicubic(self) -> None:
        """FallbackEnhancer produces higher Laplacian variance than naive bicubic upscaling."""
        img = _create_textured_test_pattern(100, 100)
        # Downscale to simulate low-res input
        lr = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)

        # Naive bicubic 4x upscaling
        bicubic = cv2.resize(lr, (200, 200), interpolation=cv2.INTER_CUBIC)
        gray_bicubic = cv2.cvtColor(bicubic, cv2.COLOR_RGB2GRAY)
        var_bicubic = cv2.Laplacian(gray_bicubic, cv2.CV_64F).var()

        # Fallback enhancer
        fallback = FallbackEnhancer(unsharp_strength=0.7, texture_boost=0.3)
        enhanced = fallback.enhance(lr, target_w=200, target_h=200)
        gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        var_enhanced = cv2.Laplacian(gray_enhanced, cv2.CV_64F).var()

        self.assertEqual(enhanced.shape, (200, 200, 3))
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertGreater(
            var_enhanced,
            var_bicubic * 1.3,
            f"Expected Fallback var ({var_enhanced:.2f}) > Bicubic var * 1.3 ({var_bicubic * 1.3:.2f})",
        )

    def test_fallback_enhancer_empty_frame_raises(self) -> None:
        """Empty frame input raises ValueError."""
        fallback = FallbackEnhancer()
        with self.assertRaises(ValueError):
            fallback.enhance(np.array([]))


class TestMemorySafeTiling(unittest.TestCase):
    """Test dynamic overlapping spatial tiling and cosine feathering."""

    def test_feather_weights_unity_property(self) -> None:
        """Raised cosine feathering weights remain bounded in [0, 1]."""
        w = _make_1d_feather(100, pad_start=20, pad_end=20)
        self.assertEqual(len(w), 100)
        self.assertAlmostEqual(float(w[0]), 0.0, places=2)
        self.assertAlmostEqual(float(w[-1]), 0.0, places=2)
        self.assertAlmostEqual(float(w[50]), 1.0, places=5)
        self.assertTrue(np.all(w >= 0.0) and np.all(w <= 1.0))

    def test_tile_intervals_coverage(self) -> None:
        """Tile intervals span the entire axis length without gaps."""
        intervals = _get_tile_intervals(total_len=500, tile_len=150, overlap=30)
        self.assertEqual(intervals[0][0], 0)
        self.assertEqual(intervals[-1][1], 500)
        for i in range(len(intervals) - 1):
            # Check overlap between adjacent tiles
            self.assertLess(intervals[i + 1][0], intervals[i][1])

    def test_tiling_identity_reconstruction(self) -> None:
        """Tiling an image with an identity function reconstructs the original image seamlessly."""
        img = np.random.randint(0, 255, (320, 480, 3), dtype=np.uint8)
        reconstructed = tile_process(
            img,
            process_fn=lambda t: t,
            tile_size=128,
            overlap=24,
            scale=1,
        )
        self.assertEqual(reconstructed.shape, img.shape)
        diff = np.max(np.abs(reconstructed.astype(float) - img.astype(float)))
        self.assertLessEqual(diff, 1.0, f"Max reconstruction error too high: {diff}")

    def test_tiling_seamless_gradient_no_boundary_artifacts(self) -> None:
        """Linear gradient upscaling exhibits continuous smooth transition across tile seams."""
        h, w = 300, 400
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        gradient = (0.3 * x_grid + 0.4 * y_grid).astype(np.float32)

        def linear2x(tile: np.ndarray) -> np.ndarray:
            return cv2.resize(tile, (tile.shape[1] * 2, tile.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

        tiled = tile_process(gradient, linear2x, tile_size=100, overlap=25, scale=2)
        ground_truth = linear2x(gradient)

        diff = np.abs(tiled - ground_truth)
        self.assertLess(float(diff.max()), 0.01)
        self.assertLess(float(diff.mean()), 0.001)

    def test_tiling_scale_multiplier(self) -> None:
        """Tiling with 4x scaling produces exact (H*4, W*4) dimensions."""
        img = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)

        def mock_scale4(tile: np.ndarray) -> np.ndarray:
            return cv2.resize(tile, (tile.shape[1] * 4, tile.shape[0] * 4), interpolation=cv2.INTER_NEAREST)

        out = tile_process(img, mock_scale4, tile_size=64, overlap=16, scale=4)
        self.assertEqual(out.shape, (150 * 4, 200 * 4, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_tiling_small_frame_fast_path(self) -> None:
        """Frames smaller than tile_size execute process_fn directly without tiling overhead."""
        called_count = 0

        def counting_fn(tile: np.ndarray) -> np.ndarray:
            nonlocal called_count
            called_count += 1
            return cv2.resize(tile, (tile.shape[1] * 2, tile.shape[0] * 2))

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        out = tile_process(img, counting_fn, tile_size=100, overlap=16, scale=2)
        self.assertEqual(called_count, 1)
        self.assertEqual(out.shape, (100, 100, 3))

    def test_tiling_invalid_arguments(self) -> None:
        """Invalid tile_size and overlap parameters raise ValueError."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            tile_process(img, lambda t: t, tile_size=32, overlap=32)
        with self.assertRaises(ValueError):
            tile_process(img, lambda t: t, tile_size=16, overlap=32)

    def test_tiling_memory_footprint_under_500mb(self) -> None:
        """Dynamic tiling processes 2K output with peak RAM footprint strictly under 500MB."""
        tracemalloc.start()
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        def mock_lanczos2x(tile: np.ndarray) -> np.ndarray:
            return cv2.resize(tile, (tile.shape[1] * 2, tile.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)

        # 720p -> 1440p (2K) output: (1440, 2560, 3)
        out = tile_process(frame, mock_lanczos2x, tile_size=256, overlap=32, scale=2)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024.0 * 1024.0)
        self.assertEqual(out.shape, (1440, 2560, 3))
        self.assertLess(
            peak_mb,
            500.0,
            f"Peak memory ({peak_mb:.2f} MB) exceeded 500 MB limit!",
        )


class TestSpeedModeEnhancement(unittest.TestCase):
    """Test Real-ESRGAN Compact SRVGGNet ONNX speed mode."""

    def setUp(self) -> None:
        self.enhancer = SpeedEnhancer()

    def test_speed_mode_sharpness_gain_over_bicubic(self) -> None:
        """Speed mode produces measurable Laplacian variance and Sobel gradient gain over bicubic."""
        img = _create_textured_test_pattern(128, 128)
        lr = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        # Naive bicubic 4x
        bicubic = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)
        var_bicubic = cv2.Laplacian(cv2.cvtColor(bicubic, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        sobel_bicubic = np.mean(np.abs(cv2.Sobel(cv2.cvtColor(bicubic, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 1)))

        # Speed mode
        enhanced = self.enhancer.enhance(lr, target_w=256, target_h=256)
        var_enhanced = cv2.Laplacian(cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        sobel_enhanced = np.mean(np.abs(cv2.Sobel(cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 1)))

        self.assertEqual(enhanced.shape, (256, 256, 3))
        # Genuine AI restoration yields substantial edge and texture sharpening
        self.assertGreater(
            var_enhanced,
            var_bicubic * 1.5,
            f"Laplacian var gain insufficient: {var_enhanced:.2f} vs {var_bicubic:.2f}",
        )
        self.assertGreater(
            sobel_enhanced,
            sobel_bicubic * 1.2,
            f"Sobel gradient gain insufficient: {sobel_enhanced:.2f} vs {sobel_bicubic:.2f}",
        )

    def test_enhance_frame_speed_contract(self) -> None:
        """PROJECT.md interface contract enhance_frame_speed returns correct shape and dtype."""
        f = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        out = enhance_frame_speed(f, target_w=160, target_h=120)
        self.assertEqual(out.shape, (120, 160, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_speed_mode_tiled_processing(self) -> None:
        """Speed mode correctly splits and stitches larger inputs through ONNX tiling."""
        tiled_enhancer = SpeedEnhancer(tile_size=64, overlap=16)
        f = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        out = tiled_enhancer.enhance(f, target_w=200, target_h=160)
        self.assertEqual(out.shape, (160, 200, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_speed_mode_fallback_on_missing_model(self) -> None:
        """Speed mode gracefully utilizes FallbackEnhancer when model path does not exist."""
        enhancer = SpeedEnhancer(model_path="nonexistent_weights.onnx")
        self.assertIsNone(enhancer.session)
        self.assertIsNotNone(enhancer.fallback)
        f = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
        out = enhancer.enhance(f, target_w=80, target_h=80)
        self.assertEqual(out.shape, (80, 80, 3))

    def test_speed_enhancer_empty_frame_raises(self) -> None:
        """Passing empty frame to SpeedEnhancer raises ValueError."""
        with self.assertRaises(ValueError):
            self.enhancer.enhance(np.array([]))


class TestQualityModeTemporalVSR(unittest.TestCase):
    """Test Temporal Video Super-Resolution (T-VSR) multi-frame consistency engine."""

    def test_quality_mode_flicker_reduction(self) -> None:
        """Quality mode produces lower inter-frame temporal variance than independent per-frame restoration."""
        np.random.seed(42)
        base = _create_textured_test_pattern(96, 96)

        # Generate sequence of 6 frames with synthetic random noise (simulating compression shimmer/flicker)
        noisy_frames = []
        for _ in range(6):
            noise = np.random.normal(0, 18, base.shape).astype(np.float32)
            f = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            noisy_frames.append(f)

        # 1. Independent per-frame enhancement (Speed mode)
        speed_enhancer = SpeedEnhancer()
        speed_outputs = [speed_enhancer.enhance(f, target_w=192, target_h=192) for f in noisy_frames]

        # 2. Temporal consistency enhancement (Quality mode)
        temporal_enhancer = TemporalVideoEnhancer(target_w=192, target_h=192, speed_enhancer=speed_enhancer)
        temporal_outputs = [temporal_enhancer.enhance_frame(f) for f in noisy_frames]

        # Measure inter-frame temporal differences across frames 1..N
        speed_diffs = [
            float(np.mean(np.abs(speed_outputs[i].astype(float) - speed_outputs[i - 1].astype(float))))
            for i in range(1, len(noisy_frames))
        ]
        temporal_diffs = [
            float(np.mean(np.abs(temporal_outputs[i].astype(float) - temporal_outputs[i - 1].astype(float))))
            for i in range(1, len(noisy_frames))
        ]

        mean_speed_diff = float(np.mean(speed_diffs))
        mean_temp_diff = float(np.mean(temporal_diffs))

        # Temporal VSR must demonstrate lower inter-frame variance (flicker reduction)
        self.assertLess(
            mean_temp_diff,
            mean_speed_diff,
            f"Expected temporal variance ({mean_temp_diff:.2f}) < per-frame variance ({mean_speed_diff:.2f})",
        )
        reduction_pct = (mean_speed_diff - mean_temp_diff) / mean_speed_diff * 100.0
        self.assertGreaterEqual(
            reduction_pct,
            15.0,
            f"Flicker reduction ({reduction_pct:.1f}%) expected to be at least 15%",
        )

    def test_quality_mode_motion_alignment(self) -> None:
        """TemporalVideoEnhancer correctly tracks and aligns moving objects across consecutive frames."""
        f1 = np.full((120, 160, 3), 40, dtype=np.uint8)
        cv2.rectangle(f1, (30, 30), (70, 70), (220, 220, 220), -1)

        f2 = np.full((120, 160, 3), 40, dtype=np.uint8)
        # Shift square by 4px right, 2px down
        cv2.rectangle(f2, (34, 32), (74, 72), (220, 220, 220), -1)

        enhancer = TemporalVideoEnhancer(target_w=160, target_h=120)
        enhancer.enhance_frame(f1)
        o2 = enhancer.enhance_frame(f2)

        self.assertEqual(o2.shape, (120, 160, 3))
        # Center of new square location (54, 52) should be bright
        self.assertGreater(float(np.mean(o2[52, 54])), 180.0)
        # Background location (10, 10) should be dark
        self.assertLess(float(np.mean(o2[10, 10])), 70.0)

    def test_quality_mode_scene_cut_resets_history(self) -> None:
        """Scene cut flag (is_cut=True) cleanly initializes new sequence without cross-scene ghosting."""
        temporal_enhancer = TemporalVideoEnhancer(target_w=128, target_h=128)

        # Scene A: solid dark frame
        scene_a = np.zeros((64, 64, 3), dtype=np.uint8)
        out_a = temporal_enhancer.enhance_frame(scene_a)
        self.assertEqual(out_a.shape, (128, 128, 3))

        # Scene B: solid bright frame with is_cut=True
        scene_b = np.full((64, 64, 3), 255, dtype=np.uint8)
        out_b = temporal_enhancer.enhance_frame(scene_b, is_cut=True)

        # Because is_cut was True, out_b should not contain blended residue from scene_a
        self.assertGreater(
            float(np.mean(out_b)),
            240.0,
            "Scene B output was corrupted by previous scene A buffer",
        )

    def test_quality_mode_reset(self) -> None:
        """reset() clears internal rolling buffers."""
        enhancer = TemporalVideoEnhancer(target_w=100, target_h=100)
        enhancer.enhance_frame(np.zeros((50, 50, 3), dtype=np.uint8))
        self.assertIsNotNone(enhancer.prev_lr)
        self.assertIsNotNone(enhancer.prev_hr)

        enhancer.reset()
        self.assertIsNone(enhancer.prev_lr)
        self.assertIsNone(enhancer.prev_hr)

    def test_quality_enhancer_empty_frame_raises(self) -> None:
        """Passing empty frame to TemporalVideoEnhancer raises ValueError."""
        enhancer = TemporalVideoEnhancer(target_w=100, target_h=100)
        with self.assertRaises(ValueError):
            enhancer.enhance_frame(np.array([]))


class TestVideoEnhancerPresetsAndPipeline(unittest.TestCase):
    """Test unified VideoEnhancer facade, presets, and dimension rules."""

    def test_calculate_target_dimensions_presets(self) -> None:
        """Presets preserve aspect ratio proportionally and enforce even dimensions."""
        enhancer = VideoEnhancer(mode="speed", target_resolution="1080p")

        # Standard 480p 16:9 (854x480) -> 1080p (1920x1080)
        tw, th = enhancer.calculate_target_dimensions(854, 480)
        self.assertEqual((tw, th), (1920, 1080))
        self.assertEqual(tw % 2, 0)
        self.assertEqual(th % 2, 0)

        # 4:3 (640x480) -> 1080p bounding box (1440x1080)
        tw, th = enhancer.calculate_target_dimensions(640, 480)
        self.assertEqual((tw, th), (1440, 1080))
        self.assertEqual(tw % 2, 0)
        self.assertEqual(th % 2, 0)

        # 2K Preset (1440p)
        enhancer.target_resolution = "2K"
        tw, th = enhancer.calculate_target_dimensions(854, 480)
        self.assertEqual((tw, th), (2560, 1440))

        # 4K Preset (2160p)
        enhancer.target_resolution = "4K"
        tw, th = enhancer.calculate_target_dimensions(854, 480)
        self.assertEqual((tw, th), (3840, 2160))

        # Original Preset
        enhancer.target_resolution = "Original"
        tw, th = enhancer.calculate_target_dimensions(853, 479)
        self.assertEqual((tw, th), (852, 478))  # Enforces even

        # Portrait orientation (720x1280) @ 1080p -> 1080x1920
        enhancer.target_resolution = "1080p"
        tw, th = enhancer.calculate_target_dimensions(720, 1280)
        self.assertEqual((tw, th), (1080, 1920))

    def test_video_enhancer_single_frame_modes(self) -> None:
        """VideoEnhancer processes single frames in both speed and quality modes."""
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)

        # Speed mode
        enhancer_s = VideoEnhancer(mode="speed", target_resolution="Original")
        out_s = enhancer_s.enhance_frame(frame)
        self.assertEqual(out_s.shape, (60, 80, 3))

        # Quality mode
        enhancer_q = VideoEnhancer(mode="quality", target_resolution="Original")
        out_q = enhancer_q.enhance_frame(frame)
        self.assertEqual(out_q.shape, (60, 80, 3))

    def test_video_enhancer_sequence_streaming(self) -> None:
        """enhance_sequence streams enhanced frames with scene cuts."""
        enhancer = VideoEnhancer(mode="quality", target_resolution="Original")
        frames = [np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8) for _ in range(4)]
        scene_cuts = [False, False, True, False]

        results = list(enhancer.enhance_sequence(frames, scene_cuts=scene_cuts))
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r.shape, (40, 40, 3))
            self.assertEqual(r.dtype, np.uint8)

    def test_video_enhancer_reset(self) -> None:
        """VideoEnhancer reset clears cached dimensions and internal temporal engine state."""
        enhancer = VideoEnhancer(mode="quality", target_resolution="Original")
        enhancer.enhance_frame(np.zeros((50, 50, 3), dtype=np.uint8))
        self.assertIsNotNone(enhancer._cached_target_dims)
        self.assertIsNotNone(enhancer._temporal_engine)
        self.assertIsNotNone(enhancer._temporal_engine.prev_lr)

        enhancer.reset()
        self.assertIsNone(enhancer._cached_target_dims)
        self.assertIsNone(enhancer._temporal_engine.prev_lr)

    def test_video_enhancer_empty_frame_raises(self) -> None:
        """Passing empty frame to VideoEnhancer raises ValueError."""
        enhancer = VideoEnhancer(mode="speed")
        with self.assertRaises(ValueError):
            enhancer.enhance_frame(np.array([]))

    def test_video_enhancer_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with self.assertRaises(ValueError):
            VideoEnhancer(mode="super_quantum")


if __name__ == "__main__":
    unittest.main()
