"""Adversarial stress test suite for AI Quality Enhancement Pipeline.

Tests:
1. Non-standard odd and prime dimensions (e.g. 503x331, 317x239, 131x109, 71x53)
2. Dynamic dimension changes across consecutive frames in TemporalVideoEnhancer
3. Dynamic tiling boundary continuity under 2K and 4K upscaling (verifying no seam lines)
4. Peak RAM footprint under 2K and 4K upscaling (< 500MB limit)
5. Extreme noise (Gaussian, impulse) and high compression (JPEG Q=5) stability (no NaNs/Infs)
"""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path
import pytest
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.enhancement.speed_mode import SpeedEnhancer, enhance_frame_speed
from app.ai.enhancement.quality_mode import TemporalVideoEnhancer
from app.ai.enhancement.enhancer import VideoEnhancer
from app.ai.tiling import tile_process


class TestOddAndPrimeDimensions:
    """Stress test enhancement models and tiling on non-standard odd and prime dimensions."""

    @pytest.mark.parametrize(
        "width,height",
        [
            (503, 331),
            (317, 239),
            (131, 109),
            (71, 53),
        ],
    )
    def test_speed_mode_odd_prime_dimensions(self, width: int, height: int):
        """SpeedEnhancer must handle prime/odd dimensions without shape mismatches or NaN output."""
        speed = SpeedEnhancer()
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Enhance with explicit target dimensions
        out = speed.enhance(frame, target_w=width, target_h=height)
        assert out.shape == (height, width, 3)
        assert out.dtype == np.uint8
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    @pytest.mark.parametrize(
        "width,height",
        [
            (317, 239),
            (131, 109),
        ],
    )
    def test_temporal_mode_odd_prime_dimensions(self, width: int, height: int):
        """TemporalVideoEnhancer must process prime/odd dimensions without optical flow or remapping failure."""
        speed = SpeedEnhancer()
        temp = TemporalVideoEnhancer(target_w=width, target_h=height, speed_enhancer=speed)

        f1 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        f2 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        o1 = temp.enhance_frame(f1)
        o2 = temp.enhance_frame(f2)

        assert o1.shape == (height, width, 3)
        assert o2.shape == (height, width, 3)
        assert not np.isnan(o2).any()
        assert not np.isinf(o2).any()

    def test_video_enhancer_preset_dimension_parity(self):
        """VideoEnhancer must enforce strictly even output dimensions for video encoder compliance."""
        enh = VideoEnhancer(mode="speed", target_resolution="1080p")
        # Odd dimensions 503x331
        tw, th = enh.calculate_target_dimensions(503, 331)
        assert tw % 2 == 0, f"Target width {tw} must be even"
        assert th % 2 == 0, f"Target height {th} must be even"

        # Prime dimensions 317x239
        tw, th = enh.calculate_target_dimensions(317, 239)
        assert tw % 2 == 0, f"Target width {tw} must be even"
        assert th % 2 == 0, f"Target height {th} must be even"

    def test_temporal_consecutive_frame_dimension_change_exposure(self):
        """Expose Bug 3: TemporalVideoEnhancer crashes with cv2.error if consecutive frames change dimensions."""
        temp = TemporalVideoEnhancer(target_w=128, target_h=128)
        f1 = np.zeros((64, 64, 3), dtype=np.uint8)
        f2 = np.zeros((80, 80, 3), dtype=np.uint8)

        temp.enhance_frame(f1)
        # In current implementation, passing f2 with different shape raises cv2.error in calc()
        # It should gracefully handle/reset instead of crashing.
        try:
            o2 = temp.enhance_frame(f2)
            assert o2.shape == (128, 128, 3)
        except Exception as exc:
            pytest.fail(f"VULNERABILITY CONFIRMED: TemporalVideoEnhancer crashed on changing frame dimensions: {exc}")


class TestDynamicTilingAndMemoryLimits:
    """Stress test dynamic tiling boundaries and peak memory limits under 2K and 4K upscaling."""

    def test_tiling_boundary_continuity_no_seams(self):
        """Dynamic tiling must exhibit seamless cosine feathering with zero boundary edge spikes."""
        h, w = 500, 600
        # Create continuous 2D linear gradient
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        gradient = ((x_grid / w + y_grid / h) * 128.0).astype(np.uint8)
        grad_rgb = np.stack([gradient, gradient, gradient], axis=-1)

        # Run through tile_process with standard 256 tile size and 32 overlap
        tiled = tile_process(
            grad_rgb,
            lambda t: cv2.resize(t, (t.shape[1] * 2, t.shape[0] * 2), interpolation=cv2.INTER_LINEAR),
            tile_size=256,
            overlap=32,
            scale=2,
        )
        ground_truth = cv2.resize(grad_rgb, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

        # Check maximum and average reconstruction deviation across the canvas
        diff = np.abs(tiled.astype(float) - ground_truth.astype(float))
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        assert max_diff <= 2.0, f"Boundary seam line defect detected! Max diff = {max_diff}"
        assert mean_diff <= 0.5, f"Boundary feathering deviation too high! Mean diff = {mean_diff}"

    def test_peak_ram_under_2k_upscaling_limit(self):
        """Expose Bug 1a: Peak RAM during 2K upscaling (720p -> 1440p) must remain strictly < 500MB."""
        speed = SpeedEnhancer()
        f_720p = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        tracemalloc.start()
        out_2k = speed.enhance(f_720p, target_w=2560, target_h=1440)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024.0 * 1024.0)
        assert out_2k.shape == (1440, 2560, 3)
        assert peak_mb < 500.0, (
            f"VULNERABILITY CONFIRMED: 2K upscaling peak RAM ({peak_mb:.2f} MB) violated < 500MB contract!"
        )

    def test_peak_ram_under_4k_upscaling_limit(self):
        """Expose Bug 1b: Peak RAM during 4K upscaling (1080p -> 2160p) must remain strictly < 500MB."""
        speed = SpeedEnhancer()
        f_1080p = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        tracemalloc.start()
        out_4k = speed.enhance(f_1080p, target_w=3840, target_h=2160)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024.0 * 1024.0)
        assert out_4k.shape == (2160, 3840, 3)
        assert peak_mb < 500.0, (
            f"VULNERABILITY CONFIRMED: 4K upscaling peak RAM ({peak_mb:.2f} MB) violated < 500MB contract!"
        )


class TestExtremeNoiseAndCompression:
    """Stress test model numerical stability and artifact handling under hostile inputs."""

    def test_extreme_gaussian_noise_numerical_stability(self):
        """Extreme Gaussian noise (sigma=150) must not cause NaNs, Infs, or value wrap-around."""
        speed = SpeedEnhancer()
        base = np.full((128, 128, 3), 128, dtype=np.uint8)
        noise = np.random.normal(0, 150, base.shape)
        noisy = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        out = speed.enhance(noisy, target_w=256, target_h=256)
        assert out.dtype == np.uint8
        assert not np.isnan(out).any(), "NaN detected in enhanced output"
        assert not np.isinf(out).any(), "Inf detected in enhanced output"
        assert np.all(out >= 0) and np.all(out <= 255)

    def test_severe_jpeg_compression_artifact_handling(self):
        """Severe JPEG compression (Quality=5) must be enhanced without numerical instability."""
        speed = SpeedEnhancer()
        pattern = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.putText(pattern, "STRESS", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.circle(pattern, (90, 60), 25, (0, 200, 255), -1)

        # Heavily compress to JPEG Quality 5
        _, encoded = cv2.imencode(".jpg", pattern, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        out = speed.enhance(compressed, target_w=256, target_h=256)
        assert out.shape == (256, 256, 3)
        assert out.dtype == np.uint8
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_salt_and_pepper_noise(self):
        """30% impulse (salt-and-pepper) noise must not produce numerical overflows."""
        speed = SpeedEnhancer()
        f = np.full((128, 128, 3), 120, dtype=np.uint8)
        # 15% salt, 15% pepper
        mask_salt = np.random.rand(128, 128) < 0.15
        mask_pepper = np.random.rand(128, 128) > 0.85
        f[mask_salt] = 255
        f[mask_pepper] = 0

        out = speed.enhance(f, target_w=256, target_h=256)
        assert out.dtype == np.uint8
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
