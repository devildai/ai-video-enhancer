"""Adversarial stress test suite for AI Frame Interpolation Pipeline.

Tests:
1. Extreme color flashes (pure white, pure black, pure neon green, magenta, cyan)
2. Grayscale / low-saturation transitions (adversarial luminance-only shifts)
3. Non-integer framerate mappings (23.976 -> 60, 15 -> 60, 29.97 -> 60, 24 -> 59.94)
4. Single-frame and two-frame boundary sequences
5. Scene cut elimination of morphing/smearing artifacts under stress
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.interpolation.scene_cut import detect_scene_cut, compute_scene_metrics
from app.ai.interpolation.interpolator import FrameInterpolator, interpolate_pair
from app.ai.interpolation.flow_dis import DISFlowInterpolator


class TestExtremeColorFlashes:
    """Stress test interpolation across high-contrast and saturated color flashes."""

    def test_extreme_color_flash_palette(self):
        """Pure white -> pure black -> pure neon green -> magenta -> cyan transitions must be cuts."""
        white = np.full((120, 120, 3), 255, dtype=np.uint8)
        black = np.zeros((120, 120, 3), dtype=np.uint8)

        neon_green = np.zeros((120, 120, 3), dtype=np.uint8)
        neon_green[:, :, 1] = 255  # RGB (0, 255, 0)

        neon_magenta = np.zeros((120, 120, 3), dtype=np.uint8)
        neon_magenta[:, :, 0] = 255
        neon_magenta[:, :, 2] = 255  # RGB (255, 0, 255)

        neon_cyan = np.zeros((120, 120, 3), dtype=np.uint8)
        neon_cyan[:, :, 1] = 255
        neon_cyan[:, :, 2] = 255  # RGB (0, 255, 255)

        transitions = [
            ("white -> black", white, black),
            ("black -> neon green", black, neon_green),
            ("neon green -> neon magenta", neon_green, neon_magenta),
            ("neon magenta -> neon cyan", neon_cyan, neon_magenta),
            ("neon cyan -> black", neon_cyan, black),
            ("black -> white", black, white),
            ("neon green -> white", neon_green, white),
        ]

        for name, f1, f2 in transitions:
            is_cut = detect_scene_cut(f1, f2)
            assert is_cut, f"Extreme color flash {name} failed to be flagged as scene cut!"

    def test_flash_snapping_eliminates_smearing(self):
        """FrameInterpolator must snap to keyframes across extreme flashes rather than smearing."""
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        neon_green = np.zeros((100, 100, 3), dtype=np.uint8)
        neon_green[:, :, 1] = 255

        interp = FrameInterpolator(60.0, 30.0, engine="flow_dis")
        # At t=0.25 (should snap to white)
        snap_f1 = interp.interpolate_pair(white, neon_green, 0.25)
        assert np.array_equal(snap_f1, white), "Expected snap to white for t < 0.5"

        # At t=0.75 (should snap to neon green)
        snap_f2 = interp.interpolate_pair(white, neon_green, 0.75)
        assert np.array_equal(snap_f2, neon_green), "Expected snap to neon green for t >= 0.5"

        # At t=0.5 (should snap to neon green)
        snap_mid = interp.interpolate_pair(white, neon_green, 0.5)
        assert np.array_equal(snap_mid, neon_green), "Expected snap to keyframe at t=0.5"

        # Verify no intermediate smearing / blended color
        # In a smeared frame, green would be ~128 and red/blue would be non-zero
        assert not (snap_mid[0, 0, 0] > 10 and snap_mid[0, 0, 1] > 10), "Detected smearing artifact across cut!"

    def test_adversarial_grayscale_luminance_cut_exposure(self):
        """Expose Bug 2: Grayscale cuts with L1 diff between 30 and 80 fail detection due to HSV correlation."""
        # Dark gray (40) vs medium-bright gray (95): L1 diff = 55.0 (well above threshold 30.0)
        dark_gray = np.full((100, 100, 3), 40, dtype=np.uint8)
        med_gray = np.full((100, 100, 3), 95, dtype=np.uint8)

        l1, corr = compute_scene_metrics(dark_gray, med_gray)
        assert l1 == 55.0, f"Expected L1 diff 55.0, got {l1}"
        # In HSV space with Saturation=0, corr is 1.0
        assert corr == 1.0, f"Expected HSV correlation 1.0, got {corr}"

        # In current implementation, detect_scene_cut returns False because corr >= 0.85 and l1 <= 80
        cut_detected = detect_scene_cut(dark_gray, med_gray, threshold=30.0)
        # This test documents the vulnerability: an abrupt cut of 55 luminance levels fails detection!
        # If this fails (cut_detected is False), it exposes the flaw.
        assert cut_detected, (
            f"VULNERABILITY CONFIRMED: Grayscale cut with L1={l1} > threshold 30.0 failed scene cut detection!"
        )


class TestNonIntegerFramerateMappings:
    """Stress test non-integer and fractional framerate conversions."""

    @pytest.mark.parametrize(
        "src_fps,dst_fps",
        [
            (23.976, 60.0),
            (15.0, 60.0),
            (29.97, 60.0),
            (24.0, 59.94),
        ],
    )
    def test_arbitrary_framerate_exact_counts(self, src_fps: float, dst_fps: float):
        """Frame counts must precisely match round(N * dst_fps / src_fps) without drift."""
        interp = FrameInterpolator(dst_fps, src_fps, engine="flow_dis")
        factor = dst_fps / src_fps

        test_lengths = [1, 2, 5, 10, 24, 60]
        for n in test_lengths:
            expected = int(round(n * factor))
            frames = [np.full((32, 32, 3), (k * 17) % 256, dtype=np.uint8) for k in range(n)]
            interpolated = list(interp.interpolate_sequence(frames))
            assert len(interpolated) == expected, (
                f"Framerate {src_fps}->{dst_fps} count mismatch for N={n}: "
                f"expected {expected}, got {len(interpolated)}"
            )

    def test_timesteps_strictly_valid(self):
        """Fractional timesteps returned by calculate_timesteps must be in [0.0, 1.0)."""
        interp = FrameInterpolator(60.0, 23.976)
        mappings = interp.calculate_timesteps(100)
        for k, src_idx, t in mappings:
            assert 0 <= src_idx < 100, f"Source index {src_idx} out of range"
            assert 0.0 <= t < 1.0, f"Timestep t={t} not in [0.0, 1.0)"


class TestSingleAndTwoFrameSequences:
    """Stress test minimal edge case sequences (0, 1, and 2 frames)."""

    def test_single_frame_sequence(self):
        """Single-frame stream must output round(1 * factor) frames without crashing."""
        interp = FrameInterpolator(60.0, 30.0, engine="flow_dis")
        single = np.full((48, 48, 3), 77, dtype=np.uint8)
        out = list(interp.interpolate_sequence([single]))
        assert len(out) == 2
        assert np.array_equal(out[0], single)
        assert np.array_equal(out[1], single)

    def test_two_frame_sequence(self):
        """Two-frame stream must synthesize intermediate frame and tail pad correctly."""
        interp = FrameInterpolator(60.0, 30.0, engine="flow_dis")
        f0 = np.full((48, 48, 3), 10, dtype=np.uint8)
        f1 = np.full((48, 48, 3), 200, dtype=np.uint8)
        out = list(interp.interpolate_sequence([f0, f1]))
        assert len(out) == 4
        assert out[0].shape == f0.shape
        assert out[1].shape == f0.shape
        assert out[2].shape == f0.shape
        assert out[3].shape == f0.shape

    def test_empty_frame_sequence(self):
        """Empty stream must cleanly yield 0 frames without IndexError or StopIteration error."""
        interp = FrameInterpolator(60.0, 30.0, engine="flow_dis")
        out = list(interp.interpolate_sequence([]))
        assert len(out) == 0
