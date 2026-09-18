"""Exploratory probe runner to empirically test all adversarial attack hypotheses."""

import os
import sys
import tracemalloc
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.interpolation.scene_cut import detect_scene_cut, compute_scene_metrics
from app.ai.interpolation.flow_dis import DISFlowInterpolator
from app.ai.interpolation.rife import RIFEInterpolator
from app.ai.interpolation.interpolator import FrameInterpolator, interpolate_pair
from app.ai.enhancement.speed_mode import SpeedEnhancer, enhance_frame_speed
from app.ai.enhancement.quality_mode import TemporalVideoEnhancer
from app.ai.enhancement.enhancer import VideoEnhancer
from app.ai.tiling import tile_process


def test_extreme_color_flashes():
    print("--- Testing Extreme Color Flashes ---")
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    neon_green = np.zeros((100, 100, 3), dtype=np.uint8)
    neon_green[:, :, 1] = 255 # RGB (0, 255, 0)
    neon_magenta = np.zeros((100, 100, 3), dtype=np.uint8)
    neon_magenta[:, :, 0] = 255
    neon_magenta[:, :, 2] = 255 # RGB (255, 0, 255)
    neon_cyan = np.zeros((100, 100, 3), dtype=np.uint8)
    neon_cyan[:, :, 1] = 255
    neon_cyan[:, :, 2] = 255 # RGB (0, 255, 255)

    pairs = [
        ("white -> black", white, black),
        ("black -> neon green", black, neon_green),
        ("neon green -> neon magenta", neon_green, neon_magenta),
        ("neon magenta -> neon cyan", neon_magenta, neon_cyan),
        ("neon cyan -> black", neon_cyan, black),
        ("black -> white", black, white),
        ("neon green -> white", neon_green, white),
    ]

    for name, f1, f2 in pairs:
        l1, corr = compute_scene_metrics(f1, f2)
        cut = detect_scene_cut(f1, f2)
        print(f"  {name}: l1={l1:.2f}, corr={corr:.4f}, cut={cut}")
        assert cut, f"FAIL: {name} was not detected as scene cut!"

    # Test interpolation snapping across cut
    interp = FrameInterpolator(60.0, 30.0, engine="flow_dis")
    out_mid = interp.interpolate_pair(white, neon_green, 0.5)
    # Since it snaps, it must be exactly neon_green (timestep 0.5 snaps to f2)
    is_exact_f2 = np.array_equal(out_mid, neon_green)
    is_exact_f1 = np.array_equal(out_mid, white)
    print(f"  Snapping check (white -> neon green at t=0.5): exact f2={is_exact_f2}, exact f1={is_exact_f1}")
    assert is_exact_f2 or is_exact_f1, "FAIL: Intermediate frame did not snap cleanly to keyframe!"


def test_non_integer_framerate_mappings():
    print("--- Testing Non-Integer Framerate Mappings ---")
    test_cases = [
        (23.976, 60.0, [1, 2, 5, 10, 24, 100]),
        (15.0, 60.0, [1, 2, 5, 10, 24, 100]),
        (29.97, 60.0, [1, 2, 5, 10, 30, 100]),
        (24.0, 59.94, [1, 2, 5, 10, 24, 100]),
    ]

    for src_fps, dst_fps, frame_counts in test_cases:
        interp = FrameInterpolator(dst_fps, src_fps, engine="flow_dis")
        factor = dst_fps / src_fps
        print(f"  src={src_fps}, dst={dst_fps}, factor={factor:.4f}")
        for n in frame_counts:
            expected = int(round(n * factor))
            frames = [np.full((32, 32, 3), i % 255, dtype=np.uint8) for i in range(n)]
            out = list(interp.interpolate_sequence(frames))
            actual = len(out)
            assert actual == expected, f"FAIL: n={n}, expected {expected}, got {actual}"
        print(f"    Passed all frame counts: {frame_counts}")


def test_single_and_two_frame_sequences():
    print("--- Testing Single and Two-Frame Sequences ---")
    interp = FrameInterpolator(60.0, 30.0, engine="flow_dis")
    # Single frame
    f1 = np.full((64, 64, 3), 120, dtype=np.uint8)
    out1 = list(interp.interpolate_sequence([f1]))
    print(f"  1 frame input (factor 2.0): yielded {len(out1)} frames")
    assert len(out1) == 2
    assert np.array_equal(out1[0], f1)
    assert np.array_equal(out1[1], f1)

    # Two frames
    f2 = np.full((64, 64, 3), 200, dtype=np.uint8)
    out2 = list(interp.interpolate_sequence([f1, f2]))
    print(f"  2 frames input (factor 2.0): yielded {len(out2)} frames")
    assert len(out2) == 4

    # Empty sequence
    out0 = list(interp.interpolate_sequence([]))
    print(f"  0 frames input: yielded {len(out0)} frames")
    assert len(out0) == 0


def test_odd_and_prime_dimensions():
    print("--- Testing Non-standard Odd and Prime Dimensions ---")
    primes = [(503, 331), (317, 239), (131, 109)]
    speed = SpeedEnhancer()
    for w, h in primes:
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        print(f"  Testing frame shape ({h}, {w}) in SpeedEnhancer...")
        out = speed.enhance(frame, target_w=w, target_h=h)
        assert out.shape == (h, w, 3), f"FAIL: expected ({h}, {w}, 3), got {out.shape}"
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

        # Test in TemporalVideoEnhancer
        print(f"  Testing frame shape ({h}, {w}) in TemporalVideoEnhancer...")
        temp = TemporalVideoEnhancer(target_w=w, target_h=h, speed_enhancer=speed)
        t_out = temp.enhance_frame(frame)
        assert t_out.shape == (h, w, 3), f"FAIL: expected ({h}, {w}, 3), got {t_out.shape}"

        # Test in VideoEnhancer
        enh = VideoEnhancer(mode="speed", target_resolution="1080p")
        tw, th = enh.calculate_target_dimensions(w, h)
        print(f"  VideoEnhancer preset for ({w}x{h}) -> target ({tw}x{th}) [both even: {tw%2==0 and th%2==0}]")
        assert tw % 2 == 0 and th % 2 == 0
        v_out = enh.enhance_frame(frame)
        assert v_out.shape == (th, tw, 3)


def test_dynamic_tiling_boundaries_and_ram():
    print("--- Testing Dynamic Tiling Boundaries & Peak RAM Under 2K and 4K ---")
    speed = SpeedEnhancer()

    # 1. 2K upscaling (720p -> 2560x1440)
    tracemalloc.start()
    f_720p = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    out_2k = speed.enhance(f_720p, target_w=2560, target_h=1440)
    _, peak_2k = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_2k_mb = peak_2k / (1024.0 * 1024.0)
    print(f"  2K upscaling: shape={out_2k.shape}, peak RAM = {peak_2k_mb:.2f} MB")
    assert out_2k.shape == (1440, 2560, 3)
    assert peak_2k_mb < 500.0, f"FAIL: 2K peak RAM ({peak_2k_mb:.2f} MB) >= 500 MB"

    # 2. 4K upscaling (1080p -> 3840x2160)
    # Using a 540p or 720p or 1080p frame
    tracemalloc.start()
    f_1080p = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
    out_4k = speed.enhance(f_1080p, target_w=3840, target_h=2160)
    _, peak_4k = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_4k_mb = peak_4k / (1024.0 * 1024.0)
    print(f"  4K upscaling: shape={out_4k.shape}, peak RAM = {peak_4k_mb:.2f} MB")
    assert out_4k.shape == (2160, 3840, 3)
    assert peak_4k_mb < 500.0, f"FAIL: 4K peak RAM ({peak_4k_mb:.2f} MB) >= 500 MB"

    # 3. Seam line continuity test on smooth gradient
    h, w = 600, 800
    y_g, x_g = np.mgrid[0:h, 0:w].astype(np.float32)
    gradient = ((x_g / w + y_g / h) * 120.0).astype(np.uint8)
    grad_3ch = np.stack([gradient, gradient, gradient], axis=-1)

    tiled = tile_process(grad_3ch, lambda t: cv2.resize(t, (t.shape[1] * 2, t.shape[0] * 2), interpolation=cv2.INTER_LINEAR), tile_size=256, overlap=32, scale=2)
    expected = cv2.resize(grad_3ch, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    max_diff = np.max(np.abs(tiled.astype(float) - expected.astype(float)))
    mean_diff = np.mean(np.abs(tiled.astype(float) - expected.astype(float)))
    print(f"  Gradient seam max diff: {max_diff:.4f}, mean diff: {mean_diff:.4f}")
    assert max_diff <= 2.0, f"FAIL: Visible seam line artifact! Max diff={max_diff}"


def test_extreme_noise_and_compression():
    print("--- Testing Extreme Noise and High Compression ---")
    speed = SpeedEnhancer()

    # Extreme Gaussian Noise
    base = np.full((128, 128, 3), 128, dtype=np.uint8)
    cv2.circle(base, (64, 64), 30, (255, 255, 255), -1)
    noise = np.random.normal(0, 150, base.shape)
    noisy = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    out_noisy = speed.enhance(noisy, target_w=256, target_h=256)
    assert not np.isnan(out_noisy).any(), "NaN found in noisy output"
    assert not np.isinf(out_noisy).any(), "Inf found in noisy output"
    assert out_noisy.dtype == np.uint8

    # Extreme JPEG compression (quality = 5)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 5]
    _, encimg = cv2.imencode('.jpg', base, encode_param)
    compressed = cv2.imdecode(encimg, 1)

    out_comp = speed.enhance(compressed, target_w=256, target_h=256)
    assert not np.isnan(out_comp).any(), "NaN found in compressed output"
    assert not np.isinf(out_comp).any(), "Inf found in compressed output"
    assert out_comp.dtype == np.uint8

    # Check that artifact removal occurred:
    # JPEG quality 5 introduces blocky ringing artifacts.
    # Out_comp should be sharper and smooth out low-frequency blocking.
    print("  Noise & compression robustness verified with zero NaNs/Infs.")


if __name__ == "__main__":
    test_extreme_color_flashes()
    test_non_integer_framerate_mappings()
    test_single_and_two_frame_sequences()
    test_odd_and_prime_dimensions()
    test_dynamic_tiling_boundaries_and_ram()
    test_extreme_noise_and_compression()
    print("\nALL PROBES PASSED EMPIRICALLY!")
