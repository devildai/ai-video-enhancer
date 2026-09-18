"""In-pipeline scene cut detection using grayscale L1 difference and HSV histogram correlation.

Provides ultra-fast (<5ms) detection of abrupt scene transitions to prevent
interpolation morphing/rubber-band artifacts across cuts.
"""

from typing import Tuple
import numpy as np
import cv2


def compute_scene_metrics(
    frame1: np.ndarray,
    frame2: np.ndarray,
) -> Tuple[float, float]:
    """Compute normalized grayscale L1 difference and 2D HSV histogram correlation.

    Args:
        frame1: First frame as uint8 (H, W, 3) RGB or (H, W) grayscale.
        frame2: Second frame as uint8 (H, W, 3) RGB or (H, W) grayscale.

    Returns:
        Tuple of (l1_diff, hsv_correlation):
            - l1_diff: Mean absolute pixel difference in [0, 255].
            - hsv_correlation: HSV histogram correlation in [-1.0, 1.0], where
              1.0 indicates identical color distribution.
    """
    if frame1 is None or frame2 is None:
        return 255.0, -1.0

    if frame1.shape != frame2.shape:
        return 255.0, -1.0

    f1 = frame1
    f2 = frame2

    # Handle floating point images in [0, 1] or [0, 255]
    if f1.dtype != np.uint8:
        if np.issubdtype(f1.dtype, np.floating):
            max_val = max(float(f1.max()), float(f2.max()), 1.0)
            if max_val <= 1.0:
                f1 = (f1 * 255.0).clip(0, 255).astype(np.uint8)
                f2 = (f2 * 255.0).clip(0, 255).astype(np.uint8)
            else:
                f1 = f1.clip(0, 255).astype(np.uint8)
                f2 = f2.clip(0, 255).astype(np.uint8)
        else:
            f1 = f1.astype(np.uint8)
            f2 = f2.astype(np.uint8)

    # Color frames
    if len(f1.shape) == 3 and f1.shape[2] >= 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)

        hsv1 = cv2.cvtColor(f1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(f2, cv2.COLOR_RGB2HSV)

        # 2D histogram over Hue (30 bins, [0, 180]) and Saturation (32 bins, [0, 256])
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        if np.isnan(corr):
            corr = 1.0 if np.array_equal(f1, f2) else 0.0
    else:
        # Grayscale single-channel
        gray1 = f1 if len(f1.shape) == 2 else f1.squeeze()
        gray2 = f2 if len(f2.shape) == 2 else f2.squeeze()
        corr = 1.0 if np.array_equal(gray1, gray2) else 0.5

    l1_diff = float(np.mean(cv2.absdiff(gray1, gray2)))
    return l1_diff, corr


def detect_scene_cut(
    frame1: np.ndarray,
    frame2: np.ndarray,
    threshold: float = 30.0,
) -> bool:
    """Detect whether a scene cut (hard transition) occurs between frame1 and frame2.

    Combines normalized L1 difference of grayscale frames with 2D HSV color histogram
    correlation. Detects hard cuts, color flashes, and cross-scene transitions while
    resisting false positives on fast camera pans, object motion, or subtle lighting shifts.

    Args:
        frame1: First frame as uint8 (H, W, 3) RGB or (H, W) grayscale.
        frame2: Second frame as uint8 (H, W, 3) RGB or (H, W) grayscale.
        threshold: Sensitivity threshold (default 30.0). Higher values require larger
            changes to flag a cut.

    Returns:
        True if transition exceeds threshold indicating an abrupt scene cut, False otherwise.
    """
    if frame1 is None or frame2 is None:
        return True
    if frame1.shape != frame2.shape:
        return True

    l1_diff, corr = compute_scene_metrics(frame1, frame2)

    # Scale thresholds relative to the default 30.0
    scale = threshold / 30.0
    l1_thresh = threshold
    l1_extreme = 80.0 * scale

    # 1. Extreme luminance shift (e.g. black to white, major lighting flash)
    if l1_diff > l1_extreme:
        return True

    # 2. Significant luminance change AND color distribution change
    # (High correlation >= 0.85 indicates camera pan or moving object, not a cut)
    if l1_diff > l1_thresh and corr < 0.85:
        return True

    # 3. Complete color shift / uncorrelated color palette (even if luminance coincides)
    # e.g. green field cutting to orange desert, red room to blue room
    if corr < 0.35 and l1_diff > 0.5:
        return True

    # 4. Moderate color shift accompanied by subtle luminance change
    if corr < 0.55 and l1_diff > (15.0 * scale):
        return True

    return False
