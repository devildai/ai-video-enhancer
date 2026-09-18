"""RIFE v4 Timestep ONNX frame interpolation engine.

Provides AI neural frame interpolation running on CPU via ONNX Runtime with
CPUExecutionProvider. Features automated model weight downloading and caching,
resolution padding to multiples of 32, and seamless fallback to OpenCV DIS
Optical Flow when offline or model is unavailable.
"""

import os
import urllib.request
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.ai.interpolation.flow_dis import DISFlowInterpolator

logger = logging.getLogger("ai.interpolation.rife")

MODEL_PRIMARY_URL = "https://huggingface.co/walterlow/RIFE_fp32_timestep/resolve/main/RIFE_fp32_timestep.onnx"
MODEL_MIRROR_URL = "https://huggingface.co/FuryTMP/RIFE_fp32/resolve/main/RIFE_fp32.onnx"
DEFAULT_MODEL_FILENAME = "RIFE_fp32_timestep.onnx"


def get_default_model_dir() -> str:
    """Return default directory where RIFE weights are cached."""
    # Find project root or fallback to local directory
    curr = os.path.abspath(os.path.dirname(__file__))
    # Ascend up to find workspace root containing models or app
    for _ in range(5):
        if os.path.exists(os.path.join(curr, "models")) or os.path.exists(os.path.join(curr, "app")):
            return os.path.join(curr, "models", "rife")
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    return os.path.join(os.getcwd(), "models", "rife")


def get_rife_model_path(model_dir: Optional[str] = None) -> str:
    """Return absolute path to cached RIFE ONNX model."""
    target_dir = model_dir or get_default_model_dir()
    return os.path.join(target_dir, DEFAULT_MODEL_FILENAME)


def download_rife_model(
    destination_path: Optional[str] = None,
    timeout_s: int = 30,
) -> Optional[str]:
    """Download RIFE v4 Timestep ONNX model if not already present.

    Attempts primary Hugging Face source, then fallback mirror.
    Returns path if successful, None if download fails or offline.
    """
    dest = destination_path or get_rife_model_path()
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)

    # Check existing valid file (RIFE ONNX is ~20.6MB)
    if os.path.isfile(dest) and os.path.getsize(dest) > 10_000_000:
        logger.info("Found existing RIFE model weights at %s", dest)
        return dest

    temp_path = dest + ".tmp"
    urls = [MODEL_PRIMARY_URL, MODEL_MIRROR_URL]

    for url in urls:
        try:
            logger.info("Downloading RIFE weights from %s to %s", url, dest)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AIVideoEnhancementTool/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as response, open(temp_path, "wb") as out_file:
                chunk_size = 1024 * 512
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)

            if os.path.getsize(temp_path) > 10_000_000:
                if os.path.exists(dest):
                    os.remove(dest)
                os.rename(temp_path, dest)
                logger.info("Successfully downloaded RIFE model to %s", dest)
                return dest
            else:
                logger.warning("Downloaded model file too small (<10MB) from %s", url)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as exc:
            logger.warning("Failed downloading RIFE model from %s: %s", url, exc)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    logger.warning("All RIFE model download sources failed. Falling back to DIS Optical Flow.")
    return None


class RIFEInterpolator:
    """Neural Frame Interpolator using RIFE v4 Timestep ONNX.

    Runs on CPU with AVX-512 acceleration. Seamlessly falls back to DIS
    optical flow if ONNX Runtime or model weights are unavailable.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        auto_download: bool = True,
        num_threads: int = 4,
    ) -> None:
        """Initialize RIFE Interpolator.

        Args:
            model_path: Path to RIFE ONNX file. If None, uses default cache path.
            auto_download: If True and model file is absent, attempts download.
            num_threads: Number of CPU intra-op inference threads.
        """
        self.num_threads = num_threads
        self.session: Optional[Any] = None
        self.dis_fallback = DISFlowInterpolator()
        self.is_neural_ready = False
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.timestep_shape: List[Any] = []
        self.takes_combined_input = False

        if ort is None:
            logger.warning("onnxruntime is not installed. Using DIS Optical Flow fallback.")
            return

        target_path = model_path or get_rife_model_path()
        if not os.path.isfile(target_path) and auto_download:
            target_path = download_rife_model(target_path) or target_path

        if os.path.isfile(target_path) and os.path.getsize(target_path) > 10_000_000:
            try:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = self.num_threads
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                self.session = ort.InferenceSession(
                    target_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )

                inputs = self.session.get_inputs()
                self.input_names = [inp.name for inp in inputs]
                self.output_names = [out.name for out in self.session.get_outputs()]

                # Inspect whether input is combined (C=6) or separate (img0, img1)
                for inp in inputs:
                    if inp.name in ("timestep", "time", "t"):
                        self.timestep_shape = list(inp.shape)
                    elif inp.name in ("input", "inputs", "img"):
                        self.takes_combined_input = True

                self.is_neural_ready = True
                logger.info("RIFE ONNX engine initialized successfully from %s", target_path)
            except Exception as exc:
                logger.warning(
                    "Error initializing RIFE ONNX session from %s: %s. Using DIS fallback.",
                    target_path,
                    exc,
                )
                self.session = None
                self.is_neural_ready = False
        else:
            logger.warning("RIFE model weights not found at %s. Using DIS Optical Flow fallback.", target_path)

    @property
    def is_available(self) -> bool:
        """Return True if neural ONNX session is active."""
        return self.is_neural_ready and self.session is not None

    def interpolate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        """Interpolate an intermediate frame between frame1 and frame2 at timestep t.

        Args:
            frame1: First frame (uint8 RGB, H x W x 3).
            frame2: Second frame (uint8 RGB, H x W x 3).
            timestep: Continuous timestep t in [0.0, 1.0].

        Returns:
            Synthesized intermediate frame (uint8 RGB, H x W x 3).
        """
        if timestep <= 0.001:
            return frame1.copy()
        if timestep >= 0.999:
            return frame2.copy()

        # If neural model is not active, route to DIS optical flow
        if not self.is_available:
            return self.dis_fallback.interpolate(frame1, frame2, timestep)

        try:
            return self._run_rife_inference(frame1, frame2, timestep)
        except Exception as exc:
            logger.warning("RIFE neural inference encountered error (%s). Falling back to DIS.", exc)
            return self.dis_fallback.interpolate(frame1, frame2, timestep)

    def interpolate_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        timestep: float,
        is_scene_cut: bool = False,
    ) -> np.ndarray:
        """Interpolate intermediate frame with scene cut snapping.

        If is_scene_cut is True, snaps to nearest keyframe (frame1 if t < 0.5, else frame2).
        Otherwise performs neural / optical flow interpolation.
        """
        if is_scene_cut:
            return frame1.copy() if timestep < 0.5 else frame2.copy()
        return self.interpolate(frame1, frame2, timestep)

    def _run_rife_inference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        """Execute RIFE ONNX inference with padding to multiples of 32."""
        h, w = frame1.shape[:2]

        # Convert to uint8 RGB if needed
        f1 = frame1
        f2 = frame2
        if f1.dtype != np.uint8:
            if np.issubdtype(f1.dtype, np.floating) and f1.max() <= 1.0:
                f1 = (f1 * 255.0).clip(0, 255).astype(np.uint8)
                f2 = (f2 * 255.0).clip(0, 255).astype(np.uint8)
            else:
                f1 = f1.clip(0, 255).astype(np.uint8)
                f2 = f2.clip(0, 255).astype(np.uint8)

        # Pad height and width to multiples of 32
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32

        if pad_h > 0 or pad_w > 0:
            f1_pad = cv2.copyMakeBorder(f1, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            f2_pad = cv2.copyMakeBorder(f2, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            f1_pad = f1
            f2_pad = f2

        # Preprocess: normalize to [0.0, 1.0], transpose to (1, 3, H_pad, W_pad)
        i0 = (f1_pad.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        i1 = (f2_pad.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        # Prepare timestep tensor according to input shape requirement
        ts_val = float(timestep)
        if len(self.timestep_shape) == 0:
            ts_array = np.array(ts_val, dtype=np.float32)
        elif len(self.timestep_shape) == 1:
            ts_array = np.array([ts_val], dtype=np.float32)
        elif len(self.timestep_shape) == 4:
            ts_array = np.array([[[[ts_val]]]], dtype=np.float32)
        else:
            ts_array = np.array(ts_val, dtype=np.float32)

        # Build feed dict
        feed_dict: Dict[str, Any] = {}
        ts_key = "timestep" if "timestep" in self.input_names else (
            "time" if "time" in self.input_names else self.input_names[-1]
        )
        feed_dict[ts_key] = ts_array

        if self.takes_combined_input or len(self.input_names) == 2:
            input_key = self.input_names[0] if self.input_names[0] != ts_key else self.input_names[1]
            feed_dict[input_key] = np.concatenate([i0, i1], axis=1)
        else:
            # Separate inputs (e.g. img0, img1)
            feed_dict["img0"] = i0
            feed_dict["img1"] = i1

        # Run ONNX inference
        outputs = self.session.run(None, feed_dict)
        raw_out = outputs[0]

        # Crop back to original dimensions: (1, 3, H, W)
        cropped = raw_out[0, :, :h, :w]

        # Transpose back to (H, W, 3) and scale to uint8 [0, 255]
        out_frame = cropped.transpose(1, 2, 0)
        return np.clip(out_frame * 255.0, 0, 255).astype(np.uint8)
