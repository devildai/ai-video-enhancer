"""Configuration settings, directory paths, presets, and limits."""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple, Set

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage and scratch directories
UPLOAD_DIR = Path(os.getenv("ENHANCER_UPLOAD_DIR", str(BASE_DIR / "uploads")))
OUTPUT_DIR = Path(os.getenv("ENHANCER_OUTPUT_DIR", str(BASE_DIR / "outputs")))
PREVIEW_DIR = Path(os.getenv("ENHANCER_PREVIEW_DIR", str(BASE_DIR / "previews")))
TEMP_DIR = Path(os.getenv("ENHANCER_TEMP_DIR", str(BASE_DIR / "temp")))
STATIC_DIR = Path(os.getenv("ENHANCER_STATIC_DIR", str(BASE_DIR / "app" / "static")))

# Presets mapping
RESOLUTION_PRESETS: Dict[str, Optional[Tuple[int, int]]] = {
    "original": None,
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
    "2160p": (3840, 2160),
    "720p": (1280, 720),
}

FPS_PRESETS: Dict[str, Optional[float]] = {
    "original": None,
    "30": 30.0,
    "30fps": 30.0,
    "60": 60.0,
    "60fps": 60.0,
    "120": 120.0,
    "120fps": 120.0,
}

VALID_ENHANCEMENT_MODES: Set[str] = {"speed", "quality"}

ALLOWED_CONTAINER_EXTENSIONS: Set[str] = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
}

# Limits
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB

# Server defaults
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def ensure_directories() -> None:
    """Creates all required folders if they do not already exist."""
    for directory in [UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR, TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def clean_temp_directories() -> None:
    """Removes stale intermediate processing scratch files in TEMP_DIR."""
    if TEMP_DIR.exists():
        for item in TEMP_DIR.iterdir():
            try:
                if item.is_file():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
            except Exception:
                pass
