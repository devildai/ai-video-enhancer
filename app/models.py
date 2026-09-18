"""Pydantic request and response models for the AI Video Enhancement API."""

from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class VideoMetadataModel(BaseModel):
    """Metadata schema representing container and stream properties."""
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int
    codec_name: str
    has_audio: bool
    file_size: int
    aspect_ratio: Optional[str] = None
    total_frames: Optional[int] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    audio_channels: Optional[int] = None
    audio_sample_rate: Optional[int] = None


class UploadResponse(BaseModel):
    """Response returned upon successful video upload."""
    task_id: str
    file_id: str
    filename: str
    file_size: int
    metadata: VideoMetadataModel
    preview_url: str


class ProcessRequest(BaseModel):
    """Request payload to initiate video enhancement."""
    task_id: Optional[str] = None
    file_id: Optional[str] = None
    resolution: Optional[str] = None
    target_resolution: Optional[str] = None
    fps: Optional[Union[str, int, float]] = None
    target_fps: Optional[Union[str, int, float]] = None
    mode: Optional[str] = None
    enhancement_mode: Optional[str] = None

    @field_validator("mode", "enhancement_mode")
    @classmethod
    def validate_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            mode_str = str(v).lower().strip()
            if mode_str not in ("speed", "quality"):
                raise ValueError(
                    f"Invalid enhancement mode '{v}'. Must be 'speed' or 'quality'."
                )
            return mode_str
        return v

    @field_validator("fps", "target_fps")
    @classmethod
    def validate_fps(cls, v: Optional[Union[str, int, float]]) -> Optional[Union[str, float]]:
        if v is not None:
            s = str(v).lower().strip()
            if s in ("original", "source", "none", ""):
                return "Original"
            clean_s = s[:-3].strip() if s.endswith("fps") else s
            try:
                val = float(clean_s)
                if val <= 0.0:
                    raise ValueError(f"FPS must be strictly positive, got {v}")
                return val
            except ValueError:
                raise ValueError(f"Invalid FPS value '{v}'")
        return v

    @field_validator("resolution", "target_resolution")
    @classmethod
    def validate_resolution(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            res_str = str(v).lower().strip()
            valid_resolutions = {
                "original", "source", "1080p", "2k", "1440p", "4k", "2160p", "720p", "480p"
            }
            if res_str not in valid_resolutions:
                raise ValueError(
                    f"Invalid resolution preset '{v}'. Supported: Original, 1080p, 2K, 4K."
                )
            return v
        return v

    @model_validator(mode="after")
    def validate_task_reference(self) -> "ProcessRequest":
        tid = self.task_id or self.file_id
        if not tid:
            raise ValueError("Either 'task_id' or 'file_id' must be provided.")
        return self

    def get_task_id(self) -> str:
        tid = self.task_id or self.file_id
        assert tid is not None
        return str(tid)

    def get_resolution(self) -> str:
        return str(self.resolution or self.target_resolution or "Original")

    def get_fps(self) -> Optional[float]:
        val = self.fps if self.fps is not None else self.target_fps
        if val is None or str(val).lower() in ("original", "source", ""):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).lower().strip()
        clean = s[:-3].strip() if s.endswith("fps") else s
        return float(clean)

    def get_mode(self) -> str:
        return str(self.mode or self.enhancement_mode or "speed").lower()


class ProcessResponse(BaseModel):
    """Response returned when an enhancement process is launched."""
    task_id: str
    status: str = "processing"
    message: str = "Enhancement job initiated"
    config: Optional[Dict[str, Any]] = None


class TaskStatusResponse(BaseModel):
    """Current state and metrics of an enhancement task."""
    task_id: str
    status: str
    progress: float
    stage: str
    fps: float = 0.0
    eta: float = 0.0
    mode: Optional[str] = "speed"
    resolution: Optional[str] = "Original"
    target_fps: Optional[Union[float, str]] = None
    download_url: Optional[str] = None
    error: Optional[str] = None
    comparison_ready: bool = False
    original_preview: Optional[str] = None
    enhanced_preview: Optional[str] = None


class CompareFrameResponse(BaseModel):
    """URLs and readiness metadata for before/after comparison."""
    task_id: str
    ready: bool
    original_url: str
    enhanced_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class CancelResponse(BaseModel):
    """Response returned upon cancelling a task."""
    task_id: str
    status: str = "cancelled"
    message: str = "Task cancelled successfully"


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    detail: str
