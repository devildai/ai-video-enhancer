"""REST API routes for video upload, process management, preview, and download."""

import logging
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Header,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse

from app.config import (
    UPLOAD_DIR,
    OUTPUT_DIR,
    PREVIEW_DIR,
    MAX_UPLOAD_SIZE,
    ALLOWED_CONTAINER_EXTENSIONS,
)
from app.models import (
    UploadResponse,
    VideoMetadataModel,
    ProcessRequest,
    ProcessResponse,
    TaskStatusResponse,
    CompareFrameResponse,
    CancelResponse,
)
from app.media.probe import probe_video
from app.media.preview import extract_sample_frame
from app.ai.enhancement.enhancer import VideoEnhancer
from app.pipeline.task_store import task_store
from app.pipeline.runner import run_enhancement_pipeline

logger = logging.getLogger("app.api.routes")
router = APIRouter()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload video file and extract metadata",
)
async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
    """Accepts video container upload, validates stream, and extracts sample preview."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes) or filename missing.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_CONTAINER_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{ext}'. Allowed containers: "
                f"{', '.join(sorted(ALLOWED_CONTAINER_EXTENSIONS))}."
            ),
        )

    task_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{task_id}_{file.filename}"

    # Stream write uploaded file to disk while validating size
    total_bytes = 0
    try:
        with open(saved_path, "wb") as out_f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB buffer
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE:
                    saved_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds maximum limit of {MAX_UPLOAD_SIZE // (1024*1024)}MB.",
                    )
                out_f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save upload to disk: {exc}",
        )

    if total_bytes == 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # Validate container and extract metadata via ffprobe
    try:
        meta = probe_video(str(saved_path))
    except (ValueError, RuntimeError) as probe_err:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file is not a valid video container or video stream is missing. Details: {probe_err}",
        )

    # Extract initial sample frame JPEG for preview and comparison
    sample_ts = meta.duration / 2.0 if meta.duration > 0 else 0.0
    preview_orig_path = str(PREVIEW_DIR / f"{task_id}_orig.jpg")
    try:
        extract_sample_frame(str(saved_path), sample_ts, preview_orig_path)
    except Exception as sample_err:
        logger.warning("Sample frame extraction failed for %s: %s", task_id, sample_err)
        preview_orig_path = None

    # Register initial task state in TaskStore
    task_store.create_task(
        task_id=task_id,
        file_id=task_id,
        filename=file.filename,
        input_path=str(saved_path),
        original_preview_path=preview_orig_path,
        metadata=meta,
        file_size=total_bytes,
        status="queued",
        progress=0.0,
        stage="Uploaded",
    )

    metadata_model = VideoMetadataModel(
        width=meta.width,
        height=meta.height,
        fps=round(meta.fps, 2),
        duration=round(meta.duration, 2),
        nb_frames=meta.nb_frames,
        total_frames=meta.nb_frames,
        codec_name=meta.codec_name,
        video_codec=meta.codec_name,
        has_audio=meta.has_audio,
        file_size=total_bytes,
        aspect_ratio=f"{meta.width}:{meta.height}",
        audio_codec=meta.audio_codec,
        audio_channels=meta.audio_channels,
        audio_sample_rate=meta.audio_sample_rate,
    )

    return UploadResponse(
        task_id=task_id,
        file_id=task_id,
        filename=file.filename,
        file_size=total_bytes,
        metadata=metadata_model,
        preview_url=f"/api/preview/{task_id}/original",
    )


@router.post(
    "/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure presets and launch background enhancement",
)
def process_video(req: ProcessRequest) -> ProcessResponse:
    """Accepts task configuration presets and dispatches background worker."""
    task_id = req.get_task_id()
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task or file_id '{task_id}' not found.",
        )

    if task.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task '{task_id}' is already actively processing.",
        )

    resolution = req.get_resolution()
    fps = req.get_fps()
    mode = req.get_mode()

    task_store.update_task(
        task_id,
        status="queued",
        stage="Queued for processing",
        progress=0.0,
        mode=mode,
        resolution=resolution,
        target_fps=fps,
        error=None,
    )

    # Launch pipeline runner in background daemon thread
    worker_thread = threading.Thread(
        target=run_enhancement_pipeline,
        args=(task.task_id, resolution, fps, mode),
        daemon=True,
        name=f"Worker-{task.task_id[:8]}",
    )
    worker_thread.start()

    return ProcessResponse(
        task_id=task.task_id,
        status="processing",
        message="Enhancement job initiated",
        config={
            "resolution": resolution,
            "fps": fps if fps is not None else "Original",
            "mode": mode,
        },
    )


@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get current state and progress metrics of a task",
)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Poll current execution metrics, percentage, and completion URL."""
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        stage=task.stage,
        fps=task.fps,
        eta=task.eta,
        mode=task.mode or "speed",
        resolution=task.resolution or "Original",
        target_fps=task.target_fps,
        download_url=task.download_url,
        error=task.error,
        comparison_ready=task.comparison_ready,
        original_preview=f"/api/preview/{task.task_id}/original" if task.original_preview_path else None,
        enhanced_preview=f"/api/preview/{task.task_id}/enhanced" if task.enhanced_preview_path else None,
    )


@router.post(
    "/cancel/{task_id}",
    response_model=CancelResponse,
    summary="Cancel active enhancement task",
)
def cancel_task(task_id: str) -> CancelResponse:
    """Terminates active processing and removes intermediate scratch files."""
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    task_store.cancel_task(task_id)
    return CancelResponse(
        task_id=task.task_id,
        status="cancelled",
        message="Task cancelled successfully",
    )


@router.get(
    "/compare-frame/{task_id}",
    summary="Retrieve comparison frame URLs or image",
)
def get_compare_frame(
    task_id: str,
    timestamp: Optional[float] = None,
    accept: Optional[str] = Header(None),
) -> Response:
    """Generates enhanced sample frame and returns comparison metadata or image."""
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    orig_preview = task.original_preview_path
    if timestamp is not None or not orig_preview or not os.path.exists(orig_preview):
        ts = timestamp if timestamp is not None else 0.0
        new_orig = str(PREVIEW_DIR / f"{task.task_id}_orig.jpg")
        try:
            extract_sample_frame(task.input_path, ts, new_orig)
            task.original_preview_path = new_orig
            orig_preview = new_orig
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to extract sample frame: {e}",
            )

    enhanced_preview = task.enhanced_preview_path or str(PREVIEW_DIR / f"{task.task_id}_enhanced.jpg")
    if not os.path.exists(enhanced_preview) and orig_preview and os.path.exists(orig_preview):
        try:
            orig_bgr = cv2.imread(orig_preview)
            if orig_bgr is not None:
                enhancer = VideoEnhancer(
                    mode=task.mode or "speed",
                    target_resolution=task.resolution or "Original",
                )
                orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                enh_rgb = enhancer.enhance_frame(orig_rgb)
                enh_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(enhanced_preview, enh_bgr)
                task.enhanced_preview_path = enhanced_preview
                task.comparison_ready = True
        except Exception as e:
            logger.warning("Could not generate enhanced comparison frame: %s", e)

    # If client explicitly asks for image MIME type, stream JPEG directly
    if accept and "image/" in accept and os.path.exists(enhanced_preview):
        return FileResponse(enhanced_preview, media_type="image/jpeg")

    width = task.metadata.width if task.metadata else None
    height = task.metadata.height if task.metadata else None

    resp_data = CompareFrameResponse(
        task_id=task.task_id,
        ready=os.path.exists(enhanced_preview),
        original_url=f"/api/preview/{task.task_id}/original",
        enhanced_url=f"/api/preview/{task.task_id}/enhanced" if os.path.exists(enhanced_preview) else None,
        width=width,
        height=height,
    )
    return Response(content=resp_data.model_dump_json(), media_type="application/json")


@router.get("/preview/{task_id}/{frame_type}", summary="Serve original or enhanced sample frame JPEG")
def serve_preview_frame(task_id: str, frame_type: str) -> FileResponse:
    """Serves JPEG sample frame (original or enhanced)."""
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    f_type = frame_type.lower().strip()
    if f_type == "original":
        if not task.original_preview_path or not os.path.exists(task.original_preview_path):
            raise HTTPException(status_code=404, detail="Original preview frame not found.")
        return FileResponse(
            task.original_preview_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    elif f_type == "enhanced":
        enhanced_path = task.enhanced_preview_path or str(PREVIEW_DIR / f"{task.task_id}_enhanced.jpg")
        if not os.path.exists(enhanced_path):
            # Attempt on-the-fly generation if original exists
            if task.original_preview_path and os.path.exists(task.original_preview_path):
                try:
                    orig_bgr = cv2.imread(task.original_preview_path)
                    if orig_bgr is not None:
                        enhancer = VideoEnhancer(
                            mode=task.mode or "speed",
                            target_resolution=task.resolution or "Original",
                        )
                        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                        enh_rgb = enhancer.enhance_frame(orig_rgb)
                        cv2.imwrite(enhanced_path, cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR))
                        task.enhanced_preview_path = enhanced_path
                except Exception:
                    pass

        if os.path.exists(enhanced_path):
            return FileResponse(
                enhanced_path,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        raise HTTPException(status_code=404, detail="Enhanced preview frame not yet available.")
    else:
        raise HTTPException(status_code=400, detail=f"Invalid preview frame_type '{frame_type}'.")


@router.head("/download/{task_id}", summary="Check availability and headers for download")
def head_download_video(task_id: str) -> Response:
    """Handles HEAD requests for download endpoint."""
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    if task.status != "completed" or not task.output_path or not os.path.exists(task.output_path):
        raise HTTPException(status_code=404, detail="Output file not ready or not found.")

    file_size = os.path.getsize(task.output_path)
    headers = {
        "Content-Type": "video/mp4",
        "Content-Length": str(file_size),
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'attachment; filename="enhanced_{task.filename}"',
    }
    return Response(status_code=200, headers=headers)


@router.get("/download/{task_id}", summary="Download or stream completed enhanced MP4")
def download_video(task_id: str, request: Request) -> Response:
    """Streams completed MP4 with full HTTP Range request support for browser playback."""
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    if task.status == "cancelled":
        raise HTTPException(status_code=400, detail="Cannot download a cancelled task.")

    if task.status in ("queued", "processing", "pending"):
        raise HTTPException(status_code=400, detail="Video processing has not completed yet.")

    if not task.output_path or not os.path.exists(task.output_path):
        raise HTTPException(status_code=404, detail="Output video file not found on disk.")

    file_path = task.output_path
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("range")

    if range_header:
        # Handle HTTP 206 Partial Content
        # Format: bytes=start-end
        try:
            range_val = range_header.strip().lower()
            if not range_val.startswith("bytes="):
                raise ValueError("Invalid range prefix")

            byte_range = range_val[len("bytes="):].strip()
            parts = byte_range.split("-")
            start_str, end_str = parts[0], parts[1]

            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1

            if start >= file_size or end >= file_size or start > end:
                return Response(
                    status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                    headers={"Content-Range": f"bytes */{file_size}"},
                )

            chunk_size = end - start + 1

            def file_chunk_generator(path: str, offset: int, length: int):
                with open(path, "rb") as f:
                    f.seek(offset)
                    remaining = length
                    while remaining > 0:
                        read_size = min(remaining, 64 * 1024)
                        data = f.read(read_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": "video/mp4",
                "Content-Disposition": f'inline; filename="enhanced_{task.filename}"',
            }
            return StreamingResponse(
                file_chunk_generator(file_path, start, chunk_size),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers=headers,
                media_type="video/mp4",
            )
        except Exception:
            pass

    # Standard 200 OK download / streaming response
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=f"enhanced_{task.filename}",
        headers={"Accept-Ranges": "bytes"},
    )
