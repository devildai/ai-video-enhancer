"""Background pipeline execution runner coordinating decoding, enhancement, interpolation, and encoding."""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from app.config import OUTPUT_DIR, PREVIEW_DIR, TEMP_DIR
from app.media.probe import probe_video, VideoMetadata
from app.media.audio import extract_audio
from app.media.stream import VideoFrameDecoder, VideoFrameEncoder
from app.ai.enhancement.enhancer import VideoEnhancer
from app.ai.interpolation.interpolator import FrameInterpolator
from app.pipeline.task_store import task_store, TaskRecord

logger = logging.getLogger("app.pipeline.runner")


def run_enhancement_pipeline(
    task_id: str,
    resolution: str = "Original",
    fps: Optional[Union[str, float]] = "Original",
    mode: str = "speed",
) -> None:
    """Executes the full video enhancement pipeline asynchronously in a worker thread.

    Pipeline stages:
    1. Probe input video container & stream properties.
    2. Extract audio track if present.
    3. Generate enhanced preview sample frame for client comparison.
    4. Decode video stream frame-by-frame.
    5. Enhance spatial resolution and detail.
    6. Interpolate temporal frames to target frame rate.
    7. Encode output to H.264 MP4 with faststart and audio muxing.
    8. Check cancellation token on every frame.
    """
    task = task_store.get_task(task_id)
    if not task:
        logger.error("Task %s not found in store, aborting pipeline execution", task_id)
        return

    if task.cancellation_event.is_set():
        task_store.update_task(task_id, status="cancelled", stage="Cancelled")
        return

    logger.info(
        "Starting enhancement task %s [mode=%s, res=%s, fps=%s]",
        task_id,
        mode,
        resolution,
        fps,
    )

    task_store.update_task(
        task_id,
        status="processing",
        stage="Probing video container",
        progress=2.0,
        mode=mode,
        resolution=resolution,
        target_fps=fps,
    )

    output_path = str(OUTPUT_DIR / f"{task_id}_enhanced.mp4")
    audio_path: Optional[str] = None
    decoder: Optional[VideoFrameDecoder] = None
    encoder: Optional[VideoFrameEncoder] = None

    try:
        # 1. Probe video
        meta: VideoMetadata = probe_video(task.input_path)
        task_store.update_task(task_id, metadata=meta)

        if task.cancellation_event.is_set():
            task_store.update_task(task_id, status="cancelled", stage="Cancelled")
            return

        # 2. Extract audio if present
        if meta.has_audio:
            task_store.update_task(task_id, stage="Extracting audio track", progress=5.0)
            candidate_audio = str(TEMP_DIR / f"{task_id}_audio.m4a")
            try:
                has_audio = extract_audio(task.input_path, candidate_audio)
                if has_audio and os.path.exists(candidate_audio) and os.path.getsize(candidate_audio) > 0:
                    audio_path = candidate_audio
            except Exception as e:
                logger.warning("Audio extraction failed (%s), continuing with silent output", e)
                audio_path = None

        if task.cancellation_event.is_set():
            _cleanup_temp_files(audio_path, output_path)
            task_store.update_task(task_id, status="cancelled", stage="Cancelled")
            return

        # 3. Initialize AI models & resolve target dimensions
        task_store.update_task(task_id, stage="Initializing AI engines", progress=8.0)
        enhancer = VideoEnhancer(mode=mode, target_resolution=resolution)
        target_w, target_h = enhancer.calculate_target_dimensions(meta.width, meta.height)

        # Resolve target FPS
        if fps is None or str(fps).lower() in ("original", "source", "none", ""):
            target_fps = meta.fps
        else:
            s = str(fps).lower().strip()
            clean = s[:-3].strip() if s.endswith("fps") else s
            target_fps = float(clean)

        task_store.update_task(task_id, target_fps=target_fps)

        # 4. Generate enhanced preview sample frame if original preview exists
        if task.original_preview_path and os.path.exists(task.original_preview_path):
            enhanced_preview_path = str(PREVIEW_DIR / f"{task_id}_enhanced.jpg")
            try:
                orig_bgr = cv2.imread(task.original_preview_path)
                if orig_bgr is not None:
                    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                    enh_rgb = enhancer.enhance_frame(orig_rgb)
                    enh_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(enhanced_preview_path, enh_bgr)
                    task_store.update_task(
                        task_id,
                        enhanced_preview_path=enhanced_preview_path,
                        comparison_ready=True,
                    )
            except Exception as e:
                logger.warning("Could not generate enhanced sample preview: %s", e)

        if task.cancellation_event.is_set():
            _cleanup_temp_files(audio_path, output_path)
            task_store.update_task(task_id, status="cancelled", stage="Cancelled")
            return

        # 5. Initialize Frame Interpolator
        interpolator = FrameInterpolator(target_fps=target_fps, source_fps=meta.fps)

        # 6. Setup Decoder and Encoder
        decoder = VideoFrameDecoder(task.input_path)
        encoder = VideoFrameEncoder(
            output_path=output_path,
            width=target_w,
            height=target_h,
            fps=target_fps,
            audio_path=audio_path,
        )
        task_store.update_task(task_id, output_path=output_path)

        decoder._start_process()
        encoder.start()

        # 7. Streaming Pipeline Loop
        # Decoder -> Enhancer -> Interpolator -> Encoder
        def cancellable_source_stream():
            assert decoder is not None
            for in_frame in decoder:
                if task.cancellation_event.is_set():
                    break
                yield in_frame

        enhanced_stream = enhancer.enhance_sequence(cancellable_source_stream())
        interpolated_stream = interpolator.interpolate_stream(
            enhanced_stream, total_frames=meta.nb_frames
        )

        total_expected_out = max(1, int(round(meta.nb_frames * (target_fps / meta.fps))))
        frames_written = 0
        start_time = time.time()
        last_update_time = start_time

        task_store.update_task(
            task_id,
            stage=f"Processing frames [0/{total_expected_out}]",
            progress=10.0,
            fps=0.0,
            eta=0.0,
        )

        for out_frame in interpolated_stream:
            if task.cancellation_event.is_set():
                break

            encoder.write_frame(out_frame)
            frames_written += 1

            now = time.time()
            if now - last_update_time >= 0.25 or frames_written == total_expected_out:
                elapsed = max(0.001, now - start_time)
                current_fps = frames_written / elapsed
                remaining_frames = max(0, total_expected_out - frames_written)
                eta = remaining_frames / current_fps if current_fps > 0 else 0.0

                progress_pct = min(
                    99.0,
                    10.0 + 88.0 * (frames_written / total_expected_out),
                )

                if frames_written < total_expected_out * 0.4:
                    stage_name = f"Enhancing quality [{frames_written}/{total_expected_out}]"
                elif frames_written < total_expected_out * 0.85:
                    stage_name = f"Interpolating frames [{frames_written}/{total_expected_out}]"
                else:
                    stage_name = f"Encoding & Muxing [{frames_written}/{total_expected_out}]"

                task_store.update_task(
                    task_id,
                    progress=round(progress_pct, 1),
                    stage=stage_name,
                    fps=round(current_fps, 1),
                    eta=round(eta, 1),
                )
                last_update_time = now

        # 8. Handle Cancellation vs Completion
        if task.cancellation_event.is_set():
            logger.info("Task %s cancelled during frame processing", task_id)
            if encoder:
                encoder.close(raise_on_error=False)
            if decoder:
                decoder.close()
            _cleanup_temp_files(audio_path, output_path)
            task_store.update_task(task_id, status="cancelled", stage="Cancelled")
            return

        # Normal flush & close
        task_store.update_task(task_id, stage="Finalizing MP4 container", progress=99.0)
        encoder.close(raise_on_error=True)
        decoder.close()

        # Clean audio scratch file
        _cleanup_temp_files(audio_path, None)

        # 9. Verify output
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Output MP4 file is missing or empty at {output_path}")

        download_url = f"/api/download/{task_id}"
        task_store.update_task(
            task_id,
            status="completed",
            progress=100.0,
            stage="Completed",
            download_url=download_url,
            error=None,
        )
        logger.info("Task %s successfully completed. Output: %s", task_id, output_path)

    except Exception as exc:
        logger.exception("Error executing enhancement pipeline for task %s: %s", task_id, exc)
        if encoder:
            try:
                encoder.close(raise_on_error=False)
            except Exception:
                pass
        if decoder:
            try:
                decoder.close()
            except Exception:
                pass
        _cleanup_temp_files(audio_path, output_path)
        task_store.update_task(
            task_id,
            status="failed",
            stage="Failed",
            error=str(exc),
        )


def _cleanup_temp_files(audio_path: Optional[str], output_path: Optional[str]) -> None:
    """Safely cleans up temporary intermediate files."""
    for path in (audio_path, output_path):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
