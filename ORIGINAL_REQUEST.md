# Original User Request

## Initial Request — 2026-09-18T14:49:52Z

Build a browser-based AI video enhancement tool that allows a user to upload a video, then processes it through AI-powered frame interpolation (to increase FPS) and AI-powered quality enhancement (genuine detail restoration, not just resolution upscaling), with configurable output resolution and frame rate presets. This is a personal-use local tool.

Working directory: C:\Users\devil\.gemini\antigravity\scratch\video_enhancer
Integrity mode: development

## Requirements

### R1. Web Interface — FastAPI Backend + Custom Dark-Themed Frontend

A local web application with:
- A FastAPI (Python) backend serving a custom HTML/CSS/JS frontend
- Modern dark theme with smooth animations and polished visual design
- Video upload via drag-and-drop or file picker (supports common formats: MP4, MKV, AVI, MOV, WebM)
- Input video preview and metadata display (resolution, FPS, duration, codec, file size)
- Configurable output settings the user can select before processing:
  - **Resolution presets**: Original, 1080p, 2K (1440p), 4K (2160p)
  - **FPS presets**: Original, 30fps, 60fps, 120fps
  - **Enhancement mode**: "Quality" (temporal multi-frame restoration — slower) vs "Speed" (per-frame enhancement — faster)
- Real-time processing progress with percentage, current step description, and ETA displayed via WebSocket or SSE
- Side-by-side or before/after comparison of a sample frame (original vs enhanced) shown before/during processing
- Download button for the completed enhanced video
- The app runs locally on localhost and opens in the user's default browser

### R2. AI Frame Interpolation Pipeline

Increase video frame rate using AI-based optical flow frame interpolation:
- Use RIFE (Real-Time Intermediate Flow Estimation) or an equivalent open-source frame interpolation model
- Support arbitrary FPS multiplication (e.g., 24→60, 30→120, 24→120)
- Integrate scene-cut detection (via PySceneDetect or FFmpeg scene filter) to prevent interpolation artifacts across hard cuts
- Must produce smooth, natural-looking interpolated frames without visible warping or ghosting artifacts on normal footage
- Must work on CPU (ncnn/Vulkan backend acceptable for cross-platform CPU execution, though slower)

### R3. AI Video Quality Enhancement Pipeline

Genuinely enhance video quality — restore lost detail, remove compression artifacts, sharpen textures — not merely upscale resolution:
- **Quality mode**: Use a temporal video super-resolution approach (such as RealBasicVSR or similar multi-frame model) that leverages information across neighboring frames for flicker-free, temporally consistent enhancement
- **Speed mode**: Use Real-ESRGAN or equivalent per-frame enhancement model optimized for fast inference
- Support upscaling to target resolutions (1080p, 2K, 4K) from lower-resolution input
- Handle both live-action and animation/anime content reasonably well
- Must work on CPU (ncnn backend or ONNX runtime acceptable)

### R4. Video Processing & Output Pipeline

- Use FFmpeg for video decoding, encoding, and audio handling
- Preserve original audio track and synchronize it correctly with the new frame rate
- Output in H.264 or H.265 encoded MP4 format with good compression quality
- Processing pipeline order: Decode → Enhance quality → Interpolate frames → Encode output
- Handle the full pipeline for short clips (up to 1 minute input duration)
- Show clear error messages if processing fails (e.g., unsupported codec, out of memory)

## Acceptance Criteria

### Web Interface
- [ ] Running `python main.py` (or equivalent entry point) starts the FastAPI server and the app is accessible at `http://localhost:<port>` in a browser
- [ ] The UI renders with a dark theme and allows uploading a video file via drag-and-drop or file picker
- [ ] After upload, the input video's metadata (resolution, FPS, duration, file size) is displayed
- [ ] The user can select output resolution (Original / 1080p / 2K / 4K), output FPS (Original / 30 / 60 / 120), and enhancement mode (Quality / Speed) before starting processing
- [ ] During processing, a progress indicator shows the current step and percentage
- [ ] After processing completes, the enhanced video can be downloaded

### Frame Interpolation
- [ ] A 30fps input video processed with 60fps target produces an output video with exactly 60fps (verified by `ffprobe`)
- [ ] Scene cuts in the input video do not produce warping/morphing artifacts in the output (scene detection is active)
- [ ] Interpolated frames are visually smooth and temporally coherent (not just duplicated frames)

### Quality Enhancement
- [ ] A 480p input video processed with 1080p target produces a 1920×1080 output (verified by `ffprobe`)
- [ ] The enhanced output shows visibly improved detail and sharpness compared to naive bilinear/bicubic upscaling (visual inspection of sample frames)
- [ ] Quality mode produces temporally consistent results without frame-to-frame flickering

### Audio & Output
- [ ] The output video contains the original audio track, properly synchronized
- [ ] The output file is a valid MP4 playable in standard video players (VLC, browser `<video>` tag)

### CPU Compatibility
- [ ] The entire pipeline runs successfully on a system without a dedicated GPU (CPU-only execution)
- [ ] Processing completes without crashing on a 10-second 480p 30fps test clip within a reasonable time
