# Project: AI Video Enhancement Tool

## Architecture
The AI Video Enhancement Tool is a personal-use, desktop-local web application built with FastAPI (Python 3.13) and a custom dark-themed frontend (HTML5/CSS3/Vanilla JS). It enhances video resolution and increases frame rate using genuine AI models running on CPU (AVX-512 accelerated) with low memory footprints (<500MB RAM via streaming FFmpeg rawvideo pipes and dynamic overlapping tiling).

### Data Flow
1. **Client Interaction**: User uploads video (MP4/MKV/AVI/MOV/WebM) via drag-and-drop.
2. **Metadata & Preview**: Backend inspects container via `ffprobe`, extracts initial sample frame, displays metadata (resolution, FPS, duration, codec, size).
3. **Preset Configuration**: User configures:
   - Resolution: Original, 1080p, 2K, 4K
   - FPS: Original, 30fps, 60fps, 120fps
   - Enhancement Mode: Quality (Temporal multi-frame) vs Speed (Per-frame compact neural)
4. **Execution Pipeline**:
   - `FFmpeg Decoder Stream`: Pipes raw RGB24 frames to Python generator.
   - `Audio Extractor`: Demuxes source audio stream (`-c:a copy` or AAC).
   - `AI Quality Enhancer`: Reconstructs genuine high-frequency details (Speed mode: Real-ESRGAN Compact SRVGGNet; Quality mode: Temporal Video Super-Resolution with backward optical flow warping).
   - `AI Frame Interpolator`: Multiplies frame rate to target FPS using RIFE v4 Timestep ONNX / DIS optical flow with in-flight scene-cut detection snapping.
   - `FFmpeg Encoder Stream`: Streams enhanced/interpolated frames into FFmpeg encoding H.264/H.265 MP4 (`-pix_fmt yuv420p`, `-movflags +faststart`).
   - `Muxer`: Attaches preserved audio stream with zero duration drift.
5. **Real-Time Feedback & Delivery**: WebSocket and SSE stream live percentage, stage description, and ETA to client. Interactive before/after split slider compares original vs enhanced sample frame. Download button serves finished MP4.

### Code Layout
```
C:\Users\devil\.gemini\antigravity\scratch\video_enhancer\
├── app\
│   ├── __init__.py
│   ├── main.py                    # Server entrypoint & lifespan hooks
│   ├── config.py                  # Presets, directory paths, limits
│   ├── models.py                  # Pydantic schemas for requests/responses
│   ├── api\
│   │   ├── __init__.py
│   │   ├── routes.py              # REST endpoints (/upload, /process, /status, /download, etc.)
│   │   └── sse_ws.py              # SSE (/api/progress/{id}) and WebSocket (/ws/progress/{id})
│   ├── media\
│   │   ├── __init__.py
│   │   ├── probe.py               # ffprobe metadata extraction
│   │   ├── audio.py               # Audio extraction, preservation, sync
│   │   ├── stream.py              # FFmpeg rawvideo streaming decoder and encoder
│   │   └── preview.py             # Sample frame extraction and comparison preview
│   ├── ai\
│   │   ├── __init__.py
│   │   ├── weights.py             # Automated model weight downloader & cache manager
│   │   ├── tiling.py              # Overlapping dynamic tiling with cosine blending
│   │   ├── interpolation\
│   │   │   ├── __init__.py
│   │   │   ├── scene_cut.py       # L1 difference & HSV histogram scene cut detection
│   │   │   ├── rife.py            # RIFE v4 Timestep ONNX engine
│   │   │   ├── flow_dis.py        # OpenCV DIS optical flow fallback engine
│   │   │   └── interpolator.py    # Unified interpolator with arbitrary multiplier
│   │   └── enhancement\
│   │       ├── __init__.py
│   │       ├── speed_mode.py      # Real-ESRGAN Compact SRVGGNet ONNX
│   │       ├── quality_mode.py    # Temporal consistency flow-aggregated super-resolution
│   │       └── enhancer.py        # Unified resolution scaler & detail enhancer
│   ├── pipeline\
│   │   ├── __init__.py
│   │   ├── runner.py              # Streaming execution loop (Decode -> Enhance -> Interpolate -> Encode)
│   │   └── task_store.py          # In-memory task state, cancellation, and progress publisher
│   └── static\
│       ├── index.html             # Dark-themed modern web UI
│       ├── css\
│       │   └── style.css          # Design system, glassmorphism, responsive grid, animations
│       └── js\
│           └── app.js             # Upload, preview, split slider, SSE/WS listener, presets
├── tests\
│   ├── e2e\                       # Independent opaque-box test suite (Tiers 1-4)
│   └── unit\                      # Unit tests for media, interpolation, enhancement, api
├── PROJECT.md
├── ORIGINAL_REQUEST.md
└── requirements.txt
```

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Server Entrypoint & Port Binding | Starts FastAPI app on 127.0.0.1:8000 (configurable via CLI `--port`, `--no-browser`) | M4 | ORIGINAL_REQUEST.md R1 |
| 2 | Auto-Launch Browser | Automatically opens default browser to `http://localhost:<port>` on launch | M4 | ORIGINAL_REQUEST.md R1 |
| 3 | Drag-and-Drop & File Picker Upload | Accepts MP4, MKV, AVI, MOV, WebM video uploads | M5 | ORIGINAL_REQUEST.md R1 |
| 4 | Video Metadata Extraction | Uses `ffprobe` to inspect dimensions, FPS, duration, codec, size | M1 | ORIGINAL_REQUEST.md R1, R4 |
| 5 | Sample Frame Extraction | Extracts middle frame JPEG for preview and before/after comparison | M1 | ORIGINAL_REQUEST.md R1 |
| 6 | Resolution Presets | Selectable presets: Original, 1080p, 2K (1440p), 4K (2160p) | M5 | ORIGINAL_REQUEST.md R1 |
| 7 | Frame Rate Presets | Selectable presets: Original, 30fps, 60fps, 120fps | M5 | ORIGINAL_REQUEST.md R1, R2 |
| 8 | Enhancement Mode Selection | Selectable modes: Quality (temporal multi-frame) vs Speed (per-frame) | M5 | ORIGINAL_REQUEST.md R1, R3 |
| 9 | Enhancement Job Launch | `/api/process` launches asynchronous background processing task | M4 | ORIGINAL_REQUEST.md R1, R4 |
| 10 | Enhanced Sample Frame Preview | Generates enhanced sample frame for immediate comparison | M4 | ORIGINAL_REQUEST.md R1 |
| 11 | Interactive Split-View Comparison Slider | Draggable slider comparing original sample vs enhanced sample | M5 | ORIGINAL_REQUEST.md R1 |
| 12 | SSE Progress Stream | `/api/progress/{task_id}` streams percentage, stage, ETA | M4 | ORIGINAL_REQUEST.md R1 |
| 13 | WebSocket Progress Stream | `/ws/progress/{task_id}` full-duplex live progress & control | M4 | ORIGINAL_REQUEST.md R1 |
| 14 | Task Status Polling | `/api/status/{task_id}` fallback status endpoint | M4 | ORIGINAL_REQUEST.md R1 |
| 15 | Job Cancellation | `/api/cancel/{task_id}` aborts processing & cleans temp files | M4 | Survey robustness |
| 16 | Audio Demuxing & Preservation | Demuxes source audio with `-c:a copy` or AAC re-encode | M1 | ORIGINAL_REQUEST.md R4 |
| 17 | Audio-Video Synchronization | Preserves exact audio sync across frame rate multiplication | M1 | ORIGINAL_REQUEST.md R4 |
| 18 | FFmpeg Streaming Frame Decoder | Decodes video directly to raw RGB24 frames | M1 | ORIGINAL_REQUEST.md R4 |
| 19 | RIFE AI Frame Interpolation | RIFE v4 Timestep ONNX model with arbitrary continuous multiplier | M2 | ORIGINAL_REQUEST.md R2 |
| 20 | Scene-Cut Detection | In-pipeline L1/HSV scene cut detector snapping to keyframes | M2 | ORIGINAL_REQUEST.md R2 |
| 21 | Optical Flow Fallback | OpenCV DIS optical flow fallback for fast interpolation | M2 | ORIGINAL_REQUEST.md R2 |
| 22 | Speed Mode Quality Enhancement | Real-ESRGAN Compact SRVGGNet ONNX for fast detail restoration | M3 | ORIGINAL_REQUEST.md R3 |
| 23 | Quality Mode Temporal Super-Resolution | Multi-frame temporal VSR with backward flow warping & occlusion gating | M3 | ORIGINAL_REQUEST.md R3 |
| 24 | Dynamic Memory-Safe Tiling | Overlapping tiles with cosine feathering for 2K/4K under 500MB RAM | M3 | Survey architecture |
| 25 | Sequential Processing Order | Pipeline order: Decode -> Enhance quality -> Interpolate -> Encode | M4 | ORIGINAL_REQUEST.md R4 |
| 26 | FFmpeg Encoding & FastStart Muxing | Encodes H.264/H.265 MP4 with `yuv420p` and `+faststart` | M1 | ORIGINAL_REQUEST.md R4 |
| 27 | Video Download & Streaming | `/api/download/{task_id}` serves MP4 with HTTP Range headers | M4 | ORIGINAL_REQUEST.md R1 |
| 28 | Dark Theme UI System | Modern dark UI with glassmorphism, responsive grid, glow accents | M5 | ORIGINAL_REQUEST.md R1 |
| 29 | Live Speed & ETA Calculation | Rolling average FPS and time-remaining estimator | M4 | ORIGINAL_REQUEST.md R1 |
| 30 | CPU-Only Execution Guarantee | Operates smoothly on CPU without requiring CUDA or dedicated GPU | M6 | ORIGINAL_REQUEST.md AC |
| 31 | Actionable Error Handling | Rejects corrupt/non-video files, handles odd dimensions & OOM | M4 | ORIGINAL_REQUEST.md R4 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| TEST | E2E Testing Track | Independent opaque-box test suite across Tiers 1-4 (infra, runner, ~50 test cases), creating TEST_INFRA.md and TEST_READY.md | none | IN_PROGRESS |
| M1 | Core Video Media & FFmpeg Pipeline | `app/media/`: probe, audio demux/sync, streaming decoder/encoder, sample preview | none | IN_PROGRESS |
| M2 | AI Frame Interpolation Pipeline | `app/ai/interpolation/`: scene cut detector, RIFE v4 ONNX, DIS optical flow fallback, arbitrary multiplier interpolator | M1 contracts | IN_PROGRESS |
| M3 | AI Video Quality Enhancement Pipeline | `app/ai/enhancement/`: model downloader/cache, dynamic tiling, Speed mode (Real-ESRGAN Compact), Quality mode (Temporal VSR) | M1 contracts | IN_PROGRESS |
| M4 | FastAPI Backend & Pipeline Orchestration | `app/api/`, `app/pipeline/`, `app/main.py`: REST routes, SSE/WebSocket streams, task store, runner loop, browser launcher | M1, M2, M3 | PENDING |
| M5 | Modern Dark-Themed Frontend | `app/static/`: dark UI, drag-and-drop, metadata badges, presets, split comparison slider, live progress bar, download | M4 | PENDING |
| M6 | Final Integration, E2E Verification & Hardening | Pass 100% E2E tests (Tiers 1-4), Tier 5 adversarial hardening with Challengers, Forensic Integrity Audit | M1-M5, TEST | PENDING |

---

## Interface Contracts

### Media Module (`app/media/`)
- `probe_video(video_path: str) -> VideoMetadata`:
  Returns `VideoMetadata(width: int, height: int, fps: float, duration: float, nb_frames: int, codec_name: str, has_audio: bool, file_size: int)`.
- `extract_sample_frame(video_path: str, timestamp_s: float, output_path: str) -> str`:
  Extracts single JPEG frame at specified timestamp.
- `extract_audio(video_path: str, output_audio_path: str) -> bool`:
  Extracts audio stream (`.m4a` / `.aac`). Returns `False` if no audio present.
- `VideoFrameDecoder(video_path: str)`:
  Generator yielding sequential `np.ndarray` (height, width, 3) in uint8 RGB.
- `VideoFrameEncoder(output_path: str, width: int, height: int, fps: float, audio_path: Optional[str] = None)`:
  Context manager with `write_frame(frame: np.ndarray)`. On exit, flushes and muxes audio into faststart MP4.

### Frame Interpolation Module (`app/ai/interpolation/`)
- `detect_scene_cut(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 30.0) -> bool`:
  Returns True if scene transition exceeds L1/HSV correlation threshold.
- `interpolate_pair(frame1: np.ndarray, frame2: np.ndarray, timestep: float, is_scene_cut: bool = False) -> np.ndarray`:
  Generates intermediate frame at continuous timestep `t in (0, 1)`. If `is_scene_cut=True`, snaps to nearest frame.
- `FrameInterpolator(target_fps: float, source_fps: float)`:
  Processes frame stream, calculating required timesteps and yielding interpolated RGB frames at target FPS.

### Quality Enhancement Module (`app/ai/enhancement/`)
- `enhance_frame_speed(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray`:
  Applies Real-ESRGAN Compact SRVGGNet with tiling if required, outputting sharp detail at target dimensions.
- `TemporalVideoEnhancer(target_w: int, target_h: int)`:
  Stateful class maintaining rolling feature/frame buffer. Enhances frame $F_t$ using spatial features and backward optical flow from $F_{t-1}$ with occlusion masking to eliminate frame-to-frame flicker.

### Task Store & Progress (`app/pipeline/task_store.py`)
- `TaskState(task_id: str, status: str, progress: float, stage: str, fps: float, eta: float, output_url: Optional[str], error: Optional[str])`:
  Data structure updated during processing.
- `subscribe_sse(task_id: str) -> AsyncGenerator[str, None]`:
  Yields SSE events formatted as `data: {...}\n\n`.
