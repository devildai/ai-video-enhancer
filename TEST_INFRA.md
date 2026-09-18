# AI Video Enhancement Tool — Test Infrastructure Specification

## 1. Overview
This document defines the architecture, design philosophy, and execution protocol for the independent, opaque-box End-to-End (E2E) test suite of the AI Video Enhancement Tool.

The testing infrastructure implements a **4-Tier Testing Methodology** that validates external system interfaces (HTTP REST, Server-Sent Events, WebSockets, and multimedia containers) without relying on internal implementation details.

---

## 2. Test Architecture

```
tests/
├── fixtures/
│   ├── __init__.py
│   ├── generator.py            # Automated FFmpeg lavfi synthetic media generator
│   └── cache/                  # 17 cached test fixtures (MP4, MKV, AVI, MOV, WebM, audio/silent)
├── e2e/
│   ├── __init__.py
│   ├── conftest.py             # Session/function fixtures, ASGI & live client provider
│   ├── helpers.py              # ffprobe oracle, SSE parser, OpenCV sharpness & FastStart validator
│   ├── test_tier1_features.py  # Tier 1: 45 tests across 9 core features
│   ├── test_tier2_boundaries.py# Tier 2: 30 tests across 6 boundary/corner areas
│   ├── test_tier3_combinations.py # Tier 3: 7 cross-feature combination matrix tests
│   └── test_tier4_workloads.py # Tier 4: 5 real-world workload & CPU pipeline stress tests
├── unit/                       # Unit tests co-located for isolated modules
├── run_e2e_tests.py            # Standalone E2E test runner CLI
└── pytest.ini                  # Pytest configuration & registered markers
```

---

## 3. Four-Tier Testing Methodology

### Tier 1: Core Feature Coverage (>=5 tests per core feature, 45 tests total)
- **Feature 1 — Upload Container Support**: Accepts MP4, MKV, AVI, MOV, WebM containers.
- **Feature 2 — Metadata Inspection**: Accurately extracts width, height, float FPS, duration, codecs, and audio presence.
- **Feature 3 — Resolution Presets**: Supports Original, 1080p, 2K (1440p), 4K (2160p), and preserves aspect ratio.
- **Feature 4 — FPS Multiplication Presets**: Supports Original, 30fps, 60fps, 120fps, and fractional multipliers (24fps -> 60fps).
- **Feature 5 — Enhancement Modes**: Supports Speed mode, Quality mode, schema validation, and config reflection in task status.
- **Feature 6 — Progress Streaming**: Validates SSE `/api/progress/{task_id}`, WebSocket `/ws/progress/{task_id}`, and polling `/api/status/{task_id}`.
- **Feature 7 — Frame Comparison Preview**: Validates `/api/compare-frame/{task_id}` with timestamp offset, image headers, and 404 for nonexistent tasks.
- **Feature 8 — Audio Preservation & Sync**: Preserves source audio track, maintains sync drift < 0.15s, and validates AAC compatibility.
- **Feature 9 — Video Download**: Validates `/api/download/{task_id}` HTTP 200, `video/mp4` MIME type, HTTP Range header support, and FastStart atoms.

### Tier 2: Boundary & Corner Cases (>=5 tests per area, 30 tests total)
- **Area 1 — Non-Video Files**: Rejects `.txt`, corrupt video headers, `.exe`, `.py`, and missing filenames with HTTP 400.
- **Area 2 — 0-Byte Files**: Rejects empty MP4, MKV, generic binaries, returns informative error message, and avoids orphan task creation.
- **Area 3 — Odd Dimensions**: Ingests odd dimensions (e.g. 853x481, 319x241) and sanitizes to even dimensions for H.264 `yuv420p` encoding without encoder failure.
- **Area 4 — Silent Video**: Successfully processes videos with no audio track without muxer crashes.
- **Area 5 — Target FPS <= Source FPS**: Handles target FPS equal to or less than source FPS (e.g., 60fps -> 30fps) without loop or crash.
- **Area 6 — Job Cancellation**: Validates `/api/cancel/{task_id}` transitions task to "cancelled", stops processing, and cleans up resources.

### Tier 3: Cross-Feature Combinations (7 tests)
- Tests matrix permutations combining container formats, resolutions, frame rates, and audio modes:
  1. 480p 30fps -> 1080p 60fps Speed mode with audio
  2. 360p 24fps -> 1080p 60fps Quality mode with scene cuts
  3. 720p 30fps -> 2K 60fps Speed mode
  4. Silent video -> 1080p 60fps Quality mode
  5. MKV container -> 4K 30fps MP4
  6. WebM VP8 container -> 1080p 60fps MP4
  7. AVI container -> Original resolution 120fps MP4

### Tier 4: Real-World Workloads & Stress Tests (5 tests)
- **10-Second 480p 30fps Clip**: Enhanced to 1080p 60fps on CPU with audio and hard scene cuts.
- **Authoritative ffprobe Verification**: Exact 1920x1080 dimensions, exact 60.0 fps, audio stream present, and `yuv420p` pixel format.
- **FastStart Atom Verification**: Checks that `moov` atom precedes `mdat` for web streaming.
- **Audio Synchronization Drift Bound**: Audio-video sync maintained with duration delta < 0.15s.
- **HTML5 Browser Playback Compatibility**: Guaranteed browser playback compatibility via standard MP4 container and `yuv420p` pixel format.

---

## 4. Synthetic Video Fixture Generation

All test videos are generated synthetically using FFmpeg lavfi filters in `tests/fixtures/generator.py`:
- Fast generation (< 2 seconds total for entire suite).
- Completely self-contained and reproducible without external media downloads.
- Covers all container formats (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`).
- Synthesizes exact visual signals: color bars, test patterns, abrupt scene cuts, and 1000Hz sine waves.

---

## 5. Execution Protocol

### Prerequisites
- Python 3.10+ (Python 3.13 tested)
- FFmpeg 7.x and FFprobe in system PATH
- Pytest, FastAPI, HTTPX installed

### Running the Entire Suite
```powershell
python tests/run_e2e_tests.py
```
or via standard pytest:
```powershell
python -m pytest tests/e2e -v
```

### Running Specific Tiers
```powershell
python tests/run_e2e_tests.py --tier 1
python tests/run_e2e_tests.py --tier 2
python tests/run_e2e_tests.py --tier 3
python tests/run_e2e_tests.py --tier 4
```

### Testing Against a Live Server
```powershell
python tests/run_e2e_tests.py --base-url http://127.0.0.1:8000
```
or:
```powershell
$env:TEST_BASE_URL = "http://127.0.0.1:8000"
python -m pytest tests/e2e -v
```

---

## 6. Authoritative Test Oracles

| Oracle | Method | Validation Target |
|---|---|---|
| Container & Stream Metadata | `ffprobe -v quiet -print_format json` | Exact width, height, fps, codecs, duration |
| Web Streaming Optimization | Direct byte analysis for `moov` before `mdat` | FastStart MP4 readiness |
| Image Sharpness Gain | OpenCV `cv2.Laplacian(gray, cv2.CV_64F).var()` | Detail enhancement vs bicubic upscaling |
| Real-time Streaming | SSE `text/event-stream` parser & WebSocket listener | Valid JSON payloads, stage descriptions, progress % |
| Audio Sync Drift | `abs(audio_duration - video_duration)` | Delta bound < 0.15 seconds |
