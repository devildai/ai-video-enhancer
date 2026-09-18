# 🎬 AI Video Enhancer — For the Rest of Us

> **Everyone builds video enhancers for beefy GPUs and high-end PCs.**
> **I built this one for people like me — with a basic laptop and a dream.** 💻✨
>
> No GPU? No problem. This runs entirely on your CPU. Yeah, it's slower — but hey, it works.
> Go grab a coffee while your video gets the glow-up it deserves. ☕

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-purple?logo=onnx&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CPU](https://img.shields.io/badge/GPU-Not%20Required-orange)

---

## ⚡ What Does It Do?

Upload any video through your browser → pick your settings → get back an **enhanced, upscaled, smoother video**. Simple as that.

| Feature | Details |
|---------|---------|
| 🔍 **AI Quality Enhancement** | Genuine detail restoration using Real-ESRGAN neural networks — not just stretching pixels |
| 🎞️ **AI Frame Interpolation** | RIFE v4 optical flow — creates smooth intermediate frames, not duplicates |
| 📺 **Resolution Upscaling** | Up to **4K (2160p)** output from low-res input |
| 🏎️ **FPS Boost** | Up to **120fps** from any source frame rate |
| 🧠 **Two Enhancement Modes** | *Speed* (fast per-frame) or *Quality* (temporal multi-frame, flicker-free) |
| ✂️ **Scene Cut Detection** | No ugly warping artifacts at shot transitions |
| 🔊 **Audio Preserved** | Original audio stays perfectly synced |
| 🌙 **Dark Theme UI** | Clean glassmorphic design — because we code at night 🦇 |
| 💻 **100% CPU** | No GPU required. Runs on any laptop |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **FFmpeg** installed and on your PATH → [Download FFmpeg](https://ffmpeg.org/download.html)

### Installation

```bash
# Clone this repo
git clone https://github.com/devildai/ai-video-enhancer.git
cd ai-video-enhancer

# Install Python dependencies
pip install -r requirements.txt

# Run the app
python -m app.main
```

The app will automatically open your browser to `http://127.0.0.1:8000` 🎉

### First Run

On first run, the app will **automatically download** the AI model weights (~26MB total) from Hugging Face. After that, everything runs 100% offline.

---

## 🎮 How to Use

1. **Upload** — Drag & drop a video or click to browse (MP4, MKV, AVI, MOV, WebM)
2. **Configure** — Choose your output settings:
   - 📺 Resolution: Original → 1080p → 2K → **4K**
   - 🎞️ FPS: Original → 30 → 60 → **120fps**
   - ⚡ Mode: **Speed** (fast) or **Quality** (temporal, flicker-free)
3. **Process** — Hit start and watch real-time progress with ETA
4. **Download** — Grab your enhanced video!

---

## 🏗️ Architecture

```text
Browser (Dark Theme UI)
    ↕ Upload + SSE/WebSocket Progress
FastAPI Backend
    ↓
Pipeline Runner
    ↓
FFmpeg Decode → AI Enhance (Real-ESRGAN / Temporal VSR) → AI Interpolate (RIFE v4) → FFmpeg Encode
    ↓
Enhanced MP4 + Original Audio
```

---

## ⏱️ Performance Expectations

Let's be real — CPU inference is slow. But it works!

| Input | Output | Mode | Estimated Time* |
|-------|--------|------|----------------|
| 10s 480p 30fps | 1080p 60fps | Speed | ~5-10 min |
| 10s 480p 30fps | 1080p 60fps | Quality | ~15-25 min |
| 10s 480p 30fps | 4K 120fps | Speed | ~30-60 min |

*\*Times vary based on your CPU. Grab that coffee.* ☕

---

## 🧪 Testing

The project includes **203 tests** across unit and integration suites:

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/unit/          # Unit tests (116)
pytest tests/e2e/           # End-to-end tests (87)
```

---

## 🛠️ Tech Stack

- **Backend:** FastAPI + Uvicorn
- **AI Inference:** ONNX Runtime (CPU)
- **Frame Interpolation:** RIFE v4 Timestep ONNX
- **Quality Enhancement:** Real-ESRGAN Compact SRVGGNet ONNX
- **Temporal Stabilization:** DIS Optical Flow + Occlusion Masking
- **Video Processing:** FFmpeg (pipe streaming, zero temp files)
- **Frontend:** Vanilla HTML/CSS/JS with glassmorphic dark theme

---

## 🙏 Credits & Acknowledgments

This project stands on the shoulders of giants. Huge thanks to these amazing open-source projects:

| Project | Author | What It Does | Link |
|---------|--------|--------------|------|
| **Real-ESRGAN** | Xintao Wang (xinntao) | AI image/video super-resolution & quality restoration | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| **RIFE** | Zhewei Huang (hzwer) | Real-time intermediate flow estimation for frame interpolation | [GitHub](https://github.com/hzwer/arXiv2020-RIFE) |
| **Practical-RIFE** | Zhewei Huang (hzwer) | Production-ready RIFE models | [GitHub](https://github.com/hzwer/Practical-RIFE) |
| **ONNX Runtime** | Microsoft | Cross-platform ML inference engine | [GitHub](https://github.com/microsoft/onnxruntime) |
| **FastAPI** | Sebastián Ramírez (tiangolo) | Modern Python web framework | [GitHub](https://github.com/fastapi/fastapi) |
| **FFmpeg** | FFmpeg Team | The backbone of all video processing | [Website](https://ffmpeg.org/) |
| **OpenCV** | OpenCV Team | Computer vision library (DIS optical flow, image processing) | [GitHub](https://github.com/opencv/opencv) |
| **Hugging Face** | Hugging Face | Model hosting & distribution | [Website](https://huggingface.co/) |

### 🤖 AI-Assisted Development

This project was largely developed with the help of AI coding tools. I believe in transparency — AI helped write the code, but the idea, direction, and countless hours of debugging and testing were very much human. 🧠🤝🤖

---

## 📝 License

MIT License — do whatever you want with it. If you make something cool, let me know!

---

## 💬 Contributing

Found a bug? Want to add GPU support? Have a better model? PRs are welcome!

1. Fork it
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⭐ Star This Repo

If this helped you enhance a video on your potato laptop, drop a ⭐ — it means the world! 🥔✨

---

<p align="center">
  <i>Made with ❤️ for everyone who doesn't have a GPU</i>
</p>