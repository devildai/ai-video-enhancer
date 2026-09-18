"""Server entrypoint, lifespan hooks, static file mounting, and CLI runner."""

import argparse
import logging
import os
import shutil
import sys
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    BASE_DIR,
    STATIC_DIR,
    DEFAULT_HOST,
    DEFAULT_PORT,
    ensure_directories,
    clean_temp_directories,
)
from app.api.routes import router as api_router
from app.api.sse_ws import router as sse_router, websocket_progress_stream
from app.pipeline.task_store import task_store
from app.media.probe import _find_binary

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Application lifespan manager initializing directories and cleaning scratch files."""
    logger.info("Initializing AI Video Enhancement backend...")

    # 1. Ensure required storage directories exist
    ensure_directories()

    # 2. Clean stale temporary scratch files on startup
    clean_temp_directories()

    # 3. Check FFmpeg and FFprobe binary availability
    ffprobe_bin = _find_binary("ffprobe")
    ffmpeg_bin = _find_binary("ffmpeg")
    logger.info("Using FFprobe binary: %s", ffprobe_bin)
    logger.info("Using FFmpeg binary: %s", ffmpeg_bin)

    # 4. Auto-launch browser if enabled
    no_browser = getattr(app_instance.state, "no_browser", False)
    host = getattr(app_instance.state, "host", DEFAULT_HOST)
    port = getattr(app_instance.state, "port", DEFAULT_PORT)

    if not no_browser:
        target_url = f"http://127.0.0.1:{port}"
        logger.info("Scheduling auto-launch of default browser to %s in 1.0s...", target_url)

        def launch_browser():
            try:
                webbrowser.open(target_url)
            except Exception as e:
                logger.warning("Browser auto-launch failed: %s", e)

        threading.Timer(1.0, launch_browser).start()

    yield

    # Shutdown sequence
    logger.info("Shutting down backend, cleaning active tasks and temp files...")
    task_store.cleanup_all_active_tasks()
    clean_temp_directories()


# Create FastAPI application instance
app = FastAPI(
    title="AI Video Enhancement Tool",
    description="Local web tool for AI-powered video super-resolution and frame interpolation",
    version="1.0.0",
    lifespan=lifespan,
)

# App state defaults
app.state.host = DEFAULT_HOST
app.state.port = DEFAULT_PORT
app.state.no_browser = False

# Enable CORS for local client interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api")
app.include_router(sse_router, prefix="/api")


# Mount WebSocket endpoint directly at /ws/progress/{task_id}
@app.websocket("/ws/progress/{task_id}")
async def ws_progress_endpoint(websocket: WebSocket, task_id: str) -> None:
    """Direct WebSocket route for progress streaming and cancellation."""
    await websocket_progress_stream(websocket, task_id)


# Mount static assets directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse, summary="Serve UI dashboard")
def read_root():
    """Serves the main application dark-themed single-page frontend."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>AI Video Enhancement Tool</h1><p>Frontend static files not found.</p>")


def main():
    """CLI entry point for running the server."""
    parser = argparse.ArgumentParser(description="AI Video Enhancement Tool")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open default browser")

    args = parser.parse_args()

    app.state.host = args.host
    app.state.port = args.port
    app.state.no_browser = args.no_browser

    print(f"Starting AI Video Enhancement Tool on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
