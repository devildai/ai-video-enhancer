"""Adversarial probe script for Challenger 2:
- Subprocess & Concurrency stress tests
- Media & Streaming stress tests
"""

import io
import os
import sys
import time
import uuid
import shutil
import asyncio
import threading
import subprocess
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.config import UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR, TEMP_DIR
from app.media.probe import probe_video
from app.media.audio import extract_audio, align_audio_sync
from app.media.stream import VideoFrameDecoder, VideoFrameEncoder
from app.pipeline.task_store import task_store
from app.pipeline.runner import run_enhancement_pipeline
from tests.fixtures.generator import generate_synthetic_video, run_ffmpeg

app.state.no_browser = True
client = TestClient(app)

def get_ffmpeg_pids():
    """Get list of running ffmpeg process PIDs using Windows tasklist."""
    try:
        res = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq ffmpeg.exe"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        pids = []
        for line in res.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2 and "ffmpeg" in parts[0].lower():
                pid_str = parts[1].strip(' "')
                if pid_str.isdigit():
                    pids.append(int(pid_str))
        return set(pids)
    except Exception as e:
        print(f"Error checking tasklist: {e}")
        return set()

def probe_range_requests():
    print("\n=== PROBING HTTP RANGE REQUESTS ===")
    # 1. Create a dummy completed task
    test_id = f"test-range-{uuid.uuid4().hex[:8]}"
    test_out = OUTPUT_DIR / f"{test_id}_enhanced.mp4"
    generate_synthetic_video(test_out, width=320, height=240, fps=30, duration=0.5, has_audio=False)
    file_size = test_out.stat().st_size
    print(f"Created test video at {test_out}, size={file_size} bytes")

    task_store.create_task(
        task_id=test_id,
        file_id=test_id,
        filename="test.mp4",
        input_path=str(test_out),
        output_path=str(test_out),
        status="completed",
        progress=100.0,
        stage="Completed",
    )

    range_tests = [
        ("bytes=0-100", 206, "Valid 101-byte range"),
        ("bytes=999999-100", 416, "Invalid range: start > end"),
        ("bytes=-0", 416, "Invalid range: suffix length 0 (RFC 9110)"),
        ("bytes=999999-", 416, "Invalid range: start >= file_size"),
        ("bytes=0-9999999", 206, "Range end >= file_size (RFC 9110 clamp to EOF)"),
        ("bytes=-100", 206, "Valid suffix range: last 100 bytes"),
        ("bytes=100-", 206, "Valid open-ended range: from 100 to EOF"),
        ("bytes=0-0", 206, "Valid single-byte range: first byte"),
        ("bytes=abc-def", 200, "Malformed range: ignore or 416"),
        ("bytes=-", 200, "Malformed range: both empty"),
        ("bytes=1-2-3", 200, "Malformed range: extra dashes"),
        ("invalidunit=0-10", 200, "Unsupported unit: ignore per RFC 9110"),
    ]

    results = []
    for header_val, expected_status, desc in range_tests:
        resp = client.get(f"/api/download/{test_id}", headers={"Range": header_val})
        actual_status = resp.status_code
        cr = resp.headers.get("content-range")
        cl = resp.headers.get("content-length")
        print(f"  Header: '{header_val}' -> Status: {actual_status} (expected {expected_status}), Content-Range: {cr}, Content-Length: {cl}")
        results.append({
            "header": header_val,
            "expected": expected_status,
            "actual": actual_status,
            "content_range": cr,
            "content_length": cl,
            "desc": desc,
        })

    # Cleanup test file
    test_out.unlink(missing_ok=True)
    return results

def probe_corrupt_uploads():
    print("\n=== PROBING CORRUPT & TRUNCATED UPLOADS ===")
    cases = [
        ("empty.mp4", b"", 400, "0-byte empty file"),
        ("one_byte.mp4", b"\x00", 400, "1-byte file"),
        ("random_garbage.mp4", os.urandom(1024), 400, "1KB random binary garbage"),
        ("truncated_ftyp.mp4", b"\x00\x00\x00\x18ftypmp42", 400, "Truncated ftyp box without moov"),
        ("corrupt_moov.mp4", b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isom\x00\x00\x00\x10moov\xff\xff\xff\xff", 400, "Corrupt moov box"),
        ("fake_text.mp4", b"This is not a video file at all.", 400, "Plain text masquerading as MP4"),
        ("audio_only_aac.mp4", None, 400, "Audio-only MP4 container without video stream"),
        ("truncated_real_mp4.mp4", None, 400, "Real MP4 truncated after 500 bytes"),
    ]

    # Generate an audio-only MP4
    temp_dir = Path("tests/fixtures/cache")
    temp_dir.mkdir(parents=True, exist_ok=True)
    audio_only_path = temp_dir / "audio_only.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=1", "-vn", "-c:a", "aac", str(audio_only_path)])
    with open(audio_only_path, "rb") as f:
        audio_only_bytes = f.read()

    # Generate a real valid MP4 and truncate it
    real_mp4_path = temp_dir / "real_for_trunc.mp4"
    generate_synthetic_video(real_mp4_path, width=160, height=120, fps=10, duration=0.3, has_audio=False)
    with open(real_mp4_path, "rb") as f:
        real_mp4_bytes = f.read()
    truncated_real_bytes = real_mp4_bytes[:500]

    # Update bytes in cases
    for i, (name, content, exp, desc) in enumerate(cases):
        if name == "audio_only_aac.mp4":
            cases[i] = (name, audio_only_bytes, exp, desc)
        elif name == "truncated_real_mp4.mp4":
            cases[i] = (name, truncated_real_bytes, exp, desc)

    results = []
    for fname, data, expected_code, desc in cases:
        upload_before = set(UPLOAD_DIR.glob("*"))
        resp = client.post("/api/upload", files={"file": (fname, io.BytesIO(data), "video/mp4")})
        upload_after = set(UPLOAD_DIR.glob("*"))
        new_files = upload_after - upload_before
        print(f"  Upload '{fname}' ({desc}) -> Status: {resp.status_code}, Leftover files in uploads: {len(new_files)}")
        results.append({
            "name": fname,
            "status": resp.status_code,
            "expected": expected_code,
            "detail": resp.json() if resp.status_code != 200 else None,
            "leaked_files": len(new_files),
        })
        for f in new_files:
            f.unlink(missing_ok=True)
    return results

def probe_abort_and_subprocesses():
    print("\n=== PROBING MID-PROCESSING ABORT & SUBPROCESS TERMINATION ===")
    initial_pids = get_ffmpeg_pids()
    print(f"Initial ffmpeg PIDs on system: {initial_pids}")

    # Generate a longer test video (e.g. 5 seconds, 30fps = 150 frames)
    temp_dir = Path("tests/fixtures/cache")
    test_vid = temp_dir / "long_test.mp4"
    generate_synthetic_video(test_vid, width=320, height=240, fps=30, duration=5.0, has_audio=True)

    # 1. Upload video
    with open(test_vid, "rb") as f:
        upload_resp = client.post("/api/upload", files={"file": ("long_test.mp4", f, "video/mp4")})
    assert upload_resp.status_code == 200
    task_id = upload_resp.json()["task_id"]
    print(f"Uploaded video, task_id: {task_id}")

    # 2. Launch process
    proc_resp = client.post("/api/process", json={"task_id": task_id, "resolution": "Original", "fps": "60fps", "mode": "speed"})
    assert proc_resp.status_code == 200
    print(f"Launched process for task {task_id}")

    # Wait until it is actively processing
    time.sleep(0.5)
    active_pids = get_ffmpeg_pids()
    print(f"Active ffmpeg PIDs during execution: {active_pids} (New PIDs: {active_pids - initial_pids})")

    # Send abort/cancel
    cancel_resp = client.post(f"/api/cancel/{task_id}")
    print(f"Cancel request status: {cancel_resp.status_code}, body: {cancel_resp.json()}")

    # Wait for thread to exit and clean up
    time.sleep(1.5)
    post_cancel_pids = get_ffmpeg_pids()
    zombie_pids = post_cancel_pids - initial_pids
    print(f"Post-cancel ffmpeg PIDs: {post_cancel_pids} (Orphaned/Zombie: {zombie_pids})")

    # Check temp and output files
    temp_files = list(TEMP_DIR.glob(f"*{task_id}*"))
    output_files = list(OUTPUT_DIR.glob(f"*{task_id}*"))
    print(f"Remaining temp files for {task_id}: {temp_files}")
    print(f"Remaining output files for {task_id}: {output_files}")

    # Check task status
    st_resp = client.get(f"/api/status/{task_id}")
    print(f"Final task status: {st_resp.json()['status']}")

    return {
        "zombie_pids": list(zombie_pids),
        "temp_files_count": len(temp_files),
        "output_files_count": len(output_files),
        "final_status": st_resp.json()["status"],
    }

def probe_complex_audio():
    print("\n=== PROBING COMPLEX AUDIO STREAMS ===")
    temp_dir = Path("tests/fixtures/cache")
    temp_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("audio_5_1.mp4", 6, 48000, "aac", "5.1 surround 48kHz AAC"),
        ("audio_96k.mp4", 2, 96000, "aac", "Stereo 96kHz AAC"),
        ("audio_22k.mp4", 1, 22050, "aac", "Mono 22.05kHz AAC"),
        ("audio_mp3.mp4", 2, 44100, "mp3", "Stereo 44.1kHz MP3 in MP4"),
        ("audio_ac3.mp4", 6, 48000, "ac3", "5.1 surround 48kHz AC3 in MP4"),
    ]

    results = []
    for fname, channels, srate, acodec, desc in configs:
        out_vid = temp_dir / fname
        if not out_vid.exists() or out_vid.stat().st_size == 0:
            cmd = [
                "-f", "lavfi", "-i", f"testsrc=duration=1.0:size=320x240:rate=30",
                "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={srate}:duration=1.0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
            ]
            if channels == 6:
                # generate 5.1 layout: FL+FR+FC+LFE+BL+BR
                cmd.extend(["-filter_complex", f"[1:a]pan=5.1|c0=c0|c1=c0|c2=c0|c3=c0|c4=c0|c5=c0[aout]", "-map", "0:v", "-map", "[aout]"])
            else:
                cmd.extend(["-ac", str(channels), "-map", "0:v", "-map", "1:a"])

            cmd.extend(["-c:a", acodec, str(out_vid)])
            run_ffmpeg(cmd)

        meta = probe_video(str(out_vid))
        print(f"  {fname} ({desc}): has_audio={meta.has_audio}, channels={meta.audio_channels}, sample_rate={meta.audio_sample_rate}, codec={meta.audio_codec}")

        # Test extraction
        extracted_audio = temp_dir / f"extracted_{fname}.m4a"
        has_extracted = extract_audio(str(out_vid), str(extracted_audio))
        print(f"    extract_audio -> {has_extracted}, size={extracted_audio.stat().st_size if extracted_audio.exists() else 0}")

        # Test audio alignment to 2.5s (duration stretch)
        aligned_audio = temp_dir / f"aligned_{fname}.m4a"
        align_audio_sync(2.5, str(extracted_audio), str(aligned_audio))
        # probe aligned
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(aligned_audio)],
            stdout=subprocess.PIPE, text=True
        )
        aligned_dur = float(res.stdout.strip() or 0)
        print(f"    align_audio_sync (target 2.5s) -> duration={aligned_dur:.3f}s (diff={abs(aligned_dur - 2.5):.4f}s)")

        results.append({
            "name": fname,
            "desc": desc,
            "has_audio": meta.has_audio,
            "channels": meta.audio_channels,
            "sample_rate": meta.audio_sample_rate,
            "codec": meta.audio_codec,
            "extracted": has_extracted,
            "aligned_duration": aligned_dur,
            "sync_drift": abs(aligned_dur - 2.5),
        })

    return results

if __name__ == "__main__":
    r_range = probe_range_requests()
    r_corrupt = probe_corrupt_uploads()
    r_abort = probe_abort_and_subprocesses()
    r_audio = probe_complex_audio()
    print("\n--- ALL PROBES COMPLETE ---")
