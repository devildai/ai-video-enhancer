#!/usr/bin/env python3
"""E2E Test Runner for AI Video Enhancement Tool.

Runs all 4 tiers of independent opaque-box E2E tests:
  Tier 1: Core Feature Coverage
  Tier 2: Boundary & Corner Cases
  Tier 3: Cross-Feature Combinations
  Tier 4: Real-World Workloads

Usage:
  python tests/run_e2e_tests.py [--tier {1,2,3,4}] [--base-url http://127.0.0.1:8000] [-v]
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.generator import create_all_standard_fixtures


def check_prerequisites():
    """Verify that python, ffmpeg, and ffprobe are installed and discoverable."""
    print("=" * 70)
    print("AI Video Enhancement Tool -- E2E Test Suite Runner")
    print("=" * 70)
    print(f"[+] Python Version: {sys.version.split()[0]}")

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path:
        print("[!] ERROR: 'ffmpeg' executable not found in PATH.", file=sys.stderr)
        sys.exit(1)
    if not ffprobe_path:
        print("[!] ERROR: 'ffprobe' executable not found in PATH.", file=sys.stderr)
        sys.exit(1)

    print(f"[+] FFmpeg: {ffmpeg_path}")
    print(f"[+] FFprobe: {ffprobe_path}")


def prepare_fixtures():
    """Pre-generate all synthetic video fixtures needed for the test run."""
    print("\n[+] Preparing synthetic video fixtures via FFmpeg lavfi...")
    fixtures = create_all_standard_fixtures()
    print(f"[+] Successfully verified {len(fixtures)} standard video fixtures.")
    return fixtures


def main():
    parser = argparse.ArgumentParser(description="Run E2E tests for AI Video Enhancement Tool.")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run tests only for a specific tier (1, 2, 3, or 4).")
    parser.add_argument("--base-url", type=str, help="Target a live running server (e.g. http://127.0.0.1:8000).")
    parser.add_argument("-k", type=str, help="Pytest keyword expression filter.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose pytest output.")
    parser.add_argument("--html", type=str, help="Generate HTML test report.")
    args = parser.parse_args()

    check_prerequisites()
    prepare_fixtures()

    if args.base_url:
        os.environ["TEST_BASE_URL"] = args.base_url
        print(f"[+] Testing against live server: {args.base_url}")
    else:
        print("[+] Testing mode: In-process ASGI / media contracts")

    # Build pytest argument list
    pytest_args = [sys.executable, "-m", "pytest", "tests/e2e"]

    if args.tier:
        pytest_args.extend(["-m", f"tier{args.tier}"])
        print(f"[+] Running Tier {args.tier} tests only.")
    else:
        print("[+] Running All Tiers (1-4).")

    if args.k:
        pytest_args.extend(["-k", args.k])

    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("--tb=short")

    print("\n" + "-" * 70)
    print("Executing pytest command: " + " ".join(pytest_args))
    print("-" * 70 + "\n")

    result = subprocess.run(pytest_args, cwd=str(PROJECT_ROOT))

    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("[PASS] E2E TEST RUN SUCCESSFUL -- All tests passed or cleanly evaluated.")
    else:
        print(f"[FAIL] E2E TEST RUN FAILED (Exit Code {result.returncode})", file=sys.stderr)
    print("=" * 70)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
