"""Pytest configuration and shared fixtures for E2E tests."""
import os
import sys
import shutil
import tempfile
from pathlib import Path
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.generator import create_all_standard_fixtures

# Try importing the FastAPI application
try:
    from app.main import app
    APP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    app = None
    APP_AVAILABLE = False


@pytest.fixture(scope="session")
def standard_fixtures() -> dict[str, Path]:
    """Ensure all synthetic test fixtures are generated and return path dictionary."""
    return create_all_standard_fixtures()


@pytest.fixture
def client():
    """Provides an HTTP test client for interacting with the backend.

    Supports either:
    1. Live server if TEST_BASE_URL is set (e.g. TEST_BASE_URL=http://127.0.0.1:8000)
    2. In-process FastAPI TestClient if app.main is implemented
    3. Graceful skip if app.main is pending implementation (Milestone M4)
    """
    base_url = os.environ.get("TEST_BASE_URL")
    if base_url:
        import httpx
        with httpx.Client(base_url=base_url, timeout=60.0) as http_client:
            yield http_client
    elif APP_AVAILABLE and app is not None:
        from fastapi.testclient import TestClient
        with TestClient(app) as test_client:
            yield test_client
    else:
        pytest.skip("FastAPI application (app.main) is pending implementation (Milestone M4) and TEST_BASE_URL is not set.")


@pytest.fixture
def sample_mp4(standard_fixtures) -> Path:
    return standard_fixtures["mp4_valid"]


@pytest.fixture
def sample_mkv(standard_fixtures) -> Path:
    return standard_fixtures["mkv_valid"]


@pytest.fixture
def sample_avi(standard_fixtures) -> Path:
    return standard_fixtures["avi_valid"]


@pytest.fixture
def sample_mov(standard_fixtures) -> Path:
    return standard_fixtures["mov_valid"]


@pytest.fixture
def sample_webm(standard_fixtures) -> Path:
    return standard_fixtures["webm_valid"]


@pytest.fixture
def silent_mp4(standard_fixtures) -> Path:
    return standard_fixtures["silent_mp4"]


@pytest.fixture
def odd_dim_mp4(standard_fixtures) -> Path:
    return standard_fixtures["odd_dim_853x481"]


@pytest.fixture
def scene_cut_mp4(standard_fixtures) -> Path:
    return standard_fixtures["scene_cut_2s"]


@pytest.fixture
def workload_10s_mp4(standard_fixtures) -> Path:
    return standard_fixtures["workload_10s"]


@pytest.fixture
def corrupt_file(standard_fixtures) -> Path:
    return standard_fixtures["corrupt_header"]


@pytest.fixture
def text_file(standard_fixtures) -> Path:
    return standard_fixtures["text_file"]


@pytest.fixture
def zero_byte_file(standard_fixtures) -> Path:
    return standard_fixtures["zero_byte"]


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for test output files, cleaned up afterwards."""
    temp_dir = Path(tempfile.mkdtemp(prefix="enhancer_e2e_"))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
