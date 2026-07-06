import json
import shutil
import sys
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add backend directory to sys.path to allow absolute imports relative to backend/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app.core.database as db_mod
from app.core.database import UserDatabaseManager
from app.core.auth import create_access_token
from main import app

client = TestClient(app)

# Resolve data directories relative to backend root to verify outputs
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"

# Minimal valid 1-page PDF file byte stream with longer text content to verify chunk splitting
MINIMAL_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 3 3] /Resources << >> /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 93 >>\nstream\nBT /F1 1 Tf (This is a longer text string that will definitely be split into multiple chunks.) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000062 00000 n \n0000000121 00000 n \n0000000224 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n296\n%%EOF"


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolates user database for each test case and cleans up test data directories."""
    test_db = tmp_path / "test_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db
    
    # Clean up test directories to ensure absolute isolation
    for parent_dir in [UPLOADS_DIR, CHUNKS_DIR, BASE_DIR / "data" / "vectorstore"]:
        user_dir = parent_dir / "user-123"
        if user_dir.exists():
            try:
                shutil.rmtree(user_dir)
            except Exception:
                pass

    UserDatabaseManager.initialize_db()
    yield

    # Teardown cleanup
    for parent_dir in [UPLOADS_DIR, CHUNKS_DIR, BASE_DIR / "data" / "vectorstore"]:
        user_dir = parent_dir / "user-123"
        if user_dir.exists():
            try:
                shutil.rmtree(user_dir)
            except Exception:
                pass

    db_mod.DB_PATH = old_db_path


@pytest.fixture
def auth_headers():
    """Registers a test user and returns request headers with a valid Bearer JWT."""
    username = "testuploader"
    UserDatabaseManager.create_user(
        user_id="user-123",
        username=username,
        hashed_password="hashedpassword"
    )
    token = create_access_token({"sub": username})
    return {"Authorization": f"Bearer {token}"}


def test_health_check():
    """Verify that the health check endpoint returns 200 OK and expected payload (no auth required)."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_unauthorized():
    """Verify that upload requests without credentials return HTTP 401 Unauthorized."""
    files = {"file": ("test.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 401


def test_upload_invalid_extension(auth_headers):
    """Verify that unsupported extensions are immediately rejected with HTTP 400."""
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    response = client.post("/upload", files=files, headers=auth_headers)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_valid_pdf(auth_headers):
    """Verify that uploading a valid PDF successfully processes, stores, and chunks it."""
    files = {"file": ("test.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    response = client.post("/upload", files=files, headers=auth_headers)

    assert response.status_code == 202
    payload = response.json()
    assert "job_id" in payload
    assert payload["status"] == "pending"

    job_id = payload["job_id"]

    # Poll status endpoint until background ingestion completes
    status = "pending"
    for _ in range(50):
        status_res = client.get(f"/upload/{job_id}/status", headers=auth_headers)
        assert status_res.status_code == 200
        status_data = status_res.json()
        status = status_data["status"]
        if status in ["complete", "failed"]:
            break
        time.sleep(0.1)

    assert status == "complete"
    document_id = job_id

    # Verify that raw file exists in uploads folder
    raw_file_path = UPLOADS_DIR / "user-123" / f"{document_id}.pdf"
    assert raw_file_path.exists(), f"Raw file not persisted at {raw_file_path}"

    # Verify that serialized chunks exist in chunks folder
    chunks_file_path = CHUNKS_DIR / "user-123" / f"{document_id}.json"
    assert chunks_file_path.exists(), f"Chunks metadata not persisted at {chunks_file_path}"

    # Verify JSON content structure
    with open(chunks_file_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        assert metadata["document_id"] == document_id
        assert metadata["source_filename"] == "test.pdf"
        assert metadata["total_chunks"] > 0
        assert len(metadata["chunks"]) == metadata["total_chunks"]

        first_chunk = metadata["chunks"][0]
        assert "chunk_id" in first_chunk
        assert "text" in first_chunk
        assert "char_length" in first_chunk
        assert first_chunk["page_index"] == 0


def test_upload_size_limit_header(auth_headers):
    """Verify that content-length headers exceeding 50 MB are immediately rejected."""
    files = {"file": ("test.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    # Send content-length header indicating 60 MB and merge auth headers
    headers = {"content-length": str(60 * 1024 * 1024)}
    headers.update(auth_headers)
    
    response = client.post("/upload", files=files, headers=headers)
    assert response.status_code == 400
    assert "exceeds the 50 MB limit" in response.json()["detail"]


def test_upload_chunking_parameter_overrides(auth_headers):
    """Verify that dynamic query overrides affect the character splitting dimensions."""
    files = {"file": ("test_override.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    # Force tiny chunks of 5 characters and 2 overlap
    response = client.post("/upload?chunk_size=5&chunk_overlap=2", files=files, headers=auth_headers)
    assert response.status_code == 202

    payload = response.json()
    job_id = payload["job_id"]

    # Poll status endpoint until background ingestion completes
    status = "pending"
    for _ in range(50):
        status_res = client.get(f"/upload/{job_id}/status", headers=auth_headers)
        assert status_res.status_code == 200
        status_data = status_res.json()
        status = status_data["status"]
        if status in ["complete", "failed"]:
            break
        time.sleep(0.1)

    assert status == "complete"
    document_id = job_id

    chunks_file_path = CHUNKS_DIR / "user-123" / f"{document_id}.json"
    with open(chunks_file_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        assert metadata["total_chunks"] > 1
        for chunk in metadata["chunks"]:
            assert chunk["char_length"] <= 5
