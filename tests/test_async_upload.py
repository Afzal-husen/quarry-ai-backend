import json
import shutil
import sys
import time
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app.core.database as db_mod
from app.core.database import UserDatabaseManager
from app.core.auth import create_access_token
from main import app

client = TestClient(app)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

MINIMAL_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 3 3] /Resources << >> /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 93 >>\nstream\nBT /F1 1 Tf (This is a longer text string that will definitely be split into multiple chunks.) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000009 00000 n \n0000000062 00000 n \n0000000121 00000 n \n0000000224 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n296\n%%EOF"


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolates user database for each test case and cleans up test data directories."""
    test_db = tmp_path / "test_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db

    # Clean up test directories to ensure absolute isolation
    for user_id in ["user-123", "user-456"]:
        for parent_dir in [UPLOADS_DIR, CHUNKS_DIR, VECTORSTORE_DIR]:
            user_dir = parent_dir / user_id
            if user_dir.exists():
                try:
                    shutil.rmtree(user_dir)
                except Exception:
                    pass

    UserDatabaseManager.initialize_db()
    yield

    # Teardown cleanup
    for user_id in ["user-123", "user-456"]:
        for parent_dir in [UPLOADS_DIR, CHUNKS_DIR, VECTORSTORE_DIR]:
            user_dir = parent_dir / user_id
            if user_dir.exists():
                try:
                    shutil.rmtree(user_dir)
                except Exception:
                    pass

    db_mod.DB_PATH = old_db_path


@pytest.fixture
def auth_headers():
    """Registers user-123 and returns request headers with a valid Bearer JWT."""
    username = "testuploader"
    UserDatabaseManager.create_user(
        user_id="user-123",
        username=username,
        hashed_password="hashedpassword"
    )
    token = create_access_token({"sub": username})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def other_auth_headers():
    """Registers user-456 and returns request headers with a valid Bearer JWT."""
    username = "otheruser"
    UserDatabaseManager.create_user(
        user_id="user-456",
        username=username,
        hashed_password="hashedpassword"
    )
    token = create_access_token({"sub": username})
    return {"Authorization": f"Bearer {token}"}


def test_status_endpoint_security_and_not_found(auth_headers, other_auth_headers):
    """Verify authorization rules and not found exceptions on job status endpoint."""
    # 1. Invalid UUID
    res_422 = client.get("/upload/not-a-uuid/status", headers=auth_headers)
    assert res_422.status_code == 422
    assert "must be a valid UUID string" in res_422.json()["detail"]

    # 2. Valid UUID but not found (404)
    fake_job_id = str(uuid.uuid4())
    res_404 = client.get(f"/upload/{fake_job_id}/status", headers=auth_headers)
    assert res_404.status_code == 404
    assert f"Job '{fake_job_id}' not found." in res_404.json()["detail"]

    # 3. Create a valid job for user-123
    files = {"file": ("test.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    upload_res = client.post("/upload", files=files, headers=auth_headers)
    assert upload_res.status_code == 202
    job_id = upload_res.json()["job_id"]

    # 4. Check status from other user user-456 (should return 403)
    res_403 = client.get(f"/upload/{job_id}/status", headers=other_auth_headers)
    assert res_403.status_code == 403
    assert "Forbidden" in res_403.json()["detail"]


def test_async_ingestion_failure_and_hard_cleanup(auth_headers):
    """Verify that corrupt files trigger background failures and execute a hard cleanup on disk."""
    # 1. Upload corrupt file which will fail PDF parsing
    corrupt_pdf_bytes = b"completely invalid pdf content that cannot be parsed"
    files = {"file": ("corrupt.pdf", corrupt_pdf_bytes, "application/pdf")}
    response = client.post("/upload", files=files, headers=auth_headers)
    assert response.status_code == 202

    job_id = response.json()["job_id"]

    # 2. Poll status endpoint until background ingestion fails
    status = "pending"
    error_msg = None
    for _ in range(50):
        status_res = client.get(f"/upload/{job_id}/status", headers=auth_headers)
        assert status_res.status_code == 200
        status_data = status_res.json()
        status = status_data["status"]
        error_msg = status_data["error"]
        if status in ["complete", "failed"]:
            break
        time.sleep(0.1)

    # 3. Assert status is failed and error is populated
    assert status == "failed"
    assert error_msg is not None

    # 4. Assert hard cleanup was executed on disk (no files or dirs remain for this doc)
    raw_path = UPLOADS_DIR / "user-123" / f"{job_id}.pdf"
    chunks_path = CHUNKS_DIR / "user-123" / f"{job_id}.json"
    vs_path = VECTORSTORE_DIR / "user-123" / job_id

    assert not raw_path.exists()
    assert not chunks_path.exists()
    assert not vs_path.exists()
