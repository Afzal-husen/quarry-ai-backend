import json
import shutil
import sys
import time
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
    """Isolates user database for each test case and cleans up test directories."""
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


def test_list_documents_empty(auth_headers):
    """Verify that listing documents for a user with no uploads returns an empty list."""
    response = client.get("/documents", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 0
    assert payload["items"] == []


def test_list_documents_unauthorized():
    """Verify that listing documents without Bearer token returns 401."""
    response = client.get("/documents")
    assert response.status_code == 401


def test_document_lifecycle_flow(auth_headers, other_auth_headers):
    """Verify upload, list, reindex, delete lifecycle for a document."""
    # 1. Upload document as user-123
    files = {"file": ("lifecycle_test.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    upload_res = client.post("/upload", files=files, headers=auth_headers)
    assert upload_res.status_code == 202
    doc_id = upload_res.json()["job_id"]

    # Poll status endpoint until background ingestion completes
    status = "pending"
    for _ in range(50):
        status_res = client.get(f"/upload/{doc_id}/status", headers=auth_headers)
        assert status_res.status_code == 200
        status = status_res.json()["status"]
        if status in ["complete", "failed"]:
            break
        time.sleep(0.1)

    assert status == "complete"

    # Verify files created on disk
    raw_path = UPLOADS_DIR / "user-123" / f"{doc_id}.pdf"
    chunks_path = CHUNKS_DIR / "user-123" / f"{doc_id}.json"
    vs_path = VECTORSTORE_DIR / "user-123" / doc_id

    assert raw_path.exists()
    assert chunks_path.exists()
    assert vs_path.exists()

    # 2. List documents as user-123
    list_res = client.get("/documents", headers=auth_headers)
    assert list_res.status_code == 200
    payload = list_res.json()
    docs = payload["items"]
    assert len(docs) == 1
    doc = docs[0]
    assert doc["document_id"] == doc_id
    assert doc["filename"] == "lifecycle_test.pdf"
    assert doc["status"] == "complete"
    assert doc["can_reindex"] is True
    assert doc["upload_date"] != "unknown"

    # 3. List documents as user-456 (should be empty for other user)
    other_list_res = client.get("/documents", headers=other_auth_headers)
    assert other_list_res.status_code == 200
    other_payload = other_list_res.json()
    assert other_payload["total"] == 0
    assert other_payload["items"] == []

    # 4. Try to reindex user-123's document using user-456's token (should return 403)
    reindex_forbidden_res = client.post(f"/documents/{doc_id}/reindex", headers=other_auth_headers)
    assert reindex_forbidden_res.status_code == 403

    # 5. Try to delete user-123's document using user-456's token (should return 403)
    delete_forbidden_res = client.delete(f"/documents/{doc_id}", headers=other_auth_headers)
    assert delete_forbidden_res.status_code == 403

    # 6. Reindex document as user-123
    reindex_res = client.post(f"/documents/{doc_id}/reindex?chunk_size=10&chunk_overlap=2", headers=auth_headers)
    assert reindex_res.status_code == 200
    assert reindex_res.json()["status"] == "success"
    assert reindex_res.json()["document_id"] == doc_id

    # Check that chunks file was updated
    with open(chunks_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        for chunk in payload["chunks"]:
            assert len(chunk["text"]) <= 10

    # 7. Delete document as user-123 (should return 204)
    delete_res = client.delete(f"/documents/{doc_id}", headers=auth_headers)
    assert delete_res.status_code == 204

    # Verify all files are deleted from disk
    assert not raw_path.exists()
    assert not chunks_path.exists()
    assert not vs_path.exists()

    # 8. Try to delete again (should return 404 since it's gone)
    delete_again_res = client.delete(f"/documents/{doc_id}", headers=auth_headers)
    assert delete_again_res.status_code == 404


def test_delete_and_reindex_invalid_uuid(auth_headers):
    """Verify that using an invalid UUID string results in HTTP 422 validation error."""
    invalid_id = "not-a-uuid"
    res1 = client.delete(f"/documents/{invalid_id}", headers=auth_headers)
    assert res1.status_code == 422
    assert "must be a valid UUID string" in res1.json()["detail"]

    res2 = client.post(f"/documents/{invalid_id}/reindex", headers=auth_headers)
    assert res2.status_code == 422
    assert "must be a valid UUID string" in res2.json()["detail"]


def test_reindex_missing_raw_file(auth_headers):
    """Verify that re-indexing fails with 404 if the original raw file was deleted on disk."""
    files = {"file": ("to_be_partial.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    upload_res = client.post("/upload", files=files, headers=auth_headers)
    assert upload_res.status_code == 202
    doc_id = upload_res.json()["job_id"]

    # Poll status endpoint until background Ingestion completes
    status = "pending"
    for _ in range(50):
        status_res = client.get(f"/upload/{doc_id}/status", headers=auth_headers)
        assert status_res.status_code == 200
        status = status_res.json()["status"]
        if status in ["complete", "failed"]:
            break
        time.sleep(0.1)

    assert status == "complete"

    # Delete the raw uploaded file on disk
    raw_path = UPLOADS_DIR / "user-123" / f"{doc_id}.pdf"
    assert raw_path.exists()
    raw_path.unlink()

    # Verify status is listed as partial
    list_res = client.get("/documents", headers=auth_headers)
    payload = list_res.json()
    doc_info = payload["items"][0]
    assert doc_info["status"] == "partial"
    assert doc_info["can_reindex"] is False

    # Try to reindex, which should fail because raw file is missing
    reindex_res = client.post(f"/documents/{doc_id}/reindex", headers=auth_headers)
    assert reindex_res.status_code == 404
    assert "Original upload file is missing on disk" in reindex_res.json()["detail"]

    # Clean up remaining artifacts manually to keep disk clean
    chunks_path = CHUNKS_DIR / "user-123" / f"{doc_id}.json"
    vs_path = VECTORSTORE_DIR / "user-123" / doc_id
    if chunks_path.exists():
        chunks_path.unlink()
    if vs_path.exists():
        shutil.rmtree(vs_path)


def test_get_document_file_success(auth_headers):
    """Verify that an authenticated user can download/preview their document."""
    # Write a dummy file to UPLOADS_DIR/user-123/ and mock metadata in CHUNKS_DIR/user-123/
    doc_id = "12345678-1234-1234-1234-123456789012"
    user_uploads = UPLOADS_DIR / "user-123"
    user_uploads.mkdir(parents=True, exist_ok=True)
    file_path = user_uploads / f"{doc_id}.pdf"
    file_path.write_bytes(b"dummy pdf bytes")

    user_chunks = CHUNKS_DIR / "user-123"
    user_chunks.mkdir(parents=True, exist_ok=True)
    chunks_path = user_chunks / f"{doc_id}.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump({"document_id": doc_id, "source_filename": "test.pdf", "total_chunks": 5}, f)

    try:
        response = client.get(f"/documents/{doc_id}/file", headers=auth_headers)
        assert response.status_code == 200
        assert response.content == b"dummy pdf bytes"
        assert response.headers["content-type"] == "application/pdf"
        assert "inline" in response.headers["content-disposition"]
    finally:
        if file_path.exists():
            file_path.unlink()
        if chunks_path.exists():
            chunks_path.unlink()


def test_get_document_file_forbidden(auth_headers, other_auth_headers):
    """Verify that retrieving another tenant's document file returns 403."""
    doc_id = "12345678-1234-1234-1234-123456789012"
    user_uploads = UPLOADS_DIR / "user-123"
    user_uploads.mkdir(parents=True, exist_ok=True)
    file_path = user_uploads / f"{doc_id}.pdf"
    file_path.write_bytes(b"dummy pdf bytes")

    user_chunks = CHUNKS_DIR / "user-123"
    user_chunks.mkdir(parents=True, exist_ok=True)
    chunks_path = user_chunks / f"{doc_id}.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump({"document_id": doc_id, "source_filename": "test.pdf", "total_chunks": 5}, f)

    try:
        response = client.get(f"/documents/{doc_id}/file", headers=other_auth_headers)
        assert response.status_code == 403
    finally:
        if file_path.exists():
            file_path.unlink()
        if chunks_path.exists():
            chunks_path.unlink()


def test_get_document_file_not_found(auth_headers):
    """Verify that a non-existent document file returns 404."""
    doc_id = "99999999-9999-9999-9999-999999999999"
    response = client.get(f"/documents/{doc_id}/file", headers=auth_headers)
    assert response.status_code == 404


def test_get_document_chunks_success(auth_headers):
    """Verify that an authenticated user can retrieve document chunks metadata."""
    doc_id = "12345678-1234-1234-1234-123456789012"
    user_chunks = CHUNKS_DIR / "user-123"
    user_chunks.mkdir(parents=True, exist_ok=True)
    chunks_path = user_chunks / f"{doc_id}.json"
    metadata = {
        "document_id": doc_id,
        "source_filename": "test.pdf",
        "total_chunks": 2,
        "chunks": [{"text": "chunk1"}, {"text": "chunk2"}]
    }
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    try:
        response = client.get(f"/documents/{doc_id}/chunks", headers=auth_headers)
        assert response.status_code == 200
        payload = response.json()
        assert payload["document_id"] == doc_id
        assert payload["total_chunks"] == 2
        assert len(payload["chunks"]) == 2
    finally:
        if chunks_path.exists():
            chunks_path.unlink()


def test_get_document_chunks_forbidden(auth_headers, other_auth_headers):
    """Verify that retrieving another user's document chunks returns 403."""
    doc_id = "12345678-1234-1234-1234-123456789012"
    user_chunks = CHUNKS_DIR / "user-123"
    user_chunks.mkdir(parents=True, exist_ok=True)
    chunks_path = user_chunks / f"{doc_id}.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump({"document_id": doc_id, "source_filename": "test.pdf"}, f)

    try:
        response = client.get(f"/documents/{doc_id}/chunks", headers=other_auth_headers)
        assert response.status_code == 403
    finally:
        if chunks_path.exists():
            chunks_path.unlink()


def test_get_document_chunks_not_found(auth_headers):
    """Verify that a non-existent document chunks request returns 404."""
    doc_id = "99999999-9999-9999-9999-999999999999"
    response = client.get(f"/documents/{doc_id}/chunks", headers=auth_headers)
    assert response.status_code == 404


def test_get_document_summary_success(auth_headers):
    """Verify that an authenticated user can retrieve the summary from metadata JSON."""
    doc_id = "12345678-1234-1234-1234-123456789012"
    user_chunks = CHUNKS_DIR / "user-123"
    user_chunks.mkdir(parents=True, exist_ok=True)
    chunks_path = user_chunks / f"{doc_id}.json"
    metadata = {
        "document_id": doc_id,
        "source_filename": "test.pdf",
        "summary": "This is a mock summary text.",
        "summary_status": "completed"
    }
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    try:
        response = client.get(f"/documents/{doc_id}/summary", headers=auth_headers)
        assert response.status_code == 200
        payload = response.json()
        assert payload["document_id"] == doc_id
        assert payload["summary"] == "This is a mock summary text."
        assert payload["summary_status"] == "completed"
    finally:
        if chunks_path.exists():
            chunks_path.unlink()


def test_get_document_summary_not_found(auth_headers):
    """Verify that summary retrieval for a non-existent document returns 404."""
    doc_id = "99999999-9999-9999-9999-999999999999"
    response = client.get(f"/documents/{doc_id}/summary", headers=auth_headers)
    assert response.status_code == 404


def test_regenerate_document_summary_success(auth_headers):
    """Verify that summary regeneration returns 202 and runs background task."""
    doc_id = "12345678-1234-1234-1234-123456789012"
    user_chunks = CHUNKS_DIR / "user-123"
    user_chunks.mkdir(parents=True, exist_ok=True)
    chunks_path = user_chunks / f"{doc_id}.json"
    metadata = {
        "document_id": doc_id,
        "source_filename": "test.pdf",
        "parents": [{"text": "Parent text chunk content"}],
        "summary": "Old summary",
        "summary_status": "completed"
    }
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    from unittest.mock import MagicMock, patch
    from langchain_core.messages import AIMessage

    with patch("app.core.qa.GroqConnectionManager.get_chat_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.return_value = AIMessage(content="Newly generated summary takeaways")
        mock_model.invoke.return_value = AIMessage(content="Newly generated summary takeaways")
        mock_get_model.return_value = mock_model

        try:
            # We want to check that the API returns 202 accepted
            response = client.post(f"/documents/{doc_id}/summary/regenerate", headers=auth_headers)
            assert response.status_code == 202
            payload = response.json()
            assert payload["document_id"] == doc_id
            assert payload["status"] == "pending"

            # Wait briefly for background task to execute
            time.sleep(0.5)

            # Check that chunks file was updated with new summary
            with open(chunks_path, "r", encoding="utf-8") as f:
                updated_payload = json.load(f)
            assert updated_payload["summary"] == "Newly generated summary takeaways"
            assert updated_payload["summary_status"] == "completed"

        finally:
            if chunks_path.exists():
                chunks_path.unlink()


