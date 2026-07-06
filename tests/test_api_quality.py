"""Tests for API Quality and Developer Experience (Phase 17)."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app
from app.core.auth import create_access_token, hash_password
from app.core.database import UserDatabaseManager
import app.core.database as db_mod
import uuid
import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolate user DB for each test."""
    test_db = tmp_path / "test_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db
    UserDatabaseManager.initialize_db()
    yield
    db_mod.DB_PATH = old_db_path


def _auth_headers(username: str = "test-quality-user", user_id: str = "user-quality-1") -> dict:
    """Helper to ensure user exists in database and return access headers."""
    if not UserDatabaseManager.get_user_by_username(username):
        hp = hash_password("testpassword123")
        UserDatabaseManager.create_user(user_id, username, hp)
    token = create_access_token({"sub": username})
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Error Schema Tests (API-03)
# ---------------------------------------------------------------------------

def test_error_schema_401_unauthorized():
    """Missing auth token returns standard error schema with UNAUTHORIZED code."""
    resp = client.get("/documents")
    assert resp.status_code == 401
    payload = resp.json()
    assert "detail" in payload
    assert payload["code"] == "UNAUTHORIZED"
    assert payload["field"] is None


def test_error_schema_422_validation():
    """Invalid payload formats return standard error schema with VALIDATION_ERROR code and field."""
    # Sending invalid body to POST /query
    resp = client.post("/query", json={}, headers=_auth_headers())
    assert resp.status_code == 422
    payload = resp.json()
    assert "detail" in payload
    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["field"] in ("document_id", "document_ids", "question")


# ---------------------------------------------------------------------------
# Pagination Tests (API-02)
# ---------------------------------------------------------------------------

def test_pagination_default_values():
    """GET /documents returns paginated metadata structure even if empty."""
    resp = client.get("/documents?limit=5&offset=0", headers=_auth_headers())
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 0
    assert payload["limit"] == 5
    assert payload["offset"] == 0
    assert isinstance(payload["items"], list)


def test_pagination_clamping():
    """Out of bound limits/offsets are clamped instead of causing validation errors."""
    resp = client.get("/documents?limit=999&offset=-10",
                      headers=_auth_headers())
    assert resp.status_code == 200
    payload = resp.json()
    # Clamped max limit=100, min offset=0
    assert payload["limit"] == 100
    assert payload["offset"] == 0


@pytest.mark.enable_rate_limiting
def test_rate_limiting_upload_triggers_429():
    """Rate limiter eventually triggers 429 when limits are exceeded."""
    headers = _auth_headers()
    files = {"file": ("test_limit.pdf", b"%PDF-1.4 \n%%EOF", "application/pdf")}
    
    responses = []
    for _ in range(6):
        # Disable background task to speed up test execution
        with patch("fastapi.BackgroundTasks.add_task"):
            resp = client.post("/upload", files=files, headers=headers)
            responses.append(resp)
            
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes
    
    # Verify the 429 response structure
    r_429 = [r for r in responses if r.status_code == 429][0]
    payload = r_429.json()
    assert payload["code"] == "RATE_LIMIT_EXCEEDED"
    assert "detail" in payload
    assert payload["field"] is None
    assert "Retry-After" in r_429.headers
