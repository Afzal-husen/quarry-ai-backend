"""Tests for Observability and Structured Logging (Phase 18)."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import logging
import pytest
from fastapi.testclient import TestClient
from main import app
from app.core.database import UserDatabaseManager
from app.core.auth import create_access_token, hash_password
import app.core.database as db_mod

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


def _auth_headers(username: str = "test-obs-user", user_id: str = "user-obs-1") -> dict:
    """Helper to ensure user exists in database and return access headers."""
    if not UserDatabaseManager.get_user_by_username(username):
        hp = hash_password("testpassword123")
        UserDatabaseManager.create_user(user_id, username, hp)
    token = create_access_token({"sub": username})
    return {"Authorization": f"Bearer {token}"}


def test_structured_json_logging_format(capsys):
    """Verify that logging output is valid single-line JSON with standard fields."""
    logger = logging.getLogger("app.test")
    
    # Trigger a log event
    logger.info("Test structured logging message", extra={"custom_key": "custom_val"})
    
    captured = capsys.readouterr()
    stdout_lines = [line.strip() for line in captured.out.split("\n") if line.strip()]
    
    # We should have at least one JSON log line
    assert len(stdout_lines) > 0
    
    # Parse the log line
    log_record = json.loads(stdout_lines[0])
    assert "timestamp" in log_record
    assert log_record["level"] == "INFO"
    assert log_record["message"] == "Test structured logging message"
    assert log_record["logger"] == "app.test"
    assert log_record["custom_key"] == "custom_val"


def test_request_logging_unauthenticated(capsys):
    """Verify that unauthenticated HTTP requests generate a structured log entry with user_id=None."""
    # Read output to clear buffer
    capsys.readouterr()
    
    resp = client.get("/health")
    assert resp.status_code == 200
    
    captured = capsys.readouterr()
    stdout_lines = [line.strip() for line in captured.out.split("\n") if line.strip()]
    
    # Find request log
    req_logs = []
    for line in stdout_lines:
        try:
            data = json.loads(line)
            if data.get("message") == "Request completed":
                req_logs.append(data)
        except json.JSONDecodeError:
            continue
            
    assert len(req_logs) == 1
    log = req_logs[0]
    assert log["method"] == "GET"
    assert log["path"] == "/health"
    assert log["status_code"] == 200
    assert "duration_ms" in log
    assert log["user_id"] is None
    assert "client_ip" in log


def test_request_logging_authenticated(capsys):
    """Verify that authenticated requests include the database user_id in the log record."""
    capsys.readouterr()
    
    headers = _auth_headers(username="test-obs-user", user_id="user-obs-12345")
    resp = client.get("/documents", headers=headers)
    assert resp.status_code == 200
    
    captured = capsys.readouterr()
    stdout_lines = [line.strip() for line in captured.out.split("\n") if line.strip()]
    
    req_logs = []
    for line in stdout_lines:
        try:
            data = json.loads(line)
            if data.get("message") == "Request completed":
                req_logs.append(data)
        except json.JSONDecodeError:
            continue
            
    assert len(req_logs) == 1
    log = req_logs[0]
    assert log["method"] == "GET"
    assert log["path"] == "/documents"
    assert log["status_code"] == 200
    assert log["user_id"] == "user-obs-12345"


def test_unhandled_exception_logging_with_traceback(capsys):
    """Verify unhandled exceptions trigger 500 JSON response and log standard traceback metadata."""
    capsys.readouterr()
    
    from unittest.mock import patch, MagicMock
    
    mock_chunks_dir = MagicMock()
    mock_chunks_dir.__truediv__.return_value.exists.side_effect = Exception("Database connection loss mock error")
    
    local_client = TestClient(app, raise_server_exceptions=False)
    with patch("app.routes.documents.CHUNKS_DIR", mock_chunks_dir):
        headers = _auth_headers()
        resp = local_client.get("/documents", headers=headers)
        assert resp.status_code == 500
        
    captured = capsys.readouterr()
    stdout_lines = [line.strip() for line in captured.out.split("\n") if line.strip()]
    
    err_logs = []
    for line in stdout_lines:
        try:
            data = json.loads(line)
            if "exception" in data or "Database connection loss mock error" in data.get("message", ""):
                err_logs.append(data)
        except json.JSONDecodeError:
            continue
            
    assert len(err_logs) >= 1
    log = err_logs[0]
    assert "exception" in log
    assert "Database connection loss mock error" in log["exception"]
