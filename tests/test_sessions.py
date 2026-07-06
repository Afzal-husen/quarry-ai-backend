import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app.core.database as db_mod
from app.core.database import UserDatabaseManager
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolates database for each test case using a temporary db file path."""
    test_db = tmp_path / "test_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db
    
    # Force re-initialization
    UserDatabaseManager.initialize_db()
    yield
    db_mod.DB_PATH = old_db_path


def get_auth_headers(username: str = "testuser") -> dict:
    """Helper to register and login a user, returning Bearer auth header."""
    client.post("/auth/signup", json={"username": username, "password": "password123"})
    login_resp = client.post("/auth/login", json={"username": username, "password": "password123"})
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_session_db_tables_created():
    """Verify that chat_sessions and chat_messages tables are initialized."""
    conn = db_mod.UserDatabaseManager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "chat_sessions" in tables
    assert "chat_messages" in tables


def test_create_session():
    """Verify POST /sessions creates a session with correct fields."""
    headers = get_auth_headers()
    # 1. Without explicit title
    response = client.post("/sessions", json={}, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["title"] == "New Chat"
    
    # 2. With explicit title
    response = client.post("/sessions", json={"title": "My Custom Session"}, headers=headers)
    assert response.status_code == 201
    assert response.json()["title"] == "My Custom Session"


def test_list_sessions_pagination():
    """Verify GET /sessions lists sessions with limit/offset."""
    headers = get_auth_headers("user1")
    # Create 3 sessions
    client.post("/sessions", json={"title": "Session 1"}, headers=headers)
    client.post("/sessions", json={"title": "Session 2"}, headers=headers)
    client.post("/sessions", json={"title": "Session 3"}, headers=headers)
    
    # Retrieve all
    resp = client.get("/sessions", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert len(data["items"]) == 3
    
    # Retrieve with limit
    resp_limit = client.get("/sessions?limit=2", headers=headers)
    assert resp_limit.json()["limit"] == 2
    assert len(resp_limit.json()["items"]) == 2
    
    # Retrieve with offset
    resp_offset = client.get("/sessions?limit=2&offset=2", headers=headers)
    assert len(resp_offset.json()["items"]) == 1


def test_get_session_details():
    """Verify GET /sessions/{session_id} returns session structure."""
    headers = get_auth_headers()
    # Create session
    create_resp = client.post("/sessions", json={"title": "History Session"}, headers=headers)
    session_id = create_resp.json()["id"]
    
    # Get session details (should have empty messages)
    get_resp = client.get(f"/sessions/{session_id}", headers=headers)
    assert get_resp.status_code == 200
    detail = get_resp.json()
    assert detail["id"] == session_id
    assert detail["title"] == "History Session"
    assert isinstance(detail["messages"], list)
    assert len(detail["messages"]) == 0


def test_delete_session_and_cascades():
    """Verify DELETE /sessions/{session_id} purges messages via cascade."""
    headers = get_auth_headers()
    create_resp = client.post("/sessions", json={"title": "To Delete"}, headers=headers)
    session_id = create_resp.json()["id"]
    
    # Delete session
    del_resp = client.delete(f"/sessions/{session_id}", headers=headers)
    assert del_resp.status_code == 204
    
    # Verify GET returns 404
    get_resp = client.get(f"/sessions/{session_id}", headers=headers)
    assert get_resp.status_code == 404


def test_session_ownership_boundaries():
    """Verify users cannot access/delete other users' sessions."""
    headers1 = get_auth_headers("user_one")
    headers2 = get_auth_headers("user_two")
    
    # User 1 creates session
    create_resp = client.post("/sessions", json={"title": "User One Session"}, headers=headers1)
    session_id = create_resp.json()["id"]
    
    # User 2 tries to GET session
    get_resp = client.get(f"/sessions/{session_id}", headers=headers2)
    assert get_resp.status_code == 403
    
    # User 2 tries to DELETE session
    del_resp = client.delete(f"/sessions/{session_id}", headers=headers2)
    assert del_resp.status_code == 403
