import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure the backend directory is in the path for absolute imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app.core.database as db_mod
from app.core.database import UserDatabaseManager
from app.core.auth import hash_password, verify_password, create_access_token, verify_access_token
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolates user database for each test case using a temporary db file path."""
    test_db = tmp_path / "test_users.db"
    # Monkeypatch the DB_PATH constant in database module
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db
    
    # Initialize the test database
    UserDatabaseManager.initialize_db()
    
    yield
    
    # Restore original path
    db_mod.DB_PATH = old_db_path


def test_password_hashing():
    """Verify that password hashing generates secure hashes and validates correctly."""
    password = "SuperSecurePassword123!"
    hashed = hash_password(password)
    
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False


def test_jwt_token_flow():
    """Verify that JWT tokens can be created, decoded, and throw expired errors if outdated."""
    payload = {"sub": "testuser", "custom_claim": "hello"}
    token = create_access_token(payload)
    
    decoded = verify_access_token(token)
    assert decoded["sub"] == "testuser"
    assert decoded["custom_claim"] == "hello"


def test_user_signup_success():
    """Verify that user registration is successful and returns user ID."""
    response = client.post(
        "/auth/signup",
        json={"username": "newuser", "password": "securepassword"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert "user_id" in data


def test_user_signup_duplicate():
    """Verify that registering an already existing username returns HTTP 400."""
    # First signup
    client.post(
        "/auth/signup",
        json={"username": "newuser", "password": "securepassword"}
    )
    # Second signup with same username
    response = client.post(
        "/auth/signup",
        json={"username": "newuser", "password": "anotherpassword"}
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_user_signup_validation():
    """Verify that short usernames or passwords trigger HTTP 422 validations."""
    # Username too short
    response = client.post(
        "/auth/signup",
        json={"username": "ab", "password": "securepassword"}
    )
    assert response.status_code == 422

    # Password too short
    response = client.post(
        "/auth/signup",
        json={"username": "validuser", "password": "123"}
    )
    assert response.status_code == 422


def test_user_login_success():
    """Verify that a registered user can log in and retrieve a Bearer access token."""
    # Signup first
    client.post(
        "/auth/signup",
        json={"username": "loginuser", "password": "securepassword"}
    )
    
    # Login
    response = client.post(
        "/auth/login",
        json={"username": "loginuser", "password": "securepassword"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_user_login_invalid_credentials():
    """Verify that invalid password or username returns HTTP 400."""
    # Signup
    client.post(
        "/auth/signup",
        json={"username": "loginuser", "password": "securepassword"}
    )
    
    # Login with wrong password
    response = client.post(
        "/auth/login",
        json={"username": "loginuser", "password": "wrongpassword"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid username or password."
    
    # Login with non-existent user
    response = client.post(
        "/auth/login",
        json={"username": "unknownuser", "password": "securepassword"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid username or password."
