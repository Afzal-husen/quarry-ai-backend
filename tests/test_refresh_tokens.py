import pytest
from fastapi.testclient import TestClient
from main import app
import app.core.database as db_mod
from app.core.database import UserDatabaseManager

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolates user database for each test case."""
    test_db = tmp_path / "test_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db
    UserDatabaseManager.initialize_db()
    yield
    db_mod.DB_PATH = old_db_path


def test_jwt_refresh_token_lifecycle():
    """Verify refresh token generation, access token refresh, and logout revocation."""
    # 1. Sign up user
    signup_payload = {"username": "refreshtest", "password": "securepassword"}
    signup_res = client.post("/auth/signup", json=signup_payload)
    assert signup_res.status_code == 201

    # 2. Login user and verify refresh token is returned
    login_res = client.post("/auth/login", json=signup_payload)
    assert login_res.status_code == 200
    login_data = login_res.json()
    assert "access_token" in login_data
    assert "refresh_token" in login_data
    refresh_token = login_data["refresh_token"]

    # 3. Use refresh token to obtain a new access token
    refresh_payload = {"refresh_token": refresh_token}
    refresh_res = client.post("/auth/refresh", json=refresh_payload)
    assert refresh_res.status_code == 200
    refresh_data = refresh_res.json()
    assert "access_token" in refresh_data

    # 4. Use an invalid refresh token (should fail)
    invalid_res = client.post("/auth/refresh", json={"refresh_token": "not-a-valid-token"})
    assert invalid_res.status_code == 401
    assert "Invalid refresh token" in invalid_res.json()["detail"]

    # 5. Log out and verify revocation
    logout_res = client.post("/auth/logout", json=refresh_payload)
    assert logout_res.status_code == 200
    assert logout_res.json()["status"] == "success"

    # 6. Try refreshing again with the logged out token (should fail)
    revoked_res = client.post("/auth/refresh", json=refresh_payload)
    assert revoked_res.status_code == 401
    assert "Invalid refresh token" in revoked_res.json()["detail"]
