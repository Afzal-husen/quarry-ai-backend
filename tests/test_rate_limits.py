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


@pytest.mark.enable_rate_limiting
def test_auth_rate_limiting():
    """Verify signup and login endpoints trigger rate limit checks (5 per minute)."""
    # Attempt signup 6 times consecutively
    signup_payload = {"username": "ratelimittest", "password": "securepassword"}
    
    responses = []
    for _ in range(6):
        res = client.post("/auth/signup", json=signup_payload)
        responses.append(res)

    # At least one response should be 429 Rate Limit Exceeded
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes
    
    # Check the detail msg of the rate limited response
    lim_response = [r for r in responses if r.status_code == 429][0]
    assert "Rate limit exceeded" in lim_response.json()["detail"] or "5 per 1 minute" in lim_response.json()["detail"]
