"""Tests for POST /query/stream SSE streaming endpoint."""

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

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

# Fixed user identities used across streaming tests
STREAM_USER_ID = "user-stream-1"
STREAM_USERNAME = "stream_user1"
OTHER_USER_ID = "user-stream-other"
OTHER_USERNAME = "stream_user_other"


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolate user DB and clean up per-user data directories for each test."""
    test_db = tmp_path / "test_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db

    # Pre-clean any leftover vectorstore dirs from prior runs
    for user_id in [STREAM_USER_ID, OTHER_USER_ID]:
        for parent_dir in [UPLOADS_DIR, CHUNKS_DIR, VECTORSTORE_DIR]:
            user_dir = parent_dir / user_id
            if user_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(user_dir)
                except Exception:
                    pass

    UserDatabaseManager.initialize_db()
    yield

    # Post-test cleanup
    for user_id in [STREAM_USER_ID, OTHER_USER_ID]:
        for parent_dir in [UPLOADS_DIR, CHUNKS_DIR, VECTORSTORE_DIR]:
            user_dir = parent_dir / user_id
            if user_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(user_dir)
                except Exception:
                    pass

    db_mod.DB_PATH = old_db_path


@pytest.fixture
def auth_headers():
    """Registers the primary stream test user and returns their auth headers."""
    UserDatabaseManager.create_user(
        user_id=STREAM_USER_ID,
        username=STREAM_USERNAME,
        hashed_password="hashedpassword"
    )
    token = create_access_token({"sub": STREAM_USERNAME})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_other():
    """Registers the secondary (other-user) and returns their auth headers."""
    UserDatabaseManager.create_user(
        user_id=OTHER_USER_ID,
        username=OTHER_USERNAME,
        hashed_password="hashedpassword"
    )
    token = create_access_token({"sub": OTHER_USERNAME})
    return {"Authorization": f"Bearer {token}"}


def _make_doc_dir(user_id: str, doc_id: str) -> Path:
    """Create a fake vectorstore directory to satisfy the on-disk ownership checks."""
    doc_path = VECTORSTORE_DIR / user_id / doc_id
    doc_path.mkdir(parents=True, exist_ok=True)
    return doc_path


# ---------------------------------------------------------------------------
# Auth & ownership guard tests
# ---------------------------------------------------------------------------

def test_stream_requires_auth():
    """POST /query/stream returns 401 when no Authorization header is provided."""
    doc_id = str(uuid.uuid4())
    resp = client.post(
        "/query/stream",
        json={"document_id": doc_id, "question": "What is this?"},
    )
    assert resp.status_code == 401


def test_stream_returns_404_when_vectorstore_missing(auth_headers):
    """POST /query/stream returns 404 when the requested document does not exist on disk."""
    doc_id = str(uuid.uuid4())
    resp = client.post(
        "/query/stream",
        json={"document_id": doc_id, "question": "What is this?"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


def test_stream_returns_403_for_cross_user_document(auth_headers):
    """POST /query/stream returns 403 when document belongs to a different user."""
    doc_id = str(uuid.uuid4())
    # Create vectorstore directory owned by OTHER user — simulates cross-tenant access
    (VECTORSTORE_DIR / OTHER_USER_ID / doc_id).mkdir(parents=True, exist_ok=True)
    try:
        resp = client.post(
            "/query/stream",
            json={"document_id": doc_id, "question": "What is this?"},
            headers=auth_headers,
        )
        assert resp.status_code == 403
    finally:
        import shutil
        shutil.rmtree(VECTORSTORE_DIR / OTHER_USER_ID / doc_id, ignore_errors=True)


# ---------------------------------------------------------------------------
# Happy-path SSE stream tests
# ---------------------------------------------------------------------------

def test_stream_returns_text_event_stream_content_type(auth_headers):
    """POST /query/stream responds with Content-Type: text/event-stream."""
    doc_id = str(uuid.uuid4())
    doc_path = _make_doc_dir(STREAM_USER_ID, doc_id)
    try:
        with (
            patch("app.routes.query.vector_manager.get_hybrid_retriever") as mock_retriever,
            patch("app.routes.query.RerankManager.get_ranker"),
            patch("app.routes.query.FlashrankRerank") as mock_compressor_cls,
            patch("app.routes.query.qa_pipeline.generate_answer_stream") as mock_stream,
        ):
            mock_doc = MagicMock()
            mock_doc.page_content = "Paris is the capital of France."
            mock_doc.metadata = {
                "source_filename": "doc.pdf",
                "page_index": 0,
                "document_id": doc_id,
            }
            mock_retriever.return_value.invoke.return_value = [mock_doc]
            mock_compressor_cls.return_value.compress_documents.return_value = [mock_doc]

            async def _fake_stream(query, docs):
                yield "Paris"
                yield " is"
                yield " great."

            mock_stream.return_value = _fake_stream("q", [mock_doc])

            resp = client.post(
                "/query/stream",
                json={"document_id": doc_id, "question": "What is Paris?"},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
    finally:
        import shutil
        shutil.rmtree(doc_path, ignore_errors=True)


def test_stream_first_event_is_citations(auth_headers):
    """The first SSE event must be a citations payload."""
    doc_id = str(uuid.uuid4())
    doc_path = _make_doc_dir(STREAM_USER_ID, doc_id)
    try:
        with (
            patch("app.routes.query.vector_manager.get_hybrid_retriever") as mock_retriever,
            patch("app.routes.query.RerankManager.get_ranker"),
            patch("app.routes.query.FlashrankRerank") as mock_compressor_cls,
            patch("app.routes.query.qa_pipeline.generate_answer_stream") as mock_stream,
        ):
            mock_doc = MagicMock()
            mock_doc.page_content = "Paris is the capital of France."
            mock_doc.metadata = {
                "source_filename": "doc.pdf",
                "page_index": 0,
                "document_id": doc_id,
            }
            mock_retriever.return_value.invoke.return_value = [mock_doc]
            mock_compressor_cls.return_value.compress_documents.return_value = [mock_doc]

            async def _fake_stream(query, docs):
                yield "Paris"

            mock_stream.return_value = _fake_stream("q", [mock_doc])

            resp = client.post(
                "/query/stream",
                json={"document_id": doc_id, "question": "What is Paris?"},
                headers=auth_headers,
            )
        lines = [line for line in resp.text.splitlines() if line.startswith("data:")]
        first_payload = json.loads(lines[0].removeprefix("data:").strip())
        assert "citations" in first_payload
        assert isinstance(first_payload["citations"], list)
    finally:
        import shutil
        shutil.rmtree(doc_path, ignore_errors=True)


def test_stream_token_events_and_done(auth_headers):
    """Token events contain a 'token' key; final event is [DONE]."""
    doc_id = str(uuid.uuid4())
    doc_path = _make_doc_dir(STREAM_USER_ID, doc_id)
    try:
        with (
            patch("app.routes.query.vector_manager.get_hybrid_retriever") as mock_retriever,
            patch("app.routes.query.RerankManager.get_ranker"),
            patch("app.routes.query.FlashrankRerank") as mock_compressor_cls,
            patch("app.routes.query.qa_pipeline.generate_answer_stream") as mock_stream,
        ):
            mock_doc = MagicMock()
            mock_doc.page_content = "Paris is the capital of France."
            mock_doc.metadata = {
                "source_filename": "doc.pdf",
                "page_index": 0,
                "document_id": doc_id,
            }
            mock_retriever.return_value.invoke.return_value = [mock_doc]
            mock_compressor_cls.return_value.compress_documents.return_value = [mock_doc]

            async def _fake_stream(query, docs):
                yield "Paris"
                yield " is great."

            mock_stream.return_value = _fake_stream("q", [mock_doc])

            resp = client.post(
                "/query/stream",
                json={"document_id": doc_id, "question": "What is Paris?"},
                headers=auth_headers,
            )
        data_lines = [line for line in resp.text.splitlines() if line.startswith("data:")]
        # First event = citations (tested separately), skip it
        token_lines = data_lines[1:-1]
        done_line = data_lines[-1]
        for line in token_lines:
            payload = json.loads(line.removeprefix("data:").strip())
            assert "token" in payload
        assert done_line.strip() == "data: [DONE]"
    finally:
        import shutil
        shutil.rmtree(doc_path, ignore_errors=True)


def test_stream_missing_document_id_returns_422(auth_headers):
    """POST /query/stream returns 422 when neither document_id nor document_ids is provided."""
    resp = client.post(
        "/query/stream",
        json={"question": "What is this?"},
        headers=auth_headers,
    )
    assert resp.status_code == 422
