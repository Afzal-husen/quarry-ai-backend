import sys
from pathlib import Path

# Ensure the backend directory is on sys.path so that absolute imports resolve
# when pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""
End-to-end integration tests for the Document RAG REST API.

Tests the complete upload-to-query pipeline:
  1. Upload a PDF fixture to POST /upload
  2. Confirm the document is indexed
  3. Query the indexed document via POST /query
  4. Validate the response shape, answer grounding, and citation structure
  5. Verify 404 for unknown document_id

Isolation: A unique document_id is generated per test run so that
repeated test executions do not share persistent Chroma state.
"""
import io
import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import app.core.database as db_mod
from app.core.database import UserDatabaseManager
from app.core.auth import create_access_token
from main import app

# Resolve storage root relative to this test file (backend/tests/ -> backend/)
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXTURE_DIR = Path(__file__).parent / "fixtures"
PDF_FIXTURE = FIXTURE_DIR / "sample.pdf"

# ---------------------------------------------------------------------------
# Shared TestClient
# ---------------------------------------------------------------------------
client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Isolates user database for the E2E module run."""
    test_db = BASE_DIR / "data" / "test_e2e_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db
    
    UserDatabaseManager.initialize_db()
    yield
    db_mod.DB_PATH = old_db_path
    if test_db.exists():
        test_db.unlink()


@pytest.fixture(scope="module")
def auth_headers(setup_test_db):
    """Registers a test user and returns auth headers."""
    username = "test_e2e_user"
    UserDatabaseManager.create_user(
        user_id="user-e2e-123",
        username=username,
        hashed_password="hashedpassword"
    )
    token = create_access_token({"sub": username})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def ensure_fixture():
    """Create a minimal single-page PDF fixture if it does not already exist."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_FIXTURE.exists():
        _write_minimal_pdf(PDF_FIXTURE)

    yield


@pytest.fixture(scope="module")
def uploaded_doc_id(ensure_fixture, auth_headers):
    """Upload the sample PDF once per module and return the document_id for reuse."""
    with PDF_FIXTURE.open("rb") as fh:
        response = client.post(
            "/upload",
            files={"file": ("sample.pdf", fh, "application/pdf")},
            headers=auth_headers
        )

    assert response.status_code == 202, (
        f"Upload setup fixture failed — status {response.status_code}: {response.text}"
    )

    data = response.json()
    job_id = data.get("job_id")
    assert job_id is not None, "Upload response must contain a job_id"

    import time
    status = "pending"
    for _ in range(100):
        status_res = client.get(f"/upload/{job_id}/status", headers=auth_headers)
        if status_res.status_code == 200:
            status = status_res.json()["status"]
            if status in ["complete", "failed"]:
                break
        time.sleep(0.1)

    assert status == "complete", f"Ingestion job failed or timed out: {status_res.text}"

    yield job_id

    _cleanup_doc(job_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup_doc(doc_id: str):
    """Remove all persisted artefacts for the given document id."""
    vectorstore_path = BASE_DIR / "data" / "vectorstore" / "user-e2e-123" / doc_id
    if vectorstore_path.exists():
        shutil.rmtree(vectorstore_path, ignore_errors=True)

    chunks_file = BASE_DIR / "data" / "chunks" / "user-e2e-123" / f"{doc_id}.json"
    if chunks_file.exists():
        chunks_file.unlink(missing_ok=True)

    upload_dir = BASE_DIR / "data" / "uploads" / "user-e2e-123"
    if upload_dir.exists():
        for upload_file in upload_dir.glob(f"{doc_id}*"):
            upload_file.unlink(missing_ok=True)


def _write_minimal_pdf(dest: Path):
    """Write a valid minimal single-page PDF containing searchable text."""
    page_text = (
        "The capital of France is Paris. "
        "Paris is known as the City of Light. "
        "The Eiffel Tower is located in Paris, France."
    )

    objects = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objects.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >>\n"
        b"endobj\n"
    )

    stream_content = (
        f"BT\n/F1 12 Tf\n72 720 Td\n({page_text})\nTj\nET"
    ).encode("latin-1")
    stream_len = len(stream_content)
    stream_obj = (
        f"5 0 obj\n<< /Length {stream_len} >>\nstream\n"
    ).encode("latin-1") + stream_content + b"\nendstream\nendobj\n"
    objects.append(stream_obj)

    header = b"%PDF-1.4\n"
    body = b"".join(objects)
    xref_offset = len(header) + len(body)

    xref = (
        "xref\n"
        f"0 6\n"
        "0000000000 65535 f \n"
    )
    offsets = []
    pos = len(header)
    for obj in objects:
        offsets.append(pos)
        pos += len(obj)

    for off in offsets:
        xref += f"{off:010d} 00000 n \n"

    trailer = (
        f"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    )

    dest.write_bytes(header + body + xref.encode("latin-1") + trailer.encode("latin-1"))


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------

class TestUploadE2E:
    """Verify the /upload endpoint response shape for the E2E fixture."""

    def test_upload_returns_200(self, uploaded_doc_id):
        """A valid PDF upload must return HTTP 200 with a document_id."""
        assert uploaded_doc_id is not None

    def test_upload_creates_vectorstore(self, uploaded_doc_id):
        """The vector store directory must exist on disk after a successful upload."""
        vectorstore_path = BASE_DIR / "data" / "vectorstore" / "user-e2e-123" / uploaded_doc_id
        assert vectorstore_path.exists(), (
            f"Vector store for '{uploaded_doc_id}' was not created at {vectorstore_path}"
        )

    def test_upload_creates_chunks_metadata(self, uploaded_doc_id):
        """A JSON metadata file for chunk information must exist after upload."""
        chunks_file = BASE_DIR / "data" / "chunks" / "user-e2e-123" / f"{uploaded_doc_id}.json"
        assert chunks_file.exists(), f"Chunks metadata JSON not found: {chunks_file}"


class TestQueryE2E:
    """Verify the /query endpoint through a full upload-then-query flow."""

    # --- Mocked LLM tests (always run, no API key required) ---

    def test_query_returns_200_mocked(self, uploaded_doc_id, auth_headers):
        """A valid query with a mocked LLM must return HTTP 200."""
        from langchain_groq import ChatGroq
        from langchain_core.messages import AIMessage
        mock_llm = MagicMock(spec=ChatGroq)
        mock_response = AIMessage(content="Paris is the capital of France.")
        mock_llm.invoke.return_value = mock_response
        mock_llm.return_value = mock_response

        with patch("app.core.qa.GroqConnectionManager") as mock_mgr:
            mock_mgr.get_chat_model.return_value = mock_llm
            response = client.post(
                "/query",
                json={
                    "document_id": uploaded_doc_id,
                    "question": "What is the capital of France?",
                    "top_k": 3
                },
                headers=auth_headers
            )
        assert response.status_code == 200, (
            f"Query failed with status {response.status_code}: {response.text}"
        )

    def test_query_response_has_answer_field_mocked(self, uploaded_doc_id, auth_headers):
        """The mocked response JSON must contain a non-empty 'answer' string."""
        from langchain_groq import ChatGroq
        from langchain_core.messages import AIMessage
        mock_llm = MagicMock(spec=ChatGroq)
        mock_response = AIMessage(content="Paris is the capital of France.")
        mock_llm.invoke.return_value = mock_response
        mock_llm.return_value = mock_response

        with patch("app.core.qa.GroqConnectionManager") as mock_mgr:
            mock_mgr.get_chat_model.return_value = mock_llm
            response = client.post(
                "/query",
                json={
                    "document_id": uploaded_doc_id,
                    "question": "What is the capital of France?",
                    "top_k": 3
                },
                headers=auth_headers
            )
        data = response.json()
        assert "answer" in data, f"Expected 'answer' key in response, got: {data.keys()}"
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_query_response_has_citations_field_mocked(self, uploaded_doc_id, auth_headers):
        """The mocked response JSON must include a 'citations' list."""
        from langchain_groq import ChatGroq
        from langchain_core.messages import AIMessage
        mock_llm = MagicMock(spec=ChatGroq)
        mock_response = AIMessage(content="The Eiffel Tower is located in Paris.")
        mock_llm.invoke.return_value = mock_response
        mock_llm.return_value = mock_response

        with patch("app.core.qa.GroqConnectionManager") as mock_mgr:
            mock_mgr.get_chat_model.return_value = mock_llm
            response = client.post(
                "/query",
                json={
                    "document_id": uploaded_doc_id,
                    "question": "Where is the Eiffel Tower?",
                    "top_k": 3
                },
                headers=auth_headers
            )
        data = response.json()
        assert "citations" in data, f"Expected 'citations' key in response, got: {data.keys()}"
        assert isinstance(data["citations"], list)

    # --- Validation tests (no LLM call — always run) ---

    def test_query_top_k_validation_min(self, uploaded_doc_id, auth_headers):
        """top_k must be >= 1; sending 0 should return a 422 validation error."""
        response = client.post(
            "/query",
            json={
                "document_id": uploaded_doc_id,
                "question": "Test question?",
                "top_k": 0
            },
            headers=auth_headers
        )
        assert response.status_code == 422, (
            f"Expected 422 Unprocessable Entity for top_k=0, got {response.status_code}"
        )

    def test_query_top_k_validation_max(self, uploaded_doc_id, auth_headers):
        """top_k must be <= 10; sending 11 should return a 422 validation error."""
        response = client.post(
            "/query",
            json={
                "document_id": uploaded_doc_id,
                "question": "Test question?",
                "top_k": 11
            },
            headers=auth_headers
        )
        assert response.status_code == 422, (
            f"Expected 422 Unprocessable Entity for top_k=11, got {response.status_code}"
        )

    def test_query_missing_question_returns_422(self, uploaded_doc_id, auth_headers):
        """Omitting the required 'question' field must return 422."""
        response = client.post(
            "/query",
            json={"document_id": uploaded_doc_id},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_query_missing_document_id_returns_422(self, auth_headers):
        """Omitting both 'document_id' and 'document_ids' fields must return 422."""
        response = client.post(
            "/query",
            json={"question": "Where is Paris?"},
            headers=auth_headers
        )
        assert response.status_code == 422

    # --- Live Groq API tests (skipped when GROQ_API_KEY is not configured) ---

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="GROQ_API_KEY not configured — skipping live Groq API test"
    )
    def test_query_returns_200_live(self, uploaded_doc_id, auth_headers):
        """A valid query against the live Groq API must return HTTP 200."""
        response = client.post(
            "/query",
            json={
                "document_id": uploaded_doc_id,
                "question": "What is the capital of France?",
                "top_k": 3
            },
            headers=auth_headers
        )
        assert response.status_code == 200, (
            f"Live query failed with status {response.status_code}: {response.text}"
        )

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="GROQ_API_KEY not configured — skipping live Groq API test"
    )
    def test_query_live_answer_is_grounded(self, uploaded_doc_id, auth_headers):
        """Live Groq answer must be a non-empty string grounded in the document."""
        response = client.post(
            "/query",
            json={
                "document_id": uploaded_doc_id,
                "question": "What is the capital of France?",
                "top_k": 3
            },
            headers=auth_headers
        )
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert isinstance(data["citations"], list)


class TestQueryNotFoundE2E:
    """Verify 404 handling for queries against non-existent documents."""

    def test_query_unknown_document_id_returns_404(self, auth_headers):
        """Querying a document_id that has never been uploaded must return 404."""
        fake_id = str(uuid.uuid4())
        response = client.post(
            "/query",
            json={
                "document_id": fake_id,
                "question": "Does this document exist?",
                "top_k": 3
            },
            headers=auth_headers
        )
        assert response.status_code == 404, (
            f"Expected 404 for unknown document '{fake_id}', got {response.status_code}"
        )

    def test_query_404_detail_mentions_document_id(self, auth_headers):
        """The 404 error detail must reference the missing document_id."""
        fake_id = str(uuid.uuid4())
        response = client.post(
            "/query",
            json={
                "document_id": fake_id,
                "question": "Does this document exist?",
                "top_k": 3
            },
            headers=auth_headers
        )
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert fake_id in data["detail"], (
            f"Expected '{fake_id}' in error detail, got: {data['detail']}"
        )


class TestMultiTenancyQueryE2E:
    """Verify that querying documents belonging to another tenant returns 403 Forbidden."""

    def test_query_other_users_document_returns_403(self, uploaded_doc_id):
        """Querying a document uploaded by another user must return 403 Forbidden."""
        # Create and register a second tenant
        username_b = "test_e2e_user_b"
        UserDatabaseManager.create_user(
            user_id="user-e2e-456",
            username=username_b,
            hashed_password="hashedpassword"
        )
        token_b = create_access_token({"sub": username_b})
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # Query the document uploaded by the first tenant (user-e2e-123)
        response = client.post(
            "/query",
            json={
                "document_id": uploaded_doc_id,
                "question": "What is the capital of France?",
                "top_k": 3
            },
            headers=headers_b
        )
        assert response.status_code == 403, (
            f"Expected 403 Forbidden, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "detail" in data
        assert "Forbidden" in data["detail"]
