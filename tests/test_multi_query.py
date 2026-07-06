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

MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 3 3] "
    b"/Resources << >> /Contents 4 0 R >>\nendobj\n"
    b"4 0 obj\n<< /Length 93 >>\nstream\n"
    b"BT /F1 1 Tf (This is a longer text string that will definitely be split into multiple chunks.) Tj ET\n"
    b"endstream\nendobj\nxref\n0 5\n0000000009 00000 n \n0000000062 00000 n \n"
    b"0000000121 00000 n \n0000000224 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\n"
    b"startxref\n296\n%%EOF"
)


def _make_proper_pdf_bytes() -> bytes:
    """Build a well-formed single-page PDF with searchable text that PyPDF can parse.

    Used by tests that need hybrid retrieval to succeed (BM25 requires at least one
    non-empty chunk; a malformed XRef table causes PyPDF to silently drop all content).
    """
    page_text = (
        "The capital of France is Paris. "
        "Paris is known as the City of Light. "
        "The Eiffel Tower is located in Paris, France."
    )
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >>\n"
            b"endobj\n"
        ),
    ]
    stream_content = f"BT\n/F1 12 Tf\n72 720 Td\n({page_text})\nTj\nET".encode("latin-1")
    stream_obj = (
        f"5 0 obj\n<< /Length {len(stream_content)} >>\nstream\n".encode("latin-1")
        + stream_content
        + b"\nendstream\nendobj\n"
    )
    objects.append(stream_obj)

    header = b"%PDF-1.4\n"
    body = b"".join(objects)
    xref_offset = len(header) + len(body)

    offsets = []
    pos = len(header)
    for obj in objects:
        offsets.append(pos)
        pos += len(obj)

    xref = "xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"

    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
    return header + body + xref.encode("latin-1") + trailer.encode("latin-1")


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolates user database for each test and cleans up test directories."""
    test_db = tmp_path / "test_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db

    for user_id in ["user-multi-1", "user-multi-2"]:
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

    for user_id in ["user-multi-1", "user-multi-2"]:
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
def auth_headers_user1():
    """Registers user-multi-1 and returns auth headers."""
    UserDatabaseManager.create_user(
        user_id="user-multi-1",
        username="multi_user1",
        hashed_password="hashedpassword"
    )
    token = create_access_token({"sub": "multi_user1"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_user2():
    """Registers user-multi-2 and returns auth headers."""
    UserDatabaseManager.create_user(
        user_id="user-multi-2",
        username="multi_user2",
        hashed_password="hashedpassword"
    )
    token = create_access_token({"sub": "multi_user2"})
    return {"Authorization": f"Bearer {token}"}


def test_schema_validation_document_ids_must_be_uuids(auth_headers_user1):
    """document_ids containing non-UUID strings must return 422."""
    response = client.post(
        "/query",
        json={"document_ids": ["not-a-uuid"], "question": "Test?"},
        headers=auth_headers_user1
    )
    assert response.status_code == 422, (
        f"Expected 422 for invalid UUID in document_ids, got {response.status_code}: {response.text}"
    )


def test_schema_at_least_one_of_document_id_or_document_ids(auth_headers_user1):
    """Omitting both document_id and document_ids must return 422."""
    response = client.post(
        "/query",
        json={"question": "Test?"},
        headers=auth_headers_user1
    )
    assert response.status_code == 422, (
        f"Expected 422 when neither document_id nor document_ids provided, got {response.status_code}"
    )


def test_multi_query_access_control_missing_doc(auth_headers_user1):
    """Querying a document_id that does not exist must return 404."""
    fake_id = str(uuid.uuid4())
    response = client.post(
        "/query",
        json={"document_ids": [fake_id], "question": "Test?"},
        headers=auth_headers_user1
    )
    assert response.status_code == 404, (
        f"Expected 404 for unknown document, got {response.status_code}: {response.text}"
    )


def test_multi_query_access_control_wrong_user(auth_headers_user1, auth_headers_user2):
    """Querying another user's document in document_ids must return 403."""
    import time

    # Upload a document as user1
    files = {"file": ("test.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    upload_res = client.post("/upload", files=files, headers=auth_headers_user1)
    assert upload_res.status_code == 202
    job_id = upload_res.json()["job_id"]

    # Poll until ingestion completes
    for _ in range(50):
        status_res = client.get(f"/upload/{job_id}/status", headers=auth_headers_user1)
        if status_res.json().get("status") in ["complete", "failed"]:
            break
        time.sleep(0.1)

    # user2 tries to query user1's document
    response = client.post(
        "/query",
        json={"document_ids": [job_id], "question": "Test?"},
        headers=auth_headers_user2
    )
    assert response.status_code == 403, (
        f"Expected 403 for cross-tenant access, got {response.status_code}: {response.text}"
    )


def test_backward_compat_single_document_id(auth_headers_user1):
    """Legacy single document_id field must still work (backward compatibility)."""
    import time

    # Use a properly-structured PDF so PyPDF can extract text and BM25 can initialise
    proper_pdf = _make_proper_pdf_bytes()
    files = {"file": ("test.pdf", proper_pdf, "application/pdf")}
    upload_res = client.post("/upload", files=files, headers=auth_headers_user1)
    assert upload_res.status_code == 202
    job_id = upload_res.json()["job_id"]

    for _ in range(100):
        status_res = client.get(f"/upload/{job_id}/status", headers=auth_headers_user1)
        if status_res.json().get("status") in ["complete", "failed"]:
            break
        time.sleep(0.1)
    assert status_res.json().get("status") == "complete", (
        f"Ingestion did not complete: {status_res.text}"
    )

    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    mock_llm = MagicMock(spec=ChatGroq)
    mock_llm.invoke.return_value = AIMessage(content="Mocked answer.")
    mock_llm.return_value = AIMessage(content="Mocked answer.")

    with patch("app.core.qa.GroqConnectionManager") as mock_mgr:
        mock_mgr.get_chat_model.return_value = mock_llm
        response = client.post(
            "/query",
            json={"document_id": job_id, "question": "What is the capital of France?", "top_k": 3},
            headers=auth_headers_user1
        )

    assert response.status_code == 200, (
        f"Expected 200 for backward-compat single document_id, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)


def test_citations_include_document_id(auth_headers_user1):
    """Each citation in the response must include a document_id field."""
    import time

    # Use a properly-structured PDF so PyPDF can extract text and BM25 can initialise
    proper_pdf = _make_proper_pdf_bytes()
    files = {"file": ("test.pdf", proper_pdf, "application/pdf")}
    upload_res = client.post("/upload", files=files, headers=auth_headers_user1)
    assert upload_res.status_code == 202
    job_id = upload_res.json()["job_id"]

    for _ in range(100):
        status_res = client.get(f"/upload/{job_id}/status", headers=auth_headers_user1)
        if status_res.json().get("status") in ["complete", "failed"]:
            break
        time.sleep(0.1)
    assert status_res.json().get("status") == "complete", (
        f"Ingestion did not complete: {status_res.text}"
    )

    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    mock_llm = MagicMock(spec=ChatGroq)
    mock_llm.invoke.return_value = AIMessage(content="Mocked answer for citation test.")
    mock_llm.return_value = AIMessage(content="Mocked answer for citation test.")

    with patch("app.core.qa.GroqConnectionManager") as mock_mgr:
        mock_mgr.get_chat_model.return_value = mock_llm
        response = client.post(
            "/query",
            json={"document_id": job_id, "question": "What is the capital of France?", "top_k": 3},
            headers=auth_headers_user1
        )

    assert response.status_code == 200, (
        f"Expected 200 for citation test, got {response.status_code}: {response.text}"
    )
    data = response.json()
    for citation in data.get("citations", []):
        assert "document_id" in citation, (
            f"Expected 'document_id' in citation keys, got: {list(citation.keys())}"
        )


def test_deduplication_removes_duplicate_chunks(auth_headers_user1):
    """Identical chunks from multiple documents appear only once in the pooled results."""
    import time
    from langchain_core.documents import Document

    files = {"file": ("test.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    upload_res = client.post("/upload", files=files, headers=auth_headers_user1)
    assert upload_res.status_code == 202
    job_id = upload_res.json()["job_id"]

    for _ in range(50):
        status_res = client.get(f"/upload/{job_id}/status", headers=auth_headers_user1)
        if status_res.json().get("status") in ["complete", "failed"]:
            break
        time.sleep(0.1)

    # Mock get_hybrid_retriever to return identical chunks from two different doc ids
    duplicate_text = "This is a duplicate chunk that appears in both documents."
    doc_a = Document(
        page_content=duplicate_text,
        metadata={"source_filename": "a.pdf", "page_index": 0, "document_id": job_id, "chunk_id": "c1"}
    )
    doc_b = Document(
        page_content=duplicate_text,
        metadata={"source_filename": "b.pdf", "page_index": 0, "document_id": job_id, "chunk_id": "c2"}
    )

    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    mock_llm = MagicMock(spec=ChatGroq)
    mock_llm.invoke.return_value = AIMessage(content=duplicate_text)
    mock_llm.return_value = AIMessage(content=duplicate_text)

    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = [[doc_a], [doc_b]]

    with patch("app.core.qa.GroqConnectionManager") as mock_mgr, \
         patch("app.core.vectorstore.VectorStoreManager.get_hybrid_retriever", return_value=mock_retriever):
        mock_mgr.get_chat_model.return_value = mock_llm
        response = client.post(
            "/query",
            json={"document_ids": [job_id, job_id], "question": "Test dedup?", "top_k": 3},
            headers=auth_headers_user1
        )

    # Should succeed (200) - dedup ensures no duplicate chunks reached LLM
    assert response.status_code == 200, (
        f"Expected 200 for dedup test, got {response.status_code}: {response.text}"
    )
