import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app.core.database as db_mod
from app.core.database import UserDatabaseManager, ChatDatabaseManager
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolates database for each test case using a temporary db file path."""
    test_db = tmp_path / "test_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db
    
    UserDatabaseManager.initialize_db()
    yield
    db_mod.DB_PATH = old_db_path


def get_auth_headers(username: str = "testuser") -> dict:
    """Helper to register and login a user, returning Bearer auth header."""
    client.post("/auth/signup", json={"username": username, "password": "password123"})
    login_resp = client.post("/auth/login", json={"username": username, "password": "password123"})
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@patch("app.routes.query.RerankManager.get_ranker")
@patch("app.routes.query.FlashrankRerank")
@patch("app.routes.query.vector_manager")
@patch("app.routes.query.qa_pipeline")
def test_query_with_session_writes_to_history(mock_qa, mock_vector, mock_compressor_cls, mock_ranker, tmp_path):
    """Verify POST /query with session_id performs condensation and saves messages."""
    headers = get_auth_headers()
    
    # 1. Create a chat session
    session_resp = client.post("/sessions", json={}, headers=headers)
    session_id = session_resp.json()["id"]
    
    # Mock Document chunk list
    from langchain_core.documents import Document
    mock_docs = [
        Document(page_content="RAG represents Retrieval-Augmented Generation.", metadata={"source_filename": "rag.pdf", "page_index": 1, "chunk_id": "c1", "document_id": "doc1"})
    ]
    mock_vector.get_hybrid_retriever.return_value.invoke.return_value = mock_docs
    mock_compressor_cls.return_value.compress_documents.return_value = mock_docs
    mock_vector.resolve_parent_documents.return_value = mock_docs
    mock_vector.vectorstore_dir = tmp_path # Fake directory
    # Create fake directory on disk to pass ownership check
    
    # Setup QA mock
    mock_qa.generate_answer.return_value = {
        "answer": "RAG stands for Retrieval-Augmented Generation.",
        "citations": [{"source_filename": "rag.pdf", "page_index": 1, "document_id": "doc1", "text": "RAG represents..."}]
    }
    mock_qa.condense_query.return_value = "What is RAG?"
    mock_qa.generate_session_title.return_value = "RAG Definition"
    
    # Mocking check for document ownership
    with patch("app.routes.query.Path.exists", return_value=True), \
         patch("app.routes.query.Path.glob", return_value=[tmp_path]):
        
        # Send query
        response = client.post(
            "/query",
            json={
                "document_id": "11111111-1111-1111-1111-111111111111",
                "question": "What does RAG mean?",
                "session_id": session_id
            },
            headers=headers
        )
        
        assert response.status_code == 200
        
        # Verify title was auto-updated (first question)
        get_sess = client.get(f"/sessions/{session_id}", headers=headers)
        assert get_sess.json()["title"] == "RAG Definition"
        
        # Verify messages history size (should contain 2 messages: user and assistant)
        history = get_sess.json()["messages"]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What does RAG mean?"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "RAG stands for Retrieval-Augmented Generation."
        assert history[1]["metadata"] is not None
        assert history[1]["metadata"][0]["source_filename"] == "rag.pdf"


@patch("app.routes.query.RerankManager.get_ranker")
@patch("app.routes.query.FlashrankRerank")
@patch("app.routes.query.vector_manager")
@patch("app.routes.query.qa_pipeline")
def test_query_stream_with_session_writes_to_history(mock_qa, mock_vector, mock_compressor_cls, mock_ranker, tmp_path):
    """Verify POST /query/stream with session_id saves messages on stream completion."""
    headers = get_auth_headers()
    
    # Create chat session
    session_resp = client.post("/sessions", json={}, headers=headers)
    session_id = session_resp.json()["id"]
    
    # Mock vector/QA pipeline
    from langchain_core.documents import Document
    mock_docs = [
        Document(page_content="SSE is Server-Sent Events.", metadata={"source_filename": "sse.pdf", "page_index": 2, "chunk_id": "c2", "document_id": "doc2"})
    ]
    mock_vector.get_hybrid_retriever.return_value.invoke.return_value = mock_docs
    mock_compressor_cls.return_value.compress_documents.return_value = mock_docs
    mock_vector.resolve_parent_documents.return_value = mock_docs
    mock_vector.vectorstore_dir = tmp_path
    
    # Async generator mock for streaming
    async def fake_stream(*args, **kwargs):
        yield "SSE "
        yield "stands "
        yield "for "
        yield "Server-Sent "
        yield "Events."
        
    mock_qa.generate_answer_stream = fake_stream
    mock_qa.condense_query.return_value = "What is SSE?"
    mock_qa.generate_session_title.return_value = "SSE Intro"
    
    with patch("app.routes.query.Path.exists", return_value=True), \
         patch("app.routes.query.Path.glob", return_value=[tmp_path]):
         
        response = client.post(
            "/query/stream",
            json={
                "document_id": "11111111-1111-1111-1111-111111111111",
                "question": "What is SSE streaming?",
                "session_id": session_id
            },
            headers=headers
        )
        
        assert response.status_code == 200
        # Consume the stream iterator
        content = response.text
        assert "SSE " in content
        assert "stands " in content
        assert "for " in content
        assert "Server-Sent " in content
        assert "Events." in content
        
        # Verify DB writes occurred
        get_sess = client.get(f"/sessions/{session_id}", headers=headers)
        history = get_sess.json()["messages"]
        assert len(history) == 2
        assert history[0]["content"] == "What is SSE streaming?"
        assert history[1]["content"] == "SSE stands for Server-Sent Events."




def test_conversational_endpoints_session_ownership_boundaries():
    """Verify endpoints reject invalid or unowned session IDs."""
    headers1 = get_auth_headers("user1")
    headers2 = get_auth_headers("user2")
    
    # User 1 creates session
    sess_resp = client.post("/sessions", json={}, headers=headers1)
    session_id = sess_resp.json()["id"]
    
    # User 2 tries to query using User 1's session_id
    response = client.post(
        "/query",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question": "Hello",
            "session_id": session_id
        },
        headers=headers2
    )
    assert response.status_code == 403
    
    # Query with non-existent session_id
    response_nonexistent = client.post(
        "/query",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question": "Hello",
            "session_id": "22222222-2222-2222-2222-222222222222"
        },
        headers=headers1
    )
    assert response_nonexistent.status_code == 404
