"""Tests for Advanced Chunking Strategies (Phase 19)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app
from app.core.chunker import DocumentChunker
from app.core.vectorstore import VectorStoreManager
from app.core.database import UserDatabaseManager
from app.core.auth import create_access_token
import app.core.database as db_mod

client = TestClient(app)


def test_semantic_splitting():
    """Verify semantic splitting correctly groups sentences based on distance thresholds."""
    chunker = DocumentChunker()
    
    # Sentence boundaries:
    # 0. "This is sentence one."
    # 1. "This is sentence two."
    # 2. "This is a completely different topic about cars."
    # 3. "The engine is powerful."
    text = "This is sentence one. This is sentence two. This is a completely different topic about cars. The engine is powerful."
    
    # Combined window groups for sentence_buffer_window = 1
    # Group 0 (around sent 0): sentences[0:2] -> "This is sentence one. This is sentence two."
    # Group 1 (around sent 1): sentences[0:3] -> "This is sentence one. This is sentence two. This is a completely different topic about cars."
    # Group 2 (around sent 2): sentences[1:4] -> "This is sentence two. This is a completely different topic about cars. The engine is powerful."
    # Group 3 (around sent 3): sentences[2:4] -> "This is a completely different topic about cars. The engine is powerful."
    
    # We want a split at index 1 (between sentence two and three).
    # Cosine distances:
    # d(V0, V1) = 0.0
    # d(V1, V2) = 1.0 (exceeds threshold)
    # d(V2, V3) = 0.0
    
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ]
    
    with patch("app.core.vectorstore.EmbeddingsManager.get_embeddings", return_value=mock_embeddings):
        chunks_abs = chunker._split_semantically(
            text=text,
            sentence_buffer_window=1,
            threshold_type="absolute",
            threshold_value=0.5
        )
        
        # We expect:
        # Chunk 1: "This is sentence one. This is sentence two."
        # Chunk 2: "This is a completely different topic about cars. The engine is powerful."
        assert len(chunks_abs) == 2
        assert chunks_abs[0] == "This is sentence one. This is sentence two."
        assert chunks_abs[1] == "This is a completely different topic about cars. The engine is powerful."


def test_parent_child_metadata_structure(tmp_path):
    """Verify that ingestion serializes both parent and child chunks into the JSON metadata file."""
    chunker = DocumentChunker(default_chunk_size=100, default_chunk_overlap=10)
    doc = Document(page_content="Sentence one. Sentence two. Sentence three. Sentence four.", metadata={"page": 0})
    
    # Split documents into child chunks using character strategy
    child_docs = chunker.split_documents([doc], chunk_size=20, chunk_overlap=5)
    
    assert len(child_docs) > 0
    # Every child must have parent references
    for child in child_docs:
        assert "parent_id" in child.metadata
        assert "parent_text" in child.metadata
        assert "chunk_id" in child.metadata
        assert child.metadata["page_index"] == 0

    # Save chunks to tmp path
    dest_path = chunker.save_chunks(
        document_id="test-doc-id",
        source_filename="test_file.txt",
        chunks=child_docs,
        output_dir=tmp_path,
        chunking_strategy="character"
    )
    
    assert dest_path.exists()
    with open(dest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["document_id"] == "test-doc-id"
    assert data["source_filename"] == "test_file.txt"
    assert data["chunking_strategy"] == "character"
    assert data["total_parents"] == 1  # Standard single parent split since the content is small (< 1500 chars)
    assert len(data["parents"]) == 1
    assert data["parents"][0]["text"] == doc.page_content
    assert data["total_chunks"] == len(child_docs)
    assert len(data["chunks"]) == len(child_docs)
    
    first_chunk = data["chunks"][0]
    assert "chunk_id" in first_chunk
    assert "parent_id" in first_chunk
    assert first_chunk["parent_id"] == data["parents"][0]["parent_id"]
    assert first_chunk["page_index"] == 0
    assert "text" in first_chunk
    assert "char_length" in first_chunk


def test_parent_document_resolution(tmp_path):
    """Verify that retrieval returns parent chunk text corresponding to matching child chunks."""
    manager = VectorStoreManager()
    # Override chunks_dir with temp path to isolate test execution
    manager.chunks_dir = tmp_path
    
    user_id = "user-123"
    document_id = "doc-123"
    parent_id = "parent-99"
    
    # Create parent-child relationship payload
    payload = {
        "document_id": document_id,
        "parents": [
            {
                "parent_id": parent_id,
                "page_index": 2,
                "text": "This is the full parent block content."
            }
        ],
        "chunks": [
            {
                "chunk_id": "child-0",
                "parent_id": parent_id,
                "page_index": 2,
                "text": "parent block",
                "char_length": 12
            }
        ]
    }
    
    user_chunks_dir = tmp_path / user_id
    user_chunks_dir.mkdir(parents=True, exist_ok=True)
    with open(user_chunks_dir / f"{document_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
        
    # Mock retrieved document
    retrieved_doc = Document(
        page_content="parent block",
        metadata={
            "document_id": document_id,
            "parent_id": parent_id,
            "page_index": 2
        }
    )
    
    resolved = manager.resolve_parent_documents(user_id=user_id, documents=[retrieved_doc])
    
    assert len(resolved) == 1
    assert resolved[0].page_content == "This is the full parent block content."
    assert resolved[0].metadata["document_id"] == document_id
    assert resolved[0].metadata["parent_id"] == parent_id
    assert resolved[0].metadata["child_text"] == "parent block"
    assert resolved[0].metadata["page_index"] == 2


@pytest.fixture
def auth_headers(tmp_path):
    """Registers a test user and returns request headers with a valid Bearer JWT."""
    # Temporarily isolate database
    test_db = tmp_path / "test_users.db"
    old_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = test_db
    
    UserDatabaseManager.initialize_db()
    
    username = "testchunker"
    UserDatabaseManager.create_user(
        user_id="user-123",
        username=username,
        hashed_password="hashedpassword"
    )
    token = create_access_token({"sub": username})
    
    yield {"Authorization": f"Bearer {token}"}
    
    db_mod.DB_PATH = old_db_path


def test_api_chunking_parameters(auth_headers):
    """Verify that upload and reindex endpoints accept and validate the new parameters correctly."""
    # 1. Invalid strategy
    files = {"file": ("test.pdf", b"%PDF-1.4...", "application/pdf")}
    response = client.post(
        "/upload?chunking_strategy=invalid_strategy",
        files=files,
        headers=auth_headers
    )
    assert response.status_code == 422
    assert "chunking_strategy must be either" in response.json()["detail"]

    # 2. Invalid threshold type
    response = client.post(
        "/upload?chunking_strategy=semantic&semantic_threshold_type=invalid_type",
        files=files,
        headers=auth_headers
    )
    assert response.status_code == 422
    assert "semantic_threshold_type must be" in response.json()["detail"]

    # 3. Invalid percentile threshold (out of bounds)
    response = client.post(
        "/upload?chunking_strategy=semantic&semantic_threshold_type=percentile&semantic_threshold=150",
        files=files,
        headers=auth_headers
    )
    assert response.status_code == 422
    assert "Percentile threshold must be between" in response.json()["detail"]

    # 4. Invalid standard deviation threshold (negative or zero)
    response = client.post(
        "/upload?chunking_strategy=semantic&semantic_threshold_type=standard_deviation&semantic_threshold=-0.5",
        files=files,
        headers=auth_headers
    )
    assert response.status_code == 422
    assert "Threshold value must be greater than 0" in response.json()["detail"]

    # 5. Reindex invalid strategy
    valid_uuid = "12345678-1234-5678-1234-567812345678"
    response = client.post(
        f"/documents/{valid_uuid}/reindex?chunking_strategy=invalid_strategy",
        headers=auth_headers
    )
    assert response.status_code == 422
    assert "chunking_strategy must be either" in response.json()["detail"]
