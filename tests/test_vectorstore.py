import json
import shutil
import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path to allow absolute imports relative to backend/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.vectorstore import EmbeddingsManager, VectorStoreManager, ChromaConnectionCache


def test_embeddings_manager_singleton():
    """Verify that the EmbeddingsManager caches and returns a singleton instance."""
    emb1 = EmbeddingsManager.get_embeddings()
    emb2 = EmbeddingsManager.get_embeddings()
    assert emb1 is emb2, "EmbeddingsManager did not return the identical singleton instance!"


def test_embeddings_generation():
    """Verify that the loaded local Hugging Face model generates standard vector dimensions."""
    embeddings = EmbeddingsManager.get_embeddings()
    vector = embeddings.embed_query("Test semantic query")
    assert isinstance(vector, list)
    assert len(vector) == 384, f"Expected 384 dimensions for all-MiniLM-L6-v2, got {len(vector)}"
    assert all(isinstance(val, float) for val in vector)


def test_vectorstore_indexing_and_retrieval():
    """Verify that VectorStoreManager cleanly indexes chunks and semantically retrieves top-K matches."""
    manager = VectorStoreManager()
    document_uuid = str(uuid.uuid4())
    source_filename = "test_document.pdf"

    # 1. Setup mock chunks data
    mock_chunks = {
        "document_id": document_uuid,
        "source_filename": source_filename,
        "total_chunks": 3,
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "page_index": 0,
                "text": "Apples are sweet round red or green fruits produced by an apple tree.",
                "char_length": 68
            },
            {
                "chunk_id": "chunk-2",
                "page_index": 1,
                "text": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python.",
                "char_length": 91
            },
            {
                "chunk_id": "chunk-3",
                "page_index": 2,
                "text": "Retrieval-Augmented Generation (RAG) is a technique for optimizing the output of a LLM.",
                "char_length": 88
            }
        ]
    }

    # 2. Persist mock chunks to temp JSON file
    temp_chunks_dir = manager.chunks_dir / "test-user-123"
    temp_chunks_dir.mkdir(parents=True, exist_ok=True)
    temp_chunks_path = temp_chunks_dir / f"{document_uuid}.json"
    with open(temp_chunks_path, "w", encoding="utf-8") as f:
        json.dump(mock_chunks, f, indent=4)

    db_path = manager.vectorstore_dir / "test-user-123" / document_uuid

    try:
        # 3. Index documents into isolated Chroma vector store
        persisted_path = manager.index_document("test-user-123", document_uuid, source_filename)
        assert persisted_path == db_path
        assert db_path.exists(), f"Chroma database path '{db_path}' was not created!"

        # 4. Query isolated database and verify semantic retrieval
        retrieved_docs = manager.retrieve_relevant_chunks(
            user_id="test-user-123",
            document_id=document_uuid,
            query="Tell me about FastAPI web frameworks in Python.",
            top_k=1
        )

        assert len(retrieved_docs) == 1
        matched_doc = retrieved_docs[0]
        assert "FastAPI" in matched_doc.page_content
        assert matched_doc.metadata["document_id"] == document_uuid
        assert matched_doc.metadata["page_index"] == 1
        assert matched_doc.metadata["chunk_id"] == "chunk-2"

        # Verify semantic ranking
        retrieved_rag = manager.retrieve_relevant_chunks(
            user_id="test-user-123",
            document_id=document_uuid,
            query="What is RAG retrieval model optimization?",
            top_k=1
        )
        assert len(retrieved_rag) == 1
        assert "Retrieval-Augmented Generation" in retrieved_rag[0].page_content

    finally:
        # Evict from cache to release file locks on Windows
        ChromaConnectionCache.evict("test-user-123", document_uuid)

        # 5. Clean up temporary files on disk
        if temp_chunks_path.exists():
            temp_chunks_path.unlink()
        # Clean up mock user chunks directory if empty
        if temp_chunks_dir.exists() and not list(temp_chunks_dir.iterdir()):
            temp_chunks_dir.rmdir()
            
        if db_path.exists():
            shutil.rmtree(db_path)
        # Clean up mock user vectorstore directory if empty
        user_vector_dir = manager.vectorstore_dir / "test-user-123"
        if user_vector_dir.exists() and not list(user_vector_dir.iterdir()):
            user_vector_dir.rmdir()


def test_vectorstore_hybrid_retrieval(monkeypatch):
    """Verify that VectorStoreManager constructs EnsembleRetriever and queries correctly with RRF."""
    manager = VectorStoreManager()
    document_uuid = str(uuid.uuid4())
    source_filename = "test_hybrid.pdf"
    user_id = "test-user-hybrid"

    mock_chunks = {
        "document_id": document_uuid,
        "source_filename": source_filename,
        "total_chunks": 3,
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "page_index": 0,
                "text": "Apples are sweet round red or green fruits produced by an apple tree.",
                "char_length": 68
            },
            {
                "chunk_id": "chunk-2",
                "page_index": 1,
                "text": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python.",
                "char_length": 91
            },
            {
                "chunk_id": "chunk-3",
                "page_index": 2,
                "text": "Retrieval-Augmented Generation (RAG) is a technique for optimizing the output of a LLM.",
                "char_length": 88
            }
        ]
    }

    # Persist mock chunks to user directory
    temp_chunks_dir = manager.chunks_dir / user_id
    temp_chunks_dir.mkdir(parents=True, exist_ok=True)
    temp_chunks_path = temp_chunks_dir / f"{document_uuid}.json"
    with open(temp_chunks_path, "w", encoding="utf-8") as f:
        json.dump(mock_chunks, f, indent=4)

    # Index document in Chroma
    manager.index_document(user_id, document_uuid, source_filename)
    db_path = manager.vectorstore_dir / user_id / document_uuid

    retriever = None
    try:
        # Mock weights via monkeypatch
        monkeypatch.setenv("HYBRID_LEXICAL_WEIGHT", "0.7")
        monkeypatch.setenv("HYBRID_SEMANTIC_WEIGHT", "0.3")

        # Get retriever
        retriever = manager.get_hybrid_retriever(
            user_id=user_id,
            document_id=document_uuid,
            top_k=2
        )

        from langchain_classic.retrievers import EnsembleRetriever
        assert isinstance(retriever, EnsembleRetriever)
        assert retriever.weights == [0.7, 0.3]

        # Check keyword/lexical search query hits
        results = retriever.invoke("apples fruits")
        assert len(results) >= 1
        assert "Apples" in results[0].page_content

        # Verify multi-tenant isolation raises VectorStoreError for unauthorized user
        import pytest
        from app.core.vectorstore import VectorStoreError
        with pytest.raises(VectorStoreError):
            manager.get_hybrid_retriever(
                user_id="unauthorized-user-id",
                document_id=document_uuid,
                top_k=2
            )

    finally:
        # Evict from cache to release file locks on Windows
        ChromaConnectionCache.evict(user_id, document_uuid)

        # Clean up files on disk
        if temp_chunks_path.exists():
            temp_chunks_path.unlink()
        if temp_chunks_dir.exists() and not list(temp_chunks_dir.iterdir()):
            temp_chunks_dir.rmdir()
        if db_path.exists():
            shutil.rmtree(db_path)
        user_vector_dir = manager.vectorstore_dir / user_id
        if user_vector_dir.exists() and not list(user_vector_dir.iterdir()):
            user_vector_dir.rmdir()

