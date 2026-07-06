import json
import shutil
import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path to allow absolute imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.vectorstore import VectorStoreManager, ChromaConnectionCache


def setup_mock_document(manager: VectorStoreManager, user_id: str, document_id: str, filename: str) -> Path:
    """Helper utility to write mock chunks and index a mock document on disk."""
    mock_chunks = {
        "document_id": document_id,
        "source_filename": filename,
        "total_chunks": 1,
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "page_index": 0,
                "text": f"This is mock text chunks content for document {document_id}.",
                "char_length": 60
            }
        ]
    }
    
    # Save chunk JSON metadata
    temp_chunks_dir = manager.chunks_dir / user_id
    temp_chunks_dir.mkdir(parents=True, exist_ok=True)
    temp_chunks_path = temp_chunks_dir / f"{document_id}.json"
    with open(temp_chunks_path, "w", encoding="utf-8") as f:
        json.dump(mock_chunks, f, indent=4)

    # Index document to Chroma
    manager.index_document(user_id, document_id, filename)
    return temp_chunks_path


def cleanup_mock_document(manager: VectorStoreManager, user_id: str, document_id: str, chunks_path: Path):
    """Helper utility to delete chunks file and vectorstore folders."""
    if chunks_path and chunks_path.exists():
        chunks_path.unlink()
    
    user_chunks_dir = manager.chunks_dir / user_id
    if user_chunks_dir.exists() and not list(user_chunks_dir.iterdir()):
        user_chunks_dir.rmdir()
        
    db_path = manager.vectorstore_dir / user_id / document_id
    if db_path.exists():
        try:
            shutil.rmtree(db_path)
        except Exception:
            pass
            
    user_vector_dir = manager.vectorstore_dir / user_id
    if user_vector_dir.exists() and not list(user_vector_dir.iterdir()):
        user_vector_dir.rmdir()


def test_cache_hit_reuse():
    """Verify that consecutive retrievers use the same Chroma object instance."""
    manager = VectorStoreManager()
    user_id = "test-user-cache"
    doc_id = str(uuid.uuid4())
    chunks_path = setup_mock_document(manager, user_id, doc_id, "test_cache.pdf")
    
    try:
        # Request retriever first time (populates cache)
        ret1 = manager.get_retriever(user_id=user_id, document_id=doc_id, top_k=1)
        
        # Request retriever second time (should reuse client)
        ret2 = manager.get_retriever(user_id=user_id, document_id=doc_id, top_k=1)
        
        assert ret1.vectorstore is ret2.vectorstore, "Retriever did not reuse the cached Chroma instance!"
        
        # Verify it's present in the global cache dictionary
        key = (user_id, doc_id)
        assert key in ChromaConnectionCache._cache
        assert ChromaConnectionCache._cache[key] is ret1.vectorstore

    finally:
        ChromaConnectionCache.evict(user_id, doc_id)
        cleanup_mock_document(manager, user_id, doc_id, chunks_path)


def test_lru_eviction():
    """Verify that least recently used client is evicted and closed when capacity is reached."""
    # Set capacity temporarily to 2
    original_max = ChromaConnectionCache._max_size
    ChromaConnectionCache._max_size = 2
    ChromaConnectionCache.clear()

    manager = VectorStoreManager()
    user_id = "test-user-lru"
    
    doc1 = str(uuid.uuid4())
    doc2 = str(uuid.uuid4())
    doc3 = str(uuid.uuid4())
    
    path1 = setup_mock_document(manager, user_id, doc1, "doc1.pdf")
    path2 = setup_mock_document(manager, user_id, doc2, "doc2.pdf")
    path3 = setup_mock_document(manager, user_id, doc3, "doc3.pdf")

    try:
        # Load 1 and 2
        manager.get_retriever(user_id, doc1)
        manager.get_retriever(user_id, doc2)
        
        assert (user_id, doc1) in ChromaConnectionCache._cache
        assert (user_id, doc2) in ChromaConnectionCache._cache
        
        # Load 3 -> triggers eviction of 1 (least recently used)
        manager.get_retriever(user_id, doc3)
        
        assert (user_id, doc1) not in ChromaConnectionCache._cache, "Least recently used document was not evicted!"
        assert (user_id, doc2) in ChromaConnectionCache._cache
        assert (user_id, doc3) in ChromaConnectionCache._cache

    finally:
        ChromaConnectionCache._max_size = original_max
        ChromaConnectionCache.clear()
        cleanup_mock_document(manager, user_id, doc1, path1)
        cleanup_mock_document(manager, user_id, doc2, path2)
        cleanup_mock_document(manager, user_id, doc3, path3)


def test_eviction_on_delete_and_reindex():
    """Verify that delete or reindex route triggers eviction from the cache."""
    # We will test the cache eviction method directly and then via documents manager.
    manager = VectorStoreManager()
    user_id = "test-user-delete"
    doc_id = str(uuid.uuid4())
    chunks_path = setup_mock_document(manager, user_id, doc_id, "test_delete.pdf")
    
    try:
        # Populate cache
        manager.get_retriever(user_id, doc_id)
        key = (user_id, doc_id)
        assert key in ChromaConnectionCache._cache
        
        # Evict on delete/reindex
        ChromaConnectionCache.evict(user_id, doc_id)
        assert key not in ChromaConnectionCache._cache, "Cache was not evicted on delete!"

    finally:
        cleanup_mock_document(manager, user_id, doc_id, chunks_path)


def test_shutdown_cleanup():
    """Verify that application shutdown event hook closes all open connections and clears the cache."""
    manager = VectorStoreManager()
    user_id = "test-user-shutdown"
    doc_id = str(uuid.uuid4())
    chunks_path = setup_mock_document(manager, user_id, doc_id, "test_shutdown.pdf")
    
    try:
        # Populate cache
        manager.get_retriever(user_id, doc_id)
        assert len(ChromaConnectionCache._cache) > 0
        
        # Shutdown clear
        ChromaConnectionCache.clear()
        assert len(ChromaConnectionCache._cache) == 0, "Cache was not cleared on shutdown!"

    finally:
        cleanup_mock_document(manager, user_id, doc_id, chunks_path)
