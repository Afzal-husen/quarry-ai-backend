import json
import os
import re
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.paths import get_data_dir

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


class EmbeddingsError(Exception):
    """Exception raised for errors in the embeddings pipeline."""
    pass


class VectorStoreError(Exception):
    """Exception raised for errors in the vector store persistence or retrieval."""
    pass


class FastEmbedLangChainWrapper(Embeddings):
    """Wrapper class for FastEmbed to conform to LangChain's Embeddings interface."""

    def __init__(self, model_name: str, cache_dir: str, threads: int):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        try:
            self.client = TextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=threads
            )
        except Exception as e:
            raise EmbeddingsError(f"Failed to initialize FastEmbed model '{model_name}': {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return [list(map(float, vec)) for vec in self.client.embed(texts)]
        except Exception as e:
            raise EmbeddingsError(f"Failed to embed documents with FastEmbed: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        try:
            vec = next(iter(self.client.query_embed(text)))
            return list(map(float, vec))
        except Exception as e:
            raise EmbeddingsError(f"Failed to embed query with FastEmbed: {str(e)}") from e


class EmbeddingsManager:
    """Thread-safe singleton class to load and cache local FastEmbed embeddings."""

    _instance: Optional[Embeddings] = None
    _lock = threading.Lock()

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        """Loads and caches the FastEmbed embedding model singleton.

        This ensures that the model is only loaded into CPU memory once
        and shared across all concurrent route threads.

        Returns:
            The instantiated Embeddings object.

        Raises:
            EmbeddingsError: If the model fails to load successfully.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Retrieve the embedding model from environment
                    model_env = os.getenv(
                        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                    
                    # Custom Mapping Dictionary mapping HF IDs to FastEmbed equivalents
                    MODEL_MAP = {
                        "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
                        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
                        "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
                        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5"
                    }
                    model_name = MODEL_MAP.get(model_env, model_env)
                    
                    # Local cache directory path
                    BASE_DIR = Path(__file__).resolve().parent.parent.parent
                    cache_dir = os.getenv("FASTEMBED_CACHE_DIR", str(BASE_DIR / "data" / "models" / "fastembed"))
                    
                    # Thread configuration for resource containment on cloud containers
                    try:
                        threads = int(os.getenv("FASTEMBED_THREADS", "1"))
                    except ValueError:
                        threads = 1
                        
                    try:
                        cls._instance = FastEmbedLangChainWrapper(
                            model_name=model_name,
                            cache_dir=cache_dir,
                            threads=threads
                        )
                    except Exception as e:
                        raise EmbeddingsError(
                            f"Failed to initialize FastEmbed embedding model '{model_name}': {str(e)}"
                        ) from e
        return cls._instance


class ChromaConnectionCache:
    """Thread-safe bounded LRU cache for Chroma client instances."""

    _cache: OrderedDict = OrderedDict()
    _lock = threading.Lock()
    _max_size: Optional[int] = None

    @classmethod
    def get_max_size(cls) -> int:
        """Resolves cache limit size from environment variable, defaulting to 10."""
        if cls._max_size is not None:
            return cls._max_size
        try:
            return int(os.getenv("CHROMA_CACHE_SIZE", "10"))
        except ValueError:
            return 10

    @classmethod
    def get(cls, user_id: str, document_id: str, db_path: Path, embeddings: Any) -> Chroma:
        """Fetches an existing Chroma client from the cache or instantiates a new one.

        Evicts the oldest client using LRU policy if capacity is exceeded.
        """
        key = (user_id, document_id)
        with cls._lock:
            if key in cls._cache:
                # Move to end to mark as recently used
                cls._cache.move_to_end(key)
                return cls._cache[key]

        # Instantiate new Chroma client outside of cache lock to prevent serialization blocking
        vectorstore = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings
        )

        with cls._lock:
            # Check if another thread populated this key while we were creating it
            if key in cls._cache:
                cls._close_client(vectorstore)
                cls._cache.move_to_end(key)
                return cls._cache[key]

            # Check capacity and evict oldest if necessary
            if len(cls._cache) >= cls.get_max_size():
                cls._evict_lru_under_lock()

            cls._cache[key] = vectorstore
            return vectorstore

    @classmethod
    def _evict_lru_under_lock(cls) -> None:
        """Helper to evict the least recently used client. Must be called under lock."""
        if not cls._cache:
            return
        # Pop the first element (oldest in OrderedDict)
        oldest_key, oldest_vectorstore = cls._cache.popitem(last=False)
        cls._close_client(oldest_vectorstore)

    @classmethod
    def evict(cls, user_id: str, document_id: str) -> None:
        """Removes the Chroma client from the cache and explicitly closes its SQLite connection."""
        key = (user_id, document_id)
        with cls._lock:
            vectorstore = cls._cache.pop(key, None)
            if vectorstore:
                cls._close_client(vectorstore)

    @classmethod
    def clear(cls) -> None:
        """Closes all cached connections and clears the cache."""
        with cls._lock:
            for vectorstore in cls._cache.values():
                cls._close_client(vectorstore)
            cls._cache.clear()

    @classmethod
    def _close_client(cls, vectorstore: Chroma) -> None:
        """Helper to call close() on Chroma's internal PersistentClient."""
        client = getattr(vectorstore, "_client", None)
        if client:
            close_fn = getattr(client, "_close", None) or getattr(
                client, "close", None)
            if close_fn and callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


class VectorStoreManager:
    """Orchestrates indexing and retrieval of document chunks using isolated Chroma vector stores."""

    _bm25_cache: Dict[Tuple[str, str], BM25Retriever] = {}
    _bm25_lock = threading.Lock()

    def __init__(self):
        """Initializes the VectorStoreManager resolving backend data directories."""
        data_dir = get_data_dir()
        self.chunks_dir = data_dir / "chunks"
        self.vectorstore_dir = data_dir / "vectorstore"

    def index_document(self, user_id: str, document_id: str, source_filename: str) -> Path:
        """Reads serialized JSON chunks and indexes them inside an isolated Chroma DB folder on disk.

        Args:
            user_id: The unique UUID of the authenticated user.
            document_id: The unique UUID of the uploaded document.
            source_filename: The original name of the uploaded document.

        Returns:
            The Path where the isolated Chroma index is persisted.

        Raises:
            VectorStoreError: If chunks do not exist, or indexing persists incorrectly.
        """
        chunks_file_path = self.chunks_dir / user_id / f"{document_id}.json"
        if not chunks_file_path.exists():
            raise VectorStoreError(
                f"Ingested chunks metadata file not found at {chunks_file_path}")

        # 1. Load serialized JSON chunks
        try:
            with open(chunks_file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to read chunks metadata file: {str(e)}") from e

        chunks_list = payload.get("chunks", [])
        if not chunks_list:
            raise VectorStoreError(
                "Metadata payload contains empty chunks list.")

        # 2. Convert raw chunk dicts into standard LangChain Document objects
        documents: List[Document] = []
        for chunk in chunks_list:
            metadata: Dict[str, Any] = {
                "chunk_id": chunk["chunk_id"],
                "parent_id": chunk.get("parent_id"),
                "page_index": chunk["page_index"],
                "source_filename": source_filename,
                "document_id": document_id
            }
            doc = Document(page_content=chunk["text"], metadata=metadata)
            documents.append(doc)

        # 3. Resolve persistent directory path
        db_path = self.vectorstore_dir / user_id / document_id

        # 4. Initialize isolated Chroma vector index and persist embeddings on disk
        try:
            embeddings = EmbeddingsManager.get_embeddings()
            # Chroma handles automatic SQLite serialization upon instantiation
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=str(db_path)
            )
            # Explicitly close the database client to release on-disk file descriptors (critical for Windows)
            client = getattr(vectorstore, "_client", None)
            if client:
                close_fn = getattr(client, "close", None)
                if close_fn and callable(close_fn):
                    close_fn()
        except Exception as e:
            raise VectorStoreError(
                f"Failed to index documents into Chroma database at {db_path}: {str(e)}"
            ) from e

        return db_path

    def delete_document(self, user_id: str, document_id: str) -> None:
        """Removes the isolated Chroma DB index directory from disk and evicts caches for a user and document.

        Args:
            user_id: The unique UUID of the authenticated user.
            document_id: The unique UUID of the target document.

        Raises:
            VectorStoreError: If directory removal fails.
        """
        # Evict from BM25 memory cache
        key = (user_id, document_id)
        with self._bm25_lock:
            self._bm25_cache.pop(key, None)

        db_path = self.vectorstore_dir / user_id / document_id
        if db_path.exists():
            try:
                shutil.rmtree(db_path)
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to delete Chroma database index directory at {db_path}: {str(e)}"
                ) from e

    def get_retriever(self, user_id: str, document_id: str, top_k: int = 3) -> VectorStoreRetriever:
        """Loads an isolated Chroma DB from disk and returns a native LangChain VectorStoreRetriever.

        Args:
            user_id: The unique UUID of the authenticated user.
            document_id: The unique UUID of the target document.
            top_k: The number of relevant matching chunks to return (default 3).

        Returns:
            A native LangChain VectorStoreRetriever.

        Raises:
            VectorStoreError: If the document index directory does not exist or loading fails.
        """
        db_path = self.vectorstore_dir / user_id / document_id
        if not db_path.exists():
            raise VectorStoreError(
                f"Vector database index for document '{document_id}' does not exist on disk."
            )

        try:
            embeddings = EmbeddingsManager.get_embeddings()
            vectorstore = ChromaConnectionCache.get(
                user_id=user_id,
                document_id=document_id,
                db_path=db_path,
                embeddings=embeddings
            )
            return vectorstore.as_retriever(search_kwargs={"k": top_k})
        except Exception as e:
            raise VectorStoreError(
                f"Failed to load isolated Chroma database for document '{document_id}': {str(e)}"
            ) from e

    def retrieve_relevant_chunks(
        self,
        user_id: str,
        document_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Document]:
        """Loads an isolated Chroma DB from disk and retrieves the top-K semantically matching chunks.

        Args:
            user_id: The unique UUID of the authenticated user.
            document_id: The unique UUID of the target document.
            query: The natural language search query.
            top_k: The number of relevant matching chunks to return (default 3).

        Returns:
            A list of matching LangChain Document objects.

        Raises:
            VectorStoreError: If the document index directory does not exist or querying fails.
        """
        db_path = self.vectorstore_dir / user_id / document_id
        if not db_path.exists():
            raise VectorStoreError(
                f"Vector database index for document '{document_id}' does not exist on disk."
            )

        try:
            embeddings = EmbeddingsManager.get_embeddings()
            # Load the persistent Chroma DB instance from cache
            vectorstore = ChromaConnectionCache.get(
                user_id=user_id,
                document_id=document_id,
                db_path=db_path,
                embeddings=embeddings
            )
            results = vectorstore.similarity_search(query, k=top_k)
            return results
        except Exception as e:
            raise VectorStoreError(
                f"Failed to query isolated Chroma database for document '{document_id}': {str(e)}"
            ) from e

    def get_hybrid_retriever(
        self,
        user_id: str,
        document_id: str,
        top_k: int = 3
    ) -> EnsembleRetriever:
        """Loads a user-isolated BM25 retriever dynamically from text chunks and combines

        it with the Chroma vector store retriever using Reciprocal Rank Fusion (RRF).

        Args:
            user_id: The unique UUID of the authenticated user.
            document_id: The unique UUID of the target document.
            top_k: The number of relevant matching chunks to return (default 3).

        Returns:
            An EnsembleRetriever combining lexical and semantic search components.

        Raises:
            VectorStoreError: If the chunks or vector index does not exist.
        """
        key = (user_id, document_id)
        with self._bm25_lock:
            bm25_retriever = self._bm25_cache.get(key)

        if bm25_retriever is not None:
            # Update dynamic k parameter
            bm25_retriever.k = top_k
        else:
            # 1. Resolve paths
            chunks_file_path = self.chunks_dir / \
                user_id / f"{document_id}.json"
            if not chunks_file_path.exists():
                raise VectorStoreError(
                    f"Ingested chunks metadata file not found at {chunks_file_path}. Please upload the document first."
                )

            # 2. Load serialized JSON chunks
            try:
                with open(chunks_file_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to read chunks metadata file: {str(e)}") from e

            chunks_list = payload.get("chunks", [])
            if not chunks_list:
                raise VectorStoreError(
                    "Metadata payload contains empty chunks list.")

            # 3. Convert raw chunk dicts into standard LangChain Document objects
            documents: List[Document] = []
            for chunk in chunks_list:
                metadata: Dict[str, Any] = {
                    "chunk_id": chunk["chunk_id"],
                    "parent_id": chunk.get("parent_id"),
                    "page_index": chunk["page_index"],
                    "source_filename": payload.get("source_filename", "unknown"),
                    "document_id": document_id
                }
                doc = Document(page_content=chunk["text"], metadata=metadata)
                documents.append(doc)

            # 4. Tokenization preprocessing for case-insensitive BM25 search
            def preprocess_text(text: str) -> List[str]:
                return re.findall(r"\w+", text.lower())

            # 5. Initialize BM25 retriever dynamically
            try:
                bm25_retriever = BM25Retriever.from_documents(
                    documents=documents,
                    preprocess_func=preprocess_text
                )
                bm25_retriever.k = top_k
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to initialize BM25 retriever dynamically: {str(e)}") from e

            # Save in memory cache
            with self._bm25_lock:
                self._bm25_cache[key] = bm25_retriever

        # 6. Initialize vector retriever
        vector_retriever = self.get_retriever(
            user_id=user_id, document_id=document_id, top_k=top_k)

        # 7. Load weights from environment configurations with balanced default fallbacks
        try:
            lexical_weight = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.5"))
            semantic_weight = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.5"))
        except ValueError:
            lexical_weight = 0.5
            semantic_weight = 0.5

        # 8. Construct EnsembleRetriever with Reciprocal Rank Fusion (RRF)
        try:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[lexical_weight, semantic_weight]
            )
            return ensemble_retriever
        except Exception as e:
            raise VectorStoreError(
                f"Failed to construct hybrid EnsembleRetriever: {str(e)}") from e

    def resolve_parent_documents(self, user_id: str, documents: Sequence[Document]) -> List[Document]:
        """Resolves retrieved child chunks to their corresponding parent chunks."""
        resolved_documents = []
        loaded_payloads = {}

        for doc in documents:
            doc_id = doc.metadata.get("document_id")
            parent_id = doc.metadata.get("parent_id")

            if not doc_id or not parent_id:
                resolved_documents.append(doc)
                continue

            # Load from cache or read from disk
            if doc_id not in loaded_payloads:
                chunks_file_path = self.chunks_dir / user_id / f"{doc_id}.json"
                if not chunks_file_path.exists():
                    loaded_payloads[doc_id] = {}
                else:
                    try:
                        with open(chunks_file_path, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                            parents_list = payload.get("parents", [])
                            loaded_payloads[doc_id] = {
                                p["parent_id"]: p["text"] for p in parents_list}
                    except Exception:
                        loaded_payloads[doc_id] = {}

            parent_text = loaded_payloads[doc_id].get(parent_id)
            if parent_text:
                resolved_doc = Document(
                    page_content=parent_text,
                    metadata={
                        **doc.metadata,
                        "child_text": doc.page_content
                    }
                )
                resolved_documents.append(resolved_doc)
            else:
                resolved_documents.append(doc)

        return resolved_documents
