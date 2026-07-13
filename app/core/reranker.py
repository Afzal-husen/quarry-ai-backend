import os
import threading
from typing import Any, Optional


class RerankerError(Exception):
    """Exception raised for errors in the reranker pipeline."""
    pass


class RerankManager:
    """Thread-safe singleton class to load and cache the local FlashRank Ranker model.

    This ensures that the cross-encoder model is only loaded into CPU memory once
    and shared across all concurrent route threads.
    """

    _instance: Optional[Any] = None
    _lock = threading.Lock()

    @classmethod
    def get_ranker(cls) -> Any:
        """Loads and caches the FlashRank Ranker model singleton.

        Returns:
            The instantiated Ranker object from flashrank library.

        Raises:
            RerankerError: If the model fails to load successfully.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Eagerly import inside lock to resolve potential type/model validation issues
                    try:
                        from flashrank import Ranker
                    except ImportError as e:
                        raise RerankerError(
                            "The 'flashrank' package is not installed. Please add it to pyproject.toml."
                        ) from e

                    model_name = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
                    cache_dir = os.getenv("FLASHRANK_CACHE_DIR", "/app/models/flashrank")
                    try:
                        # Ranker will download the model automatically on first run and cache it
                        cls._instance = Ranker(model_name=model_name, cache_dir=cache_dir)
                    except Exception as e:
                        raise RerankerError(
                            f"Failed to initialize FlashRank Ranker with model '{model_name}': {str(e)}"
                        ) from e
        return cls._instance
