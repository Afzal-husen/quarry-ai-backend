import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path to allow absolute imports relative to backend/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.documents import Document
from app.core.reranker import RerankManager
from app.routes.query import router
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever



def test_rerank_manager_singleton():
    """Verify that RerankManager caches and returns a singleton instance of FlashRank Ranker."""
    ranker1 = RerankManager.get_ranker()
    ranker2 = RerankManager.get_ranker()
    assert ranker1 is ranker2, "RerankManager did not return the identical singleton instance!"
    assert isinstance(ranker1, Ranker), "Returned instance is not a flashrank Ranker!"


def test_compression_retriever():
    """Verify that FlashrankRerank compressor correctly re-ranks and limits document counts."""
    ranker = RerankManager.get_ranker()
    compressor = FlashrankRerank(client=ranker, top_n=2)

    docs = [
        Document(page_content="Paris is the capital of France.", metadata={"source_filename": "test.pdf", "page_index": 1}),
        Document(page_content="Apples are round, sweet fruits.", metadata={"source_filename": "test.pdf", "page_index": 2}),
        Document(page_content="The Eiffel Tower is a landmark in Paris, France.", metadata={"source_filename": "test.pdf", "page_index": 3}),
    ]

    compressed_docs = compressor.compress_documents(docs, "What is the capital of France?")
    
    # We requested top_n=2, so we should get exactly 2 documents
    assert len(compressed_docs) == 2
    # The first document should contain Paris
    assert "Paris" in compressed_docs[0].page_content
    # Eiffel Tower document is also very relevant to Paris, so it should rank highly
    assert any("Eiffel Tower" in d.page_content for d in compressed_docs)
