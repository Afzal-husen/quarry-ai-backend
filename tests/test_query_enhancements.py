import os
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

# Add backend directory to sys.path to allow absolute imports relative to backend/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.documents import Document
from app.core.qa import QAPipeline


@pytest.fixture(autouse=True)
def enable_query_expansion():
    with patch.dict(os.environ, {"TEST_QUERY_EXPANSION_ACTIVE": "1"}):
        yield


@patch("app.core.qa.GroqConnectionManager")
def test_generate_alternative_queries(mock_connection_manager):
    """Verify that generate_alternative_queries returns exactly 3 query variations."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    
    mock_llm = MagicMock(spec=ChatGroq)
    mock_response = AIMessage(content="query variation one\nquery variation two\nquery variation three")
    mock_llm.invoke.return_value = mock_response
    mock_connection_manager.get_chat_model.return_value = mock_llm

    pipeline = QAPipeline()
    variations = pipeline.generate_alternative_queries("What is Python?")
    
    assert len(variations) == 3
    assert variations[0] == "query variation one"
    assert variations[1] == "query variation two"
    assert variations[2] == "query variation three"


@patch("app.core.qa.GroqConnectionManager")
def test_generate_alternative_queries_greeting_skipped(mock_connection_manager):
    """Verify query expansion is skipped for generic greetings."""
    pipeline = QAPipeline()
    
    # Greetings list must return empty immediately without calling LLM
    assert pipeline.generate_alternative_queries("hello") == []
    assert pipeline.generate_alternative_queries("Hi!") == []
    assert pipeline.generate_alternative_queries("how are you?") == []
    
    assert not mock_connection_manager.get_chat_model.called


@patch("app.core.qa.GroqConnectionManager")
def test_qa_pipeline_greeting_path(mock_connection_manager):
    """Verify that greetings do not include citations or disclaimers."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    
    mock_llm = MagicMock(spec=ChatGroq)
    mock_response = AIMessage(content="Hello! How can I help you today?")
    mock_llm.invoke.return_value = mock_response
    mock_connection_manager.get_chat_model.return_value = mock_llm

    retrieved_docs = [
        Document(page_content="Context text", metadata={"source_filename": "info.pdf", "page_index": 1})
    ]

    pipeline = QAPipeline()
    result = pipeline.generate_answer("hello", retrieved_docs)
    
    assert result["answer"] == "Hello! How can I help you today?"
    assert len(result["citations"]) == 0  # No citations for greetings


@patch("app.core.qa.GroqConnectionManager")
def test_qa_pipeline_fallback_path(mock_connection_manager):
    """Verify that fallback answers include the required disclaimer and no citations."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    
    disclaimer = "Disclaimer: This information was not found in your uploaded documents and is generated using general AI knowledge."
    mock_llm = MagicMock(spec=ChatGroq)
    mock_response = AIMessage(content=f"Python is a coding language.\n\n{disclaimer}")
    mock_llm.invoke.return_value = mock_response
    mock_connection_manager.get_chat_model.return_value = mock_llm

    retrieved_docs = [
        Document(page_content="Context text", metadata={"source_filename": "info.pdf", "page_index": 1})
    ]

    pipeline = QAPipeline()
    result = pipeline.generate_answer("What is Python?", retrieved_docs)
    
    assert "Disclaimer: This information was not found" in result["answer"]
    assert len(result["citations"]) == 0  # No citations for fallbacks


@patch("app.core.qa.GroqConnectionManager")
def test_qa_pipeline_grounded_path(mock_connection_manager):
    """Verify that grounded answers include citations when citation tags are present."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    
    mock_llm = MagicMock(spec=ChatGroq)
    mock_response = AIMessage(content="FastAPI is a Python web framework [1].")
    mock_llm.invoke.return_value = mock_response
    mock_connection_manager.get_chat_model.return_value = mock_llm

    retrieved_docs = [
        Document(
            page_content="FastAPI is a modern, fast web framework.",
            metadata={"source_filename": "fastapi.pdf", "page_index": 2, "chunk_id": "c-1", "document_id": "doc-uuid"}
        )
    ]

    pipeline = QAPipeline()
    result = pipeline.generate_answer("What is FastAPI?", retrieved_docs)
    
    assert "[1]" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_filename"] == "fastapi.pdf"
    assert result["citations"][0]["page_index"] == 2
