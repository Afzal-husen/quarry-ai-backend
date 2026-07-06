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
from app.core.qa import GroqConnectionManager, QAPipeline, GroqConnectionError, InferenceError


def test_groq_connection_manager_missing_key():
    """Verify that GroqConnectionManager raises GroqConnectionError if API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        # Reset the singleton instance for clean test execution
        GroqConnectionManager._instance = None
        with pytest.raises(GroqConnectionError) as exc_info:
            GroqConnectionManager.get_chat_model()
        assert "GROQ_API_KEY is not configured" in str(exc_info.value)


@patch("app.core.qa.GroqConnectionManager")
def test_qa_pipeline_successful_generation(mock_connection_manager):
    """Verify that QAPipeline successfully structures prompts, calls model, and extracts citations."""
    # 1. Setup mock LLM response
    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    mock_llm = MagicMock(spec=ChatGroq)
    mock_response = AIMessage(content="FastAPI is a Python web framework.")
    mock_llm.invoke.return_value = mock_response
    mock_llm.return_value = mock_response
    mock_connection_manager.get_chat_model.return_value = mock_llm

    # 2. Setup mock documents
    retrieved_docs = [
        Document(
            page_content="FastAPI is a modern, fast web framework.",
            metadata={"source_filename": "fastapi_guide.pdf", "page_index": 4, "chunk_id": "c-1", "document_id": "doc-uuid-1"}
        )
    ]

    pipeline = QAPipeline()
    result = pipeline.generate_answer(
        query="What is FastAPI?",
        retrieved_docs=retrieved_docs
    )

    # 3. Assert correct answer and citations structures
    assert result["answer"] == "FastAPI is a Python web framework."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_filename"] == "fastapi_guide.pdf"
    assert result["citations"][0]["page_index"] == 4
    assert result["citations"][0]["document_id"] == "doc-uuid-1"
    assert result["citations"][0]["text"] == "FastAPI is a modern, fast web framework."

    # 4. Verify system prompt assembly contains our context
    called_prompt_value = mock_llm.invoke.call_args[0][0]
    called_messages = called_prompt_value.to_messages()
    system_msg = called_messages[0].content
    assert "fastapi_guide.pdf" in system_msg
    assert "Page 4" in system_msg
    assert "FastAPI is a modern, fast web framework." in system_msg


@patch("app.core.qa.GroqConnectionManager")
def test_qa_pipeline_strict_grounding_fallback(mock_connection_manager):
    """Verify that if LLM triggers the fallback message, the citation list is cleaned and returns empty."""
    # 1. Setup mock LLM response returning fallback message
    fallback_msg = "I am sorry, but the provided documents do not contain the answer to your question."
    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    mock_llm = MagicMock(spec=ChatGroq)
    mock_response = AIMessage(content=fallback_msg)
    mock_llm.invoke.return_value = mock_response
    mock_llm.return_value = mock_response
    mock_connection_manager.get_chat_model.return_value = mock_llm

    retrieved_docs = [
        Document(
            page_content="Some completely unrelated text about cooking apples.",
            metadata={"source_filename": "cooking.pdf", "page_index": 2, "chunk_id": "c-2"}
        )
    ]

    pipeline = QAPipeline()
    result = pipeline.generate_answer(
        query="What is the speed of light?",
        retrieved_docs=retrieved_docs
    )

    # 2. Assert fallback answer and empty citations list
    assert result["answer"] == fallback_msg
    assert result["citations"] == [], "Citations list must be empty if the answer fell back to grounding message!"
