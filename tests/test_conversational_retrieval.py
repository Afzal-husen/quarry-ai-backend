# Force IDE type checker re-analysis of imports
import os
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

# Add backend directory to sys.path to allow absolute imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.qa import QAPipeline


def test_condense_query_empty_history():
    """Verify that condense_query returns the original question if history is empty, without triggering LLM."""
    pipeline = QAPipeline()
    question = "What is Python?"
    
    # We do not patch GroqConnectionManager because it shouldn't be called at all
    result = pipeline.condense_query(chat_history=[], question=question)
    assert result == question


@patch("app.core.qa.GroqConnectionManager")
def test_condense_query_with_history(mock_connection_manager):
    """Verify that condense_query uses ChatGroq to rewrite the question based on chat history."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    
    # Mock LLM and response
    mock_llm = MagicMock(spec=ChatGroq)
    mock_response = AIMessage(content="What are the main features of Python?")
    mock_llm.invoke.return_value = mock_response
    mock_llm.return_value = mock_response
    mock_connection_manager.get_chat_model.return_value = mock_llm
    
    pipeline = QAPipeline()
    chat_history = [
        {"role": "user", "content": "Tell me about Python."},
        {"role": "assistant", "content": "Python is a high-level programming language."}
    ]
    
    result = pipeline.condense_query(chat_history=chat_history, question="What are its main features?")
    
    assert result == "What are the main features of Python?"
    
    # Assert prompt assembly mapping
    called_prompt_value = mock_llm.invoke.call_args[0][0]
    called_messages = called_prompt_value.to_messages()
    
    # messages: [system, human_history, ai_history, human_question]
    assert len(called_messages) == 4
    assert called_messages[0].content.startswith("Given the following chat history")
    assert called_messages[1].content == "Tell me about Python."
    assert called_messages[2].content == "Python is a high-level programming language."
    assert called_messages[3].content == "What are its main features?"


@patch("app.core.qa.GroqConnectionManager")
def test_condense_query_fallback_on_exception(mock_connection_manager):
    """Verify that condense_query falls back to raw question if model call raises an exception."""
    mock_connection_manager.get_chat_model.side_effect = Exception("Groq is unavailable")
    
    pipeline = QAPipeline()
    chat_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    question = "Who wrote Hamlet?"
    
    result = pipeline.condense_query(chat_history=chat_history, question=question)
    # Must fallback to raw query
    assert result == question
