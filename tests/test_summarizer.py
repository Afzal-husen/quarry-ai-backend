import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path to allow absolute imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.summarizer import DocumentSummarizer, SummarizationError

def test_summarizer_empty_text():
    """Verify that the summarizer returns a default message when text is empty."""
    with patch("app.core.qa.GroqConnectionManager.get_chat_model") as mock_get_model:
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        summarizer = DocumentSummarizer()
        result = summarizer.summarize_text("")
        assert result == "No text available to summarize."
        
        result_whitespace = summarizer.summarize_text("   \n  ")
        assert result_whitespace == "No text available to summarize."

def test_summarizer_successful_inference():
    """Verify that the summarizer correctly invokes the LangChain model chain and returns the summary."""
    from langchain_core.messages import AIMessage
    with patch("app.core.qa.GroqConnectionManager.get_chat_model") as mock_get_model:
        mock_model = MagicMock()
        # Mock the chain invocation
        # Since Chain = prompt | model | parser, model.invoke() or model() will be called by chain.invoke()
        mock_response = AIMessage(content="TL;DR: This is a test summary.\n\n### Key Takeaways\n- Takeaway 1")
        mock_model.return_value = mock_response
        mock_model.invoke.return_value = mock_response
        mock_get_model.return_value = mock_model

        summarizer = DocumentSummarizer()
        result = summarizer.summarize_text("This is the long document text to summarize.")
        
        assert "TL;DR" in result
        assert "Key Takeaways" in result
        assert mock_model.invoke.called or mock_model.called

def test_summarizer_inference_failure():
    """Verify that the summarizer raises a SummarizationError when the model call fails."""
    with patch("app.core.qa.GroqConnectionManager.get_chat_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.side_effect = Exception("Groq API Timeout")
        mock_model.invoke.side_effect = Exception("Groq API Timeout")
        mock_get_model.return_value = mock_model

        summarizer = DocumentSummarizer()
        with pytest.raises(SummarizationError) as exc_info:
            summarizer.summarize_text("Some text.")
        
        assert "Summarization inference failed" in str(exc_info.value)
