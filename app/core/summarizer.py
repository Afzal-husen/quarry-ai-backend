import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.qa import GroqConnectionManager

class SummarizationError(Exception):
    """Exception raised when document summarization fails."""
    pass

class DocumentSummarizer:
    """Handles generating concise markdown summaries of document contents using ChatGroq."""

    def __init__(self):
        """Initializes the summarizer and gets the cached ChatGroq instance."""
        try:
            self.chat_model = GroqConnectionManager.get_chat_model()
        except Exception as e:
            logging.error(f"Failed to retrieve ChatGroq model: {str(e)}")
            raise SummarizationError(f"Summarizer initialization failed: {str(e)}")

    def summarize_text(self, text: str) -> str:
        """Generates a markdown summary of the provided text.

        Args:
            text: The text content of the document to summarize.

        Returns:
            A markdown-formatted summary string containing a TL;DR paragraph
            and a list of 3-5 key takeaways.
        """
        if not text.strip():
            return "No text available to summarize."

        system_prompt = (
            "You are an expert document analyzer. Your task is to write a high-quality, "
            "concise summary of the provided document content. The summary must follow this structure:\n"
            "1. A short, informative TL;DR paragraph (2-3 sentences) capturing the core focus/purpose of the document.\n"
            "2. A section titled '### Key Takeaways' with 3-5 bullet points highlighting the most important details, metrics, or findings.\n\n"
            "Respond ONLY with the markdown summary itself. Do not include introductory or concluding conversational text."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Document Content:\n\n{text}")
        ])

        chain = prompt | self.chat_model | StrOutputParser()

        try:
            summary = chain.invoke({"text": text})
            return summary.strip()
        except Exception as e:
            logging.error(f"Error during summarization inference: {str(e)}")
            raise SummarizationError(f"Summarization inference failed: {str(e)}")
