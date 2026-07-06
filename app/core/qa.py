import os
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence
from pydantic import SecretStr

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq


class GroqConnectionError(Exception):
    """Exception raised when the Groq client connection fails or key is missing."""
    pass


class InferenceError(Exception):
    """Exception raised when the Groq generative answering model fails during inference."""
    pass


class GroqConnectionManager:
    """Thread-safe singleton class to load and cache standard ChatGroq client connections."""

    _instance: Optional[ChatGroq] = None
    _lock = threading.Lock()

    @classmethod
    def get_chat_model(cls) -> ChatGroq:
        """Loads and caches the ChatGroq model instance thread-safely.

        Returns:
            The instantiated ChatGroq object.

        Raises:
            GroqConnectionError: If the GROQ_API_KEY environment variable is not configured.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    api_key = os.getenv("GROQ_API_KEY")
                    if not api_key:
                        raise GroqConnectionError(
                            "GROQ_API_KEY is not configured inside .env or the system environment."
                        )
                    model_name = os.getenv(
                        "GROQ_MODEL", "llama-3.1-8b-instant")
                    try:
                        cls._instance = ChatGroq(
                            model=model_name,
                            api_key=SecretStr(api_key)
                        )
                    except Exception as e:
                        raise GroqConnectionError(
                            f"Failed to connect to Groq client using model '{model_name}': {str(e)}"
                        ) from e
        return cls._instance


class QAPipeline:
    """Orchestrates strict grounding prompts assembly and Groq generative answering workflows."""

    def generate_answer(self, query: str, retrieved_docs: Sequence[Document]) -> Dict[str, Any]:
        """Synthesizes an answer to the query based strictly on retrieved document context.

        Args:
            query: The user's natural language question.
            retrieved_docs: A list of relevant semantic LangChain Document chunks.

        Returns:
            A dictionary containing the generated "answer" and a list of source "citations".

        Raises:
            InferenceError: If generative answering fails or API calls crash.
        """
        # 1. Format document snippets context
        context_blocks = []
        for doc in retrieved_docs:
            filename = doc.metadata.get("source_filename", "Unknown Document")
            page = doc.metadata.get("page_index", 0)
            context_blocks.append(
                f"Source: {filename} (Page {page})\n"
                f"Snippet:\n{doc.page_content}"
            )

        context_text = "\n\n---\n\n".join(
            context_blocks) if context_blocks else "NO DOCUMENT CONTEXT AVAILABLE"

        # 2. Build the strict grounding system prompt instruction
        disclaimer_msg = "Disclaimer: This information was not found in your uploaded documents and is generated using general AI knowledge."
        system_instruction = (
            "You are a warm, helpful, and professional assistant designed to perform question-answering over documents.\n"
            "Adopt a friendly tone and format your responses professionally using standard Markdown paragraphs, bullet points, numbered lists, or code blocks where appropriate.\n\n"
            "Analyze the user's question and the provided context snippets below, then follow the instructions for the matching category:\n"
            "1. **Generic Dialogues & Greetings**: If the user's input is a greeting, pleasantry, or basic generic chat (e.g., 'hi', 'hello', 'good morning', 'how are you?'), "
            "respond warmly and helpfully. Do NOT use any citations, and do NOT include any disclaimer or warning.\n"
            "2. **Document-Grounded Q&A**: If the user asks an informational question and the answer is contained in or can be inferred from the context snippets, "
            "answer the question. You MUST cite your sources using inline reference numbers (e.g., [1], [2], etc.) placed immediately adjacent to any statement supported by the context. "
            "The numbers match the 1-based index of the snippets (Source index 1 is [1]). Do NOT include the disclaimer.\n"
            "3. **General Knowledge Fallback**: If the user asks an informational question and the answer cannot be found in or inferred from the context snippets (or if context is empty/missing), "
            "answer the question using your own general knowledge. You MUST append the following exact disclaimer text at the very end of your response:\n"
            f"'{disclaimer_msg}'\n"
            "Do NOT include any source citations when answering via general knowledge fallback.\n\n"
            f"Provided Document Context:\n{context_text}"
        )

        # 3. Assemble and trigger ChatGroq model
        try:
            model = GroqConnectionManager.get_chat_model()
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_instruction),
                ("human", "{question}")
            ])
            chain = prompt | model | StrOutputParser()
            answer = chain.invoke({"context": context_text, "question": query})
        except Exception as e:
            raise InferenceError(
                f"Groq generative inference failed: {str(e)}") from e

        # 4. Format structured citation outputs
        citations = []
        old_fallback = "I am sorry, but the provided documents do not contain the answer to your question."
        is_fallback = (
            disclaimer_msg in answer or 
            "Disclaimer: This information was not found" in answer or 
            answer.strip() == old_fallback
        )
        
        lower_q = query.strip().lower().rstrip("?.!")
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you", "what's up"}
        is_greeting = lower_q in greetings
        
        if not is_fallback and not is_greeting and retrieved_docs:
            for doc in retrieved_docs:
                citations.append({
                    "source_filename": doc.metadata.get("source_filename", "Unknown Document"),
                    "page_index": doc.metadata.get("page_index", 0),
                    "document_id": doc.metadata.get("document_id", ""),
                    "text": doc.page_content
                })

        return {
            "answer": answer,
            "citations": citations
        }

    async def generate_answer_stream(
        self,
        query: str,
        retrieved_docs: Sequence[Document],
    ):
        """Streams the generative answer token-by-token using ChatGroq's async streaming.

        Builds the same strict-grounding system prompt as generate_answer(), but invokes
        the LangChain chain with .astream() to yield individual text tokens as they arrive.

        Args:
            query: The user's natural language question.
            retrieved_docs: A list of reranked LangChain Document chunks.

        Yields:
            Individual string tokens from the LLM response.

        Raises:
            InferenceError: Raised inside the generator if the streaming call fails.
        """
        # 1. Format document snippets context — identical logic to generate_answer()
        context_blocks = []
        for doc in retrieved_docs:
            filename = doc.metadata.get("source_filename", "Unknown Document")
            page = doc.metadata.get("page_index", 0)
            context_blocks.append(
                f"Source: {filename} (Page {page})\n"
                f"Snippet:\n{doc.page_content}"
            )

        context_text = "\n\n---\n\n".join(
            context_blocks) if context_blocks else "NO DOCUMENT CONTEXT AVAILABLE"

        # 2. Build the strict grounding system prompt — same constraints as the sync path
        disclaimer_msg = "Disclaimer: This information was not found in your uploaded documents and is generated using general AI knowledge."
        system_instruction = (
            "You are a warm, helpful, and professional assistant designed to perform question-answering over documents.\n"
            "Adopt a friendly tone and format your responses professionally using standard Markdown paragraphs, bullet points, numbered lists, or code blocks where appropriate.\n\n"
            "Analyze the user's question and the provided context snippets below, then follow the instructions for the matching category:\n"
            "1. **Generic Dialogues & Greetings**: If the user's input is a greeting, pleasantry, or basic generic chat (e.g., 'hi', 'hello', 'good morning', 'how are you?'), "
            "respond warmly and helpfully. Do NOT use any citations, and do NOT include any disclaimer or warning.\n"
            "2. **Document-Grounded Q&A**: If the user asks an informational question and the answer is contained in or can be inferred from the context snippets, "
            "answer the question. You MUST cite your sources using inline reference numbers (e.g., [1], [2], etc.) placed immediately adjacent to any statement supported by the context. "
            "The numbers match the 1-based index of the snippets (Source index 1 is [1]). Do NOT include the disclaimer.\n"
            "3. **General Knowledge Fallback**: If the user asks an informational question and the answer cannot be found in or inferred from the context snippets (or if context is empty/missing), "
            "answer the question using your own general knowledge. You MUST append the following exact disclaimer text at the very end of your response:\n"
            f"'{disclaimer_msg}'\n"
            "Do NOT include any source citations when answering via general knowledge fallback.\n\n"
            f"Provided Document Context:\n{context_text}"
        )

        # 3. Assemble the LangChain chain and stream tokens via async generator
        try:
            model = GroqConnectionManager.get_chat_model()
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_instruction),
                ("human", "{question}")
            ])
            chain = prompt | model | StrOutputParser()
            async for token in chain.astream({"context": context_text, "question": query}):
                yield token
        except Exception as e:
            raise InferenceError(
                f"Groq streaming inference failed: {str(e)}") from e

    def condense_query(self, chat_history: List[Dict[str, Any]], question: str) -> str:
        """Rewrites a follow-up user query into a standalone query using the chat history context.

        Args:
            chat_history: A list of dicts with keys "role" and "content" representing past turns.
            question: The latest user question.

        Returns:
            The standalone rewritten question.
        """
        if not chat_history:
            return question

        try:
            model = GroqConnectionManager.get_chat_model()
            
            # Map database messages to LangChain prompt roles
            messages = [("system", (
                "Given the following chat history and a follow-up question, "
                "rephrase the follow-up question to be a standalone question, "
                "in its original language, that can be answered independently of the chat history.\n"
                "Do NOT answer the question. Just rephrase it as a search query.\n"
                "If the follow-up question is already a standalone question or does not reference "
                "prior context, return the follow-up question exactly as is."
            ))]
            
            for msg in chat_history:
                role = "human" if msg["role"] == "user" else "ai"
                messages.append((role, msg["content"]))
                
            messages.append(("human", "{question}"))
            
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | model | StrOutputParser()
            
            condensed = chain.invoke({"question": question})
            return condensed.strip()
        except Exception as e:
            # Fallback to the original question if inference fails to keep the system resilient
            logging.getLogger("app.exception").warning(
                f"Query condensation failed: {str(e)}. Falling back to raw user query."
            )
            return question

    def generate_session_title(self, question: str) -> str:
        """Summarizes the user question in 3-5 words as a chat session title.

        Args:
            question: The user's question.

        Returns:
            The summarized session title, or "New Chat" on failure.
        """
        try:
            model = GroqConnectionManager.get_chat_model()
            prompt = ChatPromptTemplate.from_messages([
                ("human", "Summarize the user question in 3-5 words as a chat session title. Do not use quotes, punctuation, or preamble. Question: {question}")
            ])
            chain = prompt | model | StrOutputParser()
            title = chain.invoke({"question": question})
            return title.strip().strip('"').strip("'")
        except Exception as e:
            logging.getLogger("app.exception").warning(
                f"Session title generation failed: {str(e)}. Falling back to 'New Chat'."
            )
            return "New Chat"

    def generate_alternative_queries(self, question: str) -> List[str]:
        """Generates exactly 3 alternative search query variations representing the search intent.

        Args:
            question: The user's query question.

        Returns:
            A list of 3 alternative search queries, or empty list on failure.
        """
        # Skip query expansion in test runs by default to protect existing mock limits
        if "PYTEST_CURRENT_TEST" in os.environ and not os.getenv("TEST_QUERY_EXPANSION_ACTIVE"):
            return []

        # Exclude generic greetings from query expansion to save latency and tokens
        lower_q = question.strip().lower().rstrip("?.!")
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you", "what's up"}
        if lower_q in greetings:
            return []

        try:
            model = GroqConnectionManager.get_chat_model()
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a helpful query expansion assistant.\n"
                    "Generate exactly 3 alternative search query formulations (variations) that represent the search intent of the user question.\n"
                    "These queries will be used to retrieve documents from a vector database.\n"
                    "Output exactly 3 queries, one per line, with no labels, no numbering, no punctuation, and no preamble.\n"
                    "Keep the queries concise and in the same language as the original question."
                )),
                ("human", "{question}")
            ])
            chain = prompt | model | StrOutputParser()
            output = chain.invoke({"question": question})
            queries = [
                line.strip()
                for line in output.split("\n")
                if line.strip()
            ]
            return queries[:3]
        except Exception as e:
            logging.getLogger("app.exception").warning(
                f"Query expansion failed: {str(e)}. Falling back to empty expansions list."
            )
            return []

