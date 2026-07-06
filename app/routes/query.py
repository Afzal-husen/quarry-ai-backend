import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import uuid
import time

import os
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.core.limiter import limiter
from pydantic import BaseModel, Field, field_validator, model_validator

# FlashRank Pydantic model requires Ranker to be imported first
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document


from app.core.qa import QAPipeline, GroqConnectionError, InferenceError
from app.core.vectorstore import VectorStoreManager, VectorStoreError
from app.core.auth import get_current_user
from app.core.reranker import RerankManager, RerankerError
from app.core.database import ChatDatabaseManager


router = APIRouter()

# Initialize core orchestrators
vector_manager = VectorStoreManager()
qa_pipeline = QAPipeline()


def retrieve_and_rerank_context(
    user_id: str,
    target_ids: List[str],
    rewritten_question: str,
    top_k: int
) -> Tuple[List[Document], float, float]:
    """Retrieves document context using multi-query expansion, Reciprocal Rank Fusion,
    and FlashRank reranking.

    Returns:
        A tuple of (matching_chunks, retrieval_ms, reranking_ms).
    """
    # 1. Run query expansion
    expanded_queries = qa_pipeline.generate_alternative_queries(rewritten_question)
    all_queries = [rewritten_question] + expanded_queries

    # 2. Per-document hybrid retrieval across all query variations
    start_retrieval = time.perf_counter()
    candidate_k = max(10, min(25, top_k * 3))
    
    rrf_scores = {}
    try:
        for query in all_queries:
            for doc_id in target_ids:
                base_retriever = vector_manager.get_hybrid_retriever(
                    user_id=user_id,
                    document_id=doc_id,
                    top_k=candidate_k
                )
                chunks = base_retriever.invoke(query)
                for idx, doc in enumerate(chunks):
                    rank = idx + 1
                    score = 1.0 / (60.0 + rank)
                    
                    # Deduplicate by text content and source block tracking keys
                    chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("parent_id", ""))
                    key = (doc.page_content.strip(), chunk_id)
                    if key not in rrf_scores:
                        rrf_scores[key] = [doc, score]
                    else:
                        rrf_scores[key][1] += score
    except (VectorStoreError, RerankerError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Local vector store retrieval failed: {str(e)}"
        )
    retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

    # 3. Sort by aggregated RRF score and slice to candidate_k
    start_rerank = time.perf_counter()
    sorted_rrf = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
    deduped_chunks = [item[0] for item in sorted_rrf]

    # 4. Rerank top RRF candidates using FlashRank down to top_k
    if not deduped_chunks:
        return [], retrieval_ms, 0.0

    try:
        ranker = RerankManager.get_ranker()
        compressor = FlashrankRerank(client=ranker, top_n=top_k)
        matching_chunks = compressor.compress_documents(
            deduped_chunks[:candidate_k], rewritten_question)
    except RerankerError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reranking failed: {str(e)}"
        )
    reranking_ms = (time.perf_counter() - start_rerank) * 1000

    # Resolve parent documents from child chunks if Parent-Document Retriever is active
    matching_chunks = vector_manager.resolve_parent_documents(
        user_id=user_id,
        documents=matching_chunks
    )

    return matching_chunks, retrieval_ms, reranking_ms


class QueryRequest(BaseModel):
    """Pydantic model representing the JSON request schema for document Q&A.

    Supports querying a single document via `document_id` or multiple documents
    via `document_ids`. At least one of the two fields must be provided.
    """

    document_id: Optional[str] = Field(
        None,
        description="The unique UUID of a single uploaded and processed document."
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="A list of document UUIDs to query across in a single request."
    )
    question: str = Field(
        ...,
        description="The natural language question to ask related to the document context."
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of most semantically relevant text chunks to retrieve (1-10)."
    )
    session_id: Optional[str] = Field(
        None,
        description="The unique UUID of a chat session, if querying inside a session context."
    )

    # Internal resolved field — populated by model_validator
    resolved_document_ids: List[str] = Field(
        default_factory=list, exclude=True)

    @field_validator("session_id")
    @classmethod
    def validate_session_uuid(cls, value):
        """Enforces that session_id is a valid UUID if provided."""
        if value is None:
            return value
        try:
            uuid.UUID(value)
            return value
        except ValueError as e:
            raise ValueError("session_id must be a valid UUID string.") from e

    @field_validator("document_id")
    @classmethod
    def validate_document_uuid(cls, value):
        """Enforces that document_id is a valid UUID to protect against path traversal."""
        if value is None:
            return value
        try:
            uuid.UUID(value)
            return value
        except ValueError as e:
            raise ValueError("document_id must be a valid UUID string.") from e

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids_uuids(cls, value):
        """Enforces that each string in document_ids is a valid UUID."""
        if value is None:
            return value
        for doc_id in value:
            try:
                uuid.UUID(doc_id)
            except ValueError as e:
                raise ValueError(
                    f"Each entry in document_ids must be a valid UUID string. Invalid value: '{doc_id}'"
                ) from e
        return value

    @model_validator(mode="after")
    def resolve_document_ids(self):
        """Ensures at least one document identifier is provided and resolves the working list."""
        if self.document_id is None and not self.document_ids:
            raise ValueError(
                "At least one of 'document_id' or 'document_ids' must be provided."
            )
        # document_ids takes precedence; otherwise fall back to [document_id]
        if self.document_ids:
            self.resolved_document_ids = self.document_ids
        elif self.document_id:
            self.resolved_document_ids = [self.document_id]
        return self


@router.post(
    "/query",
    summary="Query Documents",
    description="Answers questions related to one or more uploaded documents using local vectors and ChatGroq inference.",
    response_description="Returns the generated answer and a list of source citations."
)
@limiter.limit(os.getenv("RATE_LIMIT_QUERY", "30/minute"))
async def query_document(
    request: Request,
    body: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Answers questions related to one or more uploaded documents using local vectors and ChatGroq inference.

    Accepts either a single `document_id` (backward-compatible) or a list of
    `document_ids`. Retrieval runs per-document, results are pooled, deduplicated,
    and reranked before being forwarded to the LLM.

    Args:
        body: The QueryRequest Pydantic JSON model.

    Returns:
        A JSON dictionary containing the generated "answer" and a list of source "citations".
        Each citation includes `source_filename`, `page_index`, `document_id`, and `text`.
    """
    user_id = current_user["id"]
    target_ids = body.resolved_document_ids

    # Validate session_id if provided
    session_id = body.session_id
    history_messages = []
    if session_id is not None:
        session = ChatDatabaseManager.get_session_by_id(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session with ID '{session_id}' not found."
            )
        if session["user_id"] != user_id:
            raise HTTPException(
                status_code=403,
                detail="Forbidden: You do not own or have permission to access this chat session."
            )
        history_messages = ChatDatabaseManager.get_messages_by_session(
            session_id, user_id)

    # Perform query condensation using last 10 messages (5 turns)
    recent_history = history_messages[-10:]
    formatted_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in recent_history
    ]
    if session_id is not None:
        rewritten_question = qa_pipeline.condense_query(
            formatted_history, body.question)
    else:
        rewritten_question = body.question

    # 1. Enforce strict ownership boundaries for every requested document ID
    for doc_id in target_ids:
        # Find if the document exists for ANY user
        global_matches = list(
            vector_manager.vectorstore_dir.glob(f"*/{doc_id}"))
        if not global_matches:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Vector database index for document '{doc_id}' does not exist on disk. "
                    "Please upload and index the document first."
                )
            )
        # Check if the document belongs specifically to the current authenticated user
        db_path = vector_manager.vectorstore_dir / user_id / doc_id
        if not db_path.exists():
            raise HTTPException(
                status_code=403,
                detail="Forbidden: You do not own or have permission to access this document."
            )

    # 2. Retrieve document context using RRF multi-query expansion and FlashRank
    matching_chunks, retrieval_ms, reranking_ms = retrieve_and_rerank_context(
        user_id=user_id,
        target_ids=target_ids,
        rewritten_question=rewritten_question,
        top_k=body.top_k
    )

    # 5. Generate strict grounded response via ChatGroq
    start_gen = time.perf_counter()
    try:
        payload = qa_pipeline.generate_answer(
            query=rewritten_question,
            retrieved_docs=matching_chunks
        )
    except GroqConnectionError as e:
        # Expose meaningful API unconfigured/connection issues as developer status 500
        raise HTTPException(
            status_code=500,
            detail=f"Groq API connection/credentials error: {str(e)}"
        )
    except InferenceError as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generative inference failure: {str(e)}"
        )
    generation_ms = (time.perf_counter() - start_gen) * 1000

    # 6. Save to session history if session is active
    if session_id is not None:
        # Generate and update session title on first turn
        if len(history_messages) == 0:
            new_title = qa_pipeline.generate_session_title(body.question)
            ChatDatabaseManager.update_session_title(
                session_id, user_id, new_title)

        # Save user message (with raw question)
        user_msg_id = str(uuid.uuid4())
        ChatDatabaseManager.create_message(
            message_id=user_msg_id,
            session_id=session_id,
            role="user",
            content=body.question,
            metadata=None
        )

        # Save assistant message
        citations_serialized = json.dumps(payload.get("citations", []))
        assistant_msg_id = str(uuid.uuid4())
        ChatDatabaseManager.create_message(
            message_id=assistant_msg_id,
            session_id=session_id,
            role="assistant",
            content=payload.get("answer", ""),
            metadata=citations_serialized
        )

    total_ms = retrieval_ms + reranking_ms + generation_ms
    request.state.latency_breakdown = {
        "retrieval_ms": round(retrieval_ms, 2),
        "reranking_ms": round(reranking_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "total_ms": round(total_ms, 2)
    }

    # Return success payload containing answer and source page-level citations
    return JSONResponse(content=payload)


@router.post(
    "/query/stream",
    summary="Stream Query Response",
    description="Streams LLM answer tokens via Server-Sent Events (SSE) for one or more uploaded documents.",
    response_description="A Server-Sent Events streaming response."
)
@limiter.limit(os.getenv("RATE_LIMIT_QUERY", "30/minute"))
async def query_document_stream(
    request: Request,
    body: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Streams LLM answer tokens via Server-Sent Events for one or more uploaded documents.

    Runs the same auth + ownership + hybrid retrieval + reranking pipeline as POST /query,
    then streams the ChatGroq response token-by-token using the SSE protocol.

    Event sequence:
      1. data: {"citations": [...]}  — source citations emitted before streaming begins
      2. data: {"token": "..."}      — one event per LLM output token
      3. data: [DONE]                — terminal event signals end of stream

    Args:
        body: The QueryRequest Pydantic JSON model (identical schema to /query).

    Returns:
        A StreamingResponse with Content-Type: text/event-stream.
    """
    user_id = current_user["id"]
    target_ids = body.resolved_document_ids

    # Validate session_id if provided
    session_id = body.session_id
    history_messages = []
    if session_id is not None:
        session = ChatDatabaseManager.get_session_by_id(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session with ID '{session_id}' not found."
            )
        if session["user_id"] != user_id:
            raise HTTPException(
                status_code=403,
                detail="Forbidden: You do not own or have permission to access this chat session."
            )
        history_messages = ChatDatabaseManager.get_messages_by_session(
            session_id, user_id)

    # Perform query condensation using last 10 messages (5 turns)
    recent_history = history_messages[-10:]
    formatted_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in recent_history
    ]
    if session_id is not None:
        rewritten_question = qa_pipeline.condense_query(
            formatted_history, body.question)
    else:
        rewritten_question = body.question

    # Generate and update session title on first turn
    if session_id is not None and len(history_messages) == 0:
        new_title = qa_pipeline.generate_session_title(body.question)
        ChatDatabaseManager.update_session_title(
            session_id, user_id, new_title)

    # 1. Enforce strict ownership boundaries for every requested document ID
    for doc_id in target_ids:
        global_matches = list(
            vector_manager.vectorstore_dir.glob(f"*/{doc_id}"))
        if not global_matches:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Vector database index for document '{doc_id}' does not exist on disk. "
                    "Please upload and index the document first."
                )
            )
        db_path = vector_manager.vectorstore_dir / user_id / doc_id
        if not db_path.exists():
            raise HTTPException(
                status_code=403,
                detail="Forbidden: You do not own or have permission to access this document."
            )

    # 2. Retrieve document context using RRF multi-query expansion and FlashRank
    matching_chunks, retrieval_ms, reranking_ms = retrieve_and_rerank_context(
        user_id=user_id,
        target_ids=target_ids,
        rewritten_question=rewritten_question,
        top_k=body.top_k
    )

    # 5. Build citation metadata — emitted as the first SSE event before streaming begins
    citations = [
        {
            "source_filename": doc.metadata.get("source_filename", "Unknown Document"),
            "page_index": doc.metadata.get("page_index", 0),
            "document_id": doc.metadata.get("document_id", ""),
            "text": doc.page_content,
        }
        for doc in matching_chunks
    ]

    # Initialize latency breakdown dictionary
    request.state.latency_breakdown = {
        "retrieval_ms": round(retrieval_ms, 2),
        "reranking_ms": round(reranking_ms, 2),
        "generation_ms": 0.0,
        "total_ms": 0.0
    }

    # 6. Define the async SSE generator
    async def sse_generator():
        # First event: emit citation metadata so clients can render source refs immediately
        yield f"data: {json.dumps({'citations': citations})}\n\n"

        # Stream LLM tokens
        start_gen = time.perf_counter()
        success = False
        tokens_accumulated = []
        try:
            async for token in qa_pipeline.generate_answer_stream(
                query=rewritten_question,
                retrieved_docs=matching_chunks
            ):
                tokens_accumulated.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
            success = True
        except (GroqConnectionError, InferenceError) as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        finally:
            generation_ms = (time.perf_counter() - start_gen) * 1000
            total_ms = retrieval_ms + reranking_ms + generation_ms
            request.state.latency_breakdown.update({
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2)
            })

            if session_id is not None and success:
                try:
                    # Save user message
                    user_msg_id = str(uuid.uuid4())
                    ChatDatabaseManager.create_message(
                        message_id=user_msg_id,
                        session_id=session_id,
                        role="user",
                        content=body.question,
                        metadata=None
                    )

                    # Save assistant message
                    assistant_answer = "".join(tokens_accumulated)
                    is_fallback = "Disclaimer: This information was not found" in assistant_answer
                    is_greeting = not any(f"[{i+1}]" in assistant_answer for i in range(len(matching_chunks)))
                    
                    final_citations = []
                    if not is_fallback and not is_greeting:
                        final_citations = citations
                        
                    citations_serialized = json.dumps(final_citations)
                    assistant_msg_id = str(uuid.uuid4())
                    ChatDatabaseManager.create_message(
                        message_id=assistant_msg_id,
                        session_id=session_id,
                        role="assistant",
                        content=assistant_answer,
                        metadata=citations_serialized
                    )
                except Exception as db_err:
                    logging.getLogger("app.exception").error(
                        f"Failed to persist chat messages inside stream generator: {str(db_err)}"
                    )

        # Terminal event
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
