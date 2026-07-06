import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.database import ChatDatabaseManager
from app.core.auth import get_current_user

router = APIRouter()


class SessionCreateRequest(BaseModel):
    """Pydantic model representing the request payload for creating a session."""

    title: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="Optional custom title for the chat session."
    )


class SessionResponse(BaseModel):
    """Pydantic model representing a simple session resource."""

    id: str
    title: str
    created_at: str


class MessageResponse(BaseModel):
    """Pydantic model representing a chat message."""

    id: str
    role: str
    content: str
    metadata: Optional[Any] = None
    created_at: str



class SessionDetailResponse(BaseModel):
    """Pydantic model representing detailed session with chronological messages."""

    id: str
    title: str
    created_at: str
    messages: List[MessageResponse]


class SessionListResponse(BaseModel):
    """Pydantic model representing a paginated list of sessions."""

    total: int
    limit: int
    offset: int
    items: List[SessionResponse]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create Chat Session",
    description="Initializes a new conversational chat session for the authenticated user.",
    response_model=SessionResponse,
    response_description="Returns the created session details."
)
async def create_session(
    body: SessionCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Creates a new conversational chat session."""
    session_id = str(uuid.uuid4())
    title = body.title if body.title else "New Chat"
    user_id = current_user["id"]
    
    try:
        session = ChatDatabaseManager.create_session(
            session_id=session_id,
            user_id=user_id,
            title=title
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat session: {str(e)}"
        )
        
    return {
        "id": session["id"],
        "title": session["title"],
        "created_at": datetime.now(timezone.utc).isoformat()  # Mock or fetch standard time
    }


@router.get(
    "",
    summary="List Chat Sessions",
    description="Retrieves a paginated list of the authenticated user's chat sessions.",
    response_model=SessionListResponse,
    response_description="Returns paginated sessions list."
)
async def list_sessions(
    limit: int = Query(10, ge=1, le=100, description="Number of sessions to return."),
    offset: int = Query(0, ge=0, description="Offset for pagination."),
    current_user: dict = Depends(get_current_user)
):
    """Lists chat sessions with offset/limit pagination."""
    user_id = current_user["id"]
    try:
        total = ChatDatabaseManager.count_sessions(user_id=user_id)
        sessions = ChatDatabaseManager.list_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )
        
    items = [
        {
            "id": s["id"],
            "title": s["title"],
            "created_at": s["created_at"]
        }
        for s in sessions
    ]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items
    }


@router.get(
    "/{session_id}",
    summary="Get Chat Session Details",
    description="Retrieves detailed metadata and chronological message history for a session.",
    response_model=SessionDetailResponse,
    response_description="Returns session metadata and messages."
)
async def get_session_details(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves session details and its messages list."""
    user_id = current_user["id"]
    try:
        session = ChatDatabaseManager.get_session(session_id=session_id, user_id=user_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during session lookup: {str(e)}"
        )
        
    if not session:
        # Check if it exists for *any* user to determine 403 vs 404
        # We can query get_session with user_id=None or custom sql, or just do it via custom check.
        # Let's perform a direct check to see if the session exists at all
        conn = ChatDatabaseManager.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM chat_sessions WHERE id = ?;", (session_id,))
            exists = cursor.fetchone()
        finally:
            conn.close()
            
        if exists:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to this session is forbidden."
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found."
        )
        
    try:
        messages = ChatDatabaseManager.get_messages_by_session(
            session_id=session_id,
            user_id=user_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load session messages: {str(e)}"
        )
        
    formatted_messages = []
    for msg in messages:
        msg_metadata = None
        if msg["metadata"]:
            try:
                msg_metadata = json.loads(msg["metadata"])
            except Exception:
                pass
        formatted_messages.append({
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "metadata": msg_metadata,
            "created_at": msg["created_at"]
        })
        
    return {
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "messages": formatted_messages
    }


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Chat Session",
    description="Permanently deletes a chat session and all its message history.",
)
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Deletes a chat session and all cascading message associations."""
    user_id = current_user["id"]
    try:
        session = ChatDatabaseManager.get_session(session_id=session_id, user_id=user_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during session lookup: {str(e)}"
        )
        
    if not session:
        # Check if it exists for *any* user to determine 403 vs 404
        conn = ChatDatabaseManager.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM chat_sessions WHERE id = ?;", (session_id,))
            exists = cursor.fetchone()
        finally:
            conn.close()
            
        if exists:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to this session is forbidden."
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found."
        )
        
    try:
        ChatDatabaseManager.delete_session(session_id=session_id, user_id=user_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )
