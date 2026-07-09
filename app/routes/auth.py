import secrets
import uuid
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.core.database import UserDatabaseManager
from app.core.auth import hash_password, verify_password, create_access_token
from app.core.limiter import limiter

router = APIRouter(prefix="/auth", tags=["Authentication"])


class UserAuthRequest(BaseModel):
    """Pydantic model representing the JSON payload for signup and login."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Unique username for the account (3-50 characters)."
    )
    password: str = Field(
        ...,
        min_length=6,
        max_length=128,
        description="Password for the account (minimum 6 characters)."
    )


class TokenRefreshRequest(BaseModel):
    """Pydantic model representing refresh token payload."""
    refresh_token: str = Field(..., description="A valid user refresh token.")


@router.post(
    "/signup",
    status_code=status.HTTP_201_CREATED,
    summary="Register User",
    description="Registers a new user account with secure hashed password storage.",
    response_description="Returns the user ID of the newly created account."
)
@limiter.limit("5/minute")
async def signup(request: Request, response: Response, body: UserAuthRequest):
    """Registers a new user account with hashed password storage."""
    hashed = await run_in_threadpool(hash_password, body.password)
    user_id = str(uuid.uuid4())
    try:
        UserDatabaseManager.create_user(
            user_id=user_id,
            username=body.username,
            hashed_password=hashed
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed due to a database error: {str(e)}"
        ) from e

    return {
        "status": "success",
        "message": "User successfully registered.",
        "user_id": user_id
    }


@router.post(
    "/login",
    summary="Login User",
    description="Authenticates user credentials and returns a valid JWT access token.",
    response_description="Returns bearer access token details on successful authentication."
)
@limiter.limit("5/minute")
async def login(request: Request, response: Response, body: UserAuthRequest):
    """Authenticates credentials and returns a valid JWT access token."""
    user = UserDatabaseManager.get_user_by_username(body.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid username or password."
        )
    
    is_valid = await run_in_threadpool(verify_password, body.password, user["hashed_password"])
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid username or password."
        )

    # Issue access token
    access_token = create_access_token(data={"sub": user["username"]})
    
    # Generate long-lived refresh token
    refresh_token = secrets.token_hex(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    UserDatabaseManager.create_refresh_token(
        token=refresh_token,
        user_id=user["id"],
        expires_at=expires_at
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token
    }


@router.post(
    "/refresh",
    summary="Refresh Access Token",
    description="Exchange a valid refresh token for a new short-lived access token."
)
async def refresh(body: TokenRefreshRequest):
    """Refreshes short-lived JWT access token using a valid refresh token."""
    record = UserDatabaseManager.get_refresh_token(body.refresh_token)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token."
        )

    expires_at = datetime.fromisoformat(record["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        UserDatabaseManager.delete_refresh_token(body.refresh_token)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired."
        )

    user = UserDatabaseManager.get_user_by_id(record["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found associated with this refresh token."
        )

    access_token = create_access_token(data={"sub": user["username"]})
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.post(
    "/logout",
    summary="Logout User",
    description="Revokes the provided refresh token."
)
async def logout(body: TokenRefreshRequest):
    """Revokes the refresh token by removing it from the database."""
    UserDatabaseManager.delete_refresh_token(body.refresh_token)
    return {
        "status": "success",
        "message": "Successfully logged out."
    }
