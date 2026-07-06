import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.database import UserDatabaseManager

# Resolve JWT configurations from environment
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "document-rag-default-jwt-secret-key-for-development")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# OAuth2 HTTP Bearer token dependency
reusable_oauth2 = HTTPBearer()


def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt.

    Args:
        password: The plaintext password.

    Returns:
        The hashed password as a string.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a hashed password.

    Args:
        password: The plaintext password to test.
        hashed_password: The hashed password stored in the database.

    Returns:
        True if the password matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT access token.

    Args:
        data: Payload data to encode.
        expires_delta: Optional duration override for token expiry.

    Returns:
        The encoded JWT token string.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_access_token(token: str) -> dict:
    """Decodes and validates a JWT token.

    Args:
        token: The raw JWT token string.

    Returns:
        The decoded claims dictionary.

    Raises:
        HTTPException: If the token is expired, signature is invalid, or claims are malformed.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(reusable_oauth2)
) -> dict:
    """FastAPI route dependency to retrieve and authenticate the current user.

    Args:
        request: The FastAPI request object.
        credentials: The parsed Bearer token credentials.

    Returns:
        The authenticated user dictionary from the database.

    Raises:
        HTTPException: If authentication fails.
    """
    token = credentials.credentials
    payload = verify_access_token(token)
    username: Optional[str] = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing subject identifier.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = UserDatabaseManager.get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists in database.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Propagate user_id to request state for structured logging middleware
    request.state.user_id = user["id"]
    return user
