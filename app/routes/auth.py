import uuid
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.database import UserDatabaseManager
from app.core.auth import hash_password, verify_password, create_access_token

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


@router.post(
    "/signup",
    status_code=status.HTTP_201_CREATED,
    summary="Register User",
    description="Registers a new user account with secure hashed password storage.",
    response_description="Returns the user ID of the newly created account."
)
async def signup(body: UserAuthRequest):
    """Registers a new user account with hashed password storage."""
    hashed = hash_password(body.password)
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
async def login(body: UserAuthRequest):
    """Authenticates credentials and returns a valid JWT access token."""
    user = UserDatabaseManager.get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid username or password."
        )

    # Issue access token
    access_token = create_access_token(data={"sub": user["username"]})
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
