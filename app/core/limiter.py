import jwt
from fastapi import Request
from slowapi import Limiter

from app.core.auth import SECRET_KEY, ALGORITHM

def custom_rate_limit_key(request: Request) -> str:
    """Decodes the user JWT access token to get their username for rate limiting, with IP fallback."""
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        token = auth.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                return username
        except Exception:
            pass
    return request.client.host if request.client else "127.0.0.1"

# Shared Limiter instance
limiter = Limiter(key_func=custom_rate_limit_key, headers_enabled=True)
