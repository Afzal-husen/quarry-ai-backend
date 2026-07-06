import os
from contextlib import asynccontextmanager

from fastapi.responses import JSONResponse
from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from app.core.limiter import limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.database import UserDatabaseManager
from app.routes.upload import router as upload_router
from app.routes.query import router as query_router
from app.routes.auth import router as auth_router
from app.routes.documents import router as documents_router
from app.routes.sessions import router as sessions_router
from app.core.vectorstore import ChromaConnectionCache

import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
from app.core.logging_config import setup_structured_logging
from app.core.paths import get_data_dir

# Load environment configurations relative to the module root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize structured logging
setup_structured_logging()

# Initialize user database
UserDatabaseManager.initialize_db()

# Ensure local storage directories exist on server startup
# DATA_DIR env var points to Render's persistent disk mount (/data) in production
_DATA_DIR = get_data_dir()
(_DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
(_DATA_DIR / "chunks").mkdir(parents=True, exist_ok=True)
(_DATA_DIR / "vectorstore").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown hooks."""
    # Startup: nothing additional needed beyond module-level initialization
    yield
    # Shutdown: cleanly close all open cached Chroma client connections
    ChromaConnectionCache.clear()


app = FastAPI(
    title="Document RAG REST API",
    description="REST API enabling Retrieval-Augmented Generation (RAG) over uploaded PDF and DOCX files.",
    version="0.1.0",
    lifespan=lifespan,
)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware to intercept requests and log details as JSON."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        # Initialize default request state variables to prevent AttributeErrors
        request.state.user_id = None
        request.state.latency_breakdown = None

        response = await call_next(request)

        def log_request(duration_ms: float):
            user_id = getattr(request.state, "user_id", None)
            latency = getattr(request.state, "latency_breakdown", None)

            log_payload = {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "user_id": user_id,
                "client_ip": request.client.host if request.client else "127.0.0.1",
            }
            if latency:
                log_payload["latency_breakdown"] = latency

            logging.getLogger("app.request").info(
                "Request completed", extra=log_payload)

        if isinstance(response, StreamingResponse):
            original_iterator = response.body_iterator

            async def wrapped_iterator():
                try:
                    async for chunk in original_iterator:
                        yield chunk
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    log_request(duration_ms)

            response.body_iterator = wrapped_iterator()
            return response
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_request(duration_ms)
            return response


# Set up rate limiter state and middleware
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(StructuredLoggingMiddleware)
# Parse comma-separated CORS origins from env; fall back to localhost for development
_raw_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [origin.strip() for origin in _raw_cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handlers for standardized error format


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if not errors:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error",
                "code": "VALIDATION_ERROR",
                "field": None
            }
        )
    # Target D-07: first field failure's path as field, and combine validation error messages into detail
    first_err = errors[0]
    loc = first_err.get("loc", [])
    field = loc[-1] if len(loc) > 0 else None

    details = []
    for err in errors:
        msg = err.get("msg", "")
        field_loc = " -> ".join(str(l) for l in err.get("loc", []))
        details.append(f"{field_loc}: {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": "; ".join(details),
            "code": "VALIDATION_ERROR",
            "field": str(field) if field else None
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code_map = {
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_SERVER_ERROR"
    }
    code = code_map.get(exc.status_code, "BAD_REQUEST")

    headers = getattr(exc, "headers", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "code": code,
            "field": None
        },
        headers=headers
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    # Call default handler to compute Retry-After header
    response = _rate_limit_exceeded_handler(request, exc)
    headers = dict(response.headers)
    return JSONResponse(
        status_code=429,
        content={
            "detail": str(exc.detail) if exc.detail else "Rate limit exceeded.",
            "code": "RATE_LIMIT_EXCEEDED",
            "field": None
        },
        headers=headers
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log unhandled exception with traceback and request metadata
    user_id = getattr(request.state, "user_id", None)
    logging.getLogger("app.exception").error(
        f"Unhandled Exception: {str(exc)}",
        exc_info=exc,
        extra={
            "method": request.method,
            "path": request.url.path,
            "user_id": user_id,
        }
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred on the server.",
            "code": "INTERNAL_SERVER_ERROR",
            "field": None
        }
    )

# Register authentication routes
app.include_router(auth_router)

# Register ingestion routes
app.include_router(upload_router, tags=["Document Ingestion"])

# Register Q&A query routes
app.include_router(query_router, tags=["Document Q&A"])

# Register documents lifecycle routes
app.include_router(documents_router, prefix="/documents", tags=["Documents"])

# Register conversational sessions routes
app.include_router(sessions_router, prefix="/sessions",
                   tags=["Conversational Sessions"])


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect incoming root requests directly to interactive OpenAPI docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System Health"])
async def health_check():
    """Lightweight health check returning static ok status."""
    return {"status": "ok"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
