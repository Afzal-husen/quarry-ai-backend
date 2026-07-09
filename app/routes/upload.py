import os
import shutil
import threading
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.chunker import DocumentChunker
from app.core.parsers import DocumentParser, DocumentParsingError
from app.core.vectorstore import VectorStoreManager, VectorStoreError
from app.core.auth import get_current_user
from app.core.limiter import limiter
from app.core.paths import get_data_dir

# Ensure environment variables are loaded
load_dotenv()

router = APIRouter()

# Resolve storage directories from configured data root (DATA_DIR env or local fallback)
_DATA_DIR = get_data_dir()
UPLOADS_DIR = _DATA_DIR / "uploads"
CHUNKS_DIR = _DATA_DIR / "chunks"

# Max file size limit: 50 MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}

# Initialize core services
parser = DocumentParser()
default_size = int(os.getenv("CHUNK_SIZE", "500"))
default_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
chunker = DocumentChunker(default_chunk_size=default_size,
                          default_chunk_overlap=default_overlap)
vector_manager = VectorStoreManager()

# In-memory job registry and thread-safety lock
ingestion_jobs = {}
jobs_lock = threading.Lock()


def prune_expired_jobs():
    """Removes job records that are older than 24 hours to prevent memory leaks."""
    now = datetime.now(timezone.utc)
    expired_keys = []
    with jobs_lock:
        for key, job in list(ingestion_jobs.items()):
            if now - job["created_at"] > timedelta(hours=24):
                expired_keys.append(key)
        for key in expired_keys:
            del ingestion_jobs[key]


def run_ingestion_job(
    document_id: str,
    temp_file_path: Path,
    original_filename: str,
    user_id: str,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    chunking_strategy: Optional[str] = "character",
    semantic_threshold_type: Optional[str] = "percentile",
    semantic_threshold: Optional[float] = None
):
    """Synchronously executes parsing, chunking, and indexing in a background threadpool thread."""
    with jobs_lock:
        if document_id in ingestion_jobs:
            ingestion_jobs[document_id]["status"] = "processing"

    try:
        # 1. Parse Document contents
        documents = parser.parse_file(temp_file_path)

        # 2. Chunk Extracted Text Content
        split_docs = chunker.split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            semantic_threshold_type=semantic_threshold_type,
            semantic_threshold=semantic_threshold
        )

        # 3. Serialize Chunks and Save JSON Metadata locally
        chunker.save_chunks(
            document_id=document_id,
            source_filename=original_filename,
            chunks=split_docs,
            output_dir=CHUNKS_DIR / user_id,
            chunking_strategy=chunking_strategy
        )

        # 4. Index chunks into isolated Chroma vector store
        vector_manager.index_document(
            user_id=user_id,
            document_id=document_id,
            source_filename=original_filename
        )

        # 5. Generate document summary asynchronously inline (fault isolated)
        try:
            import json
            import logging
            from app.core.summarizer import DocumentSummarizer
            chunks_file_path = CHUNKS_DIR / user_id / f"{document_id}.json"
            if chunks_file_path.exists():
                with open(chunks_file_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                parents = payload.get("parents", [])
                if parents:
                    # Join parent texts
                    parent_texts = [p.get("text", "") for p in parents if p.get("text")]
                    combined_text = "\n\n".join(parent_texts)

                    # Safeguard truncation limit: 10,000 characters
                    if len(combined_text) > 10000:
                        # Truncate to first 5 parent chunks
                        truncated_parent_texts = parent_texts[:5]
                        combined_text = "\n\n".join(truncated_parent_texts)

                    # Perform summarization
                    summarizer = DocumentSummarizer()
                    summary_text = summarizer.summarize_text(combined_text)

                    # Update and save payload
                    payload["summary"] = summary_text
                    payload["summary_status"] = "completed"
                else:
                    payload["summary"] = "No content available to summarize."
                    payload["summary_status"] = "completed"

                with open(chunks_file_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=4, ensure_ascii=False)

        except Exception as summarization_err:
            import json
            import logging
            logging.error(f"Failed to generate summary for document {document_id}: {str(summarization_err)}")
            try:
                chunks_file_path = CHUNKS_DIR / user_id / f"{document_id}.json"
                if chunks_file_path.exists():
                    with open(chunks_file_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    payload["summary"] = ""
                    payload["summary_status"] = "failed"
                    with open(chunks_file_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=4, ensure_ascii=False)
            except Exception:
                pass

        # Update job status to complete
        with jobs_lock:
            if document_id in ingestion_jobs:
                ingestion_jobs[document_id]["status"] = "complete"

    except Exception as e:
        import logging
        logging.getLogger("app.exception").error(
            f"Ingestion job failed for document '{document_id}': {str(e)}",
            exc_info=True
        )
        # Perform hard cleanup on failure
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass

        chunks_file_path = CHUNKS_DIR / user_id / f"{document_id}.json"
        if chunks_file_path.exists():
            try:
                chunks_file_path.unlink()
            except Exception:
                pass

        try:
            vector_manager.delete_document(
                user_id=user_id, document_id=document_id)
        except Exception:
            pass

        # Update job status to failed with error message
        with jobs_lock:
            if document_id in ingestion_jobs:
                ingestion_jobs[document_id]["status"] = "failed"
                ingestion_jobs[document_id]["error"] = str(e)


@router.post(
    "/upload",
    status_code=202,
    summary="Upload Document",
    description="Uploads a PDF, DOC, or DOCX document and dispatches the parsing and vector indexing to the background.",
    response_description="Returns the background ingestion job ID and status."
)
@limiter.limit(os.getenv("RATE_LIMIT_UPLOAD", "5/minute"))
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Query(
        None, description="Character size of each split text block"),
    chunk_overlap: Optional[int] = Query(
        None, description="Character overlap override"),
    chunking_strategy: Optional[str] = Query(
        "character", description="Chunking strategy ('character' or 'semantic')"),
    semantic_threshold_type: Optional[str] = Query(
        "percentile", description="Semantic similarity threshold strategy ('percentile', 'standard_deviation', or 'absolute')"),
    semantic_threshold: Optional[float] = Query(
        None, description="Custom similarity threshold value"),
    current_user: dict = Depends(get_current_user)
):
    """Uploads a PDF, DOC, or DOCX document, and dispatches the parsing and vector indexing to the background.

    Args:
        request: The incoming FastAPI HTTP request.
        background_tasks: The FastAPI background task manager.
        file: The uploaded file (must be PDF or Word format, max 50 MB).
        chunk_size: Optional query parameter override for splitting chunk size.
        chunk_overlap: Optional query parameter override for splitting chunk overlap.

    Returns:
        HTTP 202 status code and the generated job ID.
    """
    # Validate strategy and threshold parameters
    if chunking_strategy not in ("character", "semantic"):
        raise HTTPException(
            status_code=422,
            detail="chunking_strategy must be either 'character' or 'semantic'."
        )
    if semantic_threshold_type not in ("percentile", "standard_deviation", "absolute"):
        raise HTTPException(
            status_code=422,
            detail="semantic_threshold_type must be 'percentile', 'standard_deviation', or 'absolute'."
        )
    if semantic_threshold is not None:
        if semantic_threshold_type == "percentile" and not (0 <= semantic_threshold <= 100):
            raise HTTPException(
                status_code=422,
                detail="Percentile threshold must be between 0 and 100."
            )
        if semantic_threshold_type in ("standard_deviation", "absolute") and semantic_threshold <= 0:
            raise HTTPException(
                status_code=422,
                detail="Threshold value must be greater than 0."
            )

    # 1. Validate File Extension
    original_filename = file.filename or "unknown"
    suffix = Path(original_filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{suffix}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 2. Check Content-Length header if present at request or file level
    content_length = request.headers.get(
        "content-length") or file.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file exceeds the 50 MB limit (declared length: {int(content_length)} bytes)."
                )
        except ValueError:
            pass

    # 3. Generate unique document UUID / job ID
    document_uuid = str(uuid.uuid4())
    saved_filename = f"{document_uuid}{suffix}"
    user_id = current_user["id"]
    user_upload_dir = UPLOADS_DIR / user_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = user_upload_dir / saved_filename

    # 4. Stream upload data chunk-by-chunk to disk to preserve memory
    total_bytes_written = 0
    try:
        with open(temp_file_path, "wb") as buffer:
            while True:
                chunk_bytes = await file.read(1024 * 1024)  # Read 1 MB chunk
                if not chunk_bytes:
                    break
                total_bytes_written += len(chunk_bytes)
                if total_bytes_written > MAX_FILE_SIZE_BYTES:
                    # Clean up file and abort
                    buffer.close()
                    if temp_file_path.exists():
                        temp_file_path.unlink()
                    raise HTTPException(
                        status_code=400,
                        detail="Uploaded file content exceeds the strict 50 MB limit."
                    )
                buffer.write(chunk_bytes)
    except HTTPException:
        raise
    except Exception as e:
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected file I/O error occurred during upload streaming: {str(e)}"
        ) from e
    finally:
        await file.close()

    # Clean up old jobs from registry
    prune_expired_jobs()

    # 5. Register the job in the registry
    with jobs_lock:
        ingestion_jobs[document_uuid] = {
            "status": "pending",
            "document_id": document_uuid,
            "filename": original_filename,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "error": None
        }

    # 6. Dispatch the ingestion pipeline to Starlette's threadpool worker
    background_tasks.add_task(
        run_ingestion_job,
        document_uuid,
        temp_file_path,
        original_filename,
        user_id,
        chunk_size,
        chunk_overlap,
        chunking_strategy,
        semantic_threshold_type,
        semantic_threshold
    )

    # 7. Return 202 Accepted immediately
    return JSONResponse(
        status_code=202,
        content={
            "job_id": document_uuid,
            "status": "pending"
        }
    )


@router.get(
    "/upload/{job_id}/status",
    summary="Get Ingestion Job Status",
    description="Retrieves the current execution status and metadata of a background ingestion job.",
    response_description="Returns job status, document ID, and error details if failed."
)
@limiter.limit(os.getenv("RATE_LIMIT_UPLOAD", "20/minute"))
async def get_job_status(
    request: Request,
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves the current execution status and metadata of a background ingestion job."""
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="job_id must be a valid UUID string."
        )

    # Clean up old jobs from registry
    prune_expired_jobs()

    with jobs_lock:
        job = ingestion_jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found."
        )

    if job["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: You do not own or have permission to access this job."
        )

    return JSONResponse(
        content={
            "status": job["status"],
            "document_id": job["document_id"],
            "error": job["error"]
        }
    )
