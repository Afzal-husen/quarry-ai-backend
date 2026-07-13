import datetime
import json
import math
import os
import re
import uuid
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Handles splitting of loaded documents into manageable chunks and saving metadata locally."""

    def __init__(self, default_chunk_size: int = 500, default_chunk_overlap: int = 50):
        """Initializes the chunker with fallback default configurations.

        Args:
            default_chunk_size: Standard fallback character size for chunks.
            default_chunk_overlap: Standard fallback overlap size between chunks.
        """
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    def _cosine_distance(self, u: List[float], v: List[float]) -> float:
        """Computes the cosine distance between two vectors."""
        dot_product = sum(x * y for x, y in zip(u, v))
        norm_u = math.sqrt(sum(x * x for x in u))
        norm_v = math.sqrt(sum(x * x for x in v))
        if norm_u == 0 or norm_v == 0:
            return 1.0
        val = dot_product / (norm_u * norm_v)
        val = max(-1.0, min(1.0, val))  # Clamp to avoid float precision domain error
        return 1.0 - val

    def _split_semantically(
        self,
        text: str,
        sentence_buffer_window: int = 1,
        threshold_type: str = "percentile",
        threshold_value: Optional[float] = None
    ) -> List[str]:
        """Splits text semantically based on embedding distance shifts between sliding window sentence groupings."""
        # 1. Segment sentences
        sentence_split_regex = re.compile(r'(?<=[\.\!\?])\s+')
        sentences = [s.strip() for s in sentence_split_regex.split(text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        # 2. Build combined window groups
        combined_groups = []
        for i in range(len(sentences)):
            start = max(0, i - sentence_buffer_window)
            end = i + sentence_buffer_window + 1
            combined = " ".join(sentences[start:end])
            combined_groups.append(combined)

        # 3. Compute embeddings
        from app.core.vectorstore import EmbeddingsManager
        try:
            embeddings_model = EmbeddingsManager.get_embeddings()
            embeddings = embeddings_model.embed_documents(combined_groups)
        except Exception:
            # Fallback if embeddings computation fails: group into pairs of sentences
            chunks = []
            for i in range(0, len(sentences), 2):
                chunks.append(" ".join(sentences[i:i+2]))
            return chunks

        # 4. Compute cosine distances between consecutive sentence embeddings
        distances = []
        for i in range(len(embeddings) - 1):
            distances.append(self._cosine_distance(embeddings[i], embeddings[i+1]))

        if not distances:
            return [" ".join(sentences)]

        # 5. Resolve threshold boundary
        if threshold_type == "percentile":
            q = 0.95
            if threshold_value is not None:
                q = threshold_value / 100.0 if threshold_value > 1.0 else threshold_value
            sorted_dist = sorted(distances)
            idx = min(len(sorted_dist) - 1, int(len(sorted_dist) * q))
            threshold = sorted_dist[idx]
        elif threshold_type == "standard_deviation":
            k = 1.2
            if threshold_value is not None:
                k = threshold_value
            mean = sum(distances) / len(distances)
            variance = sum((x - mean) ** 2 for x in distances) / len(distances)
            std = math.sqrt(variance)
            threshold = mean + k * std
        elif threshold_type == "absolute":
            threshold = threshold_value if threshold_value is not None else 0.4
        else:
            threshold = 0.4

        # 6. Assemble chunks
        chunks = []
        current_chunk = []
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if i < len(distances) and distances[i] > threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def split_documents(
        self,
        docs: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_strategy: Optional[str] = "character",
        semantic_threshold_type: Optional[str] = "percentile",
        semantic_threshold: Optional[float] = None
    ) -> List[Document]:
        """Splits a list of LangChain Document objects into smaller character or semantic chunks

        nested within parent chunks.
        """
        strategy = chunking_strategy if chunking_strategy in ("character", "semantic") else "character"

        # Parent Splitter (Standard recursive character splitter)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len
        )

        child_docs = []

        for doc in docs:
            # 1. Split raw page text into parent chunks
            parent_chunks = parent_splitter.split_documents([doc])

            for parent_chunk in parent_chunks:
                parent_id = str(uuid.uuid4())
                page_index = parent_chunk.metadata.get("page", doc.metadata.get("page", 0))

                # 2. Split parent chunk into child chunks
                if strategy == "character":
                    size = chunk_size if chunk_size is not None else self.default_chunk_size
                    overlap = chunk_overlap if chunk_overlap is not None else self.default_chunk_overlap
                    if overlap >= size:
                        overlap = max(0, size - 1)

                    child_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=size,
                        chunk_overlap=overlap,
                        length_function=len
                    )
                    sub_chunks = child_splitter.split_documents([parent_chunk])
                    child_texts = [sub.page_content for sub in sub_chunks]
                else:
                    # Semantic chunking inside parent chunk boundaries
                    buffer_window = chunk_overlap if chunk_overlap is not None else 1
                    t_type = semantic_threshold_type if semantic_threshold_type is not None else "percentile"
                    child_texts = self._split_semantically(
                        text=parent_chunk.page_content,
                        sentence_buffer_window=buffer_window,
                        threshold_type=t_type,
                        threshold_value=semantic_threshold
                    )

                # 3. Create Child Document objects with parent references in metadata
                for c_text in child_texts:
                    child_metadata = {
                        "chunk_id": str(uuid.uuid4()),
                        "parent_id": parent_id,
                        "parent_text": parent_chunk.page_content,
                        "page_index": page_index,
                    }
                    child_doc = Document(
                        page_content=c_text,
                        metadata=child_metadata
                    )
                    child_docs.append(child_doc)

            # Explicitly delete references to reduce peak ingestion spikes
            del parent_chunks

        # Collect garbage once at the end of document splitting to reclaim memory
        # without causing multiple blocking per-page CPU/GIL freezes
        import gc
        gc.collect()

        return child_docs

    def save_chunks(
        self,
        document_id: str,
        source_filename: str,
        chunks: List[Document],
        output_dir: Path,
        uploaded_at: Optional[str] = None,
        chunking_strategy: Optional[str] = "character"
    ) -> Path:
        """Serializes and persists the parent and child chunked documents inside a structured JSON metadata format."""
        output_dir.mkdir(parents=True, exist_ok=True)
        destination_path = output_dir / f"{document_id}.json"

        if uploaded_at is None:
            uploaded_at = datetime.datetime.now(timezone.utc).isoformat()

        # Build parents list by tracking unique parent_ids
        parents_map = {}
        serialized_chunks = []

        for chunk in chunks:
            parent_id = chunk.metadata.get("parent_id")
            parent_text = chunk.metadata.get("parent_text")
            page_index = chunk.metadata.get("page_index", 0)

            if parent_id and parent_id not in parents_map:
                parents_map[parent_id] = {
                    "parent_id": parent_id,
                    "page_index": page_index,
                    "text": parent_text
                }

            chunk_data = {
                "chunk_id": chunk.metadata.get("chunk_id", str(uuid.uuid4())),
                "parent_id": parent_id,
                "page_index": page_index,
                "text": chunk.page_content,
                "char_length": len(chunk.page_content)
            }
            serialized_chunks.append(chunk_data)

        payload = {
            "document_id": document_id,
            "source_filename": source_filename,
            "uploaded_at": uploaded_at,
            "chunking_strategy": chunking_strategy,
            "total_parents": len(parents_map),
            "total_chunks": len(serialized_chunks),
            "parents": list(parents_map.values()),
            "chunks": serialized_chunks,
            "summary": "",
            "summary_status": "pending"
        }

        with open(destination_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        return destination_path
