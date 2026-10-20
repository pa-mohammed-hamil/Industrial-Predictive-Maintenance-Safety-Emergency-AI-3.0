
"""
app/ai/rag/ingestion/ingestion.py

Industrial RAG Document Ingestion Pipeline
==========================================

Python 3.11+

Pipeline:
    Document Upload
        -> Validation
        -> Document Processing
        -> Metadata Enrichment
        -> Chunking
        -> Embedding Generation
        -> BM25 Indexing
        -> Vector Storage
        -> Ingestion Registry

Features:
- PDF, DOCX, TXT, MD, CSV, JSON, HTML, image OCR
- Tenant-aware ingestion
- Equipment/asset/site metadata
- Page-level source references
- Content-hash duplicate detection
- Document versioning
- BM25 and dense vector indexing
- Batch ingestion
- Async processing
- Retry-safe duplicate handling
- Document deletion
- Pipeline statistics

Production note:
    This module uses an in-memory registry.
    Replace it with PostgreSQL for durable ingestion
    state and distributed worker coordination.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from app.ai.rag.ingestion.document_processing import (
    DocumentProcessor,
    DocumentProcessingConfig,
    ProcessedDocument,
)

from app.ai.rag.ingestion.chunking import (
    IndustrialDocumentChunker,
    ChunkingConfig,
    ChunkingStrategy,
    DocumentInput,
    DocumentChunk,
)

from app.ai.rag.retrieval.bm25 import (
    BM25Retriever,
    BM25Document,
)

from app.ai.rag.embeddings import EmbeddingService


logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Base ingestion error."""


class IngestionValidationError(IngestionError):
    """Invalid ingestion request."""


class IngestionIndexingError(IngestionError):
    """Indexing failed."""


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass(slots=True)
class IngestionConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150

    max_file_size_mb: int = 30
    max_pages: int = 300

    enable_embeddings: bool = True
    enable_bm25: bool = True
    enable_ocr: bool = False

    skip_duplicates: bool = True
    allow_reindex: bool = True

    max_batch_size: int = 20

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise IngestionValidationError(
                "chunk_size must be positive"
            )

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise IngestionValidationError(
                "Invalid chunk_overlap"
            )

        if self.max_batch_size <= 0:
            raise IngestionValidationError(
                "max_batch_size must be positive"
            )


@dataclass(slots=True)
class IngestionRequest:
    company_id: str
    filename: str
    data: bytes

    document_id: str | None = None
    equipment_id: str | None = None
    asset_id: str | None = None
    site_id: str | None = None

    document_version: str = "v1"

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestionRecord:
    ingestion_id: str
    company_id: str
    document_id: str
    filename: str
    content_hash: str
    document_version: str

    status: IngestionStatus
    chunk_count: int
    indexed_at: datetime | None

    equipment_id: str | None = None
    asset_id: str | None = None
    site_id: str | None = None

    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingestion_id": self.ingestion_id,
            "company_id": self.company_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "content_hash": self.content_hash,
            "document_version": self.document_version,
            "status": self.status.value,
            "chunk_count": self.chunk_count,
            "indexed_at": (
                self.indexed_at.isoformat()
                if self.indexed_at
                else None
            ),
            "equipment_id": self.equipment_id,
            "asset_id": self.asset_id,
            "site_id": self.site_id,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class VectorStore(Protocol):
    async def upsert(
        self,
        *,
        company_id: str,
        vectors: Sequence[
            tuple[str, Sequence[float], Mapping[str, Any]]
        ],
    ) -> None:
        ...

    async def delete(
        self,
        *,
        company_id: str,
        chunk_ids: Sequence[str],
    ) -> None:
        ...


class InMemoryVectorStore:
    """
    In-memory vector adapter for development.

    Production:
        Replace with Qdrant, pgvector, Milvus,
        Weaviate or Pinecone.
    """

    def __init__(self) -> None:
        self.vectors: dict[
            tuple[str, str],
            tuple[list[float], dict[str, Any]],
        ] = {}

    async def upsert(
        self,
        *,
        company_id: str,
        vectors: Sequence[
            tuple[str, Sequence[float], Mapping[str, Any]]
        ],
    ) -> None:
        for chunk_id, vector, metadata in vectors:
            self.vectors[
                (company_id, chunk_id)
            ] = (
                list(vector),
                dict(metadata),
            )

    async def delete(
        self,
        *,
        company_id: str,
        chunk_ids: Sequence[str],
    ) -> None:
        for chunk_id in chunk_ids:
            self.vectors.pop(
                (company_id, chunk_id),
                None,
            )


class IndustrialRAGIngestionPipeline:
    """
    End-to-end industrial RAG ingestion.

    The pipeline stores source metadata and chunk IDs
    so documents can be reindexed or deleted later.
    """

    def __init__(
        self,
        *,
        config: IngestionConfig | None = None,
        document_processor: DocumentProcessor | None = None,
        chunker: IndustrialDocumentChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        bm25: BM25Retriever | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.config = config or IngestionConfig()

        self.document_processor = (
            document_processor
            or DocumentProcessor(
                DocumentProcessingConfig(
                    max_file_size_mb=(
                        self.config.max_file_size_mb
                    ),
                    max_pages=self.config.max_pages,
                    enable_ocr=self.config.enable_ocr,
                )
            )
        )

        self.chunker = (
            chunker
            or IndustrialDocumentChunker(
                ChunkingConfig(
                    strategy=ChunkingStrategy.RECURSIVE,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
            )
        )

        self.embedding_service = embedding_service
        self.bm25 = bm25 or BM25Retriever()
        self.vector_store = (
            vector_store or InMemoryVectorStore()
        )

        self._records: dict[
            tuple[str, str], IngestionRecord
        ] = {}

        self._document_chunks: dict[
            tuple[str, str], list[str]
        ] = {}

        self._lock = asyncio.Lock()

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def _require_text(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise IngestionValidationError(
                f"{name} must be a nonempty string"
            )
        return value.strip()

    def _validate_request(
        self,
        request: IngestionRequest,
    ) -> None:
        self._require_text(
            request.company_id,
            "company_id",
        )

        self._require_text(
            request.filename,
            "filename",
        )

        self._require_text(
            request.document_version,
            "document_version",
        )

        if not isinstance(request.data, bytes):
            raise IngestionValidationError(
                "data must be bytes"
            )

        if not request.data:
            raise IngestionValidationError(
                "Document data cannot be empty"
            )

        maximum = (
            self.config.max_file_size_mb
            * 1024
            * 1024
        )

        if len(request.data) > maximum:
            raise IngestionValidationError(
                "Document exceeds file size limit"
            )

    # =====================================================
    # CHUNK GENERATION
    # =====================================================

    def _generate_chunks(
        self,
        document: ProcessedDocument,
        request: IngestionRequest,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for page in document.pages:
            if not page.text.strip():
                continue

            metadata = {
                **document.metadata,
                **request.metadata,
                "document_id": document.document_id,
                "document_version": (
                    request.document_version
                ),
                "equipment_id": request.equipment_id,
                "asset_id": request.asset_id,
                "site_id": request.site_id,
                "page": page.page_number,
                "title": document.filename,
                "content_hash": document.content_hash,
            }

            metadata.update(page.metadata)

            page_input = DocumentInput(
                document_id=(
                    f"{document.document_id}:"
                    f"{request.document_version}:"
                    f"page:{page.page_number}"
                ),
                company_id=request.company_id,
                text=page.text,
                metadata=metadata,
            )

            page_chunks = self.chunker.chunk_document(
                page_input
            )

            chunks.extend(page_chunks)

        return chunks

    # =====================================================
    # EMBEDDING GENERATION
    # =====================================================

    async def _embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        company_id: str,
    ) -> list[list[float]]:
        if not self.config.enable_embeddings:
            return []

        if self.embedding_service is None:
            raise IngestionIndexingError(
                "Embedding service is required"
            )

        vectors = await self.embedding_service.aembed_documents(
            [chunk.text for chunk in chunks],
            company_id=company_id,
        )

        if len(vectors) != len(chunks):
            raise IngestionIndexingError(
                "Embedding count mismatch"
            )

        return vectors

    # =====================================================
    # INGEST BYTES
    # =====================================================

    async def ingest(
        self,
        request: IngestionRequest,
    ) -> IngestionRecord:
        self._validate_request(request)

        company_id = request.company_id

        content_hash = hashlib.sha256(
            request.data
        ).hexdigest()

        document_id = (
            request.document_id
            or f"doc_{content_hash[:24]}"
        )

        key = (company_id, document_id)

        async with self._lock:
            existing = self._records.get(key)

            if existing is not None:
                if (
                    self.config.skip_duplicates
                    and existing.content_hash == content_hash
                    and existing.document_version
                    == request.document_version
                ):
                    return existing

                if not self.config.allow_reindex:
                    raise IngestionValidationError(
                        "Document already exists"
                    )

        record = IngestionRecord(
            ingestion_id=str(uuid.uuid4()),
            company_id=company_id,
            document_id=document_id,
            filename=request.filename,
            content_hash=content_hash,
            document_version=request.document_version,
            status=IngestionStatus.PROCESSING,
            chunk_count=0,
            indexed_at=None,
            equipment_id=request.equipment_id,
            asset_id=request.asset_id,
            site_id=request.site_id,
            metadata=dict(request.metadata),
        )

        try:
            processed = await asyncio.to_thread(
                self.document_processor.process_bytes,
                data=request.data,
                filename=request.filename,
                company_id=company_id,
                document_id=document_id,
                metadata=request.metadata,
            )

            chunks = await asyncio.to_thread(
                self._generate_chunks,
                processed,
                request,
            )

            if not chunks:
                raise IngestionIndexingError(
                    "No searchable chunks generated"
                )

            vectors = await self._embed_chunks(
                chunks,
                company_id=company_id,
            )

            # Commit replacement under the pipeline lock.
            # For durable stores, use a staging index and
            # atomic active-version switch instead.
            async with self._lock:
                old_chunk_ids = self._document_chunks.get(
                    key,
                    [],
                )

                if old_chunk_ids:
                    await self.vector_store.delete(
                        company_id=company_id,
                        chunk_ids=old_chunk_ids,
                    )

                    if self.config.enable_bm25:
                        for chunk_id in old_chunk_ids:
                            self.bm25.remove_document(
                                chunk_id,
                                company_id=company_id,
                            )

                if self.config.enable_bm25:
                    for chunk in chunks:
                        self.bm25.add_document(
                            BM25Document(
                                id=chunk.id,
                                text=chunk.text,
                                company_id=company_id,
                                metadata=dict(chunk.metadata),
                            )
                        )

                if vectors:
                    await self.vector_store.upsert(
                        company_id=company_id,
                        vectors=[
                            (
                                chunk.id,
                                vector,
                                dict(chunk.metadata),
                            )
                            for chunk, vector
                            in zip(chunks, vectors)
                        ],
                    )

                self._document_chunks[key] = [
                    chunk.id
                    for chunk in chunks
                ]

                record.status = IngestionStatus.COMPLETED
                record.chunk_count = len(chunks)
                record.indexed_at = datetime.now(
                    timezone.utc
                )

                self._records[key] = record

            return record

        except Exception as exc:
            record.status = IngestionStatus.FAILED
            record.error = str(exc)

            logger.exception(
                "Document ingestion failed: company=%s document=%s",
                company_id,
                document_id,
            )

            # Do not overwrite the previous completed
            # record when a reindex attempt fails.
            async with self._lock:
                if key not in self._records:
                    self._records[key] = record

            raise IngestionIndexingError(
                "Document ingestion failed"
            ) from exc

    # =====================================================
    # INGEST FILE
    # =====================================================

    async def ingest_file(
        self,
        path: str | Path,
        *,
        company_id: str,
        document_id: str | None = None,
        equipment_id: str | None = None,
        asset_id: str | None = None,
        site_id: str | None = None,
        document_version: str = "v1",
        metadata: Mapping[str, Any] | None = None,
    ) -> IngestionRecord:
        path = Path(path)

        if not path.is_file():
            raise IngestionValidationError(
                "File does not exist"
            )

        maximum = (
            self.config.max_file_size_mb
            * 1024
            * 1024
        )

        def read_limited() -> bytes:
            with path.open("rb") as stream:
                return stream.read(maximum + 1)

        data = await asyncio.to_thread(
            read_limited
        )

        return await self.ingest(
            IngestionRequest(
                company_id=company_id,
                filename=path.name,
                data=data,
                document_id=document_id,
                equipment_id=equipment_id,
                asset_id=asset_id,
                site_id=site_id,
                document_version=document_version,
                metadata=dict(metadata or {}),
            )
        )

    # =====================================================
    # BATCH INGESTION
    # =====================================================

    async def ingest_batch(
        self,
        requests: Sequence[IngestionRequest],
    ) -> list[IngestionRecord]:
        if len(requests) > self.config.max_batch_size:
            raise IngestionValidationError(
                "Batch size exceeds configured limit"
            )

        results = []

        for request in requests:
            record = await self.ingest(request)
            results.append(record)

        return results

    # =====================================================
    # GET RECORD
    # =====================================================

    async def get_record(
        self,
        document_id: str,
        *,
        company_id: str,
    ) -> IngestionRecord | None:
        async with self._lock:
            return self._records.get(
                (company_id, document_id)
            )

    # =====================================================
    # LIST RECORDS
    # =====================================================

    async def list_records(
        self,
        *,
        company_id: str,
        status: IngestionStatus | None = None,
    ) -> list[IngestionRecord]:
        async with self._lock:
            return [
                record
                for (tenant, _), record
                in self._records.items()
                if (
                    tenant == company_id
                    and (
                        status is None
                        or record.status == status
                    )
                )
            ]

    # =====================================================
    # DELETE DOCUMENT
    # =====================================================

    async def delete_document(
        self,
        document_id: str,
        *,
        company_id: str,
    ) -> bool:
        key = (company_id, document_id)

        async with self._lock:
            record = self._records.get(key)

            if record is None:
                return False

            chunk_ids = self._document_chunks.get(
                key,
                [],
            )

            await self.vector_store.delete(
                company_id=company_id,
                chunk_ids=chunk_ids,
            )

            if self.config.enable_bm25:
                for chunk_id in chunk_ids:
                    self.bm25.remove_document(
                        chunk_id,
                        company_id=company_id,
                    )

            self._document_chunks.pop(
                key,
                None,
            )

            record.status = IngestionStatus.DELETED
            record.chunk_count = 0

            return True

    # =====================================================
    # STATISTICS
    # =====================================================

    async def get_statistics(
        self,
        *,
        company_id: str,
    ) -> dict[str, Any]:
        records = await self.list_records(
            company_id=company_id,
        )

        return {
            "company_id": company_id,
            "total_documents": len(records),
            "completed_documents": sum(
                record.status == IngestionStatus.COMPLETED
                for record in records
            ),
            "failed_documents": sum(
                record.status == IngestionStatus.FAILED
                for record in records
            ),
            "deleted_documents": sum(
                record.status == IngestionStatus.DELETED
                for record in records
            ),
            "total_chunks": sum(
                record.chunk_count
                for record in records
                if record.status == IngestionStatus.COMPLETED
            ),
            "bm25_enabled": self.config.enable_bm25,
            "embeddings_enabled": (
                self.config.enable_embeddings
            ),
        }


# =========================================================
# EXAMPLE USAGE
# =========================================================

async def example() -> None:
    pipeline = IndustrialRAGIngestionPipeline(
        config=IngestionConfig(
            enable_embeddings=False,
            enable_bm25=True,
            chunk_size=400,
            chunk_overlap=60,
        )
    )

    sample_text = (
        "Industrial Pump Maintenance Manual\n\n"
        "Pump PUMP-201 has excessive vibration "
        "of 12.5 mm/s. Possible causes include "
        "bearing wear, rotor imbalance, "
        "misalignment and cavitation.\n\n"
        "Inspect the equipment according to "
        "approved maintenance procedures."
    )

    record = await pipeline.ingest(
        IngestionRequest(
            company_id="company-001",
            filename="pump_manual.txt",
            data=sample_text.encode("utf-8"),
            equipment_id="PUMP-201",
            asset_id="ASSET-201",
            site_id="plant-01",
            metadata={
                "manufacturer": "Example Manufacturer",
                "category": "maintenance_manual",
            },
        )
    )

    print(record.to_dict())

    print(
        await pipeline.get_statistics(
            company_id="company-001"
        )
    )


if __name__ == "__main__":
    asyncio.run(example())
