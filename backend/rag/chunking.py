
"""
app/ai/rag/ingestion/chunking.py

Industrial Document Chunking Engine
===================================

Python 3.11+
Standard library only

Features:
- Recursive text chunking
- Fixed-size text chunking
- Paragraph-aware chunking
- Sentence-aware splitting
- Parent-child hierarchical chunking
- Configurable chunk size and overlap
- Industrial metadata preservation
- Document/page citations
- Stable chunk identifiers
- Batch document processing
- Token-aware chunking via optional tokenizer
- Chunk statistics
- Empty-content filtering

Suitable for:
- Equipment manuals
- Maintenance records
- Safety SOPs
- Inspection reports
- Industrial troubleshooting guides
- Incident reports
- Technical specifications
- RAG and hybrid retrieval pipelines

Note:
Semantic grouping here is structure-aware, not
embedding-based semantic segmentation.
"""

from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class ChunkingError(Exception):
    """Base document chunking error."""


class ChunkingValidationError(ChunkingError):
    """Invalid chunking configuration or input."""


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    PARENT_CHILD = "parent_child"


@dataclass(slots=True)
class DocumentInput:
    document_id: str
    company_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    id: str
    document_id: str
    company_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    level: str = "child"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "company_id": self.company_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": dict(self.metadata),
            "parent_id": self.parent_id,
            "level": self.level,
        }


@dataclass(slots=True)
class ChunkingConfig:
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 1000
    chunk_overlap: int = 150
    parent_chunk_size: int = 3000
    min_chunk_size: int = 1
    preserve_whitespace: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str):
            self.strategy = ChunkingStrategy(self.strategy)

        if self.chunk_size <= 0:
            raise ChunkingValidationError(
                "chunk_size must be positive"
            )

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ChunkingValidationError(
                "chunk_overlap must be >= 0 and < chunk_size"
            )

        if self.parent_chunk_size < self.chunk_size:
            raise ChunkingValidationError(
                "parent_chunk_size must be >= chunk_size"
            )

        if self.min_chunk_size <= 0:
            raise ChunkingValidationError(
                "min_chunk_size must be positive"
            )


class IndustrialDocumentChunker:
    """
    Configurable document chunking engine.

    Character offsets refer to the original document
    after optional whitespace normalization.

    Use preserve_whitespace=True when exact original
    source offsets are required.
    """

    SEPARATORS = (
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " ",
    )

    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\\s+"
    )

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        tokenizer: Callable[[str], Sequence[Any]] | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = tokenizer

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _require_text(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ChunkingValidationError(
                f"{name} must be a nonempty string"
            )
        return value

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _chunk_id(
        company_id: str,
        document_id: str,
        index: int,
        start: int,
        end: int,
        text: str,
        level: str,
    ) -> str:
        raw = (
            f"{company_id}|{document_id}|{index}|"
            f"{start}|{end}|{level}|{text}"
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:24]

        return f"chunk_{digest}"

    def _measure(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text)

        return len(self.tokenizer(text))

    def _validate_document(
        self,
        document: DocumentInput,
    ) -> None:
        if not isinstance(document, DocumentInput):
            raise ChunkingValidationError(
                "Expected DocumentInput"
            )

        self._require_text(
            document.document_id,
            "document_id",
        )

        self._require_text(
            document.company_id,
            "company_id",
        )

        if not isinstance(document.text, str):
            raise ChunkingValidationError(
                "Document text must be a string"
            )

    # =====================================================
    # FIXED-SIZE SPLITTING
    # =====================================================

    @staticmethod
    def _fixed_spans(
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        if not text:
            return []

        spans = []
        step = chunk_size - overlap

        for start in range(0, len(text), step):
            end = min(
                start + chunk_size,
                len(text),
            )

            spans.append((start, end))

            if end == len(text):
                break

        return spans

    # =====================================================
    # RECURSIVE SPLITTING
    # =====================================================

    def _recursive_spans(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        """
        Structure-aware chunk boundaries.

        Prefers:
        1. Paragraph boundaries
        2. Line boundaries
        3. Sentence boundaries
        4. Word boundaries
        5. Hard character boundaries
        """
        if not text:
            return []

        spans: list[tuple[int, int]] = []

        start = 0
        length = len(text)

        while start < length:
            hard_end = min(
                start + chunk_size,
                length,
            )

            if hard_end == length:
                spans.append((start, length))
                break

            end = hard_end

            search_start = start + max(
                1,
                chunk_size // 2,
            )

            for separator in self.SEPARATORS:
                boundary = text.rfind(
                    separator,
                    search_start,
                    hard_end,
                )

                if boundary != -1:
                    candidate = boundary + len(separator)

                    if candidate > start:
                        end = candidate
                        break

            if end <= start:
                end = hard_end

            spans.append((start, end))

            next_start = max(
                start + 1,
                end - overlap,
            )

            start = next_start

        return spans

    # =====================================================
    # PARAGRAPH SPLITTING
    # =====================================================

    def _paragraph_spans(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        return self._recursive_spans(
            text,
            chunk_size,
            overlap,
        )

    # =====================================================
    # SENTENCE SPLITTING
    # =====================================================

    def _sentence_spans(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        if not text:
            return []

        sentence_ends = [
            match.end()
            for match in re.finditer(
                r"[.!?](?:\\s+|$)",
                text,
            )
        ]

        sentence_ends.append(len(text))
        sentence_ends = sorted(set(sentence_ends))

        spans = []
        start = 0

        while start < len(text):
            hard_end = min(
                start + chunk_size,
                len(text),
            )

            candidates = [
                end
                for end in sentence_ends
                if start < end <= hard_end
            ]

            end = (
                max(candidates)
                if candidates
                else hard_end
            )

            spans.append((start, end))

            if end == len(text):
                break

            start = max(
                start + 1,
                end - overlap,
            )

        return spans

    # =====================================================
    # STRATEGY ROUTER
    # =====================================================

    def _get_spans(
        self,
        text: str,
        *,
        strategy: ChunkingStrategy,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        if strategy == ChunkingStrategy.FIXED:
            return self._fixed_spans(
                text,
                chunk_size,
                overlap,
            )

        if strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_spans(
                text,
                chunk_size,
                overlap,
            )

        if strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_spans(
                text,
                chunk_size,
                overlap,
            )

        return self._recursive_spans(
            text,
            chunk_size,
            overlap,
        )

    # =====================================================
    # BUILD CHUNKS
    # =====================================================

    def _build_chunks(
        self,
        document: DocumentInput,
        text: str,
        spans: list[tuple[int, int]],
        *,
        level: str,
        parent_id: str | None = None,
        base_offset: int = 0,
    ) -> list[DocumentChunk]:
        chunks = []

        for start, end in spans:
            raw = text[start:end]

            if not raw.strip():
                continue

            # Preserve offsets even when trimming whitespace.
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())

            adjusted_start = start + leading
            adjusted_end = end - trailing

            content = text[
                adjusted_start:adjusted_end
            ]

            if (
                self._measure(content)
                < self.config.min_chunk_size
            ):
                continue

            absolute_start = (
                base_offset + adjusted_start
            )
            absolute_end = (
                base_offset + adjusted_end
            )

            index = len(chunks)

            chunk_id = self._chunk_id(
                document.company_id,
                document.document_id,
                index,
                absolute_start,
                absolute_end,
                content,
                level,
            )

            metadata = dict(document.metadata)

            metadata.update({
                "chunk_level": level,
                "chunk_index": index,
                "start_char": absolute_start,
                "end_char": absolute_end,
                "character_count": len(content),
                "unit_count": self._measure(content),
                "parent_id": parent_id,
            })

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document.document_id,
                    company_id=document.company_id,
                    text=content,
                    chunk_index=index,
                    start_char=absolute_start,
                    end_char=absolute_end,
                    metadata=metadata,
                    parent_id=parent_id,
                    level=level,
                )
            )

        return chunks

    # =====================================================
    # MAIN CHUNKING
    # =====================================================

    def chunk_document(
        self,
        document: DocumentInput,
    ) -> list[DocumentChunk]:
        self._validate_document(document)

        text = (
            document.text
            if self.config.preserve_whitespace
            else self._normalize_text(document.text)
        )

        if not text.strip():
            return []

        if (
            self.config.strategy
            == ChunkingStrategy.PARENT_CHILD
        ):
            return self._parent_child_chunks(
                document,
                text,
            )

        spans = self._get_spans(
            text,
            strategy=self.config.strategy,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )

        return self._build_chunks(
            document,
            text,
            spans,
            level="child",
        )

    # =====================================================
    # PARENT-CHILD CHUNKING
    # =====================================================

    def _parent_child_chunks(
        self,
        document: DocumentInput,
        text: str,
    ) -> list[DocumentChunk]:
        parent_spans = self._recursive_spans(
            text,
            self.config.parent_chunk_size,
            0,
        )

        parents = self._build_chunks(
            document,
            text,
            parent_spans,
            level="parent",
        )

        result: list[DocumentChunk] = []

        for parent in parents:
            result.append(parent)

            child_spans = self._recursive_spans(
                parent.text,
                self.config.chunk_size,
                self.config.chunk_overlap,
            )

            children = self._build_chunks(
                document,
                parent.text,
                child_spans,
                level="child",
                parent_id=parent.id,
                base_offset=parent.start_char,
            )

            result.extend(children)

        return result

    # =====================================================
    # BATCH DOCUMENT CHUNKING
    # =====================================================

    def chunk_documents(
        self,
        documents: Sequence[DocumentInput],
    ) -> list[DocumentChunk]:
        result: list[DocumentChunk] = []

        for document in documents:
            result.extend(
                self.chunk_document(document)
            )

        return result

    # =====================================================
    # CHUNK STATISTICS
    # =====================================================

    def get_statistics(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> dict[str, Any]:
        if not chunks:
            return {
                "total_chunks": 0,
                "parent_chunks": 0,
                "child_chunks": 0,
                "average_size": 0.0,
                "minimum_size": 0,
                "maximum_size": 0,
                "document_count": 0,
            }

        sizes = [
            self._measure(chunk.text)
            for chunk in chunks
        ]

        return {
            "total_chunks": len(chunks),
            "parent_chunks": sum(
                chunk.level == "parent"
                for chunk in chunks
            ),
            "child_chunks": sum(
                chunk.level == "child"
                for chunk in chunks
            ),
            "average_size": sum(sizes) / len(sizes),
            "minimum_size": min(sizes),
            "maximum_size": max(sizes),
            "document_count": len({
                (chunk.company_id, chunk.document_id)
                for chunk in chunks
            }),
            "strategy": self.config.strategy.value,
        }


# =========================================================
# EXAMPLE
# =========================================================

if __name__ == "__main__":
    document = DocumentInput(
        document_id="motor-maintenance-manual-001",
        company_id="company-001",
        text=(
            "Motor Bearing Maintenance\n\n"
            "Excessive bearing temperature can indicate "
            "lubrication failure or mechanical overload.\n\n"
            "Vibration Monitoring\n\n"
            "High vibration may result from misalignment, "
            "imbalance, or bearing wear.\n\n"
            "Safety Procedure\n\n"
            "Before inspection, follow approved lockout "
            "tagout procedures and verify isolation."
        ),
        metadata={
            "document_type": "maintenance_manual",
            "asset_id": "motor-101",
            "page": 12,
            "source": "motor_manual.pdf",
        },
    )

    config = ChunkingConfig(
        strategy=ChunkingStrategy.PARENT_CHILD,
        chunk_size=120,
        chunk_overlap=20,
        parent_chunk_size=300,
    )

    chunker = IndustrialDocumentChunker(config)

    chunks = chunker.chunk_document(document)

    for chunk in chunks:
        print({
            "id": chunk.id,
            "level": chunk.level,
            "parent_id": chunk.parent_id,
            "text": chunk.text,
        })

    print(chunker.get_statistics(chunks))
