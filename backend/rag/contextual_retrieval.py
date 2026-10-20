
"""
app/ai/rag/retrieval/contextual_retrieval.py

Industrial Contextual Retrieval Engine
======================================

Python 3.11+
Standard library only

Features:
- Contextual retrieval over ranked chunks
- Parent-child chunk expansion
- Adjacent chunk expansion
- Metadata and tenant filtering
- BM25/vector/hybrid result normalization
- Retrieval score preservation
- Duplicate evidence removal
- Configurable context budgets
- Citation-ready source metadata
- Context construction for LLM prompts
- Optional contextual enrichment callback

This module does not generate embeddings or invoke LLMs.
Retrieval adapters provide the ranked candidates.

Security:
- Every request requires company_id.
- Permission checks can be injected.
- Document content is untrusted evidence.
- Source text must never override system instructions.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import re

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence


class ContextualRetrievalError(Exception):
    """Base contextual retrieval exception."""


class ContextualRetrievalValidationError(ContextualRetrievalError):
    """Invalid retrieval input."""


@dataclass(slots=True)
class RetrievalChunk:
    chunk_id: str
    document_id: str
    company_id: str
    text: str

    score: float = 0.0
    source: str = "unknown"

    parent_id: str | None = None
    chunk_index: int | None = None

    title: str | None = None
    page: int | None = None
    section: str | None = None
    document_version: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextualEvidence:
    evidence_id: str
    chunk_id: str
    document_id: str
    company_id: str

    text: str
    score: float
    retrieval_source: str

    parent_id: str | None = None
    title: str | None = None
    page: int | None = None
    section: str | None = None
    document_version: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "company_id": self.company_id,
            "text": self.text,
            "score": self.score,
            "retrieval_source": self.retrieval_source,
            "parent_id": self.parent_id,
            "title": self.title,
            "page": self.page,
            "section": self.section,
            "document_version": self.document_version,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ContextualRetrievalResult:
    query: str
    company_id: str
    evidence: list[ContextualEvidence]
    context: str
    total_candidates: int
    selected_count: int
    context_units: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "company_id": self.company_id,
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "context": self.context,
            "total_candidates": self.total_candidates,
            "selected_count": self.selected_count,
            "context_units": self.context_units,
        }


@dataclass(slots=True)
class ContextualRetrievalConfig:
    top_k: int = 8
    candidate_k: int = 30

    max_context_units: int = 12000
    max_chunk_units: int = 2500

    expand_parents: bool = True
    expand_neighbors: bool = True
    neighbor_window: int = 1

    min_score: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "top_k",
            "candidate_k",
            "max_context_units",
            "max_chunk_units",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ContextualRetrievalValidationError(
                    f"{name} must be a positive integer"
                )

        if self.candidate_k < self.top_k:
            raise ContextualRetrievalValidationError(
                "candidate_k must be >= top_k"
            )

        if (
            isinstance(self.neighbor_window, bool)
            or not isinstance(self.neighbor_window, int)
            or self.neighbor_window < 0
        ):
            raise ContextualRetrievalValidationError(
                "neighbor_window must be >= 0"
            )

        if (
            self.min_score is not None
            and not math.isfinite(self.min_score)
        ):
            raise ContextualRetrievalValidationError(
                "min_score must be finite"
            )


class RetrievalProvider(Protocol):
    async def search(
        self,
        query: str,
        *,
        company_id: str,
        top_k: int,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievalChunk]:
        ...


class ChunkStore(Protocol):
    async def get_chunk(
        self,
        chunk_id: str,
        *,
        company_id: str,
    ) -> RetrievalChunk | None:
        ...

    async def get_neighbors(
        self,
        chunk: RetrievalChunk,
        *,
        company_id: str,
        window: int,
    ) -> Sequence[RetrievalChunk]:
        ...


PermissionChecker = Callable[
    [RetrievalChunk, str],
    bool | Awaitable[bool],
]

ContextEnricher = Callable[
    [str, RetrievalChunk],
    str | Awaitable[str],
]


class ContextualRetriever:
    """
    Retrieval orchestration with context expansion.

    Required:
        provider: ranked candidate retrieval provider

    Optional:
        chunk_store: parent/neighbor lookup
        permission_checker: document-level authorization
        context_enricher: contextual text generation

    The context_enricher is expected to return only
    grounded context derived from trusted document
    metadata and the supplied source chunk.
    """

    def __init__(
        self,
        provider: RetrievalProvider,
        *,
        chunk_store: ChunkStore | None = None,
        config: ContextualRetrievalConfig | None = None,
        permission_checker: PermissionChecker | None = None,
        context_enricher: ContextEnricher | None = None,
        tokenizer: Callable[[str], Sequence[Any]] | None = None,
    ) -> None:
        self.provider = provider
        self.chunk_store = chunk_store
        self.config = config or ContextualRetrievalConfig()

        self.permission_checker = permission_checker
        self.context_enricher = context_enricher
        self.tokenizer = tokenizer

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _require_text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ContextualRetrievalValidationError(
                f"{name} must be a nonempty string"
            )
        return value.strip()

    @staticmethod
    async def _resolve(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _measure(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text)
        return len(self.tokenizer(text))

    def _truncate(self, text: str, limit: int) -> str:
        if self._measure(text) <= limit:
            return text

        if self.tokenizer is None:
            return text[:limit].rstrip()

        # Tokenizer callback returns token units but may
        # not support decoding. Use a conservative
        # character-boundary search.
        low, high = 0, len(text)
        best = ""

        while low <= high:
            middle = (low + high) // 2
            candidate = text[:middle]

            if self._measure(candidate) <= limit:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1

        return best.rstrip()

    @staticmethod
    def _evidence_id(chunk: RetrievalChunk) -> str:
        raw = (
            f"{chunk.company_id}|"
            f"{chunk.document_id}|"
            f"{chunk.document_version or ''}|"
            f"{chunk.chunk_id}"
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:24]

    @staticmethod
    def _identity(chunk: RetrievalChunk) -> tuple[str, ...]:
        return (
            chunk.company_id,
            chunk.document_id,
            chunk.document_version or "",
            chunk.chunk_id,
        )

    @staticmethod
    def _matches_filters(
        chunk: RetrievalChunk,
        filters: Mapping[str, Any] | None,
    ) -> bool:
        if not filters:
            return True

        for key, expected in filters.items():
            actual = chunk.metadata.get(key)

            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False

        return True

    async def _authorized(
        self,
        chunk: RetrievalChunk,
        company_id: str,
        filters: Mapping[str, Any] | None,
    ) -> bool:
        if chunk.company_id != company_id:
            return False

        if not self._matches_filters(chunk, filters):
            return False

        if self.permission_checker is None:
            return True

        return bool(
            await self._resolve(
                self.permission_checker(
                    chunk,
                    company_id,
                )
            )
        )

    @staticmethod
    def _validate_chunk(chunk: RetrievalChunk) -> None:
        if not isinstance(chunk, RetrievalChunk):
            raise ContextualRetrievalValidationError(
                "Provider must return RetrievalChunk objects"
            )

        if not math.isfinite(chunk.score):
            raise ContextualRetrievalValidationError(
                "Retrieval score must be finite"
            )

    # =====================================================
    # CONTEXT EXPANSION
    # =====================================================

    async def _expand(
        self,
        chunk: RetrievalChunk,
        *,
        company_id: str,
        metadata_filters: Mapping[str, Any] | None,
    ) -> list[RetrievalChunk]:
        expanded = [chunk]

        if self.chunk_store is None:
            return expanded

        if (
            self.config.expand_parents
            and chunk.parent_id
        ):
            parent = await self.chunk_store.get_chunk(
                chunk.parent_id,
                company_id=company_id,
            )

            if parent is not None:
                self._validate_chunk(parent)

                if await self._authorized(
                    parent,
                    company_id,
                    metadata_filters,
                ):
                    expanded.append(parent)

        if (
            self.config.expand_neighbors
            and self.config.neighbor_window > 0
        ):
            neighbors = await self.chunk_store.get_neighbors(
                chunk,
                company_id=company_id,
                window=self.config.neighbor_window,
            )

            for neighbor in neighbors:
                self._validate_chunk(neighbor)

                if await self._authorized(
                    neighbor,
                    company_id,
                    metadata_filters,
                ):
                    expanded.append(neighbor)

        return expanded

    # =====================================================
    # EVIDENCE CONVERSION
    # =====================================================

    async def _to_evidence(
        self,
        query: str,
        chunk: RetrievalChunk,
    ) -> ContextualEvidence:
        text = chunk.text

        if self.context_enricher is not None:
            enriched = await self._resolve(
                self.context_enricher(query, chunk)
            )

            if not isinstance(enriched, str):
                raise ContextualRetrievalValidationError(
                    "Context enricher must return a string"
                )

            if enriched.strip():
                text = (
                    f"Contextual note: {enriched.strip()}\n"
                    f"Original evidence:\n{chunk.text}"
                )

        text = self._truncate(
            text,
            self.config.max_chunk_units,
        )

        return ContextualEvidence(
            evidence_id=self._evidence_id(chunk),
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            company_id=chunk.company_id,
            text=text,
            score=chunk.score,
            retrieval_source=chunk.source,
            parent_id=chunk.parent_id,
            title=chunk.title,
            page=chunk.page,
            section=chunk.section,
            document_version=chunk.document_version,
            metadata=dict(chunk.metadata),
        )

    # =====================================================
    # CONTEXT FORMATTING
    # =====================================================

    @staticmethod
    def _format_evidence(
        evidence: ContextualEvidence,
        number: int,
    ) -> str:
        title = (
            evidence.title
            or evidence.document_id
        )

        location = []

        if evidence.page is not None:
            location.append(
                f"Page: {evidence.page}"
            )

        if evidence.section:
            location.append(
                f"Section: {evidence.section}"
            )

        location_text = (
            " | ".join(location)
            if location
            else "Location: unspecified"
        )

        return (
            f"[{number}] {title}\n"
            f"Document ID: {evidence.document_id}\n"
            f"Chunk ID: {evidence.chunk_id}\n"
            f"{location_text}\n"
            f"Evidence:\n{evidence.text}"
        )

    # =====================================================
    # MAIN RETRIEVAL
    # =====================================================

    async def retrieve(
        self,
        query: str,
        *,
        company_id: str,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> ContextualRetrievalResult:
        query = self._require_text(query, "query")
        company_id = self._require_text(
            company_id, "company_id"
        )

        candidates = await self.provider.search(
            query,
            company_id=company_id,
            top_k=self.config.candidate_k,
            metadata_filters=metadata_filters,
        )

        ranked: list[RetrievalChunk] = []

        for candidate in candidates:
            self._validate_chunk(candidate)

            if self.config.min_score is not None:
                if candidate.score < self.config.min_score:
                    continue

            if await self._authorized(
                candidate,
                company_id,
                metadata_filters,
            ):
                ranked.append(candidate)

        ranked.sort(
            key=lambda item: (
                -item.score,
                item.document_id,
                item.chunk_id,
            )
        )

        expanded: list[RetrievalChunk] = []
        seen: set[tuple[str, ...]] = set()

        for candidate in ranked[:self.config.top_k]:
            related = await self._expand(
                candidate,
                company_id=company_id,
                metadata_filters=metadata_filters,
            )

            for chunk in related:
                key = self._identity(chunk)

                if key in seen:
                    continue

                seen.add(key)
                expanded.append(chunk)

        evidence: list[ContextualEvidence] = []
        context_parts: list[str] = []

        used_units = 0

        for chunk in expanded:
            item = await self._to_evidence(
                query,
                chunk,
            )

            number = len(evidence) + 1

            formatted = self._format_evidence(
                item,
                number,
            )

            size = self._measure(formatted)

            remaining = (
                self.config.max_context_units
                - used_units
            )

            if remaining <= 0:
                break

            if size > remaining:
                # Keep evidence atomic so citations never
                # refer to a partially formatted source.
                continue

            evidence.append(item)
            context_parts.append(formatted)
            used_units += size

        return ContextualRetrievalResult(
            query=query,
            company_id=company_id,
            evidence=evidence,
            context="\n\n".join(context_parts),
            total_candidates=len(candidates),
            selected_count=len(evidence),
            context_units=used_units,
        )


# =========================================================
# IN-MEMORY ADAPTER FOR DEVELOPMENT AND TESTS
# =========================================================

class InMemoryChunkStore:
    def __init__(
        self,
        chunks: Sequence[RetrievalChunk],
    ) -> None:
        self._chunks = {
            (
                chunk.company_id,
                chunk.chunk_id,
            ): chunk
            for chunk in chunks
        }

    async def get_chunk(
        self,
        chunk_id: str,
        *,
        company_id: str,
    ) -> RetrievalChunk | None:
        return self._chunks.get(
            (company_id, chunk_id)
        )

    async def get_neighbors(
        self,
        chunk: RetrievalChunk,
        *,
        company_id: str,
        window: int,
    ) -> list[RetrievalChunk]:
        if chunk.chunk_index is None:
            return []

        results = []

        for candidate in self._chunks.values():
            if candidate.company_id != company_id:
                continue

            if candidate.document_id != chunk.document_id:
                continue

            if (
                candidate.document_version
                != chunk.document_version
            ):
                continue

            if candidate.chunk_index is None:
                continue

            distance = abs(
                candidate.chunk_index
                - chunk.chunk_index
            )

            if 0 < distance <= window:
                results.append(candidate)

        results.sort(
            key=lambda item: item.chunk_index
        )

        return results


class InMemoryRetrievalProvider:
    """
    Deterministic lexical provider for local testing.

    Replace with your production hybrid retrieval
    provider for BM25 + vector search + reranking.
    """

    TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

    def __init__(
        self,
        chunks: Sequence[RetrievalChunk],
    ) -> None:
        self.chunks = list(chunks)

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return set(
            cls.TOKEN_PATTERN.findall(
                text.lower()
            )
        )

    async def search(
        self,
        query: str,
        *,
        company_id: str,
        top_k: int,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievalChunk]:
        query_tokens = self._tokens(query)

        results = []

        for chunk in self.chunks:
            if chunk.company_id != company_id:
                continue

            if not ContextualRetriever._matches_filters(
                chunk,
                metadata_filters,
            ):
                continue

            document_tokens = self._tokens(
                chunk.text
            )

            score = float(
                len(
                    query_tokens
                    & document_tokens
                )
            )

            if score <= 0:
                continue

            results.append(
                RetrievalChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    company_id=chunk.company_id,
                    text=chunk.text,
                    score=score,
                    source="lexical_demo",
                    parent_id=chunk.parent_id,
                    chunk_index=chunk.chunk_index,
                    title=chunk.title,
                    page=chunk.page,
                    section=chunk.section,
                    document_version=chunk.document_version,
                    metadata=dict(chunk.metadata),
                )
            )

        results.sort(
            key=lambda item: -item.score
        )

        return results[:top_k]


# =========================================================
# EXAMPLE
# =========================================================

async def example() -> None:
    chunks = [
        RetrievalChunk(
            chunk_id="motor-parent-001",
            document_id="motor-manual",
            company_id="company-001",
            text=(
                "Motor bearing maintenance procedures "
                "include inspection of lubrication, "
                "alignment and operating temperature."
            ),
            chunk_index=0,
            title="Motor Maintenance Manual",
            page=12,
        ),
        RetrievalChunk(
            chunk_id="motor-child-001",
            document_id="motor-manual",
            company_id="company-001",
            text=(
                "Motor bearing overheating can be caused "
                "by insufficient lubrication."
            ),
            parent_id="motor-parent-001",
            chunk_index=1,
            title="Motor Maintenance Manual",
            page=12,
        ),
        RetrievalChunk(
            chunk_id="motor-child-002",
            document_id="motor-manual",
            company_id="company-001",
            text=(
                "Mechanical misalignment may increase "
                "vibration and bearing temperature."
            ),
            parent_id="motor-parent-001",
            chunk_index=2,
            title="Motor Maintenance Manual",
            page=13,
        ),
    ]

    provider = InMemoryRetrievalProvider(chunks)
    store = InMemoryChunkStore(chunks)

    retriever = ContextualRetriever(
        provider,
        chunk_store=store,
        config=ContextualRetrievalConfig(
            top_k=2,
            candidate_k=5,
            max_context_units=5000,
            expand_parents=True,
            expand_neighbors=True,
        ),
    )

    result = await retriever.retrieve(
        "Why is the motor bearing overheating?",
        company_id="company-001",
    )

    print(result.context)
    print(result.to_dict())


if __name__ == "__main__":
    import asyncio

    asyncio.run(example())
