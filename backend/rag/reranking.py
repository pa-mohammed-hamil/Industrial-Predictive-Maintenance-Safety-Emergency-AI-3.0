
"""
app/ai/rag/retrieval/reranking.py

Industrial RAG Reranking Engine
==============================

Python 3.11+

Features:
- Hugging Face CrossEncoder reranking
- Optional Cohere reranking
- Deterministic lexical fallback
- Batch processing
- Async support
- Score normalization
- Equipment and fault-code relevance boosting
- Tenant isolation
- Metadata filtering
- Model loading on demand
- Compatible with HybridRetriever.reranker

Dependencies:
    pip install sentence-transformers
    pip install cohere  # optional

Security:
    Document authorization must occur before
    passing text to external reranking providers.

Note:
    Lexical fallback is a retrieval heuristic,
    not an ML relevance model.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import threading

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class RerankingError(Exception):
    """Base reranking error."""


class RerankingValidationError(RerankingError):
    """Invalid reranking request."""


class RerankingProviderError(RerankingError):
    """Reranking provider failed."""


# ============================================================
# ENUMS
# ============================================================

class RerankingProvider(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    LEXICAL = "lexical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class RerankingConfig:
    provider: RerankingProvider = (
        RerankingProvider.CROSS_ENCODER
    )

    model_name: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    batch_size: int = 16
    max_documents: int = 100

    normalize_scores: bool = True

    enable_identifier_boost: bool = True
    identifier_boost: float = 0.10

    device: str | None = None

    cohere_api_key: str | None = None
    cohere_model: str = "rerank-v3.5"

    def __post_init__(self) -> None:
        if isinstance(self.provider, str):
            self.provider = RerankingProvider(
                self.provider
            )

        if self.batch_size <= 0:
            raise RerankingValidationError(
                "batch_size must be positive"
            )

        if self.max_documents <= 0:
            raise RerankingValidationError(
                "max_documents must be positive"
            )

        if not 0 <= self.identifier_boost <= 1:
            raise RerankingValidationError(
                "identifier_boost must be between 0 and 1"
            )


# ============================================================
# DOCUMENT SCHEMA
# ============================================================

@dataclass(slots=True)
class RerankDocument:
    chunk_id: str
    document_id: str
    company_id: str
    text: str

    retrieval_score: float = 0.0

    equipment_id: str | None = None
    asset_id: str | None = None
    site_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class RerankResult:
    chunk_id: str
    document_id: str
    company_id: str
    text: str

    score: float
    rank: int
    retrieval_score: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "company_id": self.company_id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "retrieval_score": self.retrieval_score,
            "metadata": dict(self.metadata),
        }


# ============================================================
# INDUSTRIAL IDENTIFIER EXTRACTOR
# ============================================================

class IndustrialIdentifierExtractor:
    """
    Extract likely industrial identifiers.

    Examples:
        PUMP-201
        MOTOR-101
        PLC-12
        E104
        F-203
    """

    PATTERN = re.compile(
        r"\b(?:[A-Z]{2,12}[-_]\d{1,8}|"
        r"[A-Z]\d{2,6})\b",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> set[str]:
        return {
            match.upper()
            for match in self.PATTERN.findall(text)
        }


# ============================================================
# LEXICAL RERANKER
# ============================================================

class LexicalReranker:
    """
    Deterministic fallback relevance scorer.
    """

    TOKEN_PATTERN = re.compile(
        r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*"
    )

    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        query_tokens = set(
            token.lower()
            for token in self.TOKEN_PATTERN.findall(
                query
            )
        )

        document_tokens = set(
            token.lower()
            for token in self.TOKEN_PATTERN.findall(
                document
            )
        )

        if not query_tokens:
            return 0.0

        overlap = (
            query_tokens & document_tokens
        )

        return (
            len(overlap)
            / len(query_tokens)
        )


# ============================================================
# MAIN RERANKING SERVICE
# ============================================================

class IndustrialReranker:
    """
    Industrial document reranking service.

    Methods:
        rerank(query, documents)
            Compatible with HybridRetriever.

        rerank_documents(...)
            Returns structured results.
    """

    def __init__(
        self,
        config: RerankingConfig | None = None,
    ) -> None:
        self.config = config or RerankingConfig()

        self._model: Any = None
        self._cohere_client: Any = None

        self._load_lock = threading.RLock()

        self.identifier_extractor = (
            IndustrialIdentifierExtractor()
        )

        self.lexical = LexicalReranker()

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_cross_encoder(self) -> Any:
        with self._load_lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import (
                    CrossEncoder,
                )
            except ImportError as exc:
                raise RerankingProviderError(
                    "Install sentence-transformers"
                ) from exc

            try:
                self._model = CrossEncoder(
                    self.config.model_name,
                    device=self.config.device,
                )
            except Exception as exc:
                raise RerankingProviderError(
                    "CrossEncoder loading failed"
                ) from exc

            return self._model

    def _load_cohere_client(self) -> Any:
        with self._load_lock:
            if self._cohere_client is not None:
                return self._cohere_client

            try:
                import cohere
            except ImportError as exc:
                raise RerankingProviderError(
                    "Install cohere"
                ) from exc

            api_key = (
                self.config.cohere_api_key
                or os.getenv("COHERE_API_KEY")
            )

            if not api_key:
                raise RerankingProviderError(
                    "COHERE_API_KEY is required"
                )

            self._cohere_client = cohere.ClientV2(
                api_key=api_key
            )

            return self._cohere_client

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_query(query: str) -> None:
        if (
            not isinstance(query, str)
            or not query.strip()
        ):
            raise RerankingValidationError(
                "query must be nonempty"
            )

    @staticmethod
    def _validate_documents(
        documents: Sequence[str],
    ) -> None:
        if isinstance(documents, (str, bytes)):
            raise RerankingValidationError(
                "documents must be a sequence of strings"
            )

        for document in documents:
            if not isinstance(document, str):
                raise RerankingValidationError(
                    "All documents must be strings"
                )

    # ========================================================
    # CROSS ENCODER
    # ========================================================

    def _cross_encoder_scores(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        model = self._load_cross_encoder()

        pairs = [
            (query, document)
            for document in documents
        ]

        scores: list[float] = []

        for start in range(
            0,
            len(pairs),
            self.config.batch_size,
        ):
            batch = pairs[
                start:start + self.config.batch_size
            ]

            output = model.predict(
                batch,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            scores.extend(
                float(value)
                for value in output
            )

        return scores

    # ========================================================
    # COHERE
    # ========================================================

    def _cohere_scores(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        client = self._load_cohere_client()

        response = client.rerank(
            model=self.config.cohere_model,
            query=query,
            documents=list(documents),
            top_n=len(documents),
        )

        scores = [
            0.0
            for _ in documents
        ]

        for result in response.results:
            index = int(result.index)

            if not 0 <= index < len(scores):
                raise RerankingProviderError(
                    "Invalid Cohere result index"
                )

            scores[index] = float(
                result.relevance_score
            )

        return scores

    # ========================================================
    # LEXICAL
    # ========================================================

    def _lexical_scores(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        return [
            self.lexical.score(
                query,
                document,
            )
            for document in documents
        ]

    # ========================================================
    # SCORE NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize(
        scores: Sequence[float],
    ) -> list[float]:
        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if math.isclose(minimum, maximum):
            return [0.5 for _ in scores]

        return [
            (score - minimum)
            / (maximum - minimum)
            for score in scores
        ]

    # ========================================================
    # IDENTIFIER BOOST
    # ========================================================

    def _apply_identifier_boost(
        self,
        query: str,
        documents: Sequence[str],
        scores: Sequence[float],
    ) -> list[float]:
        identifiers = (
            self.identifier_extractor.extract(
                query
            )
        )

        if not identifiers:
            return list(scores)

        boosted = []

        for document, score in zip(
            documents,
            scores,
        ):
            document_ids = (
                self.identifier_extractor.extract(
                    document
                )
            )

            overlap = (
                identifiers & document_ids
            )

            boost = (
                self.config.identifier_boost
                * len(overlap)
                / len(identifiers)
            )

            boosted.append(
                score + boost
            )

        return boosted

    # ========================================================
    # SCORE GENERATION
    # ========================================================

    def _score_documents(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        self._validate_query(query)
        self._validate_documents(documents)

        if not documents:
            return []

        if len(documents) > self.config.max_documents:
            raise RerankingValidationError(
                "Too many documents for reranking"
            )

        if (
            self.config.provider
            == RerankingProvider.CROSS_ENCODER
        ):
            scores = self._cross_encoder_scores(
                query,
                documents,
            )

        elif (
            self.config.provider
            == RerankingProvider.COHERE
        ):
            scores = self._cohere_scores(
                query,
                documents,
            )

        else:
            scores = self._lexical_scores(
                query,
                documents,
            )

        if len(scores) != len(documents):
            raise RerankingProviderError(
                "Provider returned incorrect score count"
            )

        if not all(
            math.isfinite(score)
            for score in scores
        ):
            raise RerankingProviderError(
                "Provider returned nonfinite scores"
            )

        if self.config.normalize_scores:
            scores = self._normalize(
                scores
            )

        if self.config.enable_identifier_boost:
            scores = self._apply_identifier_boost(
                query,
                documents,
                scores,
            )

        return scores

    # ========================================================
    # HYBRID RETRIEVER INTERFACE
    # ========================================================

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        """
        Compatible with:

            HybridRetriever(
                reranker=IndustrialReranker(...)
            )

        Returns one score per input document.
        """
        return await asyncio.to_thread(
            self._score_documents,
            query,
            list(documents),
        )

    # ========================================================
    # STRUCTURED RERANKING
    # ========================================================

    async def rerank_documents(
        self,
        query: str,
        documents: Sequence[RerankDocument],
        *,
        company_id: str,
        top_k: int | None = None,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> list[RerankResult]:
        self._validate_query(query)

        if not company_id or not company_id.strip():
            raise RerankingValidationError(
                "company_id is required"
            )

        filtered: list[RerankDocument] = []

        for document in documents:
            if document.company_id != company_id:
                continue

            if metadata_filters:
                matched = True

                for key, expected in (
                    metadata_filters.items()
                ):
                    if hasattr(document, key):
                        actual = getattr(
                            document,
                            key,
                        )
                    else:
                        actual = (
                            document.metadata.get(key)
                        )

                    if actual != expected:
                        matched = False
                        break

                if not matched:
                    continue

            filtered.append(document)

        if not filtered:
            return []

        scores = await self.rerank(
            query,
            [
                document.text
                for document in filtered
            ],
        )

        ranked = sorted(
            zip(filtered, scores),
            key=lambda item: (
                -item[1],
                item[0].chunk_id,
            ),
        )

        if top_k is not None:
            if (
                isinstance(top_k, bool)
                or not isinstance(top_k, int)
                or top_k <= 0
            ):
                raise RerankingValidationError(
                    "top_k must be positive"
                )

            ranked = ranked[:top_k]

        return [
            RerankResult(
                chunk_id=document.chunk_id,
                document_id=document.document_id,
                company_id=document.company_id,
                text=document.text,
                score=score,
                rank=rank,
                retrieval_score=(
                    document.retrieval_score
                ),
                metadata=dict(document.metadata),
            )
            for rank, (document, score)
            in enumerate(ranked, start=1)
        ]


# ============================================================
# EXAMPLE
# ============================================================

async def example() -> None:
    reranker = IndustrialReranker(
        RerankingConfig(
            provider=RerankingProvider.LEXICAL,
            normalize_scores=True,
        )
    )

    documents = [
        RerankDocument(
            chunk_id="chunk-001",
            document_id="motor-manual",
            company_id="company-001",
            text=(
                "MOTOR-101 bearing overheating may "
                "be caused by poor lubrication."
            ),
            equipment_id="MOTOR-101",
        ),
        RerankDocument(
            chunk_id="chunk-002",
            document_id="pump-manual",
            company_id="company-001",
            text=(
                "PUMP-201 vibration may indicate "
                "cavitation or rotor imbalance."
            ),
            equipment_id="PUMP-201",
        ),
    ]

    results = await reranker.rerank_documents(
        "MOTOR-101 bearing overheating",
        documents,
        company_id="company-001",
        top_k=2,
    )

    for result in results:
        print(result.to_dict())


if __name__ == "__main__":
    asyncio.run(example())
