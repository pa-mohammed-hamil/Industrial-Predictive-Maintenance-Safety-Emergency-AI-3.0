
"""
app/ai/rag/embeddings.py

Industrial RAG Embedding Service
================================

Python 3.11+

Install:
    pip install sentence-transformers numpy

Optional OpenAI:
    pip install openai

Features:
- Sentence Transformers embeddings
- OpenAI embeddings
- Async embedding generation
- Batch processing
- Query and document embeddings
- L2 vector normalization
- Cosine similarity
- In-memory LRU cache
- Model dimension validation
- Embedding metadata
- Retrieval integration
- Industrial terminology support
- Model health information

Security:
- Do not send confidential industrial documents to
  external APIs without authorization.
- Keep API keys in environment variables.
- Apply tenant/document authorization before retrieval.
- Cache isolation should be considered for sensitive data.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import threading

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence


class EmbeddingError(Exception):
    """Base embedding exception."""


class EmbeddingValidationError(EmbeddingError):
    """Invalid embedding input."""


class EmbeddingProviderError(EmbeddingError):
    """Embedding provider failure."""


class EmbeddingProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


@dataclass(slots=True)
class EmbeddingConfig:
    provider: EmbeddingProvider = (
        EmbeddingProvider.SENTENCE_TRANSFORMERS
    )

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    cache_max_entries: int = 10000

    device: str | None = None
    expected_dimensions: int | None = None

    openai_api_key: str | None = None
    openai_base_url: str | None = None

    max_input_chars: int = 20000

    def __post_init__(self) -> None:
        if isinstance(self.provider, str):
            self.provider = EmbeddingProvider(
                self.provider
            )

        for name in (
            "batch_size",
            "cache_max_entries",
            "max_input_chars",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise EmbeddingValidationError(
                    f"{name} must be a positive integer"
                )

        if (
            self.expected_dimensions is not None
            and self.expected_dimensions <= 0
        ):
            raise EmbeddingValidationError(
                "expected_dimensions must be positive"
            )


@dataclass(slots=True)
class EmbeddingResult:
    text: str
    vector: list[float]
    model_name: str
    provider: str
    dimensions: int
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "vector": self.vector,
            "model_name": self.model_name,
            "provider": self.provider,
            "dimensions": self.dimensions,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


class EmbeddingCache:
    """Thread-safe bounded LRU embedding cache."""

    def __init__(
        self,
        max_entries: int = 10000,
    ) -> None:
        self.max_entries = max_entries
        self._cache: OrderedDict[
            str, list[float]
        ] = OrderedDict()

        self._lock = threading.RLock()

    def get(
        self,
        key: str,
    ) -> list[float] | None:
        with self._lock:
            vector = self._cache.get(key)

            if vector is None:
                return None

            self._cache.move_to_end(key)
            return list(vector)

    def set(
        self,
        key: str,
        vector: Sequence[float],
    ) -> None:
        with self._lock:
            self._cache[key] = list(vector)
            self._cache.move_to_end(key)

            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class EmbeddingService:
    """
    Embedding service with pluggable providers.

    Sentence Transformers:
        Local model inference.

    OpenAI:
        Remote embedding API.

    All vectors are returned as list[float].
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()

        self._model: Any = None
        self._client: Any = None

        self._model_lock = threading.RLock()

        self.cache = EmbeddingCache(
            self.config.cache_max_entries
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingValidationError(
                "Embedding text must be nonempty"
            )

        if len(text) > self.config.max_input_chars:
            raise EmbeddingValidationError(
                "Embedding text exceeds configured limit"
            )

        return text.strip()

    @staticmethod
    def _validate_vector(
        vector: Sequence[float],
    ) -> list[float]:
        values = [
            float(value)
            for value in vector
        ]

        if not values:
            raise EmbeddingValidationError(
                "Embedding vector cannot be empty"
            )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise EmbeddingValidationError(
                "Embedding contains nonfinite values"
            )

        return values

    # =====================================================
    # CACHE KEY
    # =====================================================

    def _cache_key(
        self,
        text: str,
        *,
        company_id: str | None,
        kind: str,
    ) -> str:
        raw = (
            f"{self.config.provider.value}|"
            f"{self.config.model_name}|"
            f"{self.config.normalize}|"
            f"{company_id or ''}|"
            f"{kind}|{text}"
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    # =====================================================
    # NORMALIZATION
    # =====================================================

    @staticmethod
    def normalize_vector(
        vector: Sequence[float],
    ) -> list[float]:
        values = EmbeddingService._validate_vector(
            vector
        )

        magnitude = math.sqrt(
            sum(value * value for value in values)
        )

        if magnitude == 0:
            raise EmbeddingValidationError(
                "Cannot normalize zero vector"
            )

        return [
            value / magnitude
            for value in values
        ]

    # =====================================================
    # MODEL LOADING
    # =====================================================

    def _load_sentence_transformer(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )
            except ImportError as exc:
                raise EmbeddingProviderError(
                    "Install sentence-transformers"
                ) from exc

            try:
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                )
            except Exception as exc:
                raise EmbeddingProviderError(
                    "Failed to load embedding model"
                ) from exc

            return self._model

    def _load_openai_client(self) -> Any:
        with self._model_lock:
            if self._client is not None:
                return self._client

            try:
                from openai import OpenAI
            except ImportError as exc:
                raise EmbeddingProviderError(
                    "Install openai"
                ) from exc

            api_key = (
                self.config.openai_api_key
                or os.getenv("OPENAI_API_KEY")
            )

            if not api_key:
                raise EmbeddingProviderError(
                    "OPENAI_API_KEY is required"
                )

            kwargs = {
                "api_key": api_key,
            }

            if self.config.openai_base_url:
                kwargs["base_url"] = (
                    self.config.openai_base_url
                )

            self._client = OpenAI(**kwargs)
            return self._client

    # =====================================================
    # PROVIDER INFERENCE
    # =====================================================

    def _encode_sentence_transformers(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        model = self._load_sentence_transformer()

        try:
            vectors = model.encode(
                list(texts),
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

            return [
                self._validate_vector(vector)
                for vector in vectors
            ]

        except Exception as exc:
            raise EmbeddingProviderError(
                "Sentence Transformer inference failed"
            ) from exc

    def _encode_openai(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        client = self._load_openai_client()

        try:
            response = client.embeddings.create(
                model=self.config.model_name,
                input=list(texts),
            )

            ordered = sorted(
                response.data,
                key=lambda item: item.index,
            )

            return [
                self._validate_vector(
                    item.embedding
                )
                for item in ordered
            ]

        except Exception as exc:
            raise EmbeddingProviderError(
                "Remote embedding request failed"
            ) from exc

    def _encode(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        if (
            self.config.provider
            == EmbeddingProvider.SENTENCE_TRANSFORMERS
        ):
            vectors = self._encode_sentence_transformers(
                texts
            )
        elif (
            self.config.provider
            == EmbeddingProvider.OPENAI
        ):
            vectors = self._encode_openai(
                texts
            )
        else:
            raise EmbeddingProviderError(
                "Unsupported embedding provider"
            )

        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                "Provider returned incorrect vector count"
            )

        results = []

        for vector in vectors:
            if self.config.normalize:
                vector = self.normalize_vector(
                    vector
                )

            if (
                self.config.expected_dimensions is not None
                and len(vector)
                != self.config.expected_dimensions
            ):
                raise EmbeddingValidationError(
                    "Unexpected embedding dimensions"
                )

            results.append(vector)

        return results

    # =====================================================
    # EMBED TEXTS
    # =====================================================

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        company_id: str | None = None,
        kind: str = "document",
    ) -> list[list[float]]:
        if kind not in {"document", "query"}:
            raise EmbeddingValidationError(
                "kind must be document or query"
            )

        validated = [
            self._validate_text(text)
            for text in texts
        ]

        if not validated:
            return []

        results: list[list[float] | None] = [
            None
        ] * len(validated)

        missing: dict[
            str, tuple[str, list[int]]
        ] = {}

        for index, text in enumerate(validated):
            key = self._cache_key(
                text,
                company_id=company_id,
                kind=kind,
            )

            cached = (
                self.cache.get(key)
                if self.config.cache_enabled
                else None
            )

            if cached is not None:
                results[index] = cached
            elif key in missing:
                missing[key][1].append(index)
            else:
                missing[key] = (
                    text,
                    [index],
                )

        pending = list(missing.items())

        for start in range(
            0,
            len(pending),
            self.config.batch_size,
        ):
            batch = pending[
                start:start + self.config.batch_size
            ]

            batch_texts = [
                item[1][0]
                for item in batch
            ]

            vectors = self._encode(batch_texts)

            for (
                key,
                (_, indices),
            ), vector in zip(batch, vectors):
                if self.config.cache_enabled:
                    self.cache.set(
                        key,
                        vector,
                    )

                for index in indices:
                    results[index] = list(vector)

        if any(
            vector is None
            for vector in results
        ):
            raise EmbeddingProviderError(
                "Some embeddings were not generated"
            )

        return [
            list(vector)
            for vector in results
            if vector is not None
        ]

    # =====================================================
    # QUERY EMBEDDING
    # =====================================================

    def embed_query(
        self,
        query: str,
        *,
        company_id: str | None = None,
    ) -> list[float]:
        return self.embed_texts(
            [query],
            company_id=company_id,
            kind="query",
        )[0]

    # =====================================================
    # DOCUMENT EMBEDDINGS
    # =====================================================

    def embed_documents(
        self,
        documents: Sequence[str],
        *,
        company_id: str | None = None,
    ) -> list[list[float]]:
        return self.embed_texts(
            documents,
            company_id=company_id,
            kind="document",
        )

    # =====================================================
    # ASYNC METHODS
    # =====================================================

    async def aembed_query(
        self,
        query: str,
        *,
        company_id: str | None = None,
    ) -> list[float]:
        return await asyncio.to_thread(
            self.embed_query,
            query,
            company_id=company_id,
        )

    async def aembed_documents(
        self,
        documents: Sequence[str],
        *,
        company_id: str | None = None,
    ) -> list[list[float]]:
        return await asyncio.to_thread(
            self.embed_documents,
            documents,
            company_id=company_id,
        )

    # =====================================================
    # EMBEDDING RESULT
    # =====================================================

    def embed_with_metadata(
        self,
        text: str,
        *,
        company_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        vector = self.embed_query(
            text,
            company_id=company_id,
        )

        return EmbeddingResult(
            text=text,
            vector=vector,
            model_name=self.config.model_name,
            provider=self.config.provider.value,
            dimensions=len(vector),
            created_at=datetime.now(
                timezone.utc
            ),
            metadata=dict(metadata or {}),
        )

    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    @staticmethod
    def cosine_similarity(
        vector_a: Sequence[float],
        vector_b: Sequence[float],
    ) -> float:
        a = EmbeddingService._validate_vector(
            vector_a
        )
        b = EmbeddingService._validate_vector(
            vector_b
        )

        if len(a) != len(b):
            raise EmbeddingValidationError(
                "Vector dimensions do not match"
            )

        dot = sum(
            x * y
            for x, y in zip(a, b)
        )

        norm_a = math.sqrt(
            sum(x * x for x in a)
        )

        norm_b = math.sqrt(
            sum(y * y for y in b)
        )

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    def get_model_info(self) -> dict[str, Any]:
        dimensions = self.config.expected_dimensions

        if (
            dimensions is None
            and self._model is not None
        ):
            dimensions = (
                self._model
                .get_sentence_embedding_dimension()
            )

        return {
            "provider": self.config.provider.value,
            "model_name": self.config.model_name,
            "dimensions": dimensions,
            "normalize": self.config.normalize,
            "batch_size": self.config.batch_size,
            "cache_enabled": self.config.cache_enabled,
            "cache_entries": self.cache.size(),
            "local_model_loaded": (
                self._model is not None
            ),
        }

    # =====================================================
    # CACHE MANAGEMENT
    # =====================================================

    def clear_cache(self) -> None:
        self.cache.clear()


# =========================================================
# EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":
    config = EmbeddingConfig(
        provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        model_name=(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        batch_size=16,
        normalize=True,
        expected_dimensions=384,
    )

    service = EmbeddingService(config)

    documents = [
        (
            "Motor bearing overheating can result "
            "from insufficient lubrication."
        ),
        (
            "High vibration in industrial pumps "
            "may indicate cavitation or imbalance."
        ),
        (
            "Electrical maintenance requires "
            "approved isolation procedures."
        ),
    ]

    vectors = service.embed_documents(
        documents,
        company_id="company-001",
    )

    query_vector = service.embed_query(
        "Why is the motor bearing overheating?",
        company_id="company-001",
    )

    similarities = [
        service.cosine_similarity(
            query_vector,
            vector,
        )
        for vector in vectors
    ]

    for text, score in sorted(
        zip(documents, similarities),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            round(score, 4),
            text,
        )

    print(service.get_model_info())
