
"""
app/ai/rag/retrieval/bm25.py

Industrial BM25 Retrieval Engine
================================

Features:
- BM25Okapi scoring
- Industrial document indexing
- Tokenization and normalization
- Stopword removal
- Configurable k1 and b
- Metadata filtering
- Tenant-aware retrieval
- Top-K ranking
- Batch search
- Index updates and deletions
- JSON index persistence
- Hybrid retrieval integration

Dependencies:
    Python 3.11+
    Standard library only

This implementation uses an in-memory inverted index.
"""

from __future__ import annotations

import json
import math
import re
import threading

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class BM25Error(Exception):
    """Base BM25 exception."""


class BM25ValidationError(BM25Error):
    """Invalid retrieval input."""


@dataclass(slots=True)
class BM25Document:
    id: str
    text: str
    company_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BM25Result:
    document_id: str
    text: str
    score: float
    company_id: str
    metadata: dict[str, Any]
    rank: int


DEFAULT_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but",
    "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in",
    "on", "at", "for", "from", "by",
    "with", "as", "that", "this", "it",
    "its", "into", "about", "can",
    "could", "should", "would", "will",
    "may", "might", "do", "does",
    "did", "have", "has", "had",
})


class IndustrialTokenizer:
    """
    Tokenizer preserving useful industrial identifiers.

    Examples:
        "Motor M-101 vibration 12.5 mm/s"
        -> ["motor", "m-101", "vibration",
            "12.5", "mm/s"]

    Configure stopwords according to the corpus.
    """

    TOKEN_PATTERN = re.compile(
        r"[a-zA-Z0-9]+"
        r"(?:[._/-][a-zA-Z0-9]+)*"
    )

    def __init__(
        self,
        *,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        stopwords: Iterable[str] | None = None,
    ) -> None:
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(
            DEFAULT_STOPWORDS
            if stopwords is None
            else stopwords
        )

        if lowercase:
            self.stopwords = {
                word.lower() for word in self.stopwords
            }

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise BM25ValidationError(
                "Text must be a string"
            )

        if self.lowercase:
            text = text.lower()

        tokens = self.TOKEN_PATTERN.findall(text)

        if self.remove_stopwords:
            tokens = [
                token
                for token in tokens
                if token not in self.stopwords
            ]

        return tokens


class BM25Retriever:
    """
    Thread-safe in-memory BM25 retriever.

    Formula:
        IDF(q) = ln(1 + (N - df + 0.5) / (df + 0.5))

        BM25(q, d) =
            sum(
                IDF(q) *
                tf * (k1 + 1)
                / (
                    tf +
                    k1 * (1 - b + b * |d| / avgdl)
                )
            )

    Tenant isolation:
        Every search requires company_id.

    Notes:
        Index mutations and reads are protected by RLock.
        Search results are sorted by descending score.
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: IndustrialTokenizer | None = None,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise BM25ValidationError(
                "k1 must be positive and finite"
            )

        if not math.isfinite(b) or not 0 <= b <= 1:
            raise BM25ValidationError(
                "b must be between 0 and 1"
            )

        self.k1 = float(k1)
        self.b = float(b)

        self.tokenizer = (
            tokenizer or IndustrialTokenizer()
        )

        self.documents: dict[str, BM25Document] = {}
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.term_frequencies: dict[
            str, Counter[str]
        ] = {}

        self.inverted_index: dict[
            str, set[str]
        ] = defaultdict(set)

        self.document_frequency: Counter[str] = Counter()

        self.total_tokens = 0
        self.average_document_length = 0.0

        self._lock = threading.RLock()

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def _validate_text(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise BM25ValidationError(
                f"{name} must be a nonempty string"
            )

        return value.strip()

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise BM25ValidationError(
                "top_k must be a positive integer"
            )

    # =====================================================
    # INDEX MAINTENANCE
    # =====================================================

    def _refresh_statistics(self) -> None:
        count = len(self.documents)

        self.average_document_length = (
            self.total_tokens / count
            if count
            else 0.0
        )

    def _remove_unlocked(
        self,
        document_id: str,
    ) -> bool:
        if document_id not in self.documents:
            return False

        frequencies = self.term_frequencies.pop(
            document_id
        )

        for term in frequencies:
            postings = self.inverted_index[term]
            postings.discard(document_id)

            self.document_frequency[term] -= 1

            if not postings:
                del self.inverted_index[term]
                del self.document_frequency[term]

        self.total_tokens -= self.doc_lengths.pop(
            document_id
        )

        del self.doc_tokens[document_id]
        del self.documents[document_id]

        self._refresh_statistics()
        return True

    # =====================================================
    # ADD DOCUMENT
    # =====================================================

    def add_document(
        self,
        document: BM25Document,
    ) -> None:
        if not isinstance(document, BM25Document):
            raise BM25ValidationError(
                "Expected BM25Document"
            )

        self._validate_text(
            document.id, "document.id"
        )
        self._validate_text(
            document.company_id, "company_id"
        )

        tokens = self.tokenizer.tokenize(
            document.text
        )

        frequencies = Counter(tokens)

        with self._lock:
            self._remove_unlocked(document.id)

            self.documents[document.id] = BM25Document(
                id=document.id,
                text=document.text,
                company_id=document.company_id,
                metadata=dict(document.metadata),
            )

            self.doc_tokens[document.id] = tokens
            self.doc_lengths[document.id] = len(tokens)
            self.term_frequencies[
                document.id
            ] = frequencies

            self.total_tokens += len(tokens)

            for term in frequencies:
                self.inverted_index[term].add(
                    document.id
                )
                self.document_frequency[term] += 1

            self._refresh_statistics()

    # =====================================================
    # BATCH INDEXING
    # =====================================================

    def add_documents(
        self,
        documents: Sequence[BM25Document],
    ) -> int:
        count = 0

        for document in documents:
            self.add_document(document)
            count += 1

        return count

    # =====================================================
    # REMOVE DOCUMENT
    # =====================================================

    def remove_document(
        self,
        document_id: str,
        *,
        company_id: str,
    ) -> bool:
        self._validate_text(
            company_id, "company_id"
        )

        with self._lock:
            document = self.documents.get(
                document_id
            )

            if (
                document is None
                or document.company_id != company_id
            ):
                return False

            return self._remove_unlocked(
                document_id
            )

    # =====================================================
    # IDF
    # =====================================================

    def _idf(self, term: str) -> float:
        total = len(self.documents)

        if total == 0:
            return 0.0

        df = self.document_frequency.get(
            term, 0
        )

        return math.log1p(
            (total - df + 0.5)
            / (df + 0.5)
        )

    # =====================================================
    # DOCUMENT SCORE
    # =====================================================

    def _score_document(
        self,
        document_id: str,
        query_terms: Sequence[str],
    ) -> float:
        frequencies = self.term_frequencies[
            document_id
        ]

        length = self.doc_lengths[document_id]

        avgdl = (
            self.average_document_length
            if self.average_document_length > 0
            else 1.0
        )

        score = 0.0

        for term in set(query_terms):
            tf = frequencies.get(term, 0)

            if tf == 0:
                continue

            numerator = tf * (self.k1 + 1)

            denominator = (
                tf
                + self.k1
                * (
                    1
                    - self.b
                    + self.b * length / avgdl
                )
            )

            score += (
                self._idf(term)
                * numerator
                / denominator
            )

        return score

    # =====================================================
    # METADATA FILTERING
    # =====================================================

    @staticmethod
    def _matches_metadata(
        document: BM25Document,
        filters: Mapping[str, Any] | None,
    ) -> bool:
        if not filters:
            return True

        for key, expected in filters.items():
            actual = document.metadata.get(key)

            if isinstance(expected, (set, list, tuple)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False

        return True

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query: str,
        *,
        company_id: str,
        top_k: int = 10,
        metadata_filters: Mapping[
            str, Any
        ] | None = None,
        min_score: float = 0.0,
    ) -> list[BM25Result]:
        self._validate_text(query, "query")
        self._validate_text(
            company_id, "company_id"
        )
        self._validate_top_k(top_k)

        if not math.isfinite(min_score):
            raise BM25ValidationError(
                "min_score must be finite"
            )

        query_terms = self.tokenizer.tokenize(
            query
        )

        if not query_terms:
            return []

        with self._lock:
            candidate_ids: set[str] = set()

            for term in query_terms:
                candidate_ids.update(
                    self.inverted_index.get(
                        term, set()
                    )
                )

            scored: list[
                tuple[str, float]
            ] = []

            for document_id in candidate_ids:
                document = self.documents[
                    document_id
                ]

                if document.company_id != company_id:
                    continue

                if not self._matches_metadata(
                    document,
                    metadata_filters,
                ):
                    continue

                score = self._score_document(
                    document_id,
                    query_terms,
                )

                if score > 0 and score >= min_score:
                    scored.append(
                        (document_id, score)
                    )

            scored.sort(
                key=lambda item: (
                    -item[1],
                    item[0],
                )
            )

            results: list[BM25Result] = []

            for rank, (
                document_id,
                score,
            ) in enumerate(
                scored[:top_k],
                start=1,
            ):
                document = self.documents[
                    document_id
                ]

                results.append(
                    BM25Result(
                        document_id=document.id,
                        text=document.text,
                        score=score,
                        company_id=document.company_id,
                        metadata=dict(
                            document.metadata
                        ),
                        rank=rank,
                    )
                )

            return results

    # =====================================================
    # BATCH SEARCH
    # =====================================================

    def batch_search(
        self,
        queries: Sequence[str],
        *,
        company_id: str,
        top_k: int = 10,
        metadata_filters: Mapping[
            str, Any
        ] | None = None,
    ) -> list[list[BM25Result]]:
        return [
            self.search(
                query,
                company_id=company_id,
                top_k=top_k,
                metadata_filters=metadata_filters,
            )
            for query in queries
        ]

    # =====================================================
    # GET DOCUMENT
    # =====================================================

    def get_document(
        self,
        document_id: str,
        *,
        company_id: str,
    ) -> BM25Document | None:
        with self._lock:
            document = self.documents.get(
                document_id
            )

            if (
                document is None
                or document.company_id != company_id
            ):
                return None

            return BM25Document(
                id=document.id,
                text=document.text,
                company_id=document.company_id,
                metadata=dict(document.metadata),
            )

    # =====================================================
    # CLEAR INDEX
    # =====================================================

    def clear(self) -> None:
        with self._lock:
            self.documents.clear()
            self.doc_tokens.clear()
            self.doc_lengths.clear()
            self.term_frequencies.clear()
            self.inverted_index.clear()
            self.document_frequency.clear()

            self.total_tokens = 0
            self.average_document_length = 0.0

    # =====================================================
    # STATISTICS
    # =====================================================

    def get_statistics(
        self,
        *,
        company_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if company_id is None:
                count = len(self.documents)
                total_tokens = self.total_tokens
            else:
                self._validate_text(
                    company_id, "company_id"
                )

                ids = [
                    doc_id
                    for doc_id, document
                    in self.documents.items()
                    if document.company_id
                    == company_id
                ]

                count = len(ids)
                total_tokens = sum(
                    self.doc_lengths[doc_id]
                    for doc_id in ids
                )

            return {
                "document_count": count,
                "total_tokens": total_tokens,
                "average_document_length": (
                    total_tokens / count
                    if count
                    else 0.0
                ),
                "vocabulary_size": (
                    len(self.document_frequency)
                    if company_id is None
                    else len({
                        token
                        for doc_id in ids
                        for token in self.doc_tokens[doc_id]
                    })
                ),
                "k1": self.k1,
                "b": self.b,
            }

    # =====================================================
    # SAVE INDEX
    # =====================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save source documents and tokenizer settings.

        The inverted index is reconstructed on load.

        Use trusted, access-controlled storage.
        """
        path = Path(path)

        with self._lock:
            payload = {
                "version": 1,
                "k1": self.k1,
                "b": self.b,
                "tokenizer": {
                    "lowercase": self.tokenizer.lowercase,
                    "remove_stopwords":
                        self.tokenizer.remove_stopwords,
                    "stopwords": sorted(
                        self.tokenizer.stopwords
                    ),
                },
                "documents": [
                    {
                        "id": doc.id,
                        "text": doc.text,
                        "company_id": doc.company_id,
                        "metadata": doc.metadata,
                    }
                    for doc in self.documents.values()
                ],
            }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary.replace(path)

    # =====================================================
    # LOAD INDEX
    # =====================================================

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> BM25Retriever:
        path = Path(path)

        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        if payload.get("version") != 1:
            raise BM25ValidationError(
                "Unsupported BM25 index version"
            )

        tokenizer_config = payload["tokenizer"]

        tokenizer = IndustrialTokenizer(
            lowercase=tokenizer_config["lowercase"],
            remove_stopwords=tokenizer_config[
                "remove_stopwords"
            ],
            stopwords=tokenizer_config["stopwords"],
        )

        retriever = cls(
            k1=payload["k1"],
            b=payload["b"],
            tokenizer=tokenizer,
        )

        documents = [
            BM25Document(
                id=item["id"],
                text=item["text"],
                company_id=item["company_id"],
                metadata=item.get("metadata", {}),
            )
            for item in payload["documents"]
        ]

        retriever.add_documents(documents)

        return retriever


# =========================================================
# EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":
    retriever = BM25Retriever()

    retriever.add_documents([
        BM25Document(
            id="manual-001",
            company_id="company-001",
            text=(
                "Motor bearing overheating may result "
                "from insufficient lubrication, excessive "
                "load or shaft misalignment."
            ),
            metadata={
                "document_type": "maintenance_manual",
                "asset_id": "motor-101",
                "page": 12,
            },
        ),
        BM25Document(
            id="manual-002",
            company_id="company-001",
            text=(
                "High vibration in centrifugal pumps "
                "can indicate bearing wear, cavitation "
                "or mechanical imbalance."
            ),
            metadata={
                "document_type": "troubleshooting",
                "asset_id": "pump-202",
                "page": 8,
            },
        ),
        BM25Document(
            id="safety-001",
            company_id="company-001",
            text=(
                "Before servicing electrical machinery, "
                "follow approved lockout tagout "
                "procedures and verify isolation."
            ),
            metadata={
                "document_type": "safety_procedure",
                "page": 3,
            },
        ),
    ])

    results = retriever.search(
        "motor bearing overheating lubrication",
        company_id="company-001",
        top_k=3,
    )

    for result in results:
        print(
            result.rank,
            result.document_id,
            round(result.score, 4),
            result.text,
        )

    print(retriever.get_statistics())
