
"""
app/ai/rag/citations.py

Industrial RAG Citation Engine
==============================

Python 3.11+
Standard library only

Features:
- Source citation generation
- Document and page references
- Chunk-level evidence tracking
- Parent-child chunk support
- Tenant-scoped citation registry
- Deterministic citation numbering
- Source deduplication
- Citation formatting
- Evidence snippet extraction
- Answer citation parsing
- Citation reference validation
- Citation coverage metrics
- Source attribution reports

Important:
    Citation validity does not prove factual correctness.
    Evidence entailment requires separate groundedness
    evaluation or human verification.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class CitationError(Exception):
    """Base citation error."""


class CitationValidationError(CitationError):
    """Invalid citation input."""


@dataclass(slots=True)
class CitationSource:
    document_id: str
    chunk_id: str
    company_id: str
    text: str

    title: str | None = None
    source_path: str | None = None
    source_url: str | None = None

    page: int | None = None
    section: str | None = None

    start_char: int | None = None
    end_char: int | None = None

    parent_id: str | None = None
    document_version: str | None = None

    retrieval_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    citation_id: str
    number: int
    document_id: str
    chunk_id: str
    company_id: str

    title: str
    snippet: str

    page: int | None = None
    section: str | None = None
    source_path: str | None = None
    source_url: str | None = None

    parent_id: str | None = None
    document_version: str | None = None

    start_char: int | None = None
    end_char: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def marker(self) -> str:
        return f"[{self.number}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "number": self.number,
            "marker": self.marker,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "company_id": self.company_id,
            "title": self.title,
            "snippet": self.snippet,
            "page": self.page,
            "section": self.section,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "parent_id": self.parent_id,
            "document_version": self.document_version,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class CitationValidationResult:
    valid: bool
    referenced_numbers: list[int]
    valid_numbers: list[int]
    invalid_numbers: list[int]
    unused_numbers: list[int]
    sentence_count: int
    cited_sentence_count: int
    citation_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "referenced_numbers": self.referenced_numbers,
            "valid_numbers": self.valid_numbers,
            "invalid_numbers": self.invalid_numbers,
            "unused_numbers": self.unused_numbers,
            "sentence_count": self.sentence_count,
            "cited_sentence_count": self.cited_sentence_count,
            "citation_coverage": self.citation_coverage,
        }


class CitationManager:
    """
    Tenant-scoped RAG citation manager.

    A manager instance should be created for one
    answer-generation request or isolated session.

    Never reuse the same manager across unrelated
    users without explicit request-level isolation.
    """

    CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z0-9])"
    )

    def __init__(
        self,
        *,
        company_id: str,
        max_snippet_chars: int = 350,
    ) -> None:
        if not isinstance(company_id, str) or not company_id.strip():
            raise CitationValidationError(
                "company_id must be a nonempty string"
            )

        if (
            isinstance(max_snippet_chars, bool)
            or not isinstance(max_snippet_chars, int)
            or max_snippet_chars < 20
        ):
            raise CitationValidationError(
                "max_snippet_chars must be >= 20"
            )

        self.company_id = company_id.strip()
        self.max_snippet_chars = max_snippet_chars

        self._citations: list[Citation] = []
        self._by_key: dict[tuple[str, str, str], Citation] = {}

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _require_text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise CitationValidationError(
                f"{name} must be a nonempty string"
            )
        return value.strip()

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return " ".join(text.split())

    def _snippet(self, text: str) -> str:
        cleaned = self._normalize_whitespace(text)

        if len(cleaned) <= self.max_snippet_chars:
            return cleaned

        shortened = cleaned[:self.max_snippet_chars]

        boundary = shortened.rfind(" ")

        if boundary >= self.max_snippet_chars // 2:
            shortened = shortened[:boundary]

        return shortened.rstrip() + "..."

    @staticmethod
    def _validate_offsets(source: CitationSource) -> None:
        if source.page is not None:
            if (
                isinstance(source.page, bool)
                or not isinstance(source.page, int)
                or source.page < 1
            ):
                raise CitationValidationError(
                    "page must be a positive integer"
                )

        if (
            source.start_char is not None
            or source.end_char is not None
        ):
            if (
                source.start_char is None
                or source.end_char is None
            ):
                raise CitationValidationError(
                    "Both start_char and end_char are required"
                )

            if (
                isinstance(source.start_char, bool)
                or isinstance(source.end_char, bool)
                or not isinstance(source.start_char, int)
                or not isinstance(source.end_char, int)
                or source.start_char < 0
                or source.end_char <= source.start_char
            ):
                raise CitationValidationError(
                    "Invalid source character offsets"
                )

    # =====================================================
    # REGISTER SOURCE
    # =====================================================

    def add_source(
        self,
        source: CitationSource,
    ) -> Citation:
        if not isinstance(source, CitationSource):
            raise CitationValidationError(
                "Expected CitationSource"
            )

        self._require_text(
            source.document_id, "document_id"
        )
        self._require_text(
            source.chunk_id, "chunk_id"
        )
        self._require_text(
            source.company_id, "company_id"
        )
        self._require_text(
            source.text, "source.text"
        )

        if source.company_id != self.company_id:
            raise CitationValidationError(
                "Source belongs to another company"
            )

        self._validate_offsets(source)

        key = (
            source.document_id,
            source.document_version or "",
            source.chunk_id,
        )

        existing = self._by_key.get(key)

        if existing is not None:
            return existing

        number = len(self._citations) + 1

        title = (
            source.title
            or source.source_path
            or source.document_id
        )

        citation = Citation(
            citation_id=f"citation-{number}",
            number=number,
            document_id=source.document_id,
            chunk_id=source.chunk_id,
            company_id=source.company_id,
            title=title,
            snippet=self._snippet(source.text),
            page=source.page,
            section=source.section,
            source_path=source.source_path,
            source_url=source.source_url,
            parent_id=source.parent_id,
            document_version=source.document_version,
            start_char=source.start_char,
            end_char=source.end_char,
            metadata=dict(source.metadata),
        )

        self._citations.append(citation)
        self._by_key[key] = citation

        return citation

    # =====================================================
    # REGISTER MULTIPLE SOURCES
    # =====================================================

    def add_sources(
        self,
        sources: Sequence[CitationSource],
    ) -> list[Citation]:
        return [
            self.add_source(source)
            for source in sources
        ]

    # =====================================================
    # GET CITATION
    # =====================================================

    def get_citation(
        self,
        number: int,
    ) -> Citation | None:
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            or number > len(self._citations)
        ):
            return None

        return self._citations[number - 1]

    def get_all(self) -> list[Citation]:
        return list(self._citations)

    # =====================================================
    # FORMAT CITATION
    # =====================================================

    @staticmethod
    def format_citation(
        citation: Citation,
    ) -> str:
        location = []

        if citation.page is not None:
            location.append(f"page {citation.page}")

        if citation.section:
            location.append(
                f"section {citation.section}"
            )

        location_text = (
            f" ({', '.join(location)})"
            if location
            else ""
        )

        return (
            f"{citation.marker} "
            f"{citation.title}{location_text}"
        )

    # =====================================================
    # FORMAT SOURCE LIST
    # =====================================================

    def format_sources(
        self,
        *,
        include_snippets: bool = True,
    ) -> str:
        lines = []

        for citation in self._citations:
            lines.append(
                self.format_citation(citation)
            )

            if include_snippets:
                lines.append(
                    f"    {citation.snippet}"
                )

            if citation.source_url:
                lines.append(
                    f"    URL: {citation.source_url}"
                )

        return "\n".join(lines)

    # =====================================================
    # LLM EVIDENCE CONTEXT
    # =====================================================

    def build_context(
        self,
        *,
        max_sources: int | None = None,
        max_chars_per_source: int = 4000,
    ) -> str:
        if max_chars_per_source < 1:
            raise CitationValidationError(
                "max_chars_per_source must be positive"
            )

        citations = self._citations

        if max_sources is not None:
            if max_sources < 1:
                raise CitationValidationError(
                    "max_sources must be positive"
                )
            citations = citations[:max_sources]

        blocks = []

        for citation in citations:
            header = self.format_citation(citation)

            # The full source text should be obtained
            # from the authorized document store.
            # This manager retains the registered snippet.
            content = citation.snippet[
                :max_chars_per_source
            ]

            blocks.append(
                f"{header}\n{content}"
            )

        return "\n\n".join(blocks)

    # =====================================================
    # EXTRACT CITATION NUMBERS
    # =====================================================

    @classmethod
    def extract_citation_numbers(
        cls,
        answer: str,
    ) -> list[int]:
        if not isinstance(answer, str):
            raise CitationValidationError(
                "answer must be a string"
            )

        return [
            int(match.group(1))
            for match in cls.CITATION_PATTERN.finditer(answer)
        ]

    # =====================================================
    # VALIDATE ANSWER CITATIONS
    # =====================================================

    def validate_answer(
        self,
        answer: str,
    ) -> CitationValidationResult:
        if not isinstance(answer, str):
            raise CitationValidationError(
                "answer must be a string"
            )

        referenced = sorted(set(
            self.extract_citation_numbers(answer)
        ))

        available = {
            citation.number
            for citation in self._citations
        }

        valid_numbers = sorted(
            set(referenced) & available
        )

        invalid_numbers = sorted(
            set(referenced) - available
        )

        unused_numbers = sorted(
            available - set(referenced)
        )

        sentences = [
            sentence.strip()
            for sentence in self.SENTENCE_PATTERN.split(
                answer.strip()
            )
            if sentence.strip()
        ]

        cited_sentences = sum(
            bool(self.CITATION_PATTERN.search(sentence))
            for sentence in sentences
        )

        coverage = (
            cited_sentences / len(sentences)
            if sentences
            else 0.0
        )

        return CitationValidationResult(
            valid=not invalid_numbers,
            referenced_numbers=referenced,
            valid_numbers=valid_numbers,
            invalid_numbers=invalid_numbers,
            unused_numbers=unused_numbers,
            sentence_count=len(sentences),
            cited_sentence_count=cited_sentences,
            citation_coverage=round(coverage, 4),
        )

    # =====================================================
    # EXPORT
    # =====================================================

    def export(self) -> list[dict[str, Any]]:
        return [
            citation.to_dict()
            for citation in self._citations
        ]

    # =====================================================
    # CLEAR
    # =====================================================

    def clear(self) -> None:
        self._citations.clear()
        self._by_key.clear()


# =========================================================
# EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":
    manager = CitationManager(
        company_id="company-001"
    )

    manager.add_sources([
        CitationSource(
            document_id="motor-manual-001",
            chunk_id="chunk-001",
            company_id="company-001",
            title="Industrial Motor Maintenance Manual",
            text=(
                "Excessive motor bearing temperature "
                "may be caused by insufficient lubrication, "
                "misalignment or mechanical overload."
            ),
            page=12,
            section="Bearing Troubleshooting",
            source_path="manuals/motor_manual.pdf",
            document_version="v2",
        ),
        CitationSource(
            document_id="safety-sop-001",
            chunk_id="chunk-002",
            company_id="company-001",
            title="Electrical Maintenance Safety SOP",
            text=(
                "Before maintenance, authorized personnel "
                "must follow the approved isolation and "
                "lockout-tagout procedure."
            ),
            page=4,
            section="Electrical Isolation",
        ),
    ])

    answer = (
        "Motor overheating may result from "
        "insufficient lubrication or misalignment [1]. "
        "Maintenance personnel must follow the approved "
        "isolation procedure before inspection [2]."
    )

    print("ANSWER:")
    print(answer)

    print("\nSOURCES:")
    print(manager.format_sources())

    print("\nVALIDATION:")
    print(
        manager.validate_answer(answer).to_dict()
    )

    print("\nEXPORT:")
    print(manager.export())
