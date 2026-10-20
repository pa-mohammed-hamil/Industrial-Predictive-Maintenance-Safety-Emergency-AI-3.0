# app/evaluation/citation_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from statistics import mean
from typing import Any, Iterable


# =========================================================
# Enums
# =========================================================

class CitationSourceType(str, Enum):
    MANUAL = "manual"
    SAFETY_PROCEDURE = "safety_procedure"
    SOP = "sop"
    MAINTENANCE_GUIDE = "maintenance_guide"
    TECHNICAL_DOCUMENT = "technical_document"
    INCIDENT_REPORT = "incident_report"
    WORK_ORDER = "work_order"
    SENSOR_REPORT = "sensor_report"
    INSPECTION_REPORT = "inspection_report"
    OEM_DOCUMENT = "oem_document"
    KNOWLEDGE_BASE = "knowledge_base"
    WEB = "web"
    DATABASE = "database"
    OTHER = "other"


class CitationErrorType(str, Enum):
    INVALID_SOURCE = "invalid_source"
    WRONG_SOURCE = "wrong_source"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    MISSING_CITATION = "missing_citation"
    WRONG_PASSAGE = "wrong_passage"
    DUPLICATE_CITATION = "duplicate_citation"
    LOW_RELEVANCE = "low_relevance"
    INCORRECT_PLACEMENT = "incorrect_placement"
    INCOMPLETE_CITATION = "incomplete_citation"


# =========================================================
# Citation structures
# =========================================================

@dataclass(slots=True)
class Citation:
    citation_id: str

    source_id: str
    source_type: CitationSourceType

    document_id: str | None = None
    document_title: str | None = None

    page: int | None = None

    chunk_id: str | None = None

    quoted_text: str | None = None

    source_text: str | None = None

    start_position: int | None = None
    end_position: int | None = None

    retrieval_score: float | None = None
    reranker_score: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class Claim:
    claim_id: str
    text: str

    requires_citation: bool = True

    expected_source_ids: list[str] = field(
        default_factory=list
    )

    expected_citation_ids: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class CitationAssessment:
    citation_id: str

    valid_source: bool
    correct_source: bool

    supports_claim: bool

    relevant: bool = True
    correctly_placed: bool = True

    complete: bool = True

    support_score: float = 1.0
    relevance_score: float = 1.0

    notes: str | None = None


@dataclass(slots=True)
class CitationEvaluationSample:
    sample_id: str

    answer: str

    claims: list[Claim]

    citations: list[Citation]

    assessments: list[
        CitationAssessment
    ] = field(
        default_factory=list
    )

    query: str | None = None

    asset_id: str | None = None
    site_id: str | None = None

    conversation_id: str | None = None
    retrieval_id: str | None = None

    model_name: str | None = None
    model_version: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Per-sample result
# =========================================================

@dataclass(slots=True)
class CitationSampleResult:
    sample_id: str

    claims: int
    claims_requiring_citation: int

    citations: int
    valid_citations: int
    invalid_citations: int

    supported_claims: int
    unsupported_claims: int

    citation_precision: float
    citation_recall: float
    citation_f1: float

    citation_completeness: float

    source_correctness: float
    passage_support: float
    relevance: float
    placement_accuracy: float

    duplicate_citations: int

    faithfulness: float

    overall_score: float

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================================================
# Aggregate result
# =========================================================

@dataclass(slots=True)
class CitationEvaluationResult:
    total_samples: int

    total_claims: int
    claims_requiring_citation: int

    total_citations: int

    valid_citations: int
    invalid_citations: int

    supported_claims: int
    unsupported_claims: int

    citation_precision: float
    citation_recall: float
    citation_f1: float

    citation_completeness: float

    source_correctness: float
    passage_support: float

    citation_relevance: float
    placement_accuracy: float

    faithfulness: float

    invalid_citation_rate: float
    unsupported_claim_rate: float

    duplicate_citation_rate: float

    overall_score: float

    per_sample: list[
        CitationSampleResult
    ]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["per_sample"] = [
            item.to_dict()
            for item in self.per_sample
        ]

        return result


# =========================================================
# Validation
# =========================================================

def _probability(
    value: float,
    name: str,
) -> float:

    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


def validate_sample(
    sample: CitationEvaluationSample,
) -> None:

    citation_ids = [
        citation.citation_id
        for citation in sample.citations
    ]

    if len(citation_ids) != len(
        set(citation_ids)
    ):
        raise ValueError(
            f"Duplicate citation IDs in "
            f"sample {sample.sample_id}."
        )

    assessment_ids = [
        assessment.citation_id
        for assessment in sample.assessments
    ]

    unknown = (
        set(assessment_ids)
        - set(citation_ids)
    )

    if unknown:
        raise ValueError(
            "Assessments reference unknown "
            f"citations: {sorted(unknown)}"
        )

    for assessment in sample.assessments:
        _probability(
            assessment.support_score,
            "support_score",
        )

        _probability(
            assessment.relevance_score,
            "relevance_score",
        )


# =========================================================
# Assessment lookup
# =========================================================

def build_assessment_map(
    sample: CitationEvaluationSample,
) -> dict[str, CitationAssessment]:

    return {
        assessment.citation_id:
            assessment
        for assessment
        in sample.assessments
    }


# =========================================================
# Citation validity
# =========================================================

def is_valid_citation(
    assessment: CitationAssessment,
) -> bool:

    return (
        assessment.valid_source
        and assessment.correct_source
        and assessment.supports_claim
    )


# =========================================================
# Duplicate citations
# =========================================================

def count_duplicate_citations(
    citations: Iterable[Citation],
) -> int:

    seen: set[
        tuple[
            str,
            str | None,
            int | None,
            str | None,
        ]
    ] = set()

    duplicates = 0

    for citation in citations:
        key = (
            citation.source_id,
            citation.document_id,
            citation.page,
            citation.chunk_id,
        )

        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    return duplicates


# =========================================================
# Claim support
# =========================================================

def is_claim_supported(
    claim: Claim,
    sample: CitationEvaluationSample,
    assessment_map: dict[
        str,
        CitationAssessment,
    ],
) -> bool:

    if not claim.requires_citation:
        return True

    # ---------------------------------------------
    # Explicit expected citation IDs
    # ---------------------------------------------

    if claim.expected_citation_ids:
        for citation_id in (
            claim.expected_citation_ids
        ):
            assessment = (
                assessment_map.get(
                    citation_id
                )
            )

            if (
                assessment is not None
                and
                is_valid_citation(
                    assessment
                )
            ):
                return True

        return False

    # ---------------------------------------------
    # Expected source IDs
    # ---------------------------------------------

    if claim.expected_source_ids:
        expected = set(
            claim.expected_source_ids
        )

        for citation in sample.citations:
            if (
                citation.source_id
                not in expected
            ):
                continue

            assessment = (
                assessment_map.get(
                    citation.citation_id
                )
            )

            if (
                assessment is not None
                and
                is_valid_citation(
                    assessment
                )
            ):
                return True

        return False

    # ---------------------------------------------
    # If no explicit mapping is available,
    # any valid supporting citation is accepted.
    # ---------------------------------------------

    return any(
        is_valid_citation(
            assessment
        )
        for assessment
        in sample.assessments
    )


# =========================================================
# Citation precision / recall
# =========================================================

def calculate_precision_recall(
    *,
    valid_citations: int,
    total_citations: int,
    supported_claims: int,
    claims_requiring_citation: int,
) -> tuple[
    float,
    float,
    float,
]:

    precision = (
        valid_citations
        / total_citations
        if total_citations
        else 0.0
    )

    recall = (
        supported_claims
        / claims_requiring_citation
        if claims_requiring_citation
        else 1.0
    )

    f1 = (
        (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
        if precision + recall
        else 0.0
    )

    return (
        round(
            precision,
            4,
        ),
        round(
            recall,
            4,
        ),
        round(
            f1,
            4,
        ),
    )


# =========================================================
# Faithfulness
# =========================================================

def calculate_faithfulness(
    claims_requiring_citation: list[Claim],
    supported_claim_ids: set[str],
) -> float:

    if not claims_requiring_citation:
        return 1.0

    faithful = sum(
        claim.claim_id
        in supported_claim_ids
        for claim in claims_requiring_citation
    )

    return round(
        faithful
        / len(
            claims_requiring_citation
        ),
        4,
    )


# =========================================================
# Overall score
# =========================================================

def calculate_overall_score(
    *,
    precision: float,
    recall: float,
    source_correctness: float,
    passage_support: float,
    placement_accuracy: float,
    faithfulness: float,
) -> float:
    """
    Project-specific composite citation quality score.

    Safety/grounding weights intentionally emphasize
    claim support and faithfulness.
    """

    score = (
        precision * 0.15
        + recall * 0.20
        + source_correctness * 0.15
        + passage_support * 0.20
        + placement_accuracy * 0.10
        + faithfulness * 0.20
    )

    return round(
        score,
        4,
    )


# =========================================================
# Evaluate one sample
# =========================================================

def evaluate_citation_sample(
    sample: CitationEvaluationSample,
) -> CitationSampleResult:

    validate_sample(
        sample
    )

    assessment_map = (
        build_assessment_map(
            sample
        )
    )

    claims_requiring = [
        claim
        for claim in sample.claims
        if claim.requires_citation
    ]

    # ---------------------------------------------
    # Citation metrics
    # ---------------------------------------------

    valid_citations = 0

    source_correct_values: list[
        float
    ] = []

    support_values: list[
        float
    ] = []

    relevance_values: list[
        float
    ] = []

    placement_values: list[
        float
    ] = []

    errors: list[
        dict[str, Any]
    ] = []

    for citation in sample.citations:
        assessment = (
            assessment_map.get(
                citation.citation_id
            )
        )

        if assessment is None:
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "citation_id":
                        citation.citation_id,

                    "error_type":
                        CitationErrorType
                        .INCOMPLETE_CITATION
                        .value,

                    "message":
                        "Citation has no "
                        "evaluation assessment.",
                }
            )

            continue

        if is_valid_citation(
            assessment
        ):
            valid_citations += 1

        source_correct_values.append(
            1.0
            if (
                assessment.valid_source
                and
                assessment.correct_source
            )
            else 0.0
        )

        support_values.append(
            assessment.support_score
            if assessment.supports_claim
            else 0.0
        )

        relevance_values.append(
            assessment.relevance_score
            if assessment.relevant
            else 0.0
        )

        placement_values.append(
            1.0
            if assessment.correctly_placed
            else 0.0
        )

        if not assessment.valid_source:
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "citation_id":
                        citation.citation_id,

                    "source_id":
                        citation.source_id,

                    "error_type":
                        CitationErrorType
                        .INVALID_SOURCE
                        .value,
                }
            )

        elif not assessment.correct_source:
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "citation_id":
                        citation.citation_id,

                    "source_id":
                        citation.source_id,

                    "error_type":
                        CitationErrorType
                        .WRONG_SOURCE
                        .value,
                }
            )

        if not assessment.supports_claim:
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "citation_id":
                        citation.citation_id,

                    "error_type":
                        CitationErrorType
                        .WRONG_PASSAGE
                        .value,
                }
            )

        if not assessment.relevant:
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "citation_id":
                        citation.citation_id,

                    "error_type":
                        CitationErrorType
                        .LOW_RELEVANCE
                        .value,
                }
            )

        if not assessment.correctly_placed:
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "citation_id":
                        citation.citation_id,

                    "error_type":
                        CitationErrorType
                        .INCORRECT_PLACEMENT
                        .value,
                }
            )

        if not assessment.complete:
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "citation_id":
                        citation.citation_id,

                    "error_type":
                        CitationErrorType
                        .INCOMPLETE_CITATION
                        .value,
                }
            )

    # ---------------------------------------------
    # Claim support
    # ---------------------------------------------

    supported_claim_ids: set[str] = set()

    for claim in claims_requiring:
        supported = (
            is_claim_supported(
                claim,
                sample,
                assessment_map,
            )
        )

        if supported:
            supported_claim_ids.add(
                claim.claim_id
            )

        else:
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "claim_id":
                        claim.claim_id,

                    "claim":
                        claim.text,

                    "error_type":
                        CitationErrorType
                        .MISSING_CITATION
                        .value,
                }
            )

    supported_claims = len(
        supported_claim_ids
    )

    unsupported_claims = (
        len(claims_requiring)
        - supported_claims
    )

    # ---------------------------------------------
    # Duplicate detection
    # ---------------------------------------------

    duplicates = (
        count_duplicate_citations(
            sample.citations
        )
    )

    if duplicates:
        errors.append(
            {
                "sample_id":
                    sample.sample_id,

                "error_type":
                    CitationErrorType
                    .DUPLICATE_CITATION
                    .value,

                "count":
                    duplicates,
            }
        )

    # ---------------------------------------------
    # Precision / recall
    # ---------------------------------------------

    (
        precision,
        recall,
        f1,
    ) = calculate_precision_recall(
        valid_citations=(
            valid_citations
        ),
        total_citations=len(
            sample.citations
        ),
        supported_claims=(
            supported_claims
        ),
        claims_requiring_citation=len(
            claims_requiring
        ),
    )

    completeness = recall

    source_correctness = (
        mean(
            source_correct_values
        )
        if source_correct_values
        else (
            1.0
            if not sample.citations
            else 0.0
        )
    )

    passage_support = (
        mean(
            support_values
        )
        if support_values
        else (
            1.0
            if not sample.citations
            else 0.0
        )
    )

    relevance = (
        mean(
            relevance_values
        )
        if relevance_values
        else (
            1.0
            if not sample.citations
            else 0.0
        )
    )

    placement = (
        mean(
            placement_values
        )
        if placement_values
        else (
            1.0
            if not sample.citations
            else 0.0
        )
    )

    faithfulness = (
        calculate_faithfulness(
            claims_requiring,
            supported_claim_ids,
        )
    )

    overall = calculate_overall_score(
        precision=precision,
        recall=recall,
        source_correctness=(
            source_correctness
        ),
        passage_support=(
            passage_support
        ),
        placement_accuracy=(
            placement
        ),
        faithfulness=(
            faithfulness
        ),
    )

    return CitationSampleResult(
        sample_id=sample.sample_id,

        claims=len(
            sample.claims
        ),

        claims_requiring_citation=len(
            claims_requiring
        ),

        citations=len(
            sample.citations
        ),

        valid_citations=(
            valid_citations
        ),

        invalid_citations=(
            len(sample.citations)
            - valid_citations
        ),

        supported_claims=(
            supported_claims
        ),

        unsupported_claims=(
            unsupported_claims
        ),

        citation_precision=(
            precision
        ),

        citation_recall=(
            recall
        ),

        citation_f1=(
            f1
        ),

        citation_completeness=round(
            completeness,
            4,
        ),

        source_correctness=round(
            source_correctness,
            4,
        ),

        passage_support=round(
            passage_support,
            4,
        ),

        relevance=round(
            relevance,
            4,
        ),

        placement_accuracy=round(
            placement,
            4,
        ),

        duplicate_citations=(
            duplicates
        ),

        faithfulness=(
            faithfulness
        ),

        overall_score=(
            overall
        ),

        errors=errors,
    )


# =========================================================
# Aggregate evaluation
# =========================================================

def evaluate_citations(
    samples: list[
        CitationEvaluationSample
    ],
) -> CitationEvaluationResult:

    if not samples:
        return CitationEvaluationResult(
            total_samples=0,

            total_claims=0,
            claims_requiring_citation=0,

            total_citations=0,

            valid_citations=0,
            invalid_citations=0,

            supported_claims=0,
            unsupported_claims=0,

            citation_precision=0.0,
            citation_recall=0.0,
            citation_f1=0.0,

            citation_completeness=0.0,

            source_correctness=0.0,
            passage_support=0.0,

            citation_relevance=0.0,
            placement_accuracy=0.0,

            faithfulness=0.0,

            invalid_citation_rate=0.0,
            unsupported_claim_rate=0.0,

            duplicate_citation_rate=0.0,

            overall_score=0.0,

            per_sample=[],
            errors=[],
        )

    per_sample = [
        evaluate_citation_sample(
            sample
        )
        for sample in samples
    ]

    total_claims = sum(
        result.claims
        for result in per_sample
    )

    required_claims = sum(
        result.claims_requiring_citation
        for result in per_sample
    )

    total_citations = sum(
        result.citations
        for result in per_sample
    )

    valid_citations = sum(
        result.valid_citations
        for result in per_sample
    )

    invalid_citations = (
        total_citations
        - valid_citations
    )

    supported_claims = sum(
        result.supported_claims
        for result in per_sample
    )

    unsupported_claims = (
        required_claims
        - supported_claims
    )

    duplicates = sum(
        result.duplicate_citations
        for result in per_sample
    )

    (
        precision,
        recall,
        f1,
    ) = calculate_precision_recall(
        valid_citations=(
            valid_citations
        ),
        total_citations=(
            total_citations
        ),
        supported_claims=(
            supported_claims
        ),
        claims_requiring_citation=(
            required_claims
        ),
    )

    source_correctness = _weighted_average(
        (
            result.source_correctness,
            result.citations,
        )
        for result in per_sample
    )

    passage_support = _weighted_average(
        (
            result.passage_support,
            result.citations,
        )
        for result in per_sample
    )

    relevance = _weighted_average(
        (
            result.relevance,
            result.citations,
        )
        for result in per_sample
    )

    placement = _weighted_average(
        (
            result.placement_accuracy,
            result.citations,
        )
        for result in per_sample
    )

    faithfulness = (
        supported_claims
        / required_claims
        if required_claims
        else 1.0
    )

    invalid_rate = (
        invalid_citations
        / total_citations
        if total_citations
        else 0.0
    )

    unsupported_rate = (
        unsupported_claims
        / required_claims
        if required_claims
        else 0.0
    )

    duplicate_rate = (
        duplicates
        / total_citations
        if total_citations
        else 0.0
    )

    overall = calculate_overall_score(
        precision=precision,
        recall=recall,
        source_correctness=(
            source_correctness
        ),
        passage_support=(
            passage_support
        ),
        placement_accuracy=(
            placement
        ),
        faithfulness=(
            faithfulness
        ),
    )

    errors = [
        error
        for result in per_sample
        for error in result.errors
    ]

    return CitationEvaluationResult(
        total_samples=len(samples),

        total_claims=(
            total_claims
        ),

        claims_requiring_citation=(
            required_claims
        ),

        total_citations=(
            total_citations
        ),

        valid_citations=(
            valid_citations
        ),

        invalid_citations=(
            invalid_citations
        ),

        supported_claims=(
            supported_claims
        ),

        unsupported_claims=(
            unsupported_claims
        ),

        citation_precision=(
            precision
        ),

        citation_recall=(
            recall
        ),

        citation_f1=(
            f1
        ),

        citation_completeness=(
            recall
        ),

        source_correctness=round(
            source_correctness,
            4,
        ),

        passage_support=round(
            passage_support,
            4,
        ),

        citation_relevance=round(
            relevance,
            4,
        ),

        placement_accuracy=round(
            placement,
            4,
        ),

        faithfulness=round(
            faithfulness,
            4,
        ),

        invalid_citation_rate=round(
            invalid_rate,
            4,
        ),

        unsupported_claim_rate=round(
            unsupported_rate,
            4,
        ),

        duplicate_citation_rate=round(
            duplicate_rate,
            4,
        ),

        overall_score=(
            overall
        ),

        per_sample=per_sample,
        errors=errors,
    )


# =========================================================
# Weighted average
# =========================================================

def _weighted_average(
    values: Iterable[
        tuple[float, int]
    ],
) -> float:

    weighted_sum = 0.0
    total_weight = 0

    for value, weight in values:
        if weight <= 0:
            continue

        weighted_sum += (
            float(value)
            * weight
        )

        total_weight += weight

    if total_weight == 0:
        return 1.0

    return (
        weighted_sum
        / total_weight
    )


# =========================================================
# Quality gate
# =========================================================

def citation_quality_gate(
    result: CitationEvaluationResult,
    *,
    minimum_precision: float = 0.90,
    minimum_recall: float = 0.90,
    minimum_faithfulness: float = 0.95,
    maximum_invalid_rate: float = 0.05,
    maximum_unsupported_rate: float = 0.05,
) -> dict[str, Any]:

    minimum_precision = _probability(
        minimum_precision,
        "minimum_precision",
    )

    minimum_recall = _probability(
        minimum_recall,
        "minimum_recall",
    )

    minimum_faithfulness = _probability(
        minimum_faithfulness,
        "minimum_faithfulness",
    )

    maximum_invalid_rate = _probability(
        maximum_invalid_rate,
        "maximum_invalid_rate",
    )

    maximum_unsupported_rate = (
        _probability(
            maximum_unsupported_rate,
            "maximum_unsupported_rate",
        )
    )

    checks = {
        "precision": (
            result.citation_precision
            >= minimum_precision
        ),

        "recall": (
            result.citation_recall
            >= minimum_recall
        ),

        "faithfulness": (
            result.faithfulness
            >= minimum_faithfulness
        ),

        "invalid_citation_rate": (
            result.invalid_citation_rate
            <= maximum_invalid_rate
        ),

        "unsupported_claim_rate": (
            result.unsupported_claim_rate
            <= maximum_unsupported_rate
        ),
    }

    return {
        "passed": all(
            checks.values()
        ),
        "checks": checks,
        "metrics": {
            "citation_precision":
                result.citation_precision,

            "citation_recall":
                result.citation_recall,

            "faithfulness":
                result.faithfulness,

            "invalid_citation_rate":
                result.invalid_citation_rate,

            "unsupported_claim_rate":
                result.unsupported_claim_rate,

            "overall_score":
                result.overall_score,
        },
    }


# =========================================================
# Evaluator
# =========================================================

class CitationEvaluator:

    def evaluate_sample(
        self,
        sample: CitationEvaluationSample,
    ) -> CitationSampleResult:

        return evaluate_citation_sample(
            sample
        )

    def evaluate(
        self,
        samples: list[
            CitationEvaluationSample
        ],
    ) -> CitationEvaluationResult:

        return evaluate_citations(
            samples
        )

    def quality_gate(
        self,
        result: CitationEvaluationResult,
        *,
        minimum_precision: float = 0.90,
        minimum_recall: float = 0.90,
        minimum_faithfulness: float = 0.95,
        maximum_invalid_rate: float = 0.05,
        maximum_unsupported_rate: float = 0.05,
    ) -> dict[str, Any]:

        return citation_quality_gate(
            result,

            minimum_precision=(
                minimum_precision
            ),

            minimum_recall=(
                minimum_recall
            ),

            minimum_faithfulness=(
                minimum_faithfulness
            ),

            maximum_invalid_rate=(
                maximum_invalid_rate
            ),

            maximum_unsupported_rate=(
                maximum_unsupported_rate
            ),
        )


citation_evaluator = CitationEvaluator()


__all__ = [
    "CitationSourceType",
    "CitationErrorType",
    "Citation",
    "Claim",
    "CitationAssessment",
    "CitationEvaluationSample",
    "CitationSampleResult",
    "CitationEvaluationResult",
    "build_assessment_map",
    "is_valid_citation",
    "count_duplicate_citations",
    "is_claim_supported",
    "calculate_precision_recall",
    "calculate_faithfulness",
    "calculate_overall_score",
    "evaluate_citation_sample",
    "evaluate_citations",
    "citation_quality_gate",
    "CitationEvaluator",
    "citation_evaluator",
]