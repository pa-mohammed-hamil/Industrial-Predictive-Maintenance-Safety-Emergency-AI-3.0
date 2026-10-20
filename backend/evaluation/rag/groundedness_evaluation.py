# app/evaluation/groundedness_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from statistics import mean
from typing import Any, Iterable


# =========================================================
# Enums
# =========================================================

class GroundednessLabel(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"
    CONTRADICTED = "contradicted"


class ClaimImportance(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceType(str, Enum):
    OEM_MANUAL = "oem_manual"
    MAINTENANCE_MANUAL = "maintenance_manual"
    SAFETY_PROCEDURE = "safety_procedure"
    SOP = "sop"
    TECHNICAL_DOCUMENT = "technical_document"
    SENSOR_DATA = "sensor_data"
    TELEMETRY = "telemetry"
    INSPECTION_REPORT = "inspection_report"
    INCIDENT_REPORT = "incident_report"
    WORK_ORDER = "work_order"
    DATABASE = "database"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    WEB = "web"
    OTHER = "other"


class GroundednessErrorType(str, Enum):
    UNSUPPORTED_CLAIM = "unsupported_claim"
    PARTIAL_SUPPORT = "partial_support"
    CONTRADICTION = "contradiction"
    MISSING_EVIDENCE = "missing_evidence"
    LOW_EVIDENCE_QUALITY = "low_evidence_quality"
    LOW_EVIDENCE_RELEVANCE = "low_evidence_relevance"
    CRITICAL_UNGROUNDED_CLAIM = "critical_ungrounded_claim"


# =========================================================
# Evidence
# =========================================================

@dataclass(slots=True)
class Evidence:
    evidence_id: str
    source_id: str

    text: str

    evidence_type: EvidenceType = (
        EvidenceType.OTHER
    )

    document_id: str | None = None
    document_title: str | None = None

    page: int | None = None
    chunk_id: str | None = None

    retrieval_score: float | None = None
    reranker_score: float | None = None

    source_quality: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Claims
# =========================================================

@dataclass(slots=True)
class GroundednessClaim:
    claim_id: str
    text: str

    requires_evidence: bool = True

    importance: ClaimImportance = (
        ClaimImportance.MEDIUM
    )

    expected_evidence_ids: list[str] = field(
        default_factory=list
    )

    expected_source_ids: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Claim assessment
# =========================================================

@dataclass(slots=True)
class ClaimGroundingAssessment:
    claim_id: str

    label: GroundednessLabel

    evidence_ids: list[str] = field(
        default_factory=list
    )

    entailment_score: float = 0.0
    evidence_relevance: float = 0.0

    contradiction_score: float = 0.0

    notes: str | None = None


# =========================================================
# Evaluation sample
# =========================================================

@dataclass(slots=True)
class GroundednessEvaluationSample:
    sample_id: str

    query: str
    answer: str

    claims: list[GroundednessClaim]

    evidence: list[Evidence]

    assessments: list[
        ClaimGroundingAssessment
    ]

    asset_id: str | None = None
    site_id: str | None = None

    conversation_id: str | None = None
    retrieval_id: str | None = None
    generation_id: str | None = None

    model_name: str | None = None
    model_version: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Per-sample result
# =========================================================

@dataclass(slots=True)
class GroundednessSampleResult:
    sample_id: str

    total_claims: int
    claims_requiring_evidence: int

    grounded_claims: int
    partially_grounded_claims: int
    ungrounded_claims: int
    contradicted_claims: int

    claim_groundedness: float
    weighted_groundedness: float

    faithfulness: float

    evidence_coverage: float
    evidence_relevance: float
    evidence_quality: float

    average_entailment: float
    average_contradiction: float

    unsupported_claim_rate: float
    contradiction_rate: float

    critical_claims: int
    grounded_critical_claims: int
    critical_groundedness: float | None

    overall_score: float

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================================================
# Aggregate result
# =========================================================

@dataclass(slots=True)
class GroundednessEvaluationResult:
    total_samples: int

    total_claims: int
    claims_requiring_evidence: int

    grounded_claims: int
    partially_grounded_claims: int
    ungrounded_claims: int
    contradicted_claims: int

    claim_groundedness: float
    weighted_groundedness: float

    faithfulness: float

    evidence_coverage: float
    evidence_relevance: float
    evidence_quality: float

    average_entailment: float
    average_contradiction: float

    unsupported_claim_rate: float
    contradiction_rate: float

    critical_claims: int
    grounded_critical_claims: int
    critical_groundedness: float | None

    overall_score: float

    per_sample: list[
        GroundednessSampleResult
    ]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["per_sample"] = [
            sample.to_dict()
            for sample in self.per_sample
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
    sample: GroundednessEvaluationSample,
) -> None:

    claim_ids = [
        claim.claim_id
        for claim in sample.claims
    ]

    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError(
            f"Duplicate claim IDs in {sample.sample_id}."
        )

    evidence_ids = [
        evidence.evidence_id
        for evidence in sample.evidence
    ]

    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(
            f"Duplicate evidence IDs in {sample.sample_id}."
        )

    assessment_claim_ids = [
        assessment.claim_id
        for assessment in sample.assessments
    ]

    unknown_claims = (
        set(assessment_claim_ids)
        - set(claim_ids)
    )

    if unknown_claims:
        raise ValueError(
            "Assessments reference unknown claims: "
            f"{sorted(unknown_claims)}"
        )

    evidence_id_set = set(evidence_ids)

    for evidence in sample.evidence:
        _probability(
            evidence.source_quality,
            "source_quality",
        )

        if evidence.retrieval_score is not None:
            _probability(
                evidence.retrieval_score,
                "retrieval_score",
            )

        if evidence.reranker_score is not None:
            _probability(
                evidence.reranker_score,
                "reranker_score",
            )

    for assessment in sample.assessments:
        _probability(
            assessment.entailment_score,
            "entailment_score",
        )

        _probability(
            assessment.evidence_relevance,
            "evidence_relevance",
        )

        _probability(
            assessment.contradiction_score,
            "contradiction_score",
        )

        unknown_evidence = (
            set(assessment.evidence_ids)
            - evidence_id_set
        )

        if unknown_evidence:
            raise ValueError(
                "Assessment references unknown evidence: "
                f"{sorted(unknown_evidence)}"
            )


# =========================================================
# Maps
# =========================================================

def build_claim_map(
    sample: GroundednessEvaluationSample,
) -> dict[str, GroundednessClaim]:

    return {
        claim.claim_id: claim
        for claim in sample.claims
    }


def build_evidence_map(
    sample: GroundednessEvaluationSample,
) -> dict[str, Evidence]:

    return {
        evidence.evidence_id: evidence
        for evidence in sample.evidence
    }


def build_assessment_map(
    sample: GroundednessEvaluationSample,
) -> dict[str, ClaimGroundingAssessment]:

    return {
        assessment.claim_id: assessment
        for assessment in sample.assessments
    }


# =========================================================
# Importance weighting
# =========================================================

IMPORTANCE_WEIGHTS: dict[
    ClaimImportance,
    float,
] = {
    ClaimImportance.LOW: 1.0,
    ClaimImportance.MEDIUM: 2.0,
    ClaimImportance.HIGH: 3.0,
    ClaimImportance.CRITICAL: 5.0,
}


GROUNDING_VALUES: dict[
    GroundednessLabel,
    float,
] = {
    GroundednessLabel.GROUNDED: 1.0,
    GroundednessLabel.PARTIALLY_GROUNDED: 0.5,
    GroundednessLabel.UNGROUNDED: 0.0,
    GroundednessLabel.CONTRADICTED: 0.0,
}


# =========================================================
# Claim groundedness
# =========================================================

def calculate_claim_groundedness(
    claims: Iterable[GroundednessClaim],
    assessment_map: dict[
        str,
        ClaimGroundingAssessment,
    ],
) -> float:

    required = [
        claim
        for claim in claims
        if claim.requires_evidence
    ]

    if not required:
        return 1.0

    score = 0.0

    for claim in required:
        assessment = assessment_map.get(
            claim.claim_id
        )

        if assessment is None:
            continue

        score += GROUNDING_VALUES[
            assessment.label
        ]

    return round(
        score / len(required),
        4,
    )


# =========================================================
# Weighted groundedness
# =========================================================

def calculate_weighted_groundedness(
    claims: Iterable[GroundednessClaim],
    assessment_map: dict[
        str,
        ClaimGroundingAssessment,
    ],
) -> float:

    total_weight = 0.0
    grounded_weight = 0.0

    for claim in claims:
        if not claim.requires_evidence:
            continue

        weight = IMPORTANCE_WEIGHTS[
            claim.importance
        ]

        total_weight += weight

        assessment = assessment_map.get(
            claim.claim_id
        )

        if assessment is None:
            continue

        grounded_weight += (
            weight
            * GROUNDING_VALUES[
                assessment.label
            ]
        )

    if total_weight == 0:
        return 1.0

    return round(
        grounded_weight / total_weight,
        4,
    )


# =========================================================
# Faithfulness
# =========================================================

def calculate_faithfulness(
    claims: Iterable[GroundednessClaim],
    assessment_map: dict[
        str,
        ClaimGroundingAssessment,
    ],
) -> float:
    """
    Faithfulness is strict here.

    Only fully grounded claims count as faithful.
    Partial, unsupported, contradicted, and missing
    assessments do not count as fully faithful.
    """

    required = [
        claim
        for claim in claims
        if claim.requires_evidence
    ]

    if not required:
        return 1.0

    faithful = 0

    for claim in required:
        assessment = assessment_map.get(
            claim.claim_id
        )

        if (
            assessment is not None
            and assessment.label
            == GroundednessLabel.GROUNDED
        ):
            faithful += 1

    return round(
        faithful / len(required),
        4,
    )


# =========================================================
# Evidence coverage
# =========================================================

def calculate_evidence_coverage(
    claims: Iterable[GroundednessClaim],
    assessment_map: dict[
        str,
        ClaimGroundingAssessment,
    ],
) -> float:

    required = [
        claim
        for claim in claims
        if claim.requires_evidence
    ]

    if not required:
        return 1.0

    covered = 0

    for claim in required:
        assessment = assessment_map.get(
            claim.claim_id
        )

        if (
            assessment is not None
            and assessment.evidence_ids
        ):
            covered += 1

    return round(
        covered / len(required),
        4,
    )


# =========================================================
# Critical groundedness
# =========================================================

def calculate_critical_groundedness(
    claims: Iterable[GroundednessClaim],
    assessment_map: dict[
        str,
        ClaimGroundingAssessment,
    ],
) -> tuple[int, int, float | None]:

    critical = [
        claim
        for claim in claims
        if (
            claim.requires_evidence
            and claim.importance
            == ClaimImportance.CRITICAL
        )
    ]

    if not critical:
        return 0, 0, None

    grounded = sum(
        (
            assessment_map.get(
                claim.claim_id
            ) is not None
            and
            assessment_map[
                claim.claim_id
            ].label
            == GroundednessLabel.GROUNDED
        )
        for claim in critical
    )

    return (
        len(critical),
        grounded,
        round(
            grounded / len(critical),
            4,
        ),
    )


# =========================================================
# Evidence quality
# =========================================================

def calculate_evidence_quality(
    sample: GroundednessEvaluationSample,
) -> float:

    used_evidence_ids = {
        evidence_id
        for assessment in sample.assessments
        for evidence_id in assessment.evidence_ids
    }

    if not used_evidence_ids:
        return 0.0

    evidence_map = build_evidence_map(
        sample
    )

    values = [
        evidence_map[evidence_id].source_quality
        for evidence_id in used_evidence_ids
        if evidence_id in evidence_map
    ]

    if not values:
        return 0.0

    return round(
        mean(values),
        4,
    )


# =========================================================
# Expected evidence checking
# =========================================================

def expected_evidence_satisfied(
    claim: GroundednessClaim,
    assessment: ClaimGroundingAssessment | None,
    evidence_map: dict[str, Evidence],
) -> bool:

    if not claim.requires_evidence:
        return True

    if assessment is None:
        return False

    used_ids = set(
        assessment.evidence_ids
    )

    if claim.expected_evidence_ids:
        if not (
            used_ids
            & set(
                claim.expected_evidence_ids
            )
        ):
            return False

    if claim.expected_source_ids:
        used_sources = {
            evidence_map[evidence_id].source_id
            for evidence_id in used_ids
            if evidence_id in evidence_map
        }

        if not (
            used_sources
            & set(
                claim.expected_source_ids
            )
        ):
            return False

    return True


# =========================================================
# Overall score
# =========================================================

def calculate_overall_score(
    *,
    weighted_groundedness: float,
    faithfulness: float,
    evidence_coverage: float,
    evidence_relevance: float,
    evidence_quality: float,
    contradiction_rate: float,
) -> float:
    """
    Project-specific composite score.

    Grounding and faithfulness receive the highest
    weights because unsupported maintenance/safety
    statements are high-risk.
    """

    base = (
        weighted_groundedness * 0.30
        + faithfulness * 0.25
        + evidence_coverage * 0.15
        + evidence_relevance * 0.15
        + evidence_quality * 0.15
    )

    contradiction_penalty = (
        contradiction_rate * 0.50
    )

    return round(
        max(
            0.0,
            min(
                1.0,
                base
                - contradiction_penalty,
            ),
        ),
        4,
    )


# =========================================================
# Per-sample evaluation
# =========================================================

def evaluate_groundedness_sample(
    sample: GroundednessEvaluationSample,
) -> GroundednessSampleResult:

    validate_sample(
        sample
    )

    assessment_map = (
        build_assessment_map(
            sample
        )
    )

    evidence_map = (
        build_evidence_map(
            sample
        )
    )

    required_claims = [
        claim
        for claim in sample.claims
        if claim.requires_evidence
    ]

    grounded = 0
    partial = 0
    ungrounded = 0
    contradicted = 0

    entailment_scores: list[float] = []
    relevance_scores: list[float] = []
    contradiction_scores: list[float] = []

    errors: list[
        dict[str, Any]
    ] = []

    for claim in required_claims:
        assessment = assessment_map.get(
            claim.claim_id
        )

        if assessment is None:
            ungrounded += 1

            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "claim_id":
                        claim.claim_id,

                    "claim":
                        claim.text,

                    "error_type":
                        GroundednessErrorType
                        .MISSING_EVIDENCE.value,

                    "message":
                        "No grounding assessment "
                        "exists for this claim.",
                }
            )

            continue

        entailment_scores.append(
            assessment.entailment_score
        )

        relevance_scores.append(
            assessment.evidence_relevance
        )

        contradiction_scores.append(
            assessment.contradiction_score
        )

        if (
            assessment.label
            == GroundednessLabel.GROUNDED
        ):
            grounded += 1

        elif (
            assessment.label
            == GroundednessLabel.PARTIALLY_GROUNDED
        ):
            partial += 1

            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "claim_id":
                        claim.claim_id,

                    "claim":
                        claim.text,

                    "error_type":
                        GroundednessErrorType
                        .PARTIAL_SUPPORT.value,

                    "entailment_score":
                        assessment.entailment_score,
                }
            )

        elif (
            assessment.label
            == GroundednessLabel.CONTRADICTED
        ):
            contradicted += 1

            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "claim_id":
                        claim.claim_id,

                    "claim":
                        claim.text,

                    "error_type":
                        GroundednessErrorType
                        .CONTRADICTION.value,

                    "contradiction_score":
                        assessment.contradiction_score,
                }
            )

        else:
            ungrounded += 1

            error_type = (
                GroundednessErrorType
                .CRITICAL_UNGROUNDED_CLAIM
                if claim.importance
                == ClaimImportance.CRITICAL
                else GroundednessErrorType
                .UNSUPPORTED_CLAIM
            )

            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "claim_id":
                        claim.claim_id,

                    "claim":
                        claim.text,

                    "importance":
                        claim.importance.value,

                    "error_type":
                        error_type.value,
                }
            )

        if not expected_evidence_satisfied(
            claim,
            assessment,
            evidence_map,
        ):
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "claim_id":
                        claim.claim_id,

                    "error_type":
                        GroundednessErrorType
                        .MISSING_EVIDENCE.value,

                    "message":
                        "Expected evidence/source "
                        "was not used.",
                }
            )

        if (
            assessment.evidence_relevance
            < 0.50
        ):
            errors.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "claim_id":
                        claim.claim_id,

                    "error_type":
                        GroundednessErrorType
                        .LOW_EVIDENCE_RELEVANCE
                        .value,

                    "score":
                        assessment
                        .evidence_relevance,
                }
            )

    total_required = len(
        required_claims
    )

    claim_groundedness = (
        calculate_claim_groundedness(
            sample.claims,
            assessment_map,
        )
    )

    weighted_groundedness = (
        calculate_weighted_groundedness(
            sample.claims,
            assessment_map,
        )
    )

    faithfulness = (
        calculate_faithfulness(
            sample.claims,
            assessment_map,
        )
    )

    evidence_coverage = (
        calculate_evidence_coverage(
            sample.claims,
            assessment_map,
        )
    )

    evidence_quality = (
        calculate_evidence_quality(
            sample
        )
    )

    average_entailment = (
        mean(entailment_scores)
        if entailment_scores
        else 0.0
    )

    average_relevance = (
        mean(relevance_scores)
        if relevance_scores
        else 0.0
    )

    average_contradiction = (
        mean(contradiction_scores)
        if contradiction_scores
        else 0.0
    )

    unsupported = (
        ungrounded
        + contradicted
    )

    unsupported_rate = (
        unsupported / total_required
        if total_required
        else 0.0
    )

    contradiction_rate = (
        contradicted / total_required
        if total_required
        else 0.0
    )

    (
        critical_claims,
        grounded_critical,
        critical_groundedness,
    ) = calculate_critical_groundedness(
        sample.claims,
        assessment_map,
    )

    overall = calculate_overall_score(
        weighted_groundedness=(
            weighted_groundedness
        ),
        faithfulness=faithfulness,
        evidence_coverage=(
            evidence_coverage
        ),
        evidence_relevance=(
            average_relevance
        ),
        evidence_quality=(
            evidence_quality
        ),
        contradiction_rate=(
            contradiction_rate
        ),
    )

    return GroundednessSampleResult(
        sample_id=sample.sample_id,

        total_claims=len(
            sample.claims
        ),

        claims_requiring_evidence=(
            total_required
        ),

        grounded_claims=grounded,

        partially_grounded_claims=(
            partial
        ),

        ungrounded_claims=(
            ungrounded
        ),

        contradicted_claims=(
            contradicted
        ),

        claim_groundedness=(
            claim_groundedness
        ),

        weighted_groundedness=(
            weighted_groundedness
        ),

        faithfulness=(
            faithfulness
        ),

        evidence_coverage=(
            evidence_coverage
        ),

        evidence_relevance=round(
            average_relevance,
            4,
        ),

        evidence_quality=(
            evidence_quality
        ),

        average_entailment=round(
            average_entailment,
            4,
        ),

        average_contradiction=round(
            average_contradiction,
            4,
        ),

        unsupported_claim_rate=round(
            unsupported_rate,
            4,
        ),

        contradiction_rate=round(
            contradiction_rate,
            4,
        ),

        critical_claims=(
            critical_claims
        ),

        grounded_critical_claims=(
            grounded_critical
        ),

        critical_groundedness=(
            critical_groundedness
        ),

        overall_score=(
            overall
        ),

        errors=errors,
    )


# =========================================================
# Weighted aggregate helper
# =========================================================

def _weighted_average(
    values: Iterable[
        tuple[float, int]
    ],
    *,
    default: float = 0.0,
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
        return default

    return (
        weighted_sum
        / total_weight
    )


# =========================================================
# Aggregate evaluation
# =========================================================

def evaluate_groundedness(
    samples: list[
        GroundednessEvaluationSample
    ],
) -> GroundednessEvaluationResult:

    if not samples:
        return GroundednessEvaluationResult(
            total_samples=0,

            total_claims=0,
            claims_requiring_evidence=0,

            grounded_claims=0,
            partially_grounded_claims=0,
            ungrounded_claims=0,
            contradicted_claims=0,

            claim_groundedness=0.0,
            weighted_groundedness=0.0,

            faithfulness=0.0,

            evidence_coverage=0.0,
            evidence_relevance=0.0,
            evidence_quality=0.0,

            average_entailment=0.0,
            average_contradiction=0.0,

            unsupported_claim_rate=0.0,
            contradiction_rate=0.0,

            critical_claims=0,
            grounded_critical_claims=0,
            critical_groundedness=None,

            overall_score=0.0,

            per_sample=[],
            errors=[],
        )

    per_sample = [
        evaluate_groundedness_sample(
            sample
        )
        for sample in samples
    ]

    total_claims = sum(
        result.total_claims
        for result in per_sample
    )

    required_claims = sum(
        result.claims_requiring_evidence
        for result in per_sample
    )

    grounded = sum(
        result.grounded_claims
        for result in per_sample
    )

    partial = sum(
        result.partially_grounded_claims
        for result in per_sample
    )

    ungrounded = sum(
        result.ungrounded_claims
        for result in per_sample
    )

    contradicted = sum(
        result.contradicted_claims
        for result in per_sample
    )

    critical_claims = sum(
        result.critical_claims
        for result in per_sample
    )

    grounded_critical = sum(
        result.grounded_critical_claims
        for result in per_sample
    )

    critical_groundedness = (
        round(
            grounded_critical
            / critical_claims,
            4,
        )
        if critical_claims
        else None
    )

    claim_groundedness = (
        (
            grounded
            + 0.5 * partial
        )
        / required_claims
        if required_claims
        else 1.0
    )

    faithfulness = (
        grounded / required_claims
        if required_claims
        else 1.0
    )

    unsupported = (
        ungrounded
        + contradicted
    )

    unsupported_rate = (
        unsupported / required_claims
        if required_claims
        else 0.0
    )

    contradiction_rate = (
        contradicted / required_claims
        if required_claims
        else 0.0
    )

    weighted_groundedness = (
        _weighted_average(
            (
                result.weighted_groundedness,
                result.claims_requiring_evidence,
            )
            for result in per_sample
        )
    )

    evidence_coverage = (
        _weighted_average(
            (
                result.evidence_coverage,
                result.claims_requiring_evidence,
            )
            for result in per_sample
        )
    )

    evidence_relevance = (
        _weighted_average(
            (
                result.evidence_relevance,
                result.claims_requiring_evidence,
            )
            for result in per_sample
        )
    )

    evidence_quality = (
        _weighted_average(
            (
                result.evidence_quality,
                result.claims_requiring_evidence,
            )
            for result in per_sample
        )
    )

    average_entailment = (
        _weighted_average(
            (
                result.average_entailment,
                result.claims_requiring_evidence,
            )
            for result in per_sample
        )
    )

    average_contradiction = (
        _weighted_average(
            (
                result.average_contradiction,
                result.claims_requiring_evidence,
            )
            for result in per_sample
        )
    )

    overall = calculate_overall_score(
        weighted_groundedness=(
            weighted_groundedness
        ),
        faithfulness=faithfulness,
        evidence_coverage=(
            evidence_coverage
        ),
        evidence_relevance=(
            evidence_relevance
        ),
        evidence_quality=(
            evidence_quality
        ),
        contradiction_rate=(
            contradiction_rate
        ),
    )

    errors = [
        error
        for result in per_sample
        for error in result.errors
    ]

    return GroundednessEvaluationResult(
        total_samples=len(samples),

        total_claims=(
            total_claims
        ),

        claims_requiring_evidence=(
            required_claims
        ),

        grounded_claims=(
            grounded
        ),

        partially_grounded_claims=(
            partial
        ),

        ungrounded_claims=(
            ungrounded
        ),

        contradicted_claims=(
            contradicted
        ),

        claim_groundedness=round(
            claim_groundedness,
            4,
        ),

        weighted_groundedness=round(
            weighted_groundedness,
            4,
        ),

        faithfulness=round(
            faithfulness,
            4,
        ),

        evidence_coverage=round(
            evidence_coverage,
            4,
        ),

        evidence_relevance=round(
            evidence_relevance,
            4,
        ),

        evidence_quality=round(
            evidence_quality,
            4,
        ),

        average_entailment=round(
            average_entailment,
            4,
        ),

        average_contradiction=round(
            average_contradiction,
            4,
        ),

        unsupported_claim_rate=round(
            unsupported_rate,
            4,
        ),

        contradiction_rate=round(
            contradiction_rate,
            4,
        ),

        critical_claims=(
            critical_claims
        ),

        grounded_critical_claims=(
            grounded_critical
        ),

        critical_groundedness=(
            critical_groundedness
        ),

        overall_score=(
            overall
        ),

        per_sample=(
            per_sample
        ),

        errors=errors,
    )


# =========================================================
# Production quality gate
# =========================================================

def groundedness_quality_gate(
    result: GroundednessEvaluationResult,
    *,
    minimum_groundedness: float = 0.90,
    minimum_faithfulness: float = 0.90,
    minimum_critical_groundedness: float = 1.0,
    maximum_unsupported_rate: float = 0.05,
    maximum_contradiction_rate: float = 0.01,
) -> dict[str, Any]:

    minimum_groundedness = _probability(
        minimum_groundedness,
        "minimum_groundedness",
    )

    minimum_faithfulness = _probability(
        minimum_faithfulness,
        "minimum_faithfulness",
    )

    minimum_critical_groundedness = _probability(
        minimum_critical_groundedness,
        "minimum_critical_groundedness",
    )

    maximum_unsupported_rate = _probability(
        maximum_unsupported_rate,
        "maximum_unsupported_rate",
    )

    maximum_contradiction_rate = _probability(
        maximum_contradiction_rate,
        "maximum_contradiction_rate",
    )

    checks = {
        "groundedness": (
            result.weighted_groundedness
            >= minimum_groundedness
        ),

        "faithfulness": (
            result.faithfulness
            >= minimum_faithfulness
        ),

        "unsupported_claim_rate": (
            result.unsupported_claim_rate
            <= maximum_unsupported_rate
        ),

        "contradiction_rate": (
            result.contradiction_rate
            <= maximum_contradiction_rate
        ),
    }

    if (
        result.critical_groundedness
        is not None
    ):
        checks[
            "critical_groundedness"
        ] = (
            result.critical_groundedness
            >= minimum_critical_groundedness
        )

    return {
        "passed": all(
            checks.values()
        ),

        "checks": checks,

        "metrics": {
            "weighted_groundedness":
                result.weighted_groundedness,

            "faithfulness":
                result.faithfulness,

            "critical_groundedness":
                result.critical_groundedness,

            "unsupported_claim_rate":
                result.unsupported_claim_rate,

            "contradiction_rate":
                result.contradiction_rate,

            "overall_score":
                result.overall_score,
        },
    }


# =========================================================
# Evaluator service
# =========================================================

class GroundednessEvaluator:

    def evaluate_sample(
        self,
        sample: GroundednessEvaluationSample,
    ) -> GroundednessSampleResult:

        return evaluate_groundedness_sample(
            sample
        )

    def evaluate(
        self,
        samples: list[
            GroundednessEvaluationSample
        ],
    ) -> GroundednessEvaluationResult:

        return evaluate_groundedness(
            samples
        )

    def quality_gate(
        self,
        result: GroundednessEvaluationResult,
        **kwargs: Any,
    ) -> dict[str, Any]:

        return groundedness_quality_gate(
            result,
            **kwargs,
        )


groundedness_evaluator = (
    GroundednessEvaluator()
)


__all__ = [
    "GroundednessLabel",
    "ClaimImportance",
    "EvidenceType",
    "GroundednessErrorType",
    "Evidence",
    "GroundednessClaim",
    "ClaimGroundingAssessment",
    "GroundednessEvaluationSample",
    "GroundednessSampleResult",
    "GroundednessEvaluationResult",
    "build_claim_map",
    "build_evidence_map",
    "build_assessment_map",
    "calculate_claim_groundedness",
    "calculate_weighted_groundedness",
    "calculate_faithfulness",
    "calculate_evidence_coverage",
    "calculate_critical_groundedness",
    "calculate_evidence_quality",
    "expected_evidence_satisfied",
    "calculate_overall_score",
    "evaluate_groundedness_sample",
    "evaluate_groundedness",
    "groundedness_quality_gate",
    "GroundednessEvaluator",
    "groundedness_evaluator",
]