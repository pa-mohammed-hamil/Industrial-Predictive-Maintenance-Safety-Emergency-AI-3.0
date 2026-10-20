# app/evaluation/retrieval_evaluation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite, log2
from statistics import mean
from typing import Any, Iterable, Sequence


# =========================================================
# Enums
# =========================================================

class RetrievalMethod(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    GRAPH = "graph"
    MULTI_QUERY = "multi_query"
    PARENT_CHILD = "parent_child"
    SELF_RAG = "self_rag"
    CORRECTIVE_RAG = "corrective_rag"
    OTHER = "other"


class RetrievalSourceType(str, Enum):
    OEM_MANUAL = "oem_manual"
    MAINTENANCE_MANUAL = "maintenance_manual"
    SAFETY_PROCEDURE = "safety_procedure"
    SOP = "sop"
    TECHNICAL_DOCUMENT = "technical_document"
    WORK_ORDER = "work_order"
    INCIDENT_REPORT = "incident_report"
    INSPECTION_REPORT = "inspection_report"
    SENSOR_REPORT = "sensor_report"
    KNOWLEDGE_BASE = "knowledge_base"
    DATABASE = "database"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    WEB = "web"
    OTHER = "other"


# =========================================================
# Retrieved item
# =========================================================

@dataclass(slots=True)
class RetrievedDocument:
    document_id: str
    chunk_id: str

    rank: int

    score: float | None = None
    reranker_score: float | None = None

    source_id: str | None = None

    source_type: RetrievalSourceType = (
        RetrievalSourceType.OTHER
    )

    title: str | None = None
    page: int | None = None

    text: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def unique_id(self) -> str:
        return f"{self.document_id}:{self.chunk_id}"


# =========================================================
# Relevant ground truth
# =========================================================

@dataclass(slots=True)
class RelevantDocument:
    document_id: str

    chunk_id: str | None = None

    relevance_grade: float = 1.0

    source_id: str | None = None

    required: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Query sample
# =========================================================

@dataclass(slots=True)
class RetrievalEvaluationSample:
    query_id: str
    query: str

    retrieved: list[RetrievedDocument]

    relevant: list[RelevantDocument]

    retrieval_method: RetrievalMethod = (
        RetrievalMethod.HYBRID
    )

    latency_ms: float | None = None

    asset_id: str | None = None
    site_id: str | None = None

    conversation_id: str | None = None
    retrieval_id: str | None = None

    embedding_model: str | None = None
    reranker_model: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Per-query result
# =========================================================

@dataclass(slots=True)
class RetrievalQueryResult:
    query_id: str

    retrieved_count: int
    relevant_count: int

    relevant_retrieved: int

    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]

    reciprocal_rank: float
    average_precision: float

    required_source_recall: float | None

    duplicate_rate: float

    first_relevant_rank: int | None

    latency_ms: float | None

    missed_relevant: list[str]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================================================
# Aggregate result
# =========================================================

@dataclass(slots=True)
class RetrievalEvaluationResult:
    total_queries: int

    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]

    mean_reciprocal_rank: float
    mean_average_precision: float

    required_source_recall: float | None

    zero_result_rate: float
    zero_relevant_hit_rate: float

    duplicate_rate: float

    average_first_relevant_rank: float | None

    average_latency_ms: float | None
    p95_latency_ms: float | None

    total_relevant_documents: int
    total_relevant_retrieved: int

    overall_recall: float

    per_query: list[RetrievalQueryResult]

    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["per_query"] = [
            item.to_dict()
            for item in self.per_query
        ]

        return result


# =========================================================
# Validation
# =========================================================

def _finite(
    value: float,
    name: str,
) -> float:

    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    return value


def _non_negative(
    value: float,
    name: str,
) -> float:

    value = _finite(
        value,
        name,
    )

    if value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return value


def validate_sample(
    sample: RetrievalEvaluationSample,
) -> None:

    if not sample.query_id:
        raise ValueError(
            "query_id cannot be empty."
        )

    if not sample.query.strip():
        raise ValueError(
            "query cannot be empty."
        )

    ranks: list[int] = []

    for item in sample.retrieved:
        if item.rank < 1:
            raise ValueError(
                "Retrieved document rank must be >= 1."
            )

        ranks.append(
            item.rank
        )

        if item.score is not None:
            _finite(
                item.score,
                "retrieval score",
            )

        if item.reranker_score is not None:
            _finite(
                item.reranker_score,
                "reranker score",
            )

    if len(ranks) != len(set(ranks)):
        raise ValueError(
            f"Duplicate retrieval ranks in "
            f"query {sample.query_id}."
        )

    for item in sample.relevant:
        _non_negative(
            item.relevance_grade,
            "relevance_grade",
        )

    if sample.latency_ms is not None:
        _non_negative(
            sample.latency_ms,
            "latency_ms",
        )


# =========================================================
# Matching
# =========================================================

def _matches(
    retrieved: RetrievedDocument,
    relevant: RelevantDocument,
) -> bool:
    """
    Chunk-level match when the ground truth contains a
    chunk_id; otherwise document-level relevance.
    """

    if (
        retrieved.document_id
        != relevant.document_id
    ):
        return False

    if relevant.chunk_id is None:
        return True

    return (
        retrieved.chunk_id
        == relevant.chunk_id
    )


def is_relevant(
    item: RetrievedDocument,
    relevant: Sequence[RelevantDocument],
) -> bool:

    return any(
        _matches(
            item,
            ground_truth,
        )
        for ground_truth in relevant
    )


def relevance_grade(
    item: RetrievedDocument,
    relevant: Sequence[RelevantDocument],
) -> float:

    grades = [
        ground_truth.relevance_grade
        for ground_truth in relevant
        if _matches(
            item,
            ground_truth,
        )
    ]

    if not grades:
        return 0.0

    return max(grades)


def _ordered_retrieved(
    sample: RetrievalEvaluationSample,
) -> list[RetrievedDocument]:

    return sorted(
        sample.retrieved,
        key=lambda item: item.rank,
    )


# =========================================================
# Precision@K
# =========================================================

def precision_at_k(
    sample: RetrievalEvaluationSample,
    k: int,
) -> float:

    if k < 1:
        raise ValueError(
            "k must be >= 1."
        )

    retrieved = (
        _ordered_retrieved(sample)[:k]
    )

    if not retrieved:
        return 0.0

    relevant_hits = sum(
        is_relevant(
            item,
            sample.relevant,
        )
        for item in retrieved
    )

    # Standard P@K denominator is K.
    return round(
        relevant_hits / k,
        6,
    )


# =========================================================
# Recall@K
# =========================================================

def recall_at_k(
    sample: RetrievalEvaluationSample,
    k: int,
) -> float:

    if k < 1:
        raise ValueError(
            "k must be >= 1."
        )

    if not sample.relevant:
        return 1.0

    retrieved = (
        _ordered_retrieved(sample)[:k]
    )

    matched_ground_truth: set[int] = set()

    for index, relevant in enumerate(
        sample.relevant
    ):
        if any(
            _matches(
                item,
                relevant,
            )
            for item in retrieved
        ):
            matched_ground_truth.add(
                index
            )

    return round(
        len(matched_ground_truth)
        / len(sample.relevant),
        6,
    )


# =========================================================
# Hit Rate@K
# =========================================================

def hit_rate_at_k(
    sample: RetrievalEvaluationSample,
    k: int,
) -> float:

    if k < 1:
        raise ValueError(
            "k must be >= 1."
        )

    retrieved = (
        _ordered_retrieved(sample)[:k]
    )

    return (
        1.0
        if any(
            is_relevant(
                item,
                sample.relevant,
            )
            for item in retrieved
        )
        else 0.0
    )


# =========================================================
# Reciprocal Rank / MRR
# =========================================================

def reciprocal_rank(
    sample: RetrievalEvaluationSample,
) -> float:

    for item in _ordered_retrieved(
        sample
    ):
        if is_relevant(
            item,
            sample.relevant,
        ):
            return round(
                1.0 / item.rank,
                6,
            )

    return 0.0


def first_relevant_rank(
    sample: RetrievalEvaluationSample,
) -> int | None:

    for item in _ordered_retrieved(
        sample
    ):
        if is_relevant(
            item,
            sample.relevant,
        ):
            return item.rank

    return None


# =========================================================
# Average Precision / MAP
# =========================================================

def average_precision(
    sample: RetrievalEvaluationSample,
) -> float:

    if not sample.relevant:
        return 1.0

    retrieved = _ordered_retrieved(
        sample
    )

    relevant_hits = 0
    precision_sum = 0.0

    # Avoid rewarding duplicate retrievals of the
    # same ground-truth item multiple times.
    matched_relevant: set[int] = set()

    for position, item in enumerate(
        retrieved,
        start=1,
    ):
        matched_index: int | None = None

        for index, relevant in enumerate(
            sample.relevant
        ):
            if (
                index not in matched_relevant
                and _matches(
                    item,
                    relevant,
                )
            ):
                matched_index = index
                break

        if matched_index is None:
            continue

        matched_relevant.add(
            matched_index
        )

        relevant_hits += 1

        precision_sum += (
            relevant_hits
            / position
        )

    denominator = len(
        sample.relevant
    )

    return round(
        precision_sum / denominator,
        6,
    )


# =========================================================
# DCG / NDCG
# =========================================================

def dcg_at_k(
    grades: Sequence[float],
    k: int,
) -> float:

    if k < 1:
        raise ValueError(
            "k must be >= 1."
        )

    score = 0.0

    for index, grade in enumerate(
        grades[:k],
        start=1,
    ):
        gain = (
            (2.0 ** grade)
            - 1.0
        )

        discount = log2(
            index + 1
        )

        score += (
            gain / discount
        )

    return score


def ndcg_at_k(
    sample: RetrievalEvaluationSample,
    k: int,
) -> float:

    if k < 1:
        raise ValueError(
            "k must be >= 1."
        )

    retrieved = (
        _ordered_retrieved(sample)[:k]
    )

    actual_grades = [
        relevance_grade(
            item,
            sample.relevant,
        )
        for item in retrieved
    ]

    ideal_grades = sorted(
        (
            item.relevance_grade
            for item in sample.relevant
        ),
        reverse=True,
    )[:k]

    actual_dcg = dcg_at_k(
        actual_grades,
        k,
    )

    ideal_dcg = dcg_at_k(
        ideal_grades,
        k,
    )

    if ideal_dcg == 0:
        return 1.0

    return round(
        actual_dcg / ideal_dcg,
        6,
    )


# =========================================================
# Required evidence/source recall
# =========================================================

def calculate_required_source_recall(
    sample: RetrievalEvaluationSample,
) -> float | None:

    required = [
        item
        for item in sample.relevant
        if item.required
    ]

    if not required:
        return None

    retrieved = _ordered_retrieved(
        sample
    )

    matched = sum(
        any(
            _matches(
                result,
                relevant,
            )
            for result in retrieved
        )
        for relevant in required
    )

    return round(
        matched / len(required),
        6,
    )


# =========================================================
# Duplicate retrievals
# =========================================================

def calculate_duplicate_rate(
    retrieved: Sequence[
        RetrievedDocument
    ],
) -> float:

    if not retrieved:
        return 0.0

    seen: set[str] = set()
    duplicates = 0

    for item in retrieved:
        identifier = item.unique_id

        if identifier in seen:
            duplicates += 1
        else:
            seen.add(identifier)

    return round(
        duplicates / len(retrieved),
        6,
    )


# =========================================================
# Missed relevant evidence
# =========================================================

def get_missed_relevant(
    sample: RetrievalEvaluationSample,
) -> list[str]:

    missed: list[str] = []

    for relevant in sample.relevant:
        found = any(
            _matches(
                item,
                relevant,
            )
            for item in sample.retrieved
        )

        if found:
            continue

        identifier = relevant.document_id

        if relevant.chunk_id:
            identifier += (
                f":{relevant.chunk_id}"
            )

        missed.append(
            identifier
        )

    return missed


# =========================================================
# Per-query evaluation
# =========================================================

def evaluate_retrieval_query(
    sample: RetrievalEvaluationSample,
    *,
    k_values: Sequence[int] = (
        1,
        3,
        5,
        10,
    ),
) -> RetrievalQueryResult:

    validate_sample(
        sample
    )

    if not k_values:
        raise ValueError(
            "k_values cannot be empty."
        )

    normalized_k = sorted(
        set(k_values)
    )

    if any(
        k < 1
        for k in normalized_k
    ):
        raise ValueError(
            "All k values must be >= 1."
        )

    retrieved = _ordered_retrieved(
        sample
    )

    matched_ground_truth: set[int] = set()

    for index, relevant in enumerate(
        sample.relevant
    ):
        if any(
            _matches(
                item,
                relevant,
            )
            for item in retrieved
        ):
            matched_ground_truth.add(
                index
            )

    errors: list[
        dict[str, Any]
    ] = []

    missed = get_missed_relevant(
        sample
    )

    for identifier in missed:
        errors.append(
            {
                "query_id":
                    sample.query_id,

                "type":
                    "missed_relevant_document",

                "document":
                    identifier,
            }
        )

    if not retrieved:
        errors.append(
            {
                "query_id":
                    sample.query_id,

                "type":
                    "zero_retrieval_results",
            }
        )

    duplicate_rate = (
        calculate_duplicate_rate(
            retrieved
        )
    )

    if duplicate_rate > 0:
        errors.append(
            {
                "query_id":
                    sample.query_id,

                "type":
                    "duplicate_retrieval",

                "duplicate_rate":
                    duplicate_rate,
            }
        )

    return RetrievalQueryResult(
        query_id=sample.query_id,

        retrieved_count=len(
            retrieved
        ),

        relevant_count=len(
            sample.relevant
        ),

        relevant_retrieved=len(
            matched_ground_truth
        ),

        precision_at_k={
            k: precision_at_k(
                sample,
                k,
            )
            for k in normalized_k
        },

        recall_at_k={
            k: recall_at_k(
                sample,
                k,
            )
            for k in normalized_k
        },

        hit_rate_at_k={
            k: hit_rate_at_k(
                sample,
                k,
            )
            for k in normalized_k
        },

        ndcg_at_k={
            k: ndcg_at_k(
                sample,
                k,
            )
            for k in normalized_k
        },

        reciprocal_rank=(
            reciprocal_rank(
                sample
            )
        ),

        average_precision=(
            average_precision(
                sample
            )
        ),

        required_source_recall=(
            calculate_required_source_recall(
                sample
            )
        ),

        duplicate_rate=(
            duplicate_rate
        ),

        first_relevant_rank=(
            first_relevant_rank(
                sample
            )
        ),

        latency_ms=(
            sample.latency_ms
        ),

        missed_relevant=(
            missed
        ),

        errors=errors,
    )


# =========================================================
# Percentile
# =========================================================

def _percentile(
    values: Sequence[float],
    percentile: float,
) -> float | None:

    if not values:
        return None

    ordered = sorted(
        values
    )

    percentile = max(
        0.0,
        min(
            1.0,
            percentile,
        ),
    )

    # Nearest-rank percentile.
    from math import ceil

    rank = max(
        1,
        ceil(
            percentile
            * len(ordered)
        ),
    )

    return ordered[
        rank - 1
    ]


# =========================================================
# Aggregate evaluation
# =========================================================

def evaluate_retrieval(
    samples: list[
        RetrievalEvaluationSample
    ],
    *,
    k_values: Sequence[int] = (
        1,
        3,
        5,
        10,
    ),
) -> RetrievalEvaluationResult:

    normalized_k = sorted(
        set(k_values)
    )

    if not normalized_k:
        raise ValueError(
            "k_values cannot be empty."
        )

    if any(
        k < 1
        for k in normalized_k
    ):
        raise ValueError(
            "All k values must be >= 1."
        )

    if not samples:
        return RetrievalEvaluationResult(
            total_queries=0,

            precision_at_k={
                k: 0.0
                for k in normalized_k
            },

            recall_at_k={
                k: 0.0
                for k in normalized_k
            },

            hit_rate_at_k={
                k: 0.0
                for k in normalized_k
            },

            ndcg_at_k={
                k: 0.0
                for k in normalized_k
            },

            mean_reciprocal_rank=0.0,
            mean_average_precision=0.0,

            required_source_recall=None,

            zero_result_rate=0.0,
            zero_relevant_hit_rate=0.0,

            duplicate_rate=0.0,

            average_first_relevant_rank=None,

            average_latency_ms=None,
            p95_latency_ms=None,

            total_relevant_documents=0,
            total_relevant_retrieved=0,

            overall_recall=0.0,

            per_query=[],
            errors=[],
        )

    per_query = [
        evaluate_retrieval_query(
            sample,
            k_values=normalized_k,
        )
        for sample in samples
    ]

    precision_metrics = {
        k: round(
            mean(
                result.precision_at_k[k]
                for result in per_query
            ),
            6,
        )
        for k in normalized_k
    }

    recall_metrics = {
        k: round(
            mean(
                result.recall_at_k[k]
                for result in per_query
            ),
            6,
        )
        for k in normalized_k
    }

    hit_metrics = {
        k: round(
            mean(
                result.hit_rate_at_k[k]
                for result in per_query
            ),
            6,
        )
        for k in normalized_k
    }

    ndcg_metrics = {
        k: round(
            mean(
                result.ndcg_at_k[k]
                for result in per_query
            ),
            6,
        )
        for k in normalized_k
    }

    mrr = round(
        mean(
            result.reciprocal_rank
            for result in per_query
        ),
        6,
    )

    mean_ap = round(
        mean(
            result.average_precision
            for result in per_query
        ),
        6,
    )

    required_recalls = [
        result.required_source_recall
        for result in per_query
        if (
            result.required_source_recall
            is not None
        )
    ]

    required_recall = (
        round(
            mean(required_recalls),
            6,
        )
        if required_recalls
        else None
    )

    zero_results = sum(
        result.retrieved_count == 0
        for result in per_query
    )

    zero_relevant_hits = sum(
        (
            result.relevant_count > 0
            and result.relevant_retrieved == 0
        )
        for result in per_query
    )

    duplicate_rate = round(
        mean(
            result.duplicate_rate
            for result in per_query
        ),
        6,
    )

    first_ranks = [
        result.first_relevant_rank
        for result in per_query
        if (
            result.first_relevant_rank
            is not None
        )
    ]

    average_first_rank = (
        round(
            mean(first_ranks),
            4,
        )
        if first_ranks
        else None
    )

    latencies = [
        float(result.latency_ms)
        for result in per_query
        if (
            result.latency_ms
            is not None
        )
    ]

    average_latency = (
        round(
            mean(latencies),
            4,
        )
        if latencies
        else None
    )

    p95_latency = (
        _percentile(
            latencies,
            0.95,
        )
    )

    if p95_latency is not None:
        p95_latency = round(
            p95_latency,
            4,
        )

    total_relevant = sum(
        result.relevant_count
        for result in per_query
    )

    total_retrieved_relevant = sum(
        result.relevant_retrieved
        for result in per_query
    )

    overall_recall = (
        total_retrieved_relevant
        / total_relevant
        if total_relevant
        else 1.0
    )

    errors = [
        error
        for result in per_query
        for error in result.errors
    ]

    return RetrievalEvaluationResult(
        total_queries=len(
            samples
        ),

        precision_at_k=(
            precision_metrics
        ),

        recall_at_k=(
            recall_metrics
        ),

        hit_rate_at_k=(
            hit_metrics
        ),

        ndcg_at_k=(
            ndcg_metrics
        ),

        mean_reciprocal_rank=(
            mrr
        ),

        mean_average_precision=(
            mean_ap
        ),

        required_source_recall=(
            required_recall
        ),

        zero_result_rate=round(
            zero_results
            / len(samples),
            6,
        ),

        zero_relevant_hit_rate=round(
            zero_relevant_hits
            / len(samples),
            6,
        ),

        duplicate_rate=(
            duplicate_rate
        ),

        average_first_relevant_rank=(
            average_first_rank
        ),

        average_latency_ms=(
            average_latency
        ),

        p95_latency_ms=(
            p95_latency
        ),

        total_relevant_documents=(
            total_relevant
        ),

        total_relevant_retrieved=(
            total_retrieved_relevant
        ),

        overall_recall=round(
            overall_recall,
            6,
        ),

        per_query=(
            per_query
        ),

        errors=errors,
    )


# =========================================================
# Quality gate
# =========================================================

def retrieval_quality_gate(
    result: RetrievalEvaluationResult,
    *,
    recall_k: int = 5,
    minimum_recall: float = 0.85,
    minimum_mrr: float = 0.75,
    minimum_ndcg: float = 0.80,
    minimum_required_source_recall: float = 0.95,
    maximum_zero_hit_rate: float = 0.05,
    maximum_duplicate_rate: float = 0.05,
    maximum_p95_latency_ms: float | None = None,
) -> dict[str, Any]:

    if recall_k not in result.recall_at_k:
        raise ValueError(
            f"K={recall_k} was not evaluated."
        )

    checks: dict[str, bool] = {
        f"recall_at_{recall_k}": (
            result.recall_at_k[
                recall_k
            ]
            >= minimum_recall
        ),

        "mrr": (
            result.mean_reciprocal_rank
            >= minimum_mrr
        ),

        f"ndcg_at_{recall_k}": (
            result.ndcg_at_k[
                recall_k
            ]
            >= minimum_ndcg
        ),

        "zero_relevant_hit_rate": (
            result.zero_relevant_hit_rate
            <= maximum_zero_hit_rate
        ),

        "duplicate_rate": (
            result.duplicate_rate
            <= maximum_duplicate_rate
        ),
    }

    if (
        result.required_source_recall
        is not None
    ):
        checks[
            "required_source_recall"
        ] = (
            result.required_source_recall
            >= minimum_required_source_recall
        )

    if (
        maximum_p95_latency_ms
        is not None
        and result.p95_latency_ms
        is not None
    ):
        checks[
            "p95_latency"
        ] = (
            result.p95_latency_ms
            <= maximum_p95_latency_ms
        )

    return {
        "passed": all(
            checks.values()
        ),

        "checks": checks,

        "metrics": {
            f"recall_at_{recall_k}":
                result.recall_at_k[
                    recall_k
                ],

            f"precision_at_{recall_k}":
                result.precision_at_k[
                    recall_k
                ],

            f"ndcg_at_{recall_k}":
                result.ndcg_at_k[
                    recall_k
                ],

            "mrr":
                result.mean_reciprocal_rank,

            "map":
                result.mean_average_precision,

            "required_source_recall":
                result.required_source_recall,

            "zero_relevant_hit_rate":
                result.zero_relevant_hit_rate,

            "duplicate_rate":
                result.duplicate_rate,

            "p95_latency_ms":
                result.p95_latency_ms,
        },
    }


# =========================================================
# Compare retrievers
# =========================================================

def compare_retrievers(
    retrievers: dict[
        str,
        list[RetrievalEvaluationSample],
    ],
    *,
    k: int = 5,
) -> list[dict[str, Any]]:

    results: list[
        dict[str, Any]
    ] = []

    for name, samples in retrievers.items():
        evaluation = evaluate_retrieval(
            samples,
            k_values=(
                1,
                3,
                k,
                10,
            ),
        )

        results.append(
            {
                "retriever":
                    name,

                "queries":
                    evaluation.total_queries,

                f"precision@{k}":
                    evaluation.precision_at_k[
                        k
                    ],

                f"recall@{k}":
                    evaluation.recall_at_k[
                        k
                    ],

                f"ndcg@{k}":
                    evaluation.ndcg_at_k[
                        k
                    ],

                "mrr":
                    evaluation
                    .mean_reciprocal_rank,

                "map":
                    evaluation
                    .mean_average_precision,

                "overall_recall":
                    evaluation.overall_recall,

                "required_source_recall":
                    evaluation
                    .required_source_recall,

                "zero_hit_rate":
                    evaluation
                    .zero_relevant_hit_rate,

                "p95_latency_ms":
                    evaluation
                    .p95_latency_ms,
            }
        )

    # Strong retrieval quality first.
    results.sort(
        key=lambda item: (
            item[f"recall@{k}"],
            item[f"ndcg@{k}"],
            item["mrr"],
        ),
        reverse=True,
    )

    return results


# =========================================================
# Evaluator
# =========================================================

class RetrievalEvaluator:

    def evaluate_query(
        self,
        sample: RetrievalEvaluationSample,
        *,
        k_values: Sequence[int] = (
            1,
            3,
            5,
            10,
        ),
    ) -> RetrievalQueryResult:

        return evaluate_retrieval_query(
            sample,
            k_values=k_values,
        )

    def evaluate(
        self,
        samples: list[
            RetrievalEvaluationSample
        ],
        *,
        k_values: Sequence[int] = (
            1,
            3,
            5,
            10,
        ),
    ) -> RetrievalEvaluationResult:

        return evaluate_retrieval(
            samples,
            k_values=k_values,
        )

    def quality_gate(
        self,
        result: RetrievalEvaluationResult,
        **kwargs: Any,
    ) -> dict[str, Any]:

        return retrieval_quality_gate(
            result,
            **kwargs,
        )

    def compare(
        self,
        retrievers: dict[
            str,
            list[
                RetrievalEvaluationSample
            ],
        ],
        *,
        k: int = 5,
    ) -> list[dict[str, Any]]:

        return compare_retrievers(
            retrievers,
            k=k,
        )


retrieval_evaluator = RetrievalEvaluator()


__all__ = [
    "RetrievalMethod",
    "RetrievalSourceType",
    "RetrievedDocument",
    "RelevantDocument",
    "RetrievalEvaluationSample",
    "RetrievalQueryResult",
    "RetrievalEvaluationResult",
    "validate_sample",
    "is_relevant",
    "relevance_grade",
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "first_relevant_rank",
    "average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "calculate_required_source_recall",
    "calculate_duplicate_rate",
    "get_missed_relevant",
    "evaluate_retrieval_query",
    "evaluate_retrieval",
    "retrieval_quality_gate",
    "compare_retrievers",
    "RetrievalEvaluator",
    "retrieval_evaluator",
]