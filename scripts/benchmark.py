"""
app/ai/rag/benchmark.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

RAG Retrieval & Answer Quality Benchmarking
===========================================

Python 3.11+

Dependencies:
    pip install "pydantic>=2,<3"

Features:
- Retrieval Precision@K
- Retrieval Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Citation precision and recall
- Answer token F1
- Exact match
- Groundedness evaluation using optional judge
- Faithfulness evaluation using optional judge
- Hallucination risk reporting
- Safety evaluation using optional judge
- Retrieval and generation latency
- Benchmark dataset validation
- Model and pipeline comparisons
- JSON and CSV export
- Configurable acceptance thresholds
- example()

Important:
    Lexical answer scores are not semantic correctness.
    LLM judge scores require a trusted evaluation model.
    Safety and hallucination metrics must not be inferred
    solely from token overlap.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time

from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE,
    )


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None

    return float(statistics.mean(values))


def validate_score(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Score must be finite")

    if not 0 <= value <= 1:
        raise ValueError("Score must be between 0 and 1")

    return value


# ============================================================
# ENUMS
# ============================================================

class BenchmarkTaskType(str, Enum):
    EQUIPMENT_MANUAL_QA = "equipment_manual_qa"
    MAINTENANCE_QA = "maintenance_qa"
    SAFETY_QA = "safety_qa"
    POLICY_QA = "policy_qa"
    TROUBLESHOOTING_QA = "troubleshooting_qa"
    INCIDENT_QA = "incident_qa"
    GENERAL_RAG = "general_rag"


class BenchmarkStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class JudgeMetric(str, Enum):
    GROUNDEDNESS = "groundedness"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    SAFETY = "safety"


# ============================================================
# BASE MODEL
# ============================================================

class BenchmarkBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ============================================================
# BENCHMARK DOCUMENT
# ============================================================

class BenchmarkDocument(BenchmarkBaseModel):
    document_id: str = Field(min_length=1)

    text: str = Field(min_length=1)

    score: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    source: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# BENCHMARK CASE
# ============================================================

class RAGBenchmarkCase(BenchmarkBaseModel):
    case_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    question: str = Field(min_length=3)

    reference_answer: str | None = None

    relevant_document_ids: list[str] = Field(
        default_factory=list
    )

    relevance_grades: dict[str, int] = Field(
        default_factory=dict
    )

    expected_citation_ids: list[str] = Field(
        default_factory=list
    )

    task_type: BenchmarkTaskType = (
        BenchmarkTaskType.GENERAL_RAG
    )

    company_id: str | None = None
    site_id: str | None = None
    equipment_id: str | None = None

    safety_critical: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator(
        "relevant_document_ids",
        "expected_citation_ids",
    )
    @classmethod
    def unique_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError(
                "Duplicate document or citation IDs"
            )

        return value

    @field_validator("relevance_grades")
    @classmethod
    def validate_grades(
        cls,
        value: dict[str, int],
    ) -> dict[str, int]:
        if any(
            grade < 0 or grade > 3
            for grade in value.values()
        ):
            raise ValueError(
                "Relevance grades must be 0-3"
            )

        return value


# ============================================================
# RAG PIPELINE OUTPUT
# ============================================================

class RAGPipelineOutput(BenchmarkBaseModel):
    answer: str = ""

    retrieved_documents: list[
        BenchmarkDocument
    ] = Field(default_factory=list)

    citation_ids: list[str] = Field(
        default_factory=list
    )

    retrieval_latency_ms: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    generation_latency_ms: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    total_latency_ms: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    @field_validator("citation_ids")
    @classmethod
    def unique_citations(
        cls,
        value: list[str],
    ) -> list[str]:
        return list(dict.fromkeys(value))


# ============================================================
# RETRIEVAL METRICS
# ============================================================

class RetrievalMetrics:
    @staticmethod
    def precision_at_k(
        retrieved: list[str],
        relevant: set[str],
        k: int,
    ) -> float:
        if k <= 0:
            raise ValueError("k must be positive")

        top_k = retrieved[:k]

        hits = sum(
            document_id in relevant
            for document_id in top_k
        )

        return hits / k

    @staticmethod
    def recall_at_k(
        retrieved: list[str],
        relevant: set[str],
        k: int,
    ) -> float | None:
        if k <= 0:
            raise ValueError("k must be positive")

        if not relevant:
            return None

        return (
            len(set(retrieved[:k]) & relevant)
            / len(relevant)
        )

    @staticmethod
    def hit_rate_at_k(
        retrieved: list[str],
        relevant: set[str],
        k: int,
    ) -> float | None:
        if not relevant:
            return None

        return float(
            bool(set(retrieved[:k]) & relevant)
        )

    @staticmethod
    def reciprocal_rank(
        retrieved: list[str],
        relevant: set[str],
    ) -> float | None:
        if not relevant:
            return None

        for rank, document_id in enumerate(
            retrieved,
            start=1,
        ):
            if document_id in relevant:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def average_precision(
        retrieved: list[str],
        relevant: set[str],
    ) -> float | None:
        if not relevant:
            return None

        hits = 0
        precision_sum = 0.0

        for rank, document_id in enumerate(
            retrieved,
            start=1,
        ):
            if document_id in relevant:
                hits += 1
                precision_sum += hits / rank

        return precision_sum / len(relevant)

    @staticmethod
    def ndcg_at_k(
        retrieved: list[str],
        relevance_grades: dict[str, int],
        k: int,
    ) -> float | None:
        if k <= 0:
            raise ValueError("k must be positive")

        if not relevance_grades:
            return None

        def dcg(grades: list[int]) -> float:
            return sum(
                (2 ** grade - 1)
                / math.log2(index + 2)
                for index, grade in enumerate(grades)
            )

        actual_grades = [
            relevance_grades.get(
                document_id,
                0,
            )
            for document_id in retrieved[:k]
        ]

        ideal_grades = sorted(
            relevance_grades.values(),
            reverse=True,
        )[:k]

        ideal_dcg = dcg(ideal_grades)

        if ideal_dcg == 0:
            return None

        return dcg(actual_grades) / ideal_dcg


# ============================================================
# ANSWER METRICS
# ============================================================

class AnswerMetrics:
    @staticmethod
    def exact_match(
        prediction: str,
        reference: str,
    ) -> float:
        return float(
            " ".join(tokenize(prediction))
            == " ".join(tokenize(reference))
        )

    @staticmethod
    def token_f1(
        prediction: str,
        reference: str,
    ) -> float:
        predicted_tokens = tokenize(prediction)
        reference_tokens = tokenize(reference)

        if not predicted_tokens and not reference_tokens:
            return 1.0

        if not predicted_tokens or not reference_tokens:
            return 0.0

        common = (
            Counter(predicted_tokens)
            & Counter(reference_tokens)
        )

        overlap = sum(common.values())

        if overlap == 0:
            return 0.0

        precision = overlap / len(predicted_tokens)
        recall = overlap / len(reference_tokens)

        return (
            2 * precision * recall
            / (precision + recall)
        )


# ============================================================
# CITATION METRICS
# ============================================================

class CitationMetrics:
    @staticmethod
    def precision(
        predicted: list[str],
        expected: list[str],
    ) -> float | None:
        predicted_set = set(predicted)
        expected_set = set(expected)

        if not predicted_set:
            return 1.0 if not expected_set else 0.0

        return (
            len(predicted_set & expected_set)
            / len(predicted_set)
        )

    @staticmethod
    def recall(
        predicted: list[str],
        expected: list[str],
    ) -> float | None:
        predicted_set = set(predicted)
        expected_set = set(expected)

        if not expected_set:
            return None

        return (
            len(predicted_set & expected_set)
            / len(expected_set)
        )


# ============================================================
# JUDGE RESULT
# ============================================================

class JudgeResult(BenchmarkBaseModel):
    metric: JudgeMetric

    score: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    explanation: str | None = None

    judge_model: str | None = None


JudgeCallback = Callable[
    [
        JudgeMetric,
        RAGBenchmarkCase,
        RAGPipelineOutput,
    ],
    JudgeResult,
]


PipelineCallback = Callable[
    [RAGBenchmarkCase],
    RAGPipelineOutput,
]


# ============================================================
# BENCHMARK CONFIG
# ============================================================

class RAGBenchmarkConfig(BenchmarkBaseModel):
    name: str = "industrial_rag_benchmark"

    k_values: list[int] = Field(
        default_factory=lambda: [1, 3, 5, 10]
    )

    minimum_recall_at_5: float = Field(
        default=0.80,
        ge=0,
        le=1,
    )

    minimum_mrr: float = Field(
        default=0.75,
        ge=0,
        le=1,
    )

    minimum_answer_f1: float = Field(
        default=0.60,
        ge=0,
        le=1,
    )

    minimum_groundedness: float = Field(
        default=0.85,
        ge=0,
        le=1,
    )

    minimum_safety_score: float = Field(
        default=0.95,
        ge=0,
        le=1,
    )

    maximum_average_latency_ms: float = Field(
        default=5000,
        gt=0,
    )

    @field_validator("k_values")
    @classmethod
    def validate_k_values(
        cls,
        values: list[int],
    ) -> list[int]:
        if not values:
            raise ValueError(
                "At least one k value is required"
            )

        if any(k <= 0 for k in values):
            raise ValueError(
                "All k values must be positive"
            )

        return sorted(set(values))


# ============================================================
# CASE RESULT
# ============================================================

class RAGBenchmarkCaseResult(BenchmarkBaseModel):
    case_id: str

    question: str

    retrieval_metrics: dict[
        str,
        float | None,
    ]

    answer_metrics: dict[
        str,
        float | None,
    ]

    citation_metrics: dict[
        str,
        float | None,
    ]

    judge_metrics: dict[
        str,
        float | None,
    ]

    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float

    safety_critical: bool = False

    error: str | None = None


# ============================================================
# BENCHMARK REPORT
# ============================================================

class RAGBenchmarkReport(BenchmarkBaseModel):
    benchmark_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    benchmark_name: str

    pipeline_name: str

    total_cases: int
    successful_cases: int
    failed_cases: int

    aggregate_metrics: dict[
        str,
        float | None,
    ]

    acceptance_checks: dict[
        str,
        bool | None,
    ]

    status: BenchmarkStatus

    case_results: list[
        RAGBenchmarkCaseResult
    ]

    started_at: datetime
    completed_at: datetime

    @field_validator(
        "started_at",
        "completed_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "Datetime must be timezone-aware"
            )

        return value

    def save_json(
        self,
        path: str | Path,
    ) -> None:
        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
            ),
            encoding="utf-8",
        )

    def save_csv(
        self,
        path: str | Path,
    ) -> None:
        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []

        for result in self.case_results:
            row = {
                "case_id": result.case_id,
                "question": result.question,
                "total_latency_ms": (
                    result.total_latency_ms
                ),
                "error": result.error,
            }

            row.update(result.retrieval_metrics)
            row.update(result.answer_metrics)
            row.update(result.citation_metrics)
            row.update(result.judge_metrics)

            rows.append(row)

        if not rows:
            return

        fieldnames = list(
            dict.fromkeys(
                key
                for row in rows
                for key in row
            )
        )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)


# ============================================================
# BENCHMARK ENGINE
# ============================================================

class RAGBenchmark:
    def __init__(
        self,
        config: RAGBenchmarkConfig | None = None,
        judge_callback: JudgeCallback | None = None,
    ) -> None:
        self.config = (
            config or RAGBenchmarkConfig()
        )

        self.judge_callback = judge_callback

    # --------------------------------------------------------
    # EVALUATE ONE CASE
    # --------------------------------------------------------

    def evaluate_case(
        self,
        case: RAGBenchmarkCase,
        output: RAGPipelineOutput,
        elapsed_ms: float,
    ) -> RAGBenchmarkCaseResult:
        retrieved_ids = [
            document.document_id
            for document in output.retrieved_documents
        ]

        relevant_ids = set(
            case.relevant_document_ids
        )

        retrieval_metrics = {}

        for k in self.config.k_values:
            retrieval_metrics[
                f"precision_at_{k}"
            ] = RetrievalMetrics.precision_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )

            retrieval_metrics[
                f"recall_at_{k}"
            ] = RetrievalMetrics.recall_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )

            retrieval_metrics[
                f"hit_rate_at_{k}"
            ] = RetrievalMetrics.hit_rate_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )

            retrieval_metrics[
                f"ndcg_at_{k}"
            ] = RetrievalMetrics.ndcg_at_k(
                retrieved_ids,
                case.relevance_grades,
                k,
            )

        retrieval_metrics["mrr"] = (
            RetrievalMetrics.reciprocal_rank(
                retrieved_ids,
                relevant_ids,
            )
        )

        retrieval_metrics["map"] = (
            RetrievalMetrics.average_precision(
                retrieved_ids,
                relevant_ids,
            )
        )

        answer_metrics = {
            "exact_match": None,
            "answer_f1": None,
        }

        if case.reference_answer is not None:
            answer_metrics["exact_match"] = (
                AnswerMetrics.exact_match(
                    output.answer,
                    case.reference_answer,
                )
            )

            answer_metrics["answer_f1"] = (
                AnswerMetrics.token_f1(
                    output.answer,
                    case.reference_answer,
                )
            )

        citation_metrics = {
            "citation_precision": (
                CitationMetrics.precision(
                    output.citation_ids,
                    case.expected_citation_ids,
                )
            ),
            "citation_recall": (
                CitationMetrics.recall(
                    output.citation_ids,
                    case.expected_citation_ids,
                )
            ),
        }

        judge_metrics = {
            "groundedness": None,
            "faithfulness": None,
            "answer_relevance": None,
            "safety": None,
        }

        if self.judge_callback is not None:
            for metric in JudgeMetric:
                result = self.judge_callback(
                    metric,
                    case,
                    output,
                )

                if result.metric != metric.value:
                    raise ValueError(
                        "Judge returned mismatched metric"
                    )

                judge_metrics[metric.value] = (
                    validate_score(result.score)
                )

        return RAGBenchmarkCaseResult(
            case_id=case.case_id,
            question=case.question,
            retrieval_metrics=retrieval_metrics,
            answer_metrics=answer_metrics,
            citation_metrics=citation_metrics,
            judge_metrics=judge_metrics,
            retrieval_latency_ms=(
                output.retrieval_latency_ms
            ),
            generation_latency_ms=(
                output.generation_latency_ms
            ),
            total_latency_ms=(
                output.total_latency_ms
                if output.total_latency_ms is not None
                else elapsed_ms
            ),
            safety_critical=case.safety_critical,
        )

    # --------------------------------------------------------
    # RUN BENCHMARK
    # --------------------------------------------------------

    def run(
        self,
        cases: list[RAGBenchmarkCase],
        pipeline: PipelineCallback,
        pipeline_name: str = "industrial_rag",
    ) -> RAGBenchmarkReport:
        if not cases:
            raise ValueError(
                "Benchmark dataset cannot be empty"
            )

        case_ids = [
            case.case_id
            for case in cases
        ]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "Duplicate benchmark case IDs"
            )

        started_at = utc_now()

        results = []

        for case in cases:
            start = time.perf_counter()

            try:
                output = pipeline(case)

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                result = self.evaluate_case(
                    case,
                    output,
                    elapsed_ms,
                )

            except Exception as exc:
                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                result = RAGBenchmarkCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    retrieval_metrics={},
                    answer_metrics={},
                    citation_metrics={},
                    judge_metrics={},
                    total_latency_ms=elapsed_ms,
                    safety_critical=(
                        case.safety_critical
                    ),
                    error=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

            results.append(result)

        aggregate = self._aggregate(results)

        acceptance = self._acceptance_checks(
            aggregate
        )

        failed_cases = sum(
            result.error is not None
            for result in results
        )

        successful_cases = (
            len(results) - failed_cases
        )

        if failed_cases > 0:
            status = BenchmarkStatus.FAILED
        elif any(
            value is False
            for value in acceptance.values()
        ):
            status = BenchmarkStatus.FAILED
        elif any(
            value is None
            for value in acceptance.values()
        ):
            status = BenchmarkStatus.INCOMPLETE
        else:
            status = BenchmarkStatus.PASSED

        return RAGBenchmarkReport(
            benchmark_name=self.config.name,
            pipeline_name=pipeline_name,
            total_cases=len(cases),
            successful_cases=successful_cases,
            failed_cases=failed_cases,
            aggregate_metrics=aggregate,
            acceptance_checks=acceptance,
            status=status,
            case_results=results,
            started_at=started_at,
            completed_at=utc_now(),
        )

    # --------------------------------------------------------
    # AGGREGATE METRICS
    # --------------------------------------------------------

    def _aggregate(
        self,
        results: list[RAGBenchmarkCaseResult],
    ) -> dict[str, float | None]:
        metrics: dict[
            str,
            list[float],
        ] = {}

        for result in results:
            if result.error is not None:
                continue

            combined = {
                **result.retrieval_metrics,
                **result.answer_metrics,
                **result.citation_metrics,
                **result.judge_metrics,
                "total_latency_ms": (
                    result.total_latency_ms
                ),
            }

            for name, value in combined.items():
                if value is None:
                    continue

                metrics.setdefault(
                    name,
                    [],
                ).append(value)

        names = set(metrics)

        for k in self.config.k_values:
            names.update({
                f"precision_at_{k}",
                f"recall_at_{k}",
                f"hit_rate_at_{k}",
                f"ndcg_at_{k}",
            })

        names.update({
            "mrr",
            "map",
            "exact_match",
            "answer_f1",
            "citation_precision",
            "citation_recall",
            "groundedness",
            "faithfulness",
            "answer_relevance",
            "safety",
            "total_latency_ms",
        })

        return {
            name: safe_mean(
                metrics.get(name, [])
            )
            for name in sorted(names)
        }

    # --------------------------------------------------------
    # ACCEPTANCE CHECKS
    # --------------------------------------------------------

    def _acceptance_checks(
        self,
        metrics: dict[str, float | None],
    ) -> dict[str, bool | None]:
        checks = {
            "recall_at_5": (
                self.config.minimum_recall_at_5
            ),
            "mrr": self.config.minimum_mrr,
            "answer_f1": (
                self.config.minimum_answer_f1
            ),
            "groundedness": (
                self.config.minimum_groundedness
            ),
            "safety": (
                self.config.minimum_safety_score
            ),
        }

        results: dict[
            str,
            bool | None,
        ] = {}

        for name, threshold in checks.items():
            value = metrics.get(name)

            results[name] = (
                value >= threshold
                if value is not None
                else None
            )

        latency = metrics.get(
            "total_latency_ms"
        )

        results["latency"] = (
            latency
            <= self.config.maximum_average_latency_ms
            if latency is not None
            else None
        )

        return results


# ============================================================
# BENCHMARK COMPARISON
# ============================================================

class RAGBenchmarkComparison:
    @staticmethod
    def compare(
        reports: list[RAGBenchmarkReport],
    ) -> list[dict[str, Any]]:
        rows = []

        for report in reports:
            rows.append({
                "pipeline_name": report.pipeline_name,
                "status": report.status,
                "total_cases": report.total_cases,
                "mrr": report.aggregate_metrics.get(
                    "mrr"
                ),
                "map": report.aggregate_metrics.get(
                    "map"
                ),
                "recall_at_5": (
                    report.aggregate_metrics.get(
                        "recall_at_5"
                    )
                ),
                "answer_f1": (
                    report.aggregate_metrics.get(
                        "answer_f1"
                    )
                ),
                "groundedness": (
                    report.aggregate_metrics.get(
                        "groundedness"
                    )
                ),
                "safety": (
                    report.aggregate_metrics.get(
                        "safety"
                    )
                ),
                "latency_ms": (
                    report.aggregate_metrics.get(
                        "total_latency_ms"
                    )
                ),
            })

        return rows


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    """
    Demonstration with deterministic sample outputs.

    These outputs are illustrative and do not
    represent a real model evaluation.
    """

    cases = [
        RAGBenchmarkCase(
            case_id="CASE-001",
            question=(
                "What are common causes of "
                "abnormal pump vibration?"
            ),
            reference_answer=(
                "Bearing wear, imbalance and misalignment"
            ),
            relevant_document_ids=[
                "MANUAL-001",
                "MANUAL-002",
            ],
            relevance_grades={
                "MANUAL-001": 3,
                "MANUAL-002": 2,
            },
            expected_citation_ids=[
                "MANUAL-001",
            ],
            task_type=(
                BenchmarkTaskType.EQUIPMENT_MANUAL_QA
            ),
        ),
        RAGBenchmarkCase(
            case_id="CASE-002",
            question=(
                "What should personnel verify "
                "before servicing equipment?"
            ),
            reference_answer=(
                "Approved isolation and lockout tagout "
                "procedures"
            ),
            relevant_document_ids=[
                "SAFETY-001",
            ],
            relevance_grades={
                "SAFETY-001": 3,
            },
            expected_citation_ids=[
                "SAFETY-001",
            ],
            task_type=BenchmarkTaskType.SAFETY_QA,
            safety_critical=True,
        ),
    ]

    def sample_pipeline(
        case: RAGBenchmarkCase,
    ) -> RAGPipelineOutput:
        if case.case_id == "CASE-001":
            return RAGPipelineOutput(
                answer=(
                    "Bearing wear, imbalance "
                    "and misalignment"
                ),
                retrieved_documents=[
                    BenchmarkDocument(
                        document_id="MANUAL-001",
                        text=(
                            "Abnormal vibration can result "
                            "from bearing wear."
                        ),
                    ),
                    BenchmarkDocument(
                        document_id="MANUAL-002",
                        text=(
                            "Imbalance and misalignment "
                            "can increase vibration."
                        ),
                    ),
                ],
                citation_ids=["MANUAL-001"],
                retrieval_latency_ms=25,
                generation_latency_ms=120,
                total_latency_ms=145,
            )

        return RAGPipelineOutput(
            answer=(
                "Approved isolation and lockout "
                "tagout procedures"
            ),
            retrieved_documents=[
                BenchmarkDocument(
                    document_id="SAFETY-001",
                    text=(
                        "Personnel must follow approved "
                        "isolation and lockout/tagout "
                        "procedures before servicing."
                    ),
                )
            ],
            citation_ids=["SAFETY-001"],
            retrieval_latency_ms=20,
            generation_latency_ms=110,
            total_latency_ms=130,
        )

    benchmark = RAGBenchmark(
        config=RAGBenchmarkConfig(
            name="industrial_rag_demo",
            k_values=[1, 3, 5, 10],
        )
    )

    report = benchmark.run(
        cases=cases,
        pipeline=sample_pipeline,
        pipeline_name="hybrid_rag_demo",
    )

    print(
        json.dumps(
            {
                "status": report.status,
                "total_cases": report.total_cases,
                "aggregate_metrics": (
                    report.aggregate_metrics
                ),
                "acceptance_checks": (
                    report.acceptance_checks
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    example()