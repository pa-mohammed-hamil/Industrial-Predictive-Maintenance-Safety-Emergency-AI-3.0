# app/observability/llm_observability.py

from __future__ import annotations

import asyncio
import contextvars
import logging
import math
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from time import perf_counter
from typing import Any, AsyncIterator, Callable, Mapping
from uuid import uuid4


logger = logging.getLogger(__name__)


# =========================================================
# Context
# =========================================================

_trace_id_var: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "llm_trace_id",
        default=None,
    )
)

_span_id_var: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "llm_span_id",
        default=None,
    )
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


# =========================================================
# Enums
# =========================================================

class LLMOperation(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    REASONING = "reasoning"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    RAG = "rag"
    TOOL_CALL = "tool_call"
    AGENT = "agent"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    VISION = "vision"
    OTHER = "other"


class LLMRequestStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    FALLBACK = "fallback"


class SafetyDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    REVIEW = "review"
    UNKNOWN = "unknown"


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    DENIED = "denied"


class AgentStepStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    RETRY = "retry"


# =========================================================
# Token / cost
# =========================================================

@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
        )


@dataclass(slots=True)
class CostUsage:
    input_cost: float = 0.0
    output_cost: float = 0.0
    cached_input_cost: float = 0.0
    reasoning_cost: float = 0.0

    currency: str = "USD"

    @property
    def total_cost(self) -> float:
        return (
            self.input_cost
            + self.output_cost
            + self.cached_input_cost
            + self.reasoning_cost
        )


# =========================================================
# RAG observability
# =========================================================

@dataclass(slots=True)
class RAGMetrics:
    query: str | None = None

    retrieved_documents: int = 0
    retrieved_chunks: int = 0
    context_chunks_used: int = 0

    retrieval_latency_ms: float | None = None
    reranker_latency_ms: float | None = None

    top_retrieval_score: float | None = None
    top_reranker_score: float | None = None

    citation_count: int = 0

    citation_precision: float | None = None
    citation_recall: float | None = None

    groundedness: float | None = None
    faithfulness: float | None = None

    retrieval_recall_at_k: float | None = None
    retrieval_ndcg_at_k: float | None = None
    retrieval_mrr: float | None = None

    context_tokens: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Safety
# =========================================================

@dataclass(slots=True)
class LLMSafetyMetrics:
    decision: SafetyDecision = (
        SafetyDecision.UNKNOWN
    )

    prompt_injection_detected: bool = False
    jailbreak_detected: bool = False

    pii_detected: bool = False
    pii_redacted: bool = False

    unsafe_output_detected: bool = False

    blocked_reason: str | None = None

    risk_score: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Tool observability
# =========================================================

@dataclass(slots=True)
class ToolCallMetric:
    tool_call_id: str
    tool_name: str

    status: ToolCallStatus

    latency_ms: float

    arguments_size_bytes: int = 0
    response_size_bytes: int = 0

    error_type: str | None = None
    error_message: str | None = None

    started_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Agent observability
# =========================================================

@dataclass(slots=True)
class AgentStepMetric:
    step_id: str
    agent_name: str
    step_name: str

    status: AgentStepStatus

    latency_ms: float

    tool_calls: int = 0

    input_tokens: int = 0
    output_tokens: int = 0

    error_type: str | None = None
    error_message: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# LLM request
# =========================================================

@dataclass(slots=True)
class LLMRequestMetric:
    request_id: str
    trace_id: str
    span_id: str

    operation: LLMOperation

    provider: str
    model: str

    status: LLMRequestStatus

    started_at: datetime
    completed_at: datetime

    latency_ms: float

    time_to_first_token_ms: float | None = None

    token_usage: TokenUsage = field(
        default_factory=TokenUsage
    )

    cost: CostUsage = field(
        default_factory=CostUsage
    )

    rag: RAGMetrics | None = None

    safety: LLMSafetyMetrics | None = None

    tool_calls: list[ToolCallMetric] = field(
        default_factory=list
    )

    agent_steps: list[AgentStepMetric] = field(
        default_factory=list
    )

    user_id: str | None = None
    conversation_id: str | None = None

    asset_id: str | None = None
    site_id: str | None = None

    prompt_template: str | None = None
    prompt_version: str | None = None

    temperature: float | None = None
    max_tokens: int | None = None

    retry_count: int = 0

    fallback_used: bool = False
    fallback_model: str | None = None

    error_type: str | None = None
    error_message: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["operation"] = self.operation.value
        result["status"] = self.status.value

        result["started_at"] = (
            self.started_at.isoformat()
        )

        result["completed_at"] = (
            self.completed_at.isoformat()
        )

        return result


# =========================================================
# Pricing
# =========================================================

@dataclass(slots=True)
class ModelPricing:
    input_per_million: float = 0.0
    output_per_million: float = 0.0

    cached_input_per_million: float = 0.0
    reasoning_per_million: float = 0.0

    currency: str = "USD"


class PricingRegistry:

    def __init__(self) -> None:
        self._pricing: dict[
            tuple[str, str],
            ModelPricing,
        ] = {}

    def register(
        self,
        *,
        provider: str,
        model: str,
        pricing: ModelPricing,
    ) -> None:

        self._pricing[
            (
                provider.lower(),
                model.lower(),
            )
        ] = pricing

    def get(
        self,
        provider: str,
        model: str,
    ) -> ModelPricing | None:

        return self._pricing.get(
            (
                provider.lower(),
                model.lower(),
            )
        )

    def calculate_cost(
        self,
        *,
        provider: str,
        model: str,
        usage: TokenUsage,
    ) -> CostUsage:

        pricing = self.get(
            provider,
            model,
        )

        if pricing is None:
            return CostUsage()

        million = 1_000_000.0

        return CostUsage(
            input_cost=(
                usage.input_tokens
                / million
                * pricing.input_per_million
            ),

            output_cost=(
                usage.output_tokens
                / million
                * pricing.output_per_million
            ),

            cached_input_cost=(
                usage.cached_input_tokens
                / million
                * pricing.cached_input_per_million
            ),

            reasoning_cost=(
                usage.reasoning_tokens
                / million
                * pricing.reasoning_per_million
            ),

            currency=pricing.currency,
        )


# =========================================================
# Sink
# =========================================================

class InMemoryLLMMetricStore:

    def __init__(
        self,
        *,
        max_records: int = 100_000,
    ) -> None:

        self.max_records = max_records

        self._records: list[
            LLMRequestMetric
        ] = []

        self._lock = asyncio.Lock()

    async def record(
        self,
        metric: LLMRequestMetric,
    ) -> None:

        async with self._lock:
            self._records.append(
                metric
            )

            overflow = (
                len(self._records)
                - self.max_records
            )

            if overflow > 0:
                del self._records[
                    :overflow
                ]

    async def list(
        self,
    ) -> list[LLMRequestMetric]:

        async with self._lock:
            return list(
                self._records
            )

    async def clear(self) -> None:

        async with self._lock:
            self._records.clear()


# =========================================================
# Active trace
# =========================================================

@dataclass(slots=True)
class LLMTrace:
    request_id: str

    trace_id: str
    span_id: str

    operation: LLMOperation

    provider: str
    model: str

    started_at: datetime
    started_perf: float

    user_id: str | None = None
    conversation_id: str | None = None

    asset_id: str | None = None
    site_id: str | None = None

    prompt_template: str | None = None
    prompt_version: str | None = None

    temperature: float | None = None
    max_tokens: int | None = None

    token_usage: TokenUsage = field(
        default_factory=TokenUsage
    )

    rag: RAGMetrics | None = None

    safety: LLMSafetyMetrics | None = None

    tool_calls: list[ToolCallMetric] = field(
        default_factory=list
    )

    agent_steps: list[AgentStepMetric] = field(
        default_factory=list
    )

    first_token_perf: float | None = None

    retry_count: int = 0

    fallback_used: bool = False
    fallback_model: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def mark_first_token(self) -> None:

        if self.first_token_perf is None:
            self.first_token_perf = (
                perf_counter()
            )

    def set_tokens(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:

        self.token_usage = TokenUsage(
            input_tokens=max(
                0,
                input_tokens,
            ),

            output_tokens=max(
                0,
                output_tokens,
            ),

            cached_input_tokens=max(
                0,
                cached_input_tokens,
            ),

            reasoning_tokens=max(
                0,
                reasoning_tokens,
            ),
        )

    def set_rag(
        self,
        metrics: RAGMetrics,
    ) -> None:

        self.rag = metrics

    def set_safety(
        self,
        metrics: LLMSafetyMetrics,
    ) -> None:

        self.safety = metrics

    def add_tool_call(
        self,
        metric: ToolCallMetric,
    ) -> None:

        self.tool_calls.append(
            metric
        )

    def add_agent_step(
        self,
        metric: AgentStepMetric,
    ) -> None:

        self.agent_steps.append(
            metric
        )


# =========================================================
# Percentile
# =========================================================

def percentile(
    values: list[float],
    q: float,
) -> float | None:

    if not values:
        return None

    ordered = sorted(
        values
    )

    q = max(
        0.0,
        min(
            1.0,
            q,
        ),
    )

    index = max(
        0,
        math.ceil(
            q * len(ordered)
        ) - 1,
    )

    return ordered[index]


# =========================================================
# Aggregate report
# =========================================================

@dataclass(slots=True)
class LLMObservabilityReport:
    total_requests: int

    successful_requests: int
    failed_requests: int
    blocked_requests: int

    success_rate: float
    error_rate: float

    average_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None

    average_ttft_ms: float | None
    p95_ttft_ms: float | None

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_reasoning_tokens: int

    total_cost: float
    average_cost_per_request: float

    requests_by_model: dict[str, int]
    requests_by_provider: dict[str, int]
    requests_by_operation: dict[str, int]

    average_groundedness: float | None
    average_faithfulness: float | None

    hallucination_risk_rate: float | None

    tool_calls: int
    tool_call_error_rate: float

    agent_steps: int
    agent_step_error_rate: float

    fallback_rate: float
    safety_block_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================================================
# LLM Observability
# =========================================================

class LLMObservability:

    def __init__(
        self,
        *,
        store: InMemoryLLMMetricStore | None = None,
        pricing_registry: PricingRegistry | None = None,
    ) -> None:

        self.store = (
            store
            or InMemoryLLMMetricStore()
        )

        self.pricing = (
            pricing_registry
            or PricingRegistry()
        )

    # -----------------------------------------------------
    # Trace
    # -----------------------------------------------------

    @asynccontextmanager
    async def trace(
        self,
        *,
        operation: LLMOperation,
        provider: str,
        model: str,

        trace_id: str | None = None,

        user_id: str | None = None,
        conversation_id: str | None = None,

        asset_id: str | None = None,
        site_id: str | None = None,

        prompt_template: str | None = None,
        prompt_version: str | None = None,

        temperature: float | None = None,
        max_tokens: int | None = None,

        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMTrace]:

        request_id = new_id()

        trace_id = (
            trace_id
            or _trace_id_var.get()
            or new_id()
        )

        span_id = new_id()

        trace_token = (
            _trace_id_var.set(
                trace_id
            )
        )

        span_token = (
            _span_id_var.set(
                span_id
            )
        )

        started_at = utc_now()
        started_perf = perf_counter()

        active = LLMTrace(
            request_id=request_id,

            trace_id=trace_id,
            span_id=span_id,

            operation=operation,

            provider=provider,
            model=model,

            started_at=started_at,
            started_perf=started_perf,

            user_id=user_id,
            conversation_id=conversation_id,

            asset_id=asset_id,
            site_id=site_id,

            prompt_template=prompt_template,
            prompt_version=prompt_version,

            temperature=temperature,
            max_tokens=max_tokens,

            metadata=metadata or {},
        )

        status = LLMRequestStatus.SUCCESS

        error_type: str | None = None
        error_message: str | None = None

        try:
            yield active

        except asyncio.TimeoutError as exc:
            status = LLMRequestStatus.TIMEOUT

            error_type = (
                type(exc).__name__
            )

            error_message = str(exc)

            raise

        except asyncio.CancelledError:
            status = (
                LLMRequestStatus.CANCELLED
            )

            error_type = "CancelledError"
            error_message = (
                "LLM request cancelled."
            )

            raise

        except Exception as exc:
            status = LLMRequestStatus.ERROR

            error_type = (
                type(exc).__name__
            )

            error_message = str(exc)

            raise

        finally:
            completed_at = utc_now()
            completed_perf = perf_counter()

            latency_ms = (
                completed_perf
                - started_perf
            ) * 1000.0

            ttft_ms: float | None = None

            if (
                active.first_token_perf
                is not None
            ):
                ttft_ms = (
                    active.first_token_perf
                    - started_perf
                ) * 1000.0

            if (
                active.safety is not None
                and active.safety.decision
                == SafetyDecision.BLOCK
            ):
                status = (
                    LLMRequestStatus.BLOCKED
                )

            elif (
                status
                == LLMRequestStatus.SUCCESS
                and active.fallback_used
            ):
                status = (
                    LLMRequestStatus.FALLBACK
                )

            cost = (
                self.pricing.calculate_cost(
                    provider=provider,
                    model=model,
                    usage=active.token_usage,
                )
            )

            metric = LLMRequestMetric(
                request_id=request_id,

                trace_id=trace_id,
                span_id=span_id,

                operation=operation,

                provider=provider,
                model=model,

                status=status,

                started_at=started_at,
                completed_at=completed_at,

                latency_ms=round(
                    latency_ms,
                    4,
                ),

                time_to_first_token_ms=(
                    round(
                        ttft_ms,
                        4,
                    )
                    if ttft_ms is not None
                    else None
                ),

                token_usage=(
                    active.token_usage
                ),

                cost=cost,

                rag=active.rag,
                safety=active.safety,

                tool_calls=(
                    active.tool_calls
                ),

                agent_steps=(
                    active.agent_steps
                ),

                user_id=user_id,

                conversation_id=(
                    conversation_id
                ),

                asset_id=asset_id,
                site_id=site_id,

                prompt_template=(
                    prompt_template
                ),

                prompt_version=(
                    prompt_version
                ),

                temperature=temperature,
                max_tokens=max_tokens,

                retry_count=(
                    active.retry_count
                ),

                fallback_used=(
                    active.fallback_used
                ),

                fallback_model=(
                    active.fallback_model
                ),

                error_type=error_type,
                error_message=error_message,

                metadata=active.metadata,
            )

            try:
                await self.store.record(
                    metric
                )

            except Exception:
                # Observability should not take down
                # the application request.
                logger.exception(
                    "Failed to persist LLM metric."
                )

            _trace_id_var.reset(
                trace_token
            )

            _span_id_var.reset(
                span_token
            )

    # -----------------------------------------------------
    # Tool execution
    # -----------------------------------------------------

    @asynccontextmanager
    async def tool_span(
        self,
        trace: LLMTrace,
        *,
        tool_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:

        tool_call_id = new_id()
        started_at = utc_now()
        started_perf = perf_counter()

        state: dict[str, Any] = {
            "arguments_size_bytes": 0,
            "response_size_bytes": 0,
        }

        status = ToolCallStatus.SUCCESS
        error_type = None
        error_message = None

        try:
            yield state

        except asyncio.TimeoutError as exc:
            status = ToolCallStatus.TIMEOUT
            error_type = type(exc).__name__
            error_message = str(exc)
            raise

        except Exception as exc:
            status = ToolCallStatus.ERROR
            error_type = type(exc).__name__
            error_message = str(exc)
            raise

        finally:
            latency_ms = (
                perf_counter()
                - started_perf
            ) * 1000.0

            trace.add_tool_call(
                ToolCallMetric(
                    tool_call_id=tool_call_id,

                    tool_name=tool_name,

                    status=status,

                    latency_ms=round(
                        latency_ms,
                        4,
                    ),

                    arguments_size_bytes=max(
                        0,
                        int(
                            state.get(
                                "arguments_size_bytes",
                                0,
                            )
                        ),
                    ),

                    response_size_bytes=max(
                        0,
                        int(
                            state.get(
                                "response_size_bytes",
                                0,
                            )
                        ),
                    ),

                    error_type=error_type,
                    error_message=error_message,

                    started_at=started_at,

                    metadata=metadata or {},
                )
            )

    # -----------------------------------------------------
    # Agent step
    # -----------------------------------------------------

    @asynccontextmanager
    async def agent_step(
        self,
        trace: LLMTrace,
        *,
        agent_name: str,
        step_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:

        step_id = new_id()
        started = perf_counter()

        state: dict[str, Any] = {
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        status = AgentStepStatus.SUCCESS
        error_type = None
        error_message = None

        try:
            yield state

        except Exception as exc:
            status = AgentStepStatus.ERROR

            error_type = (
                type(exc).__name__
            )

            error_message = str(exc)

            raise

        finally:
            latency_ms = (
                perf_counter()
                - started
            ) * 1000.0

            trace.add_agent_step(
                AgentStepMetric(
                    step_id=step_id,

                    agent_name=agent_name,
                    step_name=step_name,

                    status=status,

                    latency_ms=round(
                        latency_ms,
                        4,
                    ),

                    tool_calls=max(
                        0,
                        int(
                            state.get(
                                "tool_calls",
                                0,
                            )
                        ),
                    ),

                    input_tokens=max(
                        0,
                        int(
                            state.get(
                                "input_tokens",
                                0,
                            )
                        ),
                    ),

                    output_tokens=max(
                        0,
                        int(
                            state.get(
                                "output_tokens",
                                0,
                            )
                        ),
                    ),

                    error_type=error_type,
                    error_message=error_message,

                    metadata=metadata or {},
                )
            )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    async def report(
        self,
    ) -> LLMObservabilityReport:

        records = await self.store.list()

        if not records:
            return LLMObservabilityReport(
                total_requests=0,

                successful_requests=0,
                failed_requests=0,
                blocked_requests=0,

                success_rate=0.0,
                error_rate=0.0,

                average_latency_ms=None,
                p50_latency_ms=None,
                p95_latency_ms=None,
                p99_latency_ms=None,

                average_ttft_ms=None,
                p95_ttft_ms=None,

                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_reasoning_tokens=0,

                total_cost=0.0,
                average_cost_per_request=0.0,

                requests_by_model={},
                requests_by_provider={},
                requests_by_operation={},

                average_groundedness=None,
                average_faithfulness=None,

                hallucination_risk_rate=None,

                tool_calls=0,
                tool_call_error_rate=0.0,

                agent_steps=0,
                agent_step_error_rate=0.0,

                fallback_rate=0.0,
                safety_block_rate=0.0,
            )

        total = len(records)

        successful = sum(
            record.status
            in {
                LLMRequestStatus.SUCCESS,
                LLMRequestStatus.FALLBACK,
            }
            for record in records
        )

        failed = sum(
            record.status
            in {
                LLMRequestStatus.ERROR,
                LLMRequestStatus.TIMEOUT,
                LLMRequestStatus.CANCELLED,
            }
            for record in records
        )

        blocked = sum(
            record.status
            == LLMRequestStatus.BLOCKED
            for record in records
        )

        latencies = [
            record.latency_ms
            for record in records
        ]

        ttfts = [
            record.time_to_first_token_ms
            for record in records
            if (
                record.time_to_first_token_ms
                is not None
            )
        ]

        total_input = sum(
            record.token_usage.input_tokens
            for record in records
        )

        total_output = sum(
            record.token_usage.output_tokens
            for record in records
        )

        total_reasoning = sum(
            record.token_usage.reasoning_tokens
            for record in records
        )

        total_tokens = sum(
            record.token_usage.total_tokens
            for record in records
        )

        total_cost = sum(
            record.cost.total_cost
            for record in records
        )

        by_model: dict[str, int] = (
            defaultdict(int)
        )

        by_provider: dict[str, int] = (
            defaultdict(int)
        )

        by_operation: dict[str, int] = (
            defaultdict(int)
        )

        for record in records:
            by_model[
                record.model
            ] += 1

            by_provider[
                record.provider
            ] += 1

            by_operation[
                record.operation.value
            ] += 1

        groundedness_values = [
            record.rag.groundedness
            for record in records
            if (
                record.rag is not None
                and record.rag.groundedness
                is not None
            )
        ]

        faithfulness_values = [
            record.rag.faithfulness
            for record in records
            if (
                record.rag is not None
                and record.rag.faithfulness
                is not None
            )
        ]

        # Project-specific heuristic:
        # groundedness < 0.8 is treated as a
        # hallucination-risk response.
        hallucination_candidates = [
            value
            for value in groundedness_values
            if value < 0.80
        ]

        hallucination_rate = (
            len(hallucination_candidates)
            / len(groundedness_values)
            if groundedness_values
            else None
        )

        tool_calls = [
            call
            for record in records
            for call in record.tool_calls
        ]

        tool_errors = sum(
            call.status
            != ToolCallStatus.SUCCESS
            for call in tool_calls
        )

        agent_steps = [
            step
            for record in records
            for step in record.agent_steps
        ]

        agent_errors = sum(
            step.status
            == AgentStepStatus.ERROR
            for step in agent_steps
        )

        fallbacks = sum(
            record.fallback_used
            for record in records
        )

        return LLMObservabilityReport(
            total_requests=total,

            successful_requests=successful,
            failed_requests=failed,
            blocked_requests=blocked,

            success_rate=round(
                successful / total,
                6,
            ),

            error_rate=round(
                failed / total,
                6,
            ),

            average_latency_ms=round(
                mean(latencies),
                4,
            ),

            p50_latency_ms=_round_optional(
                percentile(
                    latencies,
                    0.50,
                )
            ),

            p95_latency_ms=_round_optional(
                percentile(
                    latencies,
                    0.95,
                )
            ),

            p99_latency_ms=_round_optional(
                percentile(
                    latencies,
                    0.99,
                )
            ),

            average_ttft_ms=(
                round(
                    mean(ttfts),
                    4,
                )
                if ttfts
                else None
            ),

            p95_ttft_ms=_round_optional(
                percentile(
                    ttfts,
                    0.95,
                )
            ),

            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,

            total_reasoning_tokens=(
                total_reasoning
            ),

            total_cost=round(
                total_cost,
                8,
            ),

            average_cost_per_request=round(
                total_cost / total,
                8,
            ),

            requests_by_model=dict(
                by_model
            ),

            requests_by_provider=dict(
                by_provider
            ),

            requests_by_operation=dict(
                by_operation
            ),

            average_groundedness=(
                round(
                    mean(
                        groundedness_values
                    ),
                    6,
                )
                if groundedness_values
                else None
            ),

            average_faithfulness=(
                round(
                    mean(
                        faithfulness_values
                    ),
                    6,
                )
                if faithfulness_values
                else None
            ),

            hallucination_risk_rate=(
                round(
                    hallucination_rate,
                    6,
                )
                if hallucination_rate
                is not None
                else None
            ),

            tool_calls=len(
                tool_calls
            ),

            tool_call_error_rate=round(
                tool_errors
                / len(tool_calls),
                6,
            )
            if tool_calls
            else 0.0,

            agent_steps=len(
                agent_steps
            ),

            agent_step_error_rate=round(
                agent_errors
                / len(agent_steps),
                6,
            )
            if agent_steps
            else 0.0,

            fallback_rate=round(
                fallbacks / total,
                6,
            ),

            safety_block_rate=round(
                blocked / total,
                6,
            ),
        )


# =========================================================
# Helpers
# =========================================================

def _round_optional(
    value: float | None,
    digits: int = 4,
) -> float | None:

    if value is None:
        return None

    return round(
        value,
        digits,
    )


# =========================================================
# Production SLO gate
# =========================================================

def llm_quality_gate(
    report: LLMObservabilityReport,
    *,
    minimum_success_rate: float = 0.99,
    maximum_error_rate: float = 0.01,

    maximum_p95_latency_ms: float = 10_000.0,
    maximum_p95_ttft_ms: float = 3_000.0,

    minimum_groundedness: float = 0.90,
    minimum_faithfulness: float = 0.90,

    maximum_tool_error_rate: float = 0.05,
    maximum_agent_error_rate: float = 0.05,

    maximum_hallucination_risk_rate: float = 0.05,
) -> dict[str, Any]:

    checks: dict[str, bool] = {
        "success_rate": (
            report.success_rate
            >= minimum_success_rate
        ),

        "error_rate": (
            report.error_rate
            <= maximum_error_rate
        ),

        "tool_error_rate": (
            report.tool_call_error_rate
            <= maximum_tool_error_rate
        ),

        "agent_error_rate": (
            report.agent_step_error_rate
            <= maximum_agent_error_rate
        ),
    }

    if report.p95_latency_ms is not None:
        checks[
            "p95_latency"
        ] = (
            report.p95_latency_ms
            <= maximum_p95_latency_ms
        )

    if report.p95_ttft_ms is not None:
        checks[
            "p95_ttft"
        ] = (
            report.p95_ttft_ms
            <= maximum_p95_ttft_ms
        )

    if (
        report.average_groundedness
        is not None
    ):
        checks[
            "groundedness"
        ] = (
            report.average_groundedness
            >= minimum_groundedness
        )

    if (
        report.average_faithfulness
        is not None
    ):
        checks[
            "faithfulness"
        ] = (
            report.average_faithfulness
            >= minimum_faithfulness
        )

    if (
        report.hallucination_risk_rate
        is not None
    ):
        checks[
            "hallucination_risk"
        ] = (
            report.hallucination_risk_rate
            <= maximum_hallucination_risk_rate
        )

    return {
        "passed": all(
            checks.values()
        ),

        "checks": checks,

        "metrics": {
            "success_rate":
                report.success_rate,

            "error_rate":
                report.error_rate,

            "p95_latency_ms":
                report.p95_latency_ms,

            "p95_ttft_ms":
                report.p95_ttft_ms,

            "groundedness":
                report.average_groundedness,

            "faithfulness":
                report.average_faithfulness,

            "hallucination_risk_rate":
                report.hallucination_risk_rate,

            "tool_error_rate":
                report.tool_call_error_rate,

            "agent_error_rate":
                report.agent_step_error_rate,

            "fallback_rate":
                report.fallback_rate,

            "total_cost":
                report.total_cost,
        },
    }


# =========================================================
# Prometheus-style metric export
# =========================================================

async def prometheus_snapshot(
    observability: LLMObservability,
) -> dict[str, float]:

    report = await observability.report()

    metrics: dict[str, float] = {
        "llm_requests_total":
            float(report.total_requests),

        "llm_success_rate":
            report.success_rate,

        "llm_error_rate":
            report.error_rate,

        "llm_input_tokens_total":
            float(
                report.total_input_tokens
            ),

        "llm_output_tokens_total":
            float(
                report.total_output_tokens
            ),

        "llm_tokens_total":
            float(
                report.total_tokens
            ),

        "llm_cost_total":
            report.total_cost,

        "llm_tool_error_rate":
            report.tool_call_error_rate,

        "llm_agent_error_rate":
            report.agent_step_error_rate,

        "llm_fallback_rate":
            report.fallback_rate,

        "llm_safety_block_rate":
            report.safety_block_rate,
    }

    optional = {
        "llm_latency_p95_ms":
            report.p95_latency_ms,

        "llm_ttft_p95_ms":
            report.p95_ttft_ms,

        "llm_groundedness":
            report.average_groundedness,

        "llm_faithfulness":
            report.average_faithfulness,

        "llm_hallucination_risk_rate":
            report.hallucination_risk_rate,
    }

    for name, value in optional.items():
        if value is not None:
            metrics[name] = float(value)

    return metrics


# =========================================================
# Global observability service
# =========================================================

pricing_registry = PricingRegistry()

llm_observability = LLMObservability(
    pricing_registry=pricing_registry
)


__all__ = [
    "LLMOperation",
    "LLMRequestStatus",
    "SafetyDecision",
    "ToolCallStatus",
    "AgentStepStatus",
    "TokenUsage",
    "CostUsage",
    "RAGMetrics",
    "LLMSafetyMetrics",
    "ToolCallMetric",
    "AgentStepMetric",
    "LLMRequestMetric",
    "ModelPricing",
    "PricingRegistry",
    "InMemoryLLMMetricStore",
    "LLMTrace",
    "LLMObservabilityReport",
    "LLMObservability",
    "percentile",
    "llm_quality_gate",
    "prometheus_snapshot",
    "pricing_registry",
    "llm_observability",
]