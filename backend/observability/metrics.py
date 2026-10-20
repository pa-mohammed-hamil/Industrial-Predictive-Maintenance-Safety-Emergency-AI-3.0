# app/observability/metrics.py

from __future__ import annotations

import math
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator, Mapping


# =========================================================
# Helpers
# =========================================================

def _normalize_labels(
    labels: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()

    return tuple(
        sorted(
            (str(key), str(value))
            for key, value in labels.items()
        )
    )


def _labels_to_dict(
    labels: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    return dict(labels)


def _percentile(
    values: list[float],
    q: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    q = max(0.0, min(1.0, q))

    index = max(
        0,
        math.ceil(q * len(ordered)) - 1,
    )

    return ordered[index]


# =========================================================
# Counter
# =========================================================

class Counter:
    def __init__(
        self,
        name: str,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description

        self._values: dict[
            tuple[tuple[str, str], ...],
            float,
        ] = defaultdict(float)

        self._lock = threading.RLock()

    def inc(
        self,
        amount: float = 1.0,
        *,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        if amount < 0:
            raise ValueError(
                "Counter cannot be decreased."
            )

        key = _normalize_labels(labels)

        with self._lock:
            self._values[key] += amount

    def get(
        self,
        *,
        labels: Mapping[str, Any] | None = None,
    ) -> float:
        key = _normalize_labels(labels)

        with self._lock:
            return self._values.get(key, 0.0)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "labels": _labels_to_dict(labels),
                    "value": value,
                }
                for labels, value
                in self._values.items()
            ]


# =========================================================
# Gauge
# =========================================================

class Gauge:
    def __init__(
        self,
        name: str,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description

        self._values: dict[
            tuple[tuple[str, str], ...],
            float,
        ] = defaultdict(float)

        self._lock = threading.RLock()

    def set(
        self,
        value: float,
        *,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        key = _normalize_labels(labels)

        with self._lock:
            self._values[key] = float(value)

    def inc(
        self,
        amount: float = 1.0,
        *,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        key = _normalize_labels(labels)

        with self._lock:
            self._values[key] += amount

    def dec(
        self,
        amount: float = 1.0,
        *,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        key = _normalize_labels(labels)

        with self._lock:
            self._values[key] -= amount

    def get(
        self,
        *,
        labels: Mapping[str, Any] | None = None,
    ) -> float:
        key = _normalize_labels(labels)

        with self._lock:
            return self._values.get(key, 0.0)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "labels": _labels_to_dict(labels),
                    "value": value,
                }
                for labels, value
                in self._values.items()
            ]


# =========================================================
# Histogram
# =========================================================

@dataclass(slots=True)
class HistogramStats:
    count: int
    total: float

    minimum: float | None
    maximum: float | None
    average: float | None

    p50: float | None
    p95: float | None
    p99: float | None


class Histogram:
    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        max_samples_per_series: int = 10_000,
    ) -> None:
        self.name = name
        self.description = description

        self.max_samples_per_series = (
            max_samples_per_series
        )

        self._values: dict[
            tuple[tuple[str, str], ...],
            list[float],
        ] = defaultdict(list)

        self._counts: dict[
            tuple[tuple[str, str], ...],
            int,
        ] = defaultdict(int)

        self._totals: dict[
            tuple[tuple[str, str], ...],
            float,
        ] = defaultdict(float)

        self._lock = threading.RLock()

    def observe(
        self,
        value: float,
        *,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        value = float(value)

        if not math.isfinite(value):
            return

        key = _normalize_labels(labels)

        with self._lock:
            samples = self._values[key]

            samples.append(value)

            if (
                len(samples)
                > self.max_samples_per_series
            ):
                del samples[
                    :len(samples)
                    - self.max_samples_per_series
                ]

            self._counts[key] += 1
            self._totals[key] += value

    def stats(
        self,
        *,
        labels: Mapping[str, Any] | None = None,
    ) -> HistogramStats:
        key = _normalize_labels(labels)

        with self._lock:
            samples = list(
                self._values.get(key, [])
            )

            count = self._counts.get(
                key,
                0,
            )

            total = self._totals.get(
                key,
                0.0,
            )

        if not samples:
            return HistogramStats(
                count=count,
                total=total,
                minimum=None,
                maximum=None,
                average=(
                    total / count
                    if count
                    else None
                ),
                p50=None,
                p95=None,
                p99=None,
            )

        return HistogramStats(
            count=count,
            total=total,

            minimum=min(samples),
            maximum=max(samples),

            average=(
                total / count
                if count
                else None
            ),

            p50=_percentile(
                samples,
                0.50,
            ),

            p95=_percentile(
                samples,
                0.95,
            ),

            p99=_percentile(
                samples,
                0.99,
            ),
        )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            keys = set(
                self._values
            ) | set(
                self._counts
            )

        result: list[dict[str, Any]] = []

        for key in keys:
            stats = self.stats(
                labels=dict(key)
            )

            result.append(
                {
                    "labels":
                        _labels_to_dict(key),

                    "count":
                        stats.count,

                    "sum":
                        stats.total,

                    "min":
                        stats.minimum,

                    "max":
                        stats.maximum,

                    "avg":
                        stats.average,

                    "p50":
                        stats.p50,

                    "p95":
                        stats.p95,

                    "p99":
                        stats.p99,
                }
            )

        return result


# =========================================================
# Metric registry
# =========================================================

class MetricRegistry:
    def __init__(self) -> None:
        self._counters: dict[
            str,
            Counter,
        ] = {}

        self._gauges: dict[
            str,
            Gauge,
        ] = {}

        self._histograms: dict[
            str,
            Histogram,
        ] = {}

        self._lock = threading.RLock()

    def counter(
        self,
        name: str,
        description: str = "",
    ) -> Counter:
        with self._lock:
            metric = self._counters.get(
                name
            )

            if metric is None:
                metric = Counter(
                    name,
                    description,
                )

                self._counters[name] = metric

            return metric

    def gauge(
        self,
        name: str,
        description: str = "",
    ) -> Gauge:
        with self._lock:
            metric = self._gauges.get(
                name
            )

            if metric is None:
                metric = Gauge(
                    name,
                    description,
                )

                self._gauges[name] = metric

            return metric

    def histogram(
        self,
        name: str,
        description: str = "",
    ) -> Histogram:
        with self._lock:
            metric = self._histograms.get(
                name
            )

            if metric is None:
                metric = Histogram(
                    name,
                    description,
                )

                self._histograms[name] = metric

            return metric

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(
                self._counters
            )

            gauges = dict(
                self._gauges
            )

            histograms = dict(
                self._histograms
            )

        return {
            "counters": {
                name: metric.snapshot()
                for name, metric
                in counters.items()
            },

            "gauges": {
                name: metric.snapshot()
                for name, metric
                in gauges.items()
            },

            "histograms": {
                name: metric.snapshot()
                for name, metric
                in histograms.items()
            },
        }


registry = MetricRegistry()


# =========================================================
# HTTP metrics
# =========================================================

http_requests_total = registry.counter(
    "http_requests_total"
)

http_request_errors_total = registry.counter(
    "http_request_errors_total"
)

http_requests_in_progress = registry.gauge(
    "http_requests_in_progress"
)

http_request_duration_ms = registry.histogram(
    "http_request_duration_ms"
)


# =========================================================
# Database
# =========================================================

database_queries_total = registry.counter(
    "database_queries_total"
)

database_query_errors_total = registry.counter(
    "database_query_errors_total"
)

database_query_duration_ms = registry.histogram(
    "database_query_duration_ms"
)

database_connections_active = registry.gauge(
    "database_connections_active"
)


# =========================================================
# Redis
# =========================================================

redis_operations_total = registry.counter(
    "redis_operations_total"
)

redis_errors_total = registry.counter(
    "redis_errors_total"
)

redis_operation_duration_ms = registry.histogram(
    "redis_operation_duration_ms"
)

redis_cache_hits_total = registry.counter(
    "redis_cache_hits_total"
)

redis_cache_misses_total = registry.counter(
    "redis_cache_misses_total"
)


# =========================================================
# Kafka
# =========================================================

kafka_messages_produced_total = registry.counter(
    "kafka_messages_produced_total"
)

kafka_messages_consumed_total = registry.counter(
    "kafka_messages_consumed_total"
)

kafka_errors_total = registry.counter(
    "kafka_errors_total"
)

kafka_processing_duration_ms = registry.histogram(
    "kafka_processing_duration_ms"
)

kafka_consumer_lag = registry.gauge(
    "kafka_consumer_lag"
)


# =========================================================
# MQTT / telemetry
# =========================================================

mqtt_messages_total = registry.counter(
    "mqtt_messages_total"
)

mqtt_errors_total = registry.counter(
    "mqtt_errors_total"
)

telemetry_messages_total = registry.counter(
    "telemetry_messages_total"
)

telemetry_invalid_total = registry.counter(
    "telemetry_invalid_total"
)

sensor_readings_total = registry.counter(
    "sensor_readings_total"
)

sensor_errors_total = registry.counter(
    "sensor_errors_total"
)

sensor_value = registry.gauge(
    "sensor_value"
)


# =========================================================
# ML inference
# =========================================================

model_inference_total = registry.counter(
    "model_inference_total"
)

model_inference_errors_total = registry.counter(
    "model_inference_errors_total"
)

model_inference_duration_ms = registry.histogram(
    "model_inference_duration_ms"
)

model_prediction_confidence = registry.histogram(
    "model_prediction_confidence"
)

model_active_requests = registry.gauge(
    "model_active_requests"
)


# =========================================================
# Predictive maintenance
# =========================================================

anomalies_detected_total = registry.counter(
    "anomalies_detected_total"
)

failure_predictions_total = registry.counter(
    "failure_predictions_total"
)

failure_probability = registry.gauge(
    "failure_probability"
)

rul_predictions_total = registry.counter(
    "rul_predictions_total"
)

predicted_rul = registry.gauge(
    "predicted_rul"
)

maintenance_recommendations_total = registry.counter(
    "maintenance_recommendations_total"
)

work_orders_created_total = registry.counter(
    "work_orders_created_total"
)


# =========================================================
# Computer vision / safety
# =========================================================

vision_inference_total = registry.counter(
    "vision_inference_total"
)

vision_inference_duration_ms = registry.histogram(
    "vision_inference_duration_ms"
)

defects_detected_total = registry.counter(
    "defects_detected_total"
)

fire_events_total = registry.counter(
    "fire_events_total"
)

smoke_events_total = registry.counter(
    "smoke_events_total"
)

overheating_events_total = registry.counter(
    "overheating_events_total"
)

unsafe_motion_events_total = registry.counter(
    "unsafe_motion_events_total"
)

safety_incidents_total = registry.counter(
    "safety_incidents_total"
)

emergency_events_total = registry.counter(
    "emergency_events_total"
)


# =========================================================
# RAG metrics
# =========================================================

rag_requests_total = registry.counter(
    "rag_requests_total"
)

rag_errors_total = registry.counter(
    "rag_errors_total"
)

rag_duration_ms = registry.histogram(
    "rag_duration_ms"
)

retrieval_duration_ms = registry.histogram(
    "retrieval_duration_ms"
)

reranker_duration_ms = registry.histogram(
    "reranker_duration_ms"
)

retrieved_chunks = registry.histogram(
    "retrieved_chunks"
)

retrieval_recall_at_k = registry.gauge(
    "retrieval_recall_at_k"
)

retrieval_precision_at_k = registry.gauge(
    "retrieval_precision_at_k"
)

retrieval_mrr = registry.gauge(
    "retrieval_mrr"
)

retrieval_ndcg = registry.gauge(
    "retrieval_ndcg"
)

rag_groundedness = registry.gauge(
    "rag_groundedness"
)

rag_faithfulness = registry.gauge(
    "rag_faithfulness"
)

citation_precision = registry.gauge(
    "citation_precision"
)

citation_recall = registry.gauge(
    "citation_recall"
)


# =========================================================
# LLM metrics
# =========================================================

llm_requests_total = registry.counter(
    "llm_requests_total"
)

llm_errors_total = registry.counter(
    "llm_errors_total"
)

llm_timeouts_total = registry.counter(
    "llm_timeouts_total"
)

llm_fallbacks_total = registry.counter(
    "llm_fallbacks_total"
)

llm_duration_ms = registry.histogram(
    "llm_duration_ms"
)

llm_ttft_ms = registry.histogram(
    "llm_ttft_ms"
)

llm_input_tokens_total = registry.counter(
    "llm_input_tokens_total"
)

llm_output_tokens_total = registry.counter(
    "llm_output_tokens_total"
)

llm_reasoning_tokens_total = registry.counter(
    "llm_reasoning_tokens_total"
)

llm_cost_total = registry.counter(
    "llm_cost_total"
)

llm_active_requests = registry.gauge(
    "llm_active_requests"
)

llm_safety_blocks_total = registry.counter(
    "llm_safety_blocks_total"
)

llm_hallucination_risk_total = registry.counter(
    "llm_hallucination_risk_total"
)


# =========================================================
# Agent / tool metrics
# =========================================================

agent_runs_total = registry.counter(
    "agent_runs_total"
)

agent_errors_total = registry.counter(
    "agent_errors_total"
)

agent_duration_ms = registry.histogram(
    "agent_duration_ms"
)

agent_steps_total = registry.counter(
    "agent_steps_total"
)

tool_calls_total = registry.counter(
    "tool_calls_total"
)

tool_call_errors_total = registry.counter(
    "tool_call_errors_total"
)

tool_call_duration_ms = registry.histogram(
    "tool_call_duration_ms"
)


# =========================================================
# Drift metrics
# =========================================================

data_drift_evaluations_total = registry.counter(
    "data_drift_evaluations_total"
)

data_drift_detected_total = registry.counter(
    "data_drift_detected_total"
)

data_drift_score = registry.gauge(
    "data_drift_score"
)

drifted_feature_ratio = registry.gauge(
    "drifted_feature_ratio"
)


# =========================================================
# Alerting metrics
# =========================================================

alerts_created_total = registry.counter(
    "alerts_created_total"
)

alerts_resolved_total = registry.counter(
    "alerts_resolved_total"
)

alerts_suppressed_total = registry.counter(
    "alerts_suppressed_total"
)

active_alerts = registry.gauge(
    "active_alerts"
)


# =========================================================
# Cost / business metrics
# =========================================================

avoided_cost_total = registry.counter(
    "avoided_cost_total"
)

downtime_savings_total = registry.counter(
    "downtime_savings_total"
)

maintenance_cost_total = registry.counter(
    "maintenance_cost_total"
)

maintenance_roi = registry.gauge(
    "maintenance_roi"
)

downtime_hours_avoided_total = registry.counter(
    "downtime_hours_avoided_total"
)


# =========================================================
# Timing utility
# =========================================================

@contextmanager
def observe_duration(
    histogram: Histogram,
    *,
    labels: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    started = perf_counter()

    try:
        yield

    finally:
        elapsed_ms = (
            perf_counter()
            - started
        ) * 1000.0

        histogram.observe(
            elapsed_ms,
            labels=labels,
        )


# =========================================================
# HTTP helpers
# =========================================================

@contextmanager
def track_http_request(
    *,
    method: str,
    route: str,
) -> Iterator[None]:
    base_labels = {
        "method": method.upper(),
        "route": route,
    }

    http_requests_in_progress.inc(
        labels=base_labels
    )

    started = perf_counter()

    status_code = 500

    try:
        yield
        status_code = 200

    except Exception:
        http_request_errors_total.inc(
            labels=base_labels
        )
        raise

    finally:
        elapsed = (
            perf_counter()
            - started
        ) * 1000.0

        http_requests_in_progress.dec(
            labels=base_labels
        )

        http_requests_total.inc(
            labels={
                **base_labels,
                "status":
                    status_code,
            }
        )

        http_request_duration_ms.observe(
            elapsed,
            labels=base_labels,
        )


def record_http_request(
    *,
    method: str,
    route: str,
    status_code: int,
    duration_ms: float,
) -> None:
    labels = {
        "method": method.upper(),
        "route": route,
        "status": status_code,
    }

    http_requests_total.inc(
        labels=labels
    )

    http_request_duration_ms.observe(
        duration_ms,
        labels={
            "method": method.upper(),
            "route": route,
        },
    )

    if status_code >= 500:
        http_request_errors_total.inc(
            labels={
                "method": method.upper(),
                "route": route,
            }
        )


# =========================================================
# ML helpers
# =========================================================

def record_model_inference(
    *,
    model_name: str,
    model_version: str | None,
    prediction_type: str,
    duration_ms: float,
    success: bool = True,
    confidence: float | None = None,
) -> None:
    labels = {
        "model": model_name,
        "version": model_version or "unknown",
        "prediction_type": prediction_type,
    }

    model_inference_total.inc(
        labels=labels
    )

    model_inference_duration_ms.observe(
        duration_ms,
        labels=labels,
    )

    if confidence is not None:
        model_prediction_confidence.observe(
            confidence,
            labels=labels,
        )

    if not success:
        model_inference_errors_total.inc(
            labels=labels
        )


def record_failure_prediction(
    *,
    asset_id: str,
    model_name: str,
    failure_mode: str,
    probability: float,
) -> None:
    labels = {
        "asset_id": asset_id,
        "model": model_name,
        "failure_mode": failure_mode,
    }

    failure_predictions_total.inc(
        labels=labels
    )

    failure_probability.set(
        probability,
        labels=labels,
    )


def record_rul_prediction(
    *,
    asset_id: str,
    model_name: str,
    rul: float,
    unit: str,
) -> None:
    labels = {
        "asset_id": asset_id,
        "model": model_name,
        "unit": unit,
    }

    rul_predictions_total.inc(
        labels=labels
    )

    predicted_rul.set(
        rul,
        labels=labels,
    )


# =========================================================
# RAG helpers
# =========================================================

def record_rag_request(
    *,
    retriever: str,
    duration_ms: float,
    retrieval_ms: float,
    reranker_ms: float | None = None,
    chunk_count: int = 0,
    success: bool = True,
) -> None:
    labels = {
        "retriever": retriever,
    }

    rag_requests_total.inc(
        labels=labels
    )

    rag_duration_ms.observe(
        duration_ms,
        labels=labels,
    )

    retrieval_duration_ms.observe(
        retrieval_ms,
        labels=labels,
    )

    retrieved_chunks.observe(
        chunk_count,
        labels=labels,
    )

    if reranker_ms is not None:
        reranker_duration_ms.observe(
            reranker_ms,
            labels=labels,
        )

    if not success:
        rag_errors_total.inc(
            labels=labels
        )


def record_rag_evaluation(
    *,
    model_name: str,
    recall_at_k: float | None = None,
    precision_at_k: float | None = None,
    mrr: float | None = None,
    ndcg: float | None = None,
    groundedness: float | None = None,
    faithfulness: float | None = None,
    citation_precision_value: float | None = None,
    citation_recall_value: float | None = None,
) -> None:
    labels = {
        "model": model_name,
    }

    values = (
        (
            retrieval_recall_at_k,
            recall_at_k,
        ),
        (
            retrieval_precision_at_k,
            precision_at_k,
        ),
        (
            retrieval_mrr,
            mrr,
        ),
        (
            retrieval_ndcg,
            ndcg,
        ),
        (
            rag_groundedness,
            groundedness,
        ),
        (
            rag_faithfulness,
            faithfulness,
        ),
        (
            citation_precision,
            citation_precision_value,
        ),
        (
            citation_recall,
            citation_recall_value,
        ),
    )

    for metric, value in values:
        if value is not None:
            metric.set(
                value,
                labels=labels,
            )


# =========================================================
# LLM helpers
# =========================================================

def record_llm_request(
    *,
    provider: str,
    model: str,
    operation: str,
    status: str,
    duration_ms: float,

    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,

    cost: float = 0.0,

    ttft_ms: float | None = None,

    fallback_used: bool = False,
    safety_blocked: bool = False,

    groundedness: float | None = None,
) -> None:
    labels = {
        "provider": provider,
        "model": model,
        "operation": operation,
    }

    llm_requests_total.inc(
        labels={
            **labels,
            "status": status,
        }
    )

    llm_duration_ms.observe(
        duration_ms,
        labels=labels,
    )

    if ttft_ms is not None:
        llm_ttft_ms.observe(
            ttft_ms,
            labels=labels,
        )

    if input_tokens > 0:
        llm_input_tokens_total.inc(
            input_tokens,
            labels=labels,
        )

    if output_tokens > 0:
        llm_output_tokens_total.inc(
            output_tokens,
            labels=labels,
        )

    if reasoning_tokens > 0:
        llm_reasoning_tokens_total.inc(
            reasoning_tokens,
            labels=labels,
        )

    if cost > 0:
        llm_cost_total.inc(
            cost,
            labels={
                **labels,
                "currency": "USD",
            },
        )

    if status.lower() in {
        "error",
        "timeout",
        "cancelled",
    }:
        llm_errors_total.inc(
            labels=labels
        )

    if status.lower() == "timeout":
        llm_timeouts_total.inc(
            labels=labels
        )

    if fallback_used:
        llm_fallbacks_total.inc(
            labels=labels
        )

    if safety_blocked:
        llm_safety_blocks_total.inc(
            labels=labels
        )

    if (
        groundedness is not None
        and groundedness < 0.80
    ):
        llm_hallucination_risk_total.inc(
            labels=labels
        )


# =========================================================
# Agent / tool helpers
# =========================================================

def record_agent_run(
    *,
    agent_name: str,
    duration_ms: float,
    success: bool,
    steps: int = 0,
) -> None:
    labels = {
        "agent": agent_name,
    }

    agent_runs_total.inc(
        labels=labels
    )

    agent_duration_ms.observe(
        duration_ms,
        labels=labels,
    )

    if steps > 0:
        agent_steps_total.inc(
            steps,
            labels=labels,
        )

    if not success:
        agent_errors_total.inc(
            labels=labels
        )


def record_tool_call(
    *,
    tool_name: str,
    duration_ms: float,
    success: bool,
) -> None:
    labels = {
        "tool": tool_name,
    }

    tool_calls_total.inc(
        labels=labels
    )

    tool_call_duration_ms.observe(
        duration_ms,
        labels=labels,
    )

    if not success:
        tool_call_errors_total.inc(
            labels=labels
        )


# =========================================================
# Drift helpers
# =========================================================

def record_data_drift(
    *,
    model_name: str,
    model_version: str,
    score: float,
    feature_ratio: float,
    drift_detected: bool,
    severity: str,
) -> None:
    labels = {
        "model": model_name,
        "version": model_version,
    }

    data_drift_evaluations_total.inc(
        labels=labels
    )

    data_drift_score.set(
        score,
        labels=labels,
    )

    drifted_feature_ratio.set(
        feature_ratio,
        labels=labels,
    )

    if drift_detected:
        data_drift_detected_total.inc(
            labels={
                **labels,
                "severity": severity,
            }
        )


# =========================================================
# Safety helpers
# =========================================================

def record_safety_incident(
    *,
    incident_type: str,
    severity: str,
    site_id: str | None = None,
) -> None:
    safety_incidents_total.inc(
        labels={
            "incident_type": incident_type,
            "severity": severity,
            "site_id": site_id or "unknown",
        }
    )


# =========================================================
# Cost helpers
# =========================================================

def record_avoided_cost(
    *,
    amount: float,
    currency: str,
    asset_id: str | None = None,
) -> None:
    if amount <= 0:
        return

    avoided_cost_total.inc(
        amount,
        labels={
            "currency": currency,
            "asset_id":
                asset_id or "unknown",
        },
    )


def record_maintenance_roi(
    *,
    roi_percent: float,
    asset_id: str,
) -> None:
    maintenance_roi.set(
        roi_percent,
        labels={
            "asset_id": asset_id,
        },
    )


# =========================================================
# Prometheus exposition
# =========================================================

def _escape_prometheus_label(
    value: str,
) -> str:
    return (
        value
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _format_prometheus_labels(
    labels: Mapping[str, str],
) -> str:
    if not labels:
        return ""

    content = ",".join(
        f'{key}="{_escape_prometheus_label(value)}"'
        for key, value
        in sorted(labels.items())
    )

    return "{" + content + "}"


def prometheus_text(
    metric_registry: MetricRegistry = registry,
) -> str:
    snapshot = metric_registry.snapshot()

    lines: list[str] = []

    # Counters
    for name, series in snapshot[
        "counters"
    ].items():
        for item in series:
            labels = (
                _format_prometheus_labels(
                    item["labels"]
                )
            )

            lines.append(
                f"{name}{labels} "
                f'{item["value"]}'
            )

    # Gauges
    for name, series in snapshot[
        "gauges"
    ].items():
        for item in series:
            labels = (
                _format_prometheus_labels(
                    item["labels"]
                )
            )

            lines.append(
                f"{name}{labels} "
                f'{item["value"]}'
            )

    # Histogram summaries.
    #
    # This is intentionally exported as summary-like
    # derived metrics rather than pretending these are
    # native Prometheus histogram buckets.
    for name, series in snapshot[
        "histograms"
    ].items():
        for item in series:
            labels = item["labels"]

            base = (
                _format_prometheus_labels(
                    labels
                )
            )

            lines.append(
                f"{name}_count{base} "
                f'{item["count"]}'
            )

            lines.append(
                f"{name}_sum{base} "
                f'{item["sum"]}'
            )

            for quantile_name in (
                "p50",
                "p95",
                "p99",
            ):
                value = item[
                    quantile_name
                ]

                if value is None:
                    continue

                quantile = {
                    "p50": "0.50",
                    "p95": "0.95",
                    "p99": "0.99",
                }[
                    quantile_name
                ]

                quantile_labels = {
                    **labels,
                    "quantile": quantile,
                }

                rendered = (
                    _format_prometheus_labels(
                        quantile_labels
                    )
                )

                lines.append(
                    f"{name}{rendered} "
                    f"{value}"
                )

    return "\n".join(lines) + "\n"


# =========================================================
# Dashboard summary
# =========================================================

def metrics_snapshot() -> dict[str, Any]:
    return registry.snapshot()


# =========================================================
# FastAPI endpoint
# =========================================================

def install_metrics_endpoint(
    app: Any,
    *,
    path: str = "/metrics",
) -> None:
    """
    Installs a Prometheus-compatible text endpoint.

    Example:

        install_metrics_endpoint(app)
    """

    try:
        from fastapi import Response
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is required to install "
            "the metrics endpoint."
        ) from exc

    @app.get(
        path,
        include_in_schema=False,
    )
    async def metrics_endpoint() -> Response:
        return Response(
            content=prometheus_text(),
            media_type=(
                "text/plain; "
                "version=0.0.4; "
                "charset=utf-8"
            ),
        )


# =========================================================
# FastAPI request middleware
# =========================================================

def install_metrics_middleware(
    app: Any,
) -> None:
    @app.middleware("http")
    async def metrics_middleware(
        request: Any,
        call_next: Any,
    ) -> Any:
        route = request.url.path
        method = request.method

        labels = {
            "method": method,
            "route": route,
        }

        http_requests_in_progress.inc(
            labels=labels
        )

        started = perf_counter()

        status_code = 500

        try:
            response = await call_next(
                request
            )

            status_code = (
                response.status_code
            )

            return response

        except Exception:
            http_request_errors_total.inc(
                labels=labels
            )

            raise

        finally:
            duration_ms = (
                perf_counter()
                - started
            ) * 1000.0

            http_requests_in_progress.dec(
                labels=labels
            )

            http_requests_total.inc(
                labels={
                    **labels,
                    "status":
                        status_code,
                }
            )

            http_request_duration_ms.observe(
                duration_ms,
                labels=labels,
            )


# =========================================================
# Health / observability summary
# =========================================================

def observability_summary() -> dict[str, Any]:
    snapshot = registry.snapshot()

    return {
        "status": "ok",

        "metric_families": {
            "counters": len(
                snapshot["counters"]
            ),

            "gauges": len(
                snapshot["gauges"]
            ),

            "histograms": len(
                snapshot["histograms"]
            ),
        },

        "domains": [
            "http",
            "database",
            "redis",
            "kafka",
            "mqtt",
            "telemetry",
            "sensors",
            "ml",
            "predictive_maintenance",
            "computer_vision",
            "safety",
            "rag",
            "llm",
            "agents",
            "tools",
            "drift",
            "alerting",
            "cost",
        ],
    }


__all__ = [
    # primitives
    "Counter",
    "Gauge",
    "Histogram",
    "HistogramStats",
    "MetricRegistry",
    "registry",

    # utilities
    "observe_duration",
    "prometheus_text",
    "metrics_snapshot",
    "observability_summary",

    # FastAPI
    "install_metrics_endpoint",
    "install_metrics_middleware",

    # helpers
    "record_http_request",
    "record_model_inference",
    "record_failure_prediction",
    "record_rul_prediction",
    "record_rag_request",
    "record_rag_evaluation",
    "record_llm_request",
    "record_agent_run",
    "record_tool_call",
    "record_data_drift",
    "record_safety_incident",
    "record_avoided_cost",
    "record_maintenance_roi",

    # HTTP
    "http_requests_total",
    "http_request_errors_total",
    "http_requests_in_progress",
    "http_request_duration_ms",

    # infrastructure
    "database_queries_total",
    "database_query_errors_total",
    "database_query_duration_ms",
    "database_connections_active",
    "redis_operations_total",
    "redis_errors_total",
    "redis_operation_duration_ms",
    "redis_cache_hits_total",
    "redis_cache_misses_total",
    "kafka_messages_produced_total",
    "kafka_messages_consumed_total",
    "kafka_errors_total",
    "kafka_processing_duration_ms",
    "kafka_consumer_lag",

    # telemetry
    "mqtt_messages_total",
    "mqtt_errors_total",
    "telemetry_messages_total",
    "telemetry_invalid_total",
    "sensor_readings_total",
    "sensor_errors_total",
    "sensor_value",

    # ML
    "model_inference_total",
    "model_inference_errors_total",
    "model_inference_duration_ms",
    "model_prediction_confidence",
    "model_active_requests",

    # predictive maintenance
    "anomalies_detected_total",
    "failure_predictions_total",
    "failure_probability",
    "rul_predictions_total",
    "predicted_rul",
    "maintenance_recommendations_total",
    "work_orders_created_total",

    # safety / CV
    "vision_inference_total",
    "vision_inference_duration_ms",
    "defects_detected_total",
    "fire_events_total",
    "smoke_events_total",
    "overheating_events_total",
    "unsafe_motion_events_total",
    "safety_incidents_total",
    "emergency_events_total",

    # RAG
    "rag_requests_total",
    "rag_errors_total",
    "rag_duration_ms",
    "retrieval_duration_ms",
    "reranker_duration_ms",
    "retrieved_chunks",
    "retrieval_recall_at_k",
    "retrieval_precision_at_k",
    "retrieval_mrr",
    "retrieval_ndcg",
    "rag_groundedness",
    "rag_faithfulness",
    "citation_precision",
    "citation_recall",

    # LLM
    "llm_requests_total",
    "llm_errors_total",
    "llm_timeouts_total",
    "llm_fallbacks_total",
    "llm_duration_ms",
    "llm_ttft_ms",
    "llm_input_tokens_total",
    "llm_output_tokens_total",
    "llm_reasoning_tokens_total",
    "llm_cost_total",
    "llm_active_requests",
    "llm_safety_blocks_total",
    "llm_hallucination_risk_total",

    # agents
    "agent_runs_total",
    "agent_errors_total",
    "agent_duration_ms",
    "agent_steps_total",
    "tool_calls_total",
    "tool_call_errors_total",
    "tool_call_duration_ms",

    # drift
    "data_drift_evaluations_total",
    "data_drift_detected_total",
    "data_drift_score",
    "drifted_feature_ratio",

    # alerts
    "alerts_created_total",
    "alerts_resolved_total",
    "alerts_suppressed_total",
    "active_alerts",

    # finance
    "avoided_cost_total",
    "downtime_savings_total",
    "maintenance_cost_total",
    "maintenance_roi",
    "downtime_hours_avoided_total",
]