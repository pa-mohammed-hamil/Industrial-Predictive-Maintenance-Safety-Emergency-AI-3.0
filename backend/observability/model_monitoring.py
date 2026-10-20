# app/observability/model_monitoring.py

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Deque, Mapping, Sequence
from uuid import uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def _finite(
    value: float,
    *,
    name: str,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    return value


def _probability(
    value: float,
    *,
    name: str,
) -> float:
    value = _finite(
        value,
        name=name,
    )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


def _percentile(
    values: Sequence[float],
    q: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

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


def _safe_mean(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return mean(values)


# =========================================================
# Enums
# =========================================================

class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"

    ANOMALY = "anomaly"
    FAILURE = "failure"
    RUL = "rul"

    DEFECT = "defect"
    FIRE = "fire"
    SMOKE = "smoke"
    MOTION = "motion"

    RISK = "risk"
    COST = "cost"

    EMBEDDING = "embedding"
    RERANKER = "reranker"

    OTHER = "other"


class ModelHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MonitoringSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class PredictionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    FALLBACK = "fallback"


class MonitoringEventType(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    HIGH_LATENCY = "high_latency"

    ERROR_RATE = "error_rate"
    TIMEOUT_RATE = "timeout_rate"

    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"

    SAFETY_DEGRADATION = "safety_degradation"

    STALE_MODEL = "stale_model"

    PREDICTION_ANOMALY = "prediction_anomaly"

    THROUGHPUT_DROP = "throughput_drop"

    HEALTH_CHANGE = "health_change"


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class ModelMonitoringConfig:
    max_history_per_model: int = 10_000

    health_window_minutes: int = 60

    minimum_health_samples: int = 10

    low_confidence_threshold: float = 0.60

    warning_error_rate: float = 0.02
    critical_error_rate: float = 0.10

    warning_timeout_rate: float = 0.01
    critical_timeout_rate: float = 0.05

    warning_p95_latency_ms: float = 2_000.0
    critical_p95_latency_ms: float = 10_000.0

    warning_low_confidence_rate: float = 0.20
    critical_low_confidence_rate: float = 0.50

    stale_model_days: int = 180

    enable_data_drift: bool = True
    enable_model_drift: bool = True

    enable_alerting: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True

    safety_model_types: set[
        ModelType
    ] = field(
        default_factory=lambda: {
            ModelType.FAILURE,
            ModelType.RUL,
            ModelType.FIRE,
            ModelType.SMOKE,
            ModelType.DEFECT,
            ModelType.MOTION,
            ModelType.RISK,
        }
    )


# =========================================================
# Model registration
# =========================================================

@dataclass(slots=True)
class ModelRegistration:
    model_name: str
    model_version: str

    model_type: ModelType

    deployed_at: datetime = field(
        default_factory=utc_now
    )

    training_completed_at: datetime | None = None

    stage: str = "production"

    owner: str | None = None

    safety_critical: bool = False

    expected_latency_ms: float | None = None
    minimum_confidence: float | None = None

    baseline_metrics: dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    @property
    def model_key(self) -> str:
        return (
            f"{self.model_name}:"
            f"{self.model_version}"
        )


# =========================================================
# Prediction record
# =========================================================

@dataclass(slots=True)
class PredictionRecord:
    prediction_id: str

    model_name: str
    model_version: str

    model_type: ModelType

    status: PredictionStatus

    timestamp: datetime

    latency_ms: float

    prediction: Any = None
    actual: Any = None

    confidence: float | None = None

    asset_id: str | None = None
    site_id: str | None = None

    request_id: str | None = None
    trace_id: str | None = None

    fallback_used: bool = False

    error_type: str | None = None

    input_feature_summary: dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# =========================================================
# Monitoring event
# =========================================================

@dataclass(slots=True)
class MonitoringEvent:
    event_id: str

    model_name: str
    model_version: str

    event_type: MonitoringEventType

    severity: MonitoringSeverity

    message: str

    value: float | None = None
    threshold: float | None = None

    asset_id: str | None = None
    site_id: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# =========================================================
# Health result
# =========================================================

@dataclass(slots=True)
class ModelHealth:
    model_name: str
    model_version: str

    model_type: ModelType

    status: ModelHealthStatus

    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    timeout_predictions: int

    success_rate: float
    error_rate: float
    timeout_rate: float

    average_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None

    average_confidence: float | None
    low_confidence_rate: float | None

    fallback_rate: float

    predictions_per_minute: float

    data_drift_detected: bool = False
    model_drift_detected: bool = False

    data_drift_score: float | None = None
    model_drift_score: float | None = None

    active_events: int = 0

    evaluated_at: datetime = field(
        default_factory=utc_now
    )

    recommendations: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["model_type"] = (
            self.model_type.value
        )

        data["status"] = (
            self.status.value
        )

        data["evaluated_at"] = (
            self.evaluated_at.isoformat()
        )

        return data


# =========================================================
# Fleet summary
# =========================================================

@dataclass(slots=True)
class ModelFleetHealth:
    total_models: int

    healthy_models: int
    degraded_models: int
    unhealthy_models: int
    critical_models: int
    unknown_models: int

    total_predictions: int

    average_success_rate: float
    average_error_rate: float

    models: dict[
        str,
        ModelHealth,
    ]

    evaluated_at: datetime = field(
        default_factory=utc_now
    )


# =========================================================
# Runtime prediction timer
# =========================================================

class PredictionTimer:
    def __init__(
        self,
        monitor: "ModelMonitor",
        registration: ModelRegistration,
        *,
        prediction_id: str | None = None,
        asset_id: str | None = None,
        site_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.monitor = monitor
        self.registration = registration

        self.prediction_id = (
            prediction_id
            or new_id()
        )

        self.asset_id = asset_id
        self.site_id = site_id

        self.request_id = request_id
        self.trace_id = trace_id

        self.metadata = (
            metadata or {}
        )

        self.started = 0.0

        self.prediction: Any = None
        self.actual: Any = None

        self.confidence: float | None = None

        self.fallback_used = False

        self.feature_summary: dict[
            str,
            float,
        ] = {}

    def __enter__(
        self,
    ) -> "PredictionTimer":
        self.started = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        latency_ms = (
            perf_counter()
            - self.started
        ) * 1000.0

        if exc_type is None:
            status = (
                PredictionStatus.SUCCESS
            )

            error_type = None

        elif (
            exc_type is TimeoutError
            or (
                isinstance(
                    exc_value,
                    TimeoutError,
                )
            )
        ):
            status = (
                PredictionStatus.TIMEOUT
            )

            error_type = (
                exc_type.__name__
                if exc_type
                else "TimeoutError"
            )

        else:
            status = (
                PredictionStatus.ERROR
            )

            error_type = (
                exc_type.__name__
                if exc_type
                else "UnknownError"
            )

        self.monitor.record_prediction(
            PredictionRecord(
                prediction_id=(
                    self.prediction_id
                ),

                model_name=(
                    self.registration
                    .model_name
                ),

                model_version=(
                    self.registration
                    .model_version
                ),

                model_type=(
                    self.registration
                    .model_type
                ),

                status=status,

                timestamp=utc_now(),

                latency_ms=latency_ms,

                prediction=(
                    self.prediction
                ),

                actual=(
                    self.actual
                ),

                confidence=(
                    self.confidence
                ),

                asset_id=(
                    self.asset_id
                ),

                site_id=(
                    self.site_id
                ),

                request_id=(
                    self.request_id
                ),

                trace_id=(
                    self.trace_id
                ),

                fallback_used=(
                    self.fallback_used
                ),

                error_type=(
                    error_type
                ),

                input_feature_summary=(
                    self.feature_summary
                ),

                metadata=(
                    self.metadata
                ),
            )
        )

        # Never swallow model exceptions.
        return False


# =========================================================
# Model Monitor
# =========================================================

class ModelMonitor:
    def __init__(
        self,
        *,
        config: (
            ModelMonitoringConfig
            | None
        ) = None,
    ) -> None:
        self.config = (
            config
            or ModelMonitoringConfig()
        )

        self._models: dict[
            str,
            ModelRegistration,
        ] = {}

        self._predictions: dict[
            str,
            Deque[PredictionRecord],
        ] = {}

        self._events: dict[
            str,
            Deque[MonitoringEvent],
        ] = {}

        self._data_drift: dict[
            str,
            tuple[bool, float | None],
        ] = {}

        self._model_drift: dict[
            str,
            tuple[bool, float | None],
        ] = {}

        self._last_health: dict[
            str,
            ModelHealthStatus,
        ] = {}

        self._lock = (
            threading.RLock()
        )

    # -----------------------------------------------------
    # Registration
    # -----------------------------------------------------

    def register_model(
        self,
        registration: ModelRegistration,
    ) -> None:
        key = registration.model_key

        with self._lock:
            self._models[key] = (
                registration
            )

            self._predictions.setdefault(
                key,
                deque(
                    maxlen=(
                        self.config
                        .max_history_per_model
                    )
                ),
            )

            self._events.setdefault(
                key,
                deque(
                    maxlen=5_000
                ),
            )

    def unregister_model(
        self,
        model_name: str,
        model_version: str,
    ) -> None:
        key = self._model_key(
            model_name,
            model_version,
        )

        with self._lock:
            self._models.pop(
                key,
                None,
            )

            self._predictions.pop(
                key,
                None,
            )

            self._events.pop(
                key,
                None,
            )

            self._data_drift.pop(
                key,
                None,
            )

            self._model_drift.pop(
                key,
                None,
            )

            self._last_health.pop(
                key,
                None,
            )

    def get_model(
        self,
        model_name: str,
        model_version: str,
    ) -> ModelRegistration | None:
        key = self._model_key(
            model_name,
            model_version,
        )

        with self._lock:
            return self._models.get(
                key
            )

    # -----------------------------------------------------
    # Timer
    # -----------------------------------------------------

    def prediction_timer(
        self,
        *,
        model_name: str,
        model_version: str,

        prediction_id: str | None = None,

        asset_id: str | None = None,
        site_id: str | None = None,

        request_id: str | None = None,
        trace_id: str | None = None,

        metadata: dict[
            str,
            Any,
        ] | None = None,
    ) -> PredictionTimer:
        registration = (
            self.get_model(
                model_name,
                model_version,
            )
        )

        if registration is None:
            raise KeyError(
                "Model is not registered: "
                f"{model_name}:"
                f"{model_version}"
            )

        return PredictionTimer(
            self,
            registration,

            prediction_id=(
                prediction_id
            ),

            asset_id=asset_id,
            site_id=site_id,

            request_id=request_id,
            trace_id=trace_id,

            metadata=metadata,
        )

    # -----------------------------------------------------
    # Record prediction
    # -----------------------------------------------------

    def record_prediction(
        self,
        record: PredictionRecord,
    ) -> None:
        if record.latency_ms < 0:
            raise ValueError(
                "latency_ms cannot be negative."
            )

        if record.confidence is not None:
            record.confidence = (
                _probability(
                    record.confidence,
                    name="confidence",
                )
            )

        key = self._model_key(
            record.model_name,
            record.model_version,
        )

        with self._lock:
            if key not in self._models:
                self.register_model(
                    ModelRegistration(
                        model_name=(
                            record.model_name
                        ),

                        model_version=(
                            record.model_version
                        ),

                        model_type=(
                            record.model_type
                        ),
                    )
                )

            self._predictions[
                key
            ].append(
                record
            )

        self._publish_prediction_metrics(
            record
        )

        self._evaluate_prediction_event(
            record
        )

    # -----------------------------------------------------
    # Truth update
    # -----------------------------------------------------

    def attach_ground_truth(
        self,
        *,
        prediction_id: str,
        actual: Any,
    ) -> bool:
        with self._lock:
            for records in (
                self._predictions.values()
            ):
                for record in reversed(
                    records
                ):
                    if (
                        record.prediction_id
                        == prediction_id
                    ):
                        record.actual = actual
                        return True

        return False

    # -----------------------------------------------------
    # Drift integration
    # -----------------------------------------------------

    def update_data_drift(
        self,
        *,
        model_name: str,
        model_version: str,
        detected: bool,
        score: float | None = None,
    ) -> None:
        key = self._model_key(
            model_name,
            model_version,
        )

        with self._lock:
            self._data_drift[key] = (
                detected,
                score,
            )

        if detected:
            self._emit_event(
                MonitoringEvent(
                    event_id=new_id(),

                    model_name=model_name,
                    model_version=model_version,

                    event_type=(
                        MonitoringEventType
                        .DATA_DRIFT
                    ),

                    severity=(
                        MonitoringSeverity
                        .WARNING
                    ),

                    message=(
                        "Input data drift "
                        "detected."
                    ),

                    value=score,
                )
            )

    def update_model_drift(
        self,
        *,
        model_name: str,
        model_version: str,
        detected: bool,
        score: float | None = None,
        critical: bool = False,
    ) -> None:
        key = self._model_key(
            model_name,
            model_version,
        )

        with self._lock:
            self._model_drift[key] = (
                detected,
                score,
            )

        if detected:
            self._emit_event(
                MonitoringEvent(
                    event_id=new_id(),

                    model_name=model_name,
                    model_version=model_version,

                    event_type=(
                        MonitoringEventType
                        .MODEL_DRIFT
                    ),

                    severity=(
                        MonitoringSeverity.CRITICAL
                        if critical
                        else MonitoringSeverity.HIGH
                    ),

                    message=(
                        "Model performance "
                        "drift detected."
                    ),

                    value=score,
                )
            )

    # -----------------------------------------------------
    # Window
    # -----------------------------------------------------

    def prediction_window(
        self,
        *,
        model_name: str,
        model_version: str,
        minutes: int | None = None,
    ) -> list[PredictionRecord]:
        key = self._model_key(
            model_name,
            model_version,
        )

        minutes = (
            minutes
            or self.config
            .health_window_minutes
        )

        cutoff = (
            utc_now()
            - timedelta(
                minutes=minutes
            )
        )

        with self._lock:
            records = list(
                self._predictions.get(
                    key,
                    (),
                )
            )

        return [
            record
            for record in records
            if record.timestamp >= cutoff
        ]

    # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    def evaluate_health(
        self,
        *,
        model_name: str,
        model_version: str,
        window_minutes: int | None = None,
    ) -> ModelHealth:
        key = self._model_key(
            model_name,
            model_version,
        )

        registration = (
            self.get_model(
                model_name,
                model_version,
            )
        )

        if registration is None:
            raise KeyError(
                f"Unknown model: {key}"
            )

        window_minutes = (
            window_minutes
            or self.config
            .health_window_minutes
        )

        records = self.prediction_window(
            model_name=model_name,
            model_version=model_version,
            minutes=window_minutes,
        )

        total = len(records)

        successful = sum(
            record.status
            in {
                PredictionStatus.SUCCESS,
                PredictionStatus.FALLBACK,
            }
            for record in records
        )

        failed = sum(
            record.status
            == PredictionStatus.ERROR
            for record in records
        )

        timeouts = sum(
            record.status
            == PredictionStatus.TIMEOUT
            for record in records
        )

        fallbacks = sum(
            record.fallback_used
            or record.status
            == PredictionStatus.FALLBACK
            for record in records
        )

        success_rate = (
            successful / total
            if total
            else 0.0
        )

        error_rate = (
            failed / total
            if total
            else 0.0
        )

        timeout_rate = (
            timeouts / total
            if total
            else 0.0
        )

        fallback_rate = (
            fallbacks / total
            if total
            else 0.0
        )

        latencies = [
            record.latency_ms
            for record in records
            if (
                record.status
                != PredictionStatus.REJECTED
            )
        ]

        confidences = [
            record.confidence
            for record in records
            if record.confidence
            is not None
        ]

        low_confidence_count = sum(
            confidence
            < self.config
            .low_confidence_threshold
            for confidence in confidences
        )

        low_confidence_rate = (
            low_confidence_count
            / len(confidences)
            if confidences
            else None
        )

        data_drift_detected, (
            data_drift_score
        ) = self._data_drift.get(
            key,
            (False, None),
        )

        model_drift_detected, (
            model_drift_score
        ) = self._model_drift.get(
            key,
            (False, None),
        )

        status = (
            self._calculate_health_status(
                registration=registration,

                total=total,

                error_rate=error_rate,

                timeout_rate=(
                    timeout_rate
                ),

                p95_latency_ms=(
                    _percentile(
                        latencies,
                        0.95,
                    )
                ),

                low_confidence_rate=(
                    low_confidence_rate
                ),

                data_drift_detected=(
                    data_drift_detected
                ),

                model_drift_detected=(
                    model_drift_detected
                ),
            )
        )

        events = self.events(
            model_name=model_name,
            model_version=model_version,
            since=(
                utc_now()
                - timedelta(
                    minutes=(
                        window_minutes
                    )
                )
            ),
        )

        recommendations = (
            self._recommendations(
                registration=registration,

                status=status,

                error_rate=error_rate,

                timeout_rate=(
                    timeout_rate
                ),

                p95_latency_ms=(
                    _percentile(
                        latencies,
                        0.95,
                    )
                ),

                low_confidence_rate=(
                    low_confidence_rate
                ),

                data_drift=(
                    data_drift_detected
                ),

                model_drift=(
                    model_drift_detected
                ),
            )
        )

        health = ModelHealth(
            model_name=model_name,

            model_version=(
                model_version
            ),

            model_type=(
                registration.model_type
            ),

            status=status,

            total_predictions=total,

            successful_predictions=(
                successful
            ),

            failed_predictions=failed,

            timeout_predictions=(
                timeouts
            ),

            success_rate=success_rate,
            error_rate=error_rate,
            timeout_rate=timeout_rate,

            average_latency_ms=(
                _safe_mean(
                    latencies
                )
            ),

            p50_latency_ms=(
                _percentile(
                    latencies,
                    0.50,
                )
            ),

            p95_latency_ms=(
                _percentile(
                    latencies,
                    0.95,
                )
            ),

            p99_latency_ms=(
                _percentile(
                    latencies,
                    0.99,
                )
            ),

            average_confidence=(
                _safe_mean(
                    confidences
                )
            ),

            low_confidence_rate=(
                low_confidence_rate
            ),

            fallback_rate=(
                fallback_rate
            ),

            predictions_per_minute=(
                total
                / max(
                    window_minutes,
                    1,
                )
            ),

            data_drift_detected=(
                data_drift_detected
            ),

            model_drift_detected=(
                model_drift_detected
            ),

            data_drift_score=(
                data_drift_score
            ),

            model_drift_score=(
                model_drift_score
            ),

            active_events=len(
                events
            ),

            recommendations=(
                recommendations
            ),
        )

        self._handle_health_change(
            health
        )

        return health

    # -----------------------------------------------------
    # Fleet health
    # -----------------------------------------------------

    def fleet_health(
        self,
    ) -> ModelFleetHealth:
        with self._lock:
            registrations = list(
                self._models.values()
            )

        models: dict[
            str,
            ModelHealth,
        ] = {}

        statuses = defaultdict(int)

        total_predictions = 0

        success_rates: list[float] = []
        error_rates: list[float] = []

        for registration in registrations:
            health = self.evaluate_health(
                model_name=(
                    registration.model_name
                ),

                model_version=(
                    registration.model_version
                ),
            )

            models[
                registration.model_key
            ] = health

            statuses[
                health.status
            ] += 1

            total_predictions += (
                health.total_predictions
            )

            if health.total_predictions:
                success_rates.append(
                    health.success_rate
                )

                error_rates.append(
                    health.error_rate
                )

        return ModelFleetHealth(
            total_models=len(
                registrations
            ),

            healthy_models=statuses[
                ModelHealthStatus.HEALTHY
            ],

            degraded_models=statuses[
                ModelHealthStatus.DEGRADED
            ],

            unhealthy_models=statuses[
                ModelHealthStatus.UNHEALTHY
            ],

            critical_models=statuses[
                ModelHealthStatus.CRITICAL
            ],

            unknown_models=statuses[
                ModelHealthStatus.UNKNOWN
            ],

            total_predictions=(
                total_predictions
            ),

            average_success_rate=(
                _safe_mean(
                    success_rates
                )
                or 0.0
            ),

            average_error_rate=(
                _safe_mean(
                    error_rates
                )
                or 0.0
            ),

            models=models,
        )

    # -----------------------------------------------------
    # Events
    # -----------------------------------------------------

    def events(
        self,
        *,
        model_name: str,
        model_version: str,
        since: datetime | None = None,
    ) -> list[MonitoringEvent]:
        key = self._model_key(
            model_name,
            model_version,
        )

        with self._lock:
            events = list(
                self._events.get(
                    key,
                    (),
                )
            )

        if since is None:
            return events

        return [
            event
            for event in events
            if event.timestamp >= since
        ]

    def _emit_event(
        self,
        event: MonitoringEvent,
    ) -> None:
        key = self._model_key(
            event.model_name,
            event.model_version,
        )

        with self._lock:
            self._events.setdefault(
                key,
                deque(maxlen=5_000),
            ).append(event)

        self._publish_event_log(
            event
        )

        self._publish_event_alert(
            event
        )

    # -----------------------------------------------------
    # Prediction checks
    # -----------------------------------------------------

    def _evaluate_prediction_event(
        self,
        record: PredictionRecord,
    ) -> None:
        registration = (
            self.get_model(
                record.model_name,
                record.model_version,
            )
        )

        if registration is None:
            return

        minimum_confidence = (
            registration
            .minimum_confidence
        )

        if minimum_confidence is None:
            minimum_confidence = (
                self.config
                .low_confidence_threshold
            )

        if (
            record.confidence is not None
            and record.confidence
            < minimum_confidence
        ):
            self._emit_event(
                MonitoringEvent(
                    event_id=new_id(),

                    model_name=(
                        record.model_name
                    ),

                    model_version=(
                        record.model_version
                    ),

                    event_type=(
                        MonitoringEventType
                        .LOW_CONFIDENCE
                    ),

                    severity=(
                        MonitoringSeverity
                        .WARNING
                    ),

                    message=(
                        "Low-confidence model "
                        "prediction detected."
                    ),

                    value=record.confidence,

                    threshold=(
                        minimum_confidence
                    ),

                    asset_id=(
                        record.asset_id
                    ),

                    site_id=(
                        record.site_id
                    ),

                    metadata={
                        "prediction_id":
                            record
                            .prediction_id,
                    },
                )
            )

        expected_latency = (
            registration
            .expected_latency_ms
        )

        if (
            expected_latency is not None
            and record.latency_ms
            > expected_latency
        ):
            severity = (
                MonitoringSeverity.HIGH
                if record.latency_ms
                > expected_latency * 2
                else MonitoringSeverity.WARNING
            )

            self._emit_event(
                MonitoringEvent(
                    event_id=new_id(),

                    model_name=(
                        record.model_name
                    ),

                    model_version=(
                        record.model_version
                    ),

                    event_type=(
                        MonitoringEventType
                        .HIGH_LATENCY
                    ),

                    severity=severity,

                    message=(
                        "Model inference latency "
                        "exceeded expected limit."
                    ),

                    value=(
                        record.latency_ms
                    ),

                    threshold=(
                        expected_latency
                    ),

                    asset_id=(
                        record.asset_id
                    ),

                    site_id=(
                        record.site_id
                    ),
                )
            )

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    def _calculate_health_status(
        self,
        *,
        registration: ModelRegistration,
        total: int,

        error_rate: float,
        timeout_rate: float,

        p95_latency_ms: float | None,

        low_confidence_rate: float | None,

        data_drift_detected: bool,
        model_drift_detected: bool,
    ) -> ModelHealthStatus:
        if (
            total
            < self.config
            .minimum_health_samples
        ):
            return (
                ModelHealthStatus.UNKNOWN
            )

        safety_critical = (
            registration.safety_critical
            or registration.model_type
            in self.config
            .safety_model_types
        )

        # Critical conditions
        if (
            error_rate
            >= self.config
            .critical_error_rate
        ):
            return (
                ModelHealthStatus.CRITICAL
            )

        if (
            timeout_rate
            >= self.config
            .critical_timeout_rate
        ):
            return (
                ModelHealthStatus.CRITICAL
            )

        if (
            p95_latency_ms is not None
            and p95_latency_ms
            >= self.config
            .critical_p95_latency_ms
        ):
            return (
                ModelHealthStatus.CRITICAL
            )

        if (
            model_drift_detected
            and safety_critical
        ):
            return (
                ModelHealthStatus.CRITICAL
            )

        # Unhealthy
        if model_drift_detected:
            return (
                ModelHealthStatus.UNHEALTHY
            )

        if (
            low_confidence_rate
            is not None
            and low_confidence_rate
            >= self.config
            .critical_low_confidence_rate
        ):
            return (
                ModelHealthStatus.UNHEALTHY
            )

        # Degraded
        if (
            error_rate
            >= self.config
            .warning_error_rate
        ):
            return (
                ModelHealthStatus.DEGRADED
            )

        if (
            timeout_rate
            >= self.config
            .warning_timeout_rate
        ):
            return (
                ModelHealthStatus.DEGRADED
            )

        if (
            p95_latency_ms is not None
            and p95_latency_ms
            >= self.config
            .warning_p95_latency_ms
        ):
            return (
                ModelHealthStatus.DEGRADED
            )

        if (
            low_confidence_rate
            is not None
            and low_confidence_rate
            >= self.config
            .warning_low_confidence_rate
        ):
            return (
                ModelHealthStatus.DEGRADED
            )

        if data_drift_detected:
            return (
                ModelHealthStatus.DEGRADED
            )

        return ModelHealthStatus.HEALTHY

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    def _recommendations(
        self,
        *,
        registration: ModelRegistration,
        status: ModelHealthStatus,

        error_rate: float,
        timeout_rate: float,

        p95_latency_ms: float | None,

        low_confidence_rate: float | None,

        data_drift: bool,
        model_drift: bool,
    ) -> list[str]:
        recommendations: list[str] = []

        if (
            error_rate
            >= self.config
            .warning_error_rate
        ):
            recommendations.append(
                "Investigate model inference "
                "errors and serving dependencies."
            )

        if (
            timeout_rate
            >= self.config
            .warning_timeout_rate
        ):
            recommendations.append(
                "Review model timeout limits, "
                "compute capacity and downstream "
                "dependencies."
            )

        if (
            p95_latency_ms is not None
            and p95_latency_ms
            >= self.config
            .warning_p95_latency_ms
        ):
            recommendations.append(
                "Investigate inference latency, "
                "batching, model size and serving "
                "resource utilization."
            )

        if (
            low_confidence_rate
            is not None
            and low_confidence_rate
            >= self.config
            .warning_low_confidence_rate
        ):
            recommendations.append(
                "Review low-confidence predictions "
                "and compare them with recent "
                "operating conditions."
            )

        if data_drift:
            recommendations.append(
                "Investigate feature distribution "
                "changes, sensors, preprocessing "
                "and operating conditions."
            )

        if model_drift:
            recommendations.append(
                "Validate model performance against "
                "recent ground truth and consider "
                "retraining, rollback or challenger "
                "model deployment."
            )

        if (
            registration.safety_critical
            and model_drift
        ):
            recommendations.append(
                "Route safety-critical predictions "
                "through fallback or human review "
                "until model performance is "
                "validated."
            )

        if (
            status
            == ModelHealthStatus.HEALTHY
        ):
            recommendations.append(
                "Continue normal monitoring."
            )

        return recommendations

    # -----------------------------------------------------
    # Health transition
    # -----------------------------------------------------

    def _handle_health_change(
        self,
        health: ModelHealth,
    ) -> None:
        key = self._model_key(
            health.model_name,
            health.model_version,
        )

        previous = (
            self._last_health.get(
                key
            )
        )

        self._last_health[
            key
        ] = health.status

        if (
            previous is None
            or previous == health.status
        ):
            return

        severity = {
            ModelHealthStatus.HEALTHY:
                MonitoringSeverity.INFO,

            ModelHealthStatus.UNKNOWN:
                MonitoringSeverity.INFO,

            ModelHealthStatus.DEGRADED:
                MonitoringSeverity.WARNING,

            ModelHealthStatus.UNHEALTHY:
                MonitoringSeverity.HIGH,

            ModelHealthStatus.CRITICAL:
                MonitoringSeverity.CRITICAL,
        }[
            health.status
        ]

        self._emit_event(
            MonitoringEvent(
                event_id=new_id(),

                model_name=(
                    health.model_name
                ),

                model_version=(
                    health.model_version
                ),

                event_type=(
                    MonitoringEventType
                    .HEALTH_CHANGE
                ),

                severity=severity,

                message=(
                    "Model health changed "
                    f"from {previous.value} "
                    f"to {health.status.value}."
                ),

                metadata={
                    "previous_status":
                        previous.value,

                    "current_status":
                        health.status.value,
                },
            )
        )

    # -----------------------------------------------------
    # Metrics bridge
    # -----------------------------------------------------

    def _publish_prediction_metrics(
        self,
        record: PredictionRecord,
    ) -> None:
        if not self.config.enable_metrics:
            return

        try:
            from app.observability.metrics import (
                record_model_inference,
            )
        except ImportError:
            return

        record_model_inference(
            model_name=(
                record.model_name
            ),

            model_version=(
                record.model_version
            ),

            prediction_type=(
                record.model_type.value
            ),

            duration_ms=(
                record.latency_ms
            ),

            success=(
                record.status
                in {
                    PredictionStatus.SUCCESS,
                    PredictionStatus.FALLBACK,
                }
            ),

            confidence=(
                record.confidence
            ),
        )

    # -----------------------------------------------------
    # Logging bridge
    # -----------------------------------------------------

    def _publish_event_log(
        self,
        event: MonitoringEvent,
    ) -> None:
        if not self.config.enable_logging:
            return

        try:
            from app.observability.logging import (
                LogEventType,
                get_logger,
            )
        except ImportError:
            return

        logger = get_logger(
            "industrial.model_monitoring"
        )

        data = {
            "event_id":
                event.event_id,

            "model_name":
                event.model_name,

            "model_version":
                event.model_version,

            "event_type":
                event.event_type.value,

            "severity":
                event.severity.value,

            "value":
                event.value,

            "threshold":
                event.threshold,

            "asset_id":
                event.asset_id,

            "site_id":
                event.site_id,

            **event.metadata,
        }

        if (
            event.severity
            == MonitoringSeverity.CRITICAL
        ):
            logger.critical(
                event.message,
                event_type=(
                    LogEventType
                    .MODEL_DRIFT
                ),
                data=data,
            )

        elif (
            event.severity
            == MonitoringSeverity.HIGH
        ):
            logger.error(
                event.message,
                event_type=(
                    LogEventType
                    .MODEL_DRIFT
                ),
                data=data,
            )

        elif (
            event.severity
            == MonitoringSeverity.WARNING
        ):
            logger.warning(
                event.message,
                event_type=(
                    LogEventType
                    .MODEL_DRIFT
                ),
                data=data,
            )

        else:
            logger.info(
                event.message,
                event_type=(
                    LogEventType
                    .MODEL_INFERENCE
                ),
                data=data,
            )

    # -----------------------------------------------------
    # Alert bridge
    # -----------------------------------------------------

    def _publish_event_alert(
        self,
        event: MonitoringEvent,
    ) -> None:
        if not self.config.enable_alerting:
            return

        if (
            event.severity
            not in {
                MonitoringSeverity.HIGH,
                MonitoringSeverity.CRITICAL,
            }
        ):
            return

        try:
            from app.observability.alerting import (
                alert_manager,
            )
        except ImportError:
            return

        # Keep this bridge defensive because alerting.py
        # can use a project-specific API.
        create_alert = getattr(
            alert_manager,
            "create_alert",
            None,
        )

        if not callable(create_alert):
            return

        try:
            create_alert(
                title=(
                    "Model Monitoring Alert"
                ),

                message=event.message,

                severity=(
                    event.severity.value
                ),

                category=(
                    "model_monitoring"
                ),

                source=(
                    event.model_name
                ),

                metadata={
                    "event_id":
                        event.event_id,

                    "model_name":
                        event.model_name,

                    "model_version":
                        event.model_version,

                    "event_type":
                        event.event_type.value,

                    "asset_id":
                        event.asset_id,

                    "site_id":
                        event.site_id,
                },
            )

        except Exception:
            # Monitoring must never break inference.
            return

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    @staticmethod
    def _model_key(
        model_name: str,
        model_version: str,
    ) -> str:
        return (
            f"{model_name}:"
            f"{model_version}"
        )


# =========================================================
# Model quality evaluation
# =========================================================

@dataclass(slots=True)
class PredictionQuality:
    samples: int

    accuracy: float | None = None

    mae: float | None = None
    rmse: float | None = None

    average_confidence: float | None = None

    false_negative_rate: float | None = None

    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


def evaluate_prediction_quality(
    records: Sequence[
        PredictionRecord
    ],
    *,
    positive_label: Any = 1,
) -> PredictionQuality:
    labeled = [
        record
        for record in records
        if record.actual is not None
    ]

    if not labeled:
        return PredictionQuality(
            samples=0
        )

    # Classification
    classification_records = [
        record
        for record in labeled
        if record.model_type
        in {
            ModelType.CLASSIFICATION,
            ModelType.ANOMALY,
            ModelType.FAILURE,
            ModelType.DEFECT,
            ModelType.FIRE,
            ModelType.SMOKE,
            ModelType.MOTION,
            ModelType.RISK,
        }
    ]

    accuracy: float | None = None
    fnr: float | None = None

    if classification_records:
        correct = sum(
            record.prediction
            == record.actual
            for record
            in classification_records
        )

        accuracy = (
            correct
            / len(
                classification_records
            )
        )

        tp = 0
        fn = 0

        for record in classification_records:
            actual_positive = (
                record.actual
                == positive_label
            )

            predicted_positive = (
                record.prediction
                == positive_label
            )

            if (
                actual_positive
                and predicted_positive
            ):
                tp += 1

            elif (
                actual_positive
                and not predicted_positive
            ):
                fn += 1

        if tp + fn:
            fnr = (
                fn / (tp + fn)
            )

    # Regression
    errors: list[float] = []

    for record in labeled:
        if record.model_type not in {
            ModelType.REGRESSION,
            ModelType.FORECASTING,
            ModelType.RUL,
            ModelType.COST,
        }:
            continue

        try:
            predicted = float(
                record.prediction
            )

            actual = float(
                record.actual
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            math.isfinite(predicted)
            and math.isfinite(actual)
        ):
            errors.append(
                predicted - actual
            )

    mae = None
    rmse = None

    if errors:
        mae = mean(
            abs(error)
            for error in errors
        )

        rmse = math.sqrt(
            mean(
                error * error
                for error in errors
            )
        )

    confidences = [
        record.confidence
        for record in labeled
        if record.confidence
        is not None
    ]

    return PredictionQuality(
        samples=len(labeled),

        accuracy=accuracy,

        mae=mae,
        rmse=rmse,

        average_confidence=(
            _safe_mean(
                confidences
            )
        ),

        false_negative_rate=fnr,
    )


# =========================================================
# Monitoring quality gate
# =========================================================

def model_health_gate(
    health: ModelHealth,
    *,
    allow_degraded: bool = False,
) -> dict[str, Any]:
    acceptable = {
        ModelHealthStatus.HEALTHY,
    }

    if allow_degraded:
        acceptable.add(
            ModelHealthStatus.DEGRADED
        )

    passed = (
        health.status
        in acceptable
    )

    return {
        "passed": passed,

        "model_name":
            health.model_name,

        "model_version":
            health.model_version,

        "status":
            health.status.value,

        "error_rate":
            health.error_rate,

        "timeout_rate":
            health.timeout_rate,

        "p95_latency_ms":
            health.p95_latency_ms,

        "average_confidence":
            health.average_confidence,

        "data_drift":
            health.data_drift_detected,

        "model_drift":
            health.model_drift_detected,

        "recommendations":
            health.recommendations,
    }


# =========================================================
# Global service
# =========================================================

model_monitor = ModelMonitor()


__all__ = [
    "ModelType",
    "ModelHealthStatus",
    "MonitoringSeverity",
    "PredictionStatus",
    "MonitoringEventType",

    "ModelMonitoringConfig",

    "ModelRegistration",
    "PredictionRecord",
    "MonitoringEvent",

    "ModelHealth",
    "ModelFleetHealth",

    "PredictionTimer",

    "PredictionQuality",
    "evaluate_prediction_quality",

    "ModelMonitor",

    "model_health_gate",

    "model_monitor",
]