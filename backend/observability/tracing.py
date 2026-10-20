# app/observability/tracing.py

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import inspect
import math
import threading
from contextlib import (
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    ParamSpec,
    TypeVar,
)
from uuid import uuid4


P = ParamSpec("P")
R = TypeVar("R")


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_trace_id() -> str:
    # 128-bit trace ID
    return uuid4().hex


def generate_span_id() -> str:
    # 64-bit span ID
    return uuid4().hex[:16]


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


def _safe_hash(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# =========================================================
# Enums
# =========================================================

class SpanKind(str, Enum):
    INTERNAL = "internal"

    SERVER = "server"
    CLIENT = "client"

    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class TraceComponent(str, Enum):
    API = "api"

    DATABASE = "database"
    REDIS = "redis"

    KAFKA = "kafka"
    MQTT = "mqtt"
    OPCUA = "opcua"
    PLC = "plc"

    SENSOR = "sensor"

    ML = "ml"
    DL = "dl"
    CV = "cv"

    ANOMALY = "anomaly"
    FAILURE = "failure"
    RUL = "rul"

    RAG = "rag"
    RETRIEVAL = "retrieval"
    RERANKER = "reranker"

    VECTOR_DB = "vector_db"
    KNOWLEDGE_GRAPH = "knowledge_graph"

    LLM = "llm"
    AGENT = "agent"
    TOOL = "tool"

    SAFETY = "safety"
    EMERGENCY = "emergency"

    COST = "cost"

    STORAGE = "storage"

    EXTERNAL_API = "external_api"

    WORKER = "worker"

    OTHER = "other"


# =========================================================
# Context variables
# =========================================================

_current_trace_id: contextvars.ContextVar[
    str | None
] = contextvars.ContextVar(
    "current_trace_id",
    default=None,
)

_current_span_id: contextvars.ContextVar[
    str | None
] = contextvars.ContextVar(
    "current_span_id",
    default=None,
)

_current_request_id: contextvars.ContextVar[
    str | None
] = contextvars.ContextVar(
    "current_request_id",
    default=None,
)

_current_asset_id: contextvars.ContextVar[
    str | None
] = contextvars.ContextVar(
    "current_asset_id",
    default=None,
)

_current_site_id: contextvars.ContextVar[
    str | None
] = contextvars.ContextVar(
    "current_site_id",
    default=None,
)


# =========================================================
# Context API
# =========================================================

def current_trace_id() -> str | None:
    return _current_trace_id.get()


def current_span_id() -> str | None:
    return _current_span_id.get()


def current_request_id() -> str | None:
    return _current_request_id.get()


def current_asset_id() -> str | None:
    return _current_asset_id.get()


def current_site_id() -> str | None:
    return _current_site_id.get()


def set_request_id(
    request_id: str | None,
) -> None:
    _current_request_id.set(
        request_id
    )


def set_asset_id(
    asset_id: str | None,
) -> None:
    _current_asset_id.set(
        asset_id
    )


def set_site_id(
    site_id: str | None,
) -> None:
    _current_site_id.set(
        site_id
    )


# =========================================================
# Trace event
# =========================================================

@dataclass(slots=True)
class SpanEvent:
    name: str

    timestamp: datetime = field(
        default_factory=utc_now
    )

    attributes: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp":
                self.timestamp.isoformat(),
            "attributes":
                dict(self.attributes),
        }


# =========================================================
# Span record
# =========================================================

@dataclass(slots=True)
class SpanRecord:
    trace_id: str
    span_id: str

    name: str

    component: TraceComponent

    kind: SpanKind = (
        SpanKind.INTERNAL
    )

    parent_span_id: str | None = None

    status: SpanStatus = (
        SpanStatus.UNSET
    )

    started_at: datetime = field(
        default_factory=utc_now
    )

    ended_at: datetime | None = None

    duration_ms: float | None = None

    attributes: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    events: list[
        SpanEvent
    ] = field(
        default_factory=list
    )

    error_type: str | None = None
    error_message: str | None = None

    request_id: str | None = None
    asset_id: str | None = None
    site_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id":
                self.trace_id,

            "span_id":
                self.span_id,

            "parent_span_id":
                self.parent_span_id,

            "name":
                self.name,

            "component":
                self.component.value,

            "kind":
                self.kind.value,

            "status":
                self.status.value,

            "started_at":
                self.started_at.isoformat(),

            "ended_at": (
                self.ended_at.isoformat()
                if self.ended_at
                else None
            ),

            "duration_ms":
                self.duration_ms,

            "attributes":
                dict(self.attributes),

            "events": [
                event.to_dict()
                for event in self.events
            ],

            "error_type":
                self.error_type,

            "error_message":
                self.error_message,

            "request_id":
                self.request_id,

            "asset_id":
                self.asset_id,

            "site_id":
                self.site_id,
        }


# =========================================================
# Trace summary
# =========================================================

@dataclass(slots=True)
class TraceSummary:
    trace_id: str

    span_count: int

    root_span: str | None

    total_duration_ms: float | None

    error_count: int

    components: list[str]

    started_at: datetime | None
    ended_at: datetime | None

    spans: list[
        SpanRecord
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id":
                self.trace_id,

            "span_count":
                self.span_count,

            "root_span":
                self.root_span,

            "total_duration_ms":
                self.total_duration_ms,

            "error_count":
                self.error_count,

            "components":
                self.components,

            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),

            "ended_at": (
                self.ended_at.isoformat()
                if self.ended_at
                else None
            ),

            "spans": [
                span.to_dict()
                for span in self.spans
            ],
        }


# =========================================================
# Exporter protocol
# =========================================================

class SpanExporter:
    def export(
        self,
        span: SpanRecord,
    ) -> None:
        raise NotImplementedError


# =========================================================
# In-memory exporter
# =========================================================

class InMemorySpanExporter(
    SpanExporter
):
    def __init__(
        self,
        *,
        max_spans: int = 50_000,
    ) -> None:
        self.max_spans = max_spans

        self._spans: list[
            SpanRecord
        ] = []

        self._lock = (
            threading.RLock()
        )

    def export(
        self,
        span: SpanRecord,
    ) -> None:
        with self._lock:
            self._spans.append(
                span
            )

            overflow = (
                len(self._spans)
                - self.max_spans
            )

            if overflow > 0:
                del self._spans[
                    :overflow
                ]

    def all_spans(
        self,
    ) -> list[SpanRecord]:
        with self._lock:
            return list(
                self._spans
            )

    def spans_for_trace(
        self,
        trace_id: str,
    ) -> list[SpanRecord]:
        with self._lock:
            return [
                span
                for span
                in self._spans
                if span.trace_id
                == trace_id
            ]

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


# =========================================================
# Logging exporter
# =========================================================

class LoggingSpanExporter(
    SpanExporter
):
    def export(
        self,
        span: SpanRecord,
    ) -> None:
        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.tracing"
            )

            payload = span.to_dict()

            if (
                span.status
                == SpanStatus.ERROR
            ):
                logger.error(
                    "Trace span completed with error.",
                    extra={
                        "trace": payload,
                    },
                )
            else:
                logger.info(
                    "Trace span completed.",
                    extra={
                        "trace": payload,
                    },
                )

        except Exception:
            # Tracing must never break
            # production inference.
            return


# =========================================================
# Composite exporter
# =========================================================

class CompositeSpanExporter(
    SpanExporter
):
    def __init__(
        self,
        exporters: Sequence[
            SpanExporter
        ],
    ) -> None:
        self.exporters = list(
            exporters
        )

    def export(
        self,
        span: SpanRecord,
    ) -> None:
        for exporter in self.exporters:
            try:
                exporter.export(
                    span
                )
            except Exception:
                continue


# =========================================================
# Trace span
# =========================================================

class TraceSpan:
    def __init__(
        self,
        tracer: "Tracer",
        *,
        name: str,
        component: TraceComponent,
        kind: SpanKind,
        attributes: Mapping[
            str,
            Any,
        ] | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> None:
        self.tracer = tracer

        self.name = name
        self.component = component
        self.kind = kind

        inherited_trace = (
            current_trace_id()
        )

        inherited_parent = (
            current_span_id()
        )

        self.trace_id = (
            trace_id
            or inherited_trace
            or generate_trace_id()
        )

        self.span_id = (
            generate_span_id()
        )

        self.parent_span_id = (
            parent_span_id
            if parent_span_id
            is not None
            else inherited_parent
        )

        self.record = SpanRecord(
            trace_id=self.trace_id,
            span_id=self.span_id,

            parent_span_id=(
                self.parent_span_id
            ),

            name=name,
            component=component,
            kind=kind,

            attributes=(
                self.tracer
                .sanitize_attributes(
                    attributes or {}
                )
            ),

            request_id=(
                current_request_id()
            ),

            asset_id=(
                current_asset_id()
            ),

            site_id=(
                current_site_id()
            ),
        )

        self._started_perf: (
            float | None
        ) = None

        self._trace_token: Any = None
        self._span_token: Any = None

        self._ended = False

    # -----------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------

    def start(self) -> "TraceSpan":
        if self._started_perf is not None:
            return self

        self.record.started_at = (
            utc_now()
        )

        self._started_perf = (
            perf_counter()
        )

        self._trace_token = (
            _current_trace_id.set(
                self.trace_id
            )
        )

        self._span_token = (
            _current_span_id.set(
                self.span_id
            )
        )

        return self

    def end(
        self,
        *,
        status: SpanStatus | None = None,
    ) -> None:
        if self._ended:
            return

        self._ended = True

        if self._started_perf is None:
            self.start()

        duration_ms = (
            perf_counter()
            - self._started_perf
        ) * 1000.0

        self.record.duration_ms = (
            duration_ms
        )

        self.record.ended_at = (
            utc_now()
        )

        if status is not None:
            self.record.status = status

        elif (
            self.record.status
            == SpanStatus.UNSET
        ):
            self.record.status = (
                SpanStatus.OK
            )

        self.tracer.export(
            self.record
        )

        if self._span_token is not None:
            _current_span_id.reset(
                self._span_token
            )

        if self._trace_token is not None:
            _current_trace_id.reset(
                self._trace_token
            )

    # -----------------------------------------------------
    # Attributes
    # -----------------------------------------------------

    def set_attribute(
        self,
        key: str,
        value: Any,
    ) -> None:
        sanitized = (
            self.tracer
            .sanitize_attributes(
                {
                    key: value,
                }
            )
        )

        if key in sanitized:
            self.record.attributes[
                key
            ] = sanitized[key]

    def set_attributes(
        self,
        attributes: Mapping[
            str,
            Any,
        ],
    ) -> None:
        self.record.attributes.update(
            self.tracer
            .sanitize_attributes(
                attributes
            )
        )

    # -----------------------------------------------------
    # Events
    # -----------------------------------------------------

    def add_event(
        self,
        name: str,
        *,
        attributes: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> None:
        self.record.events.append(
            SpanEvent(
                name=name,

                attributes=(
                    self.tracer
                    .sanitize_attributes(
                        attributes or {}
                    )
                ),
            )
        )

    # -----------------------------------------------------
    # Error
    # -----------------------------------------------------

    def record_exception(
        self,
        exception: BaseException,
    ) -> None:
        self.record.status = (
            SpanStatus.ERROR
        )

        self.record.error_type = (
            type(exception).__name__
        )

        self.record.error_message = (
            self.tracer
            .sanitize_error_message(
                str(exception)
            )
        )

        self.add_event(
            "exception",
            attributes={
                "exception.type":
                    type(
                        exception
                    ).__name__,

                "exception.message":
                    self.record
                    .error_message,
            },
        )

    # -----------------------------------------------------
    # Context managers
    # -----------------------------------------------------

    def __enter__(
        self,
    ) -> "TraceSpan":
        return self.start()

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        if exc_value is not None:
            self.record_exception(
                exc_value
            )

        self.end()

        return False

    async def __aenter__(
        self,
    ) -> "TraceSpan":
        return self.start()

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        if exc_value is not None:
            self.record_exception(
                exc_value
            )

        self.end()

        return False


# =========================================================
# Tracer
# =========================================================

class Tracer:
    SENSITIVE_KEYS = {
        "password",
        "passwd",
        "secret",
        "token",
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "private_key",
        "cookie",
        "session",
    }

    def __init__(
        self,
        *,
        service_name: str = (
            "industrial-ai"
        ),
        exporter: (
            SpanExporter
            | None
        ) = None,
        capture_sensitive_data: bool = False,
        max_attribute_length: int = 2_000,
    ) -> None:
        self.service_name = (
            service_name
        )

        self.exporter = (
            exporter
            or InMemorySpanExporter()
        )

        self.capture_sensitive_data = (
            capture_sensitive_data
        )

        self.max_attribute_length = (
            max_attribute_length
        )

    # -----------------------------------------------------
    # Span
    # -----------------------------------------------------

    def start_span(
        self,
        name: str,
        *,
        component: TraceComponent = (
            TraceComponent.OTHER
        ),
        kind: SpanKind = (
            SpanKind.INTERNAL
        ),
        attributes: Mapping[
            str,
            Any,
        ] | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> TraceSpan:
        return TraceSpan(
            self,

            name=name,
            component=component,
            kind=kind,

            attributes=attributes,

            trace_id=trace_id,

            parent_span_id=(
                parent_span_id
            ),
        )

    # -----------------------------------------------------
    # Export
    # -----------------------------------------------------

    def export(
        self,
        span: SpanRecord,
    ) -> None:
        try:
            self.exporter.export(
                span
            )
        except Exception:
            # Observability should not crash
            # application workloads.
            pass

        self._publish_metrics(
            span
        )

    # -----------------------------------------------------
    # Sanitization
    # -----------------------------------------------------

    def sanitize_attributes(
        self,
        attributes: Mapping[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        result: dict[
            str,
            Any,
        ] = {}

        for key, value in (
            attributes.items()
        ):
            normalized_key = (
                key.lower()
                .replace("-", "_")
            )

            sensitive = any(
                sensitive_key
                in normalized_key
                for sensitive_key
                in self.SENSITIVE_KEYS
            )

            if (
                sensitive
                and not self
                .capture_sensitive_data
            ):
                result[key] = (
                    "[REDACTED]"
                )
                continue

            result[key] = (
                self._sanitize_value(
                    value
                )
            )

        return result

    def _sanitize_value(
        self,
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if isinstance(
            value,
            (
                bool,
                int,
                float,
            ),
        ):
            return value

        if isinstance(
            value,
            Enum,
        ):
            return value.value

        if isinstance(
            value,
            datetime,
        ):
            return value.isoformat()

        if isinstance(
            value,
            str,
        ):
            if (
                len(value)
                > self.max_attribute_length
            ):
                return (
                    value[
                        :self.max_attribute_length
                    ]
                    + "..."
                )

            return value

        if isinstance(
            value,
            Mapping,
        ):
            return self.sanitize_attributes(
                value
            )

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                self._sanitize_value(
                    item
                )
                for item in value
            ][:100]

        return str(value)[
            :self.max_attribute_length
        ]

    def sanitize_error_message(
        self,
        message: str,
    ) -> str:
        if (
            len(message)
            > self.max_attribute_length
        ):
            return (
                message[
                    :self.max_attribute_length
                ]
                + "..."
            )

        return message

    # -----------------------------------------------------
    # Privacy helpers
    # -----------------------------------------------------

    @staticmethod
    def hash_identifier(
        value: str,
    ) -> str:
        return _safe_hash(
            value
        )

    # -----------------------------------------------------
    # Metrics bridge
    # -----------------------------------------------------

    @staticmethod
    def _publish_metrics(
        span: SpanRecord,
    ) -> None:
        try:
            from app.observability.metrics import (
                registry,
            )

            labels = {
                "component":
                    span.component.value,

                "span":
                    span.name,

                "status":
                    span.status.value,
            }

            registry.counter(
                "trace_spans_total"
            ).inc(
                labels=labels
            )

            if (
                span.duration_ms
                is not None
            ):
                registry.histogram(
                    "trace_span_duration_ms"
                ).observe(
                    span.duration_ms,
                    labels={
                        "component":
                            span.component.value,

                        "span":
                            span.name,
                    },
                )

            if (
                span.status
                == SpanStatus.ERROR
            ):
                registry.counter(
                    "trace_span_errors_total"
                ).inc(
                    labels={
                        "component":
                            span.component.value,

                        "span":
                            span.name,
                    }
                )

        except Exception:
            return


# =========================================================
# Global tracer
# =========================================================

memory_exporter = (
    InMemorySpanExporter()
)

tracer = Tracer(
    exporter=memory_exporter
)


# =========================================================
# Sync span helper
# =========================================================

@contextmanager
def trace_span(
    name: str,
    *,
    component: TraceComponent = (
        TraceComponent.OTHER
    ),
    kind: SpanKind = (
        SpanKind.INTERNAL
    ),
    attributes: Mapping[
        str,
        Any,
    ] | None = None,
) -> Iterator[TraceSpan]:

    with tracer.start_span(
        name,
        component=component,
        kind=kind,
        attributes=attributes,
    ) as span:
        yield span


# =========================================================
# Async span helper
# =========================================================

@asynccontextmanager
async def async_trace_span(
    name: str,
    *,
    component: TraceComponent = (
        TraceComponent.OTHER
    ),
    kind: SpanKind = (
        SpanKind.INTERNAL
    ),
    attributes: Mapping[
        str,
        Any,
    ] | None = None,
) -> AsyncIterator[TraceSpan]:

    async with tracer.start_span(
        name,
        component=component,
        kind=kind,
        attributes=attributes,
    ) as span:
        yield span


# =========================================================
# Decorator
# =========================================================

def traced(
    *,
    name: str | None = None,
    component: TraceComponent = (
        TraceComponent.OTHER
    ),
    kind: SpanKind = (
        SpanKind.INTERNAL
    ),
) -> Callable[
    [Callable[P, R]],
    Callable[P, R],
]:

    def decorator(
        function: Callable[P, R],
    ) -> Callable[P, R]:

        span_name = (
            name
            or function.__qualname__
        )

        if inspect.iscoroutinefunction(
            function
        ):

            @functools.wraps(
                function
            )
            async def async_wrapper(
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> Any:

                async with (
                    tracer.start_span(
                        span_name,

                        component=component,
                        kind=kind,
                    )
                ):
                    return await function(
                        *args,
                        **kwargs,
                    )

            return async_wrapper  # type: ignore

        @functools.wraps(
            function
        )
        def sync_wrapper(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Any:

            with tracer.start_span(
                span_name,

                component=component,
                kind=kind,
            ):
                return function(
                    *args,
                    **kwargs,
                )

        return sync_wrapper  # type: ignore

    return decorator


# =========================================================
# Trace propagation
# =========================================================

TRACE_ID_HEADER = "x-trace-id"
SPAN_ID_HEADER = "x-parent-span-id"


def inject_trace_context(
    headers: dict[str, str],
) -> dict[str, str]:
    result = dict(headers)

    trace_id = (
        current_trace_id()
    )

    span_id = (
        current_span_id()
    )

    if trace_id:
        result[
            TRACE_ID_HEADER
        ] = trace_id

    if span_id:
        result[
            SPAN_ID_HEADER
        ] = span_id

    return result


@dataclass(slots=True)
class ExtractedTraceContext:
    trace_id: str | None
    parent_span_id: str | None


def extract_trace_context(
    headers: Mapping[
        str,
        str,
    ],
) -> ExtractedTraceContext:

    normalized = {
        key.lower(): value
        for key, value
        in headers.items()
    }

    return ExtractedTraceContext(
        trace_id=normalized.get(
            TRACE_ID_HEADER
        ),

        parent_span_id=normalized.get(
            SPAN_ID_HEADER
        ),
    )


# =========================================================
# FastAPI middleware
# =========================================================

def install_tracing_middleware(
    app: Any,
) -> None:

    @app.middleware("http")
    async def tracing_middleware(
        request: Any,
        call_next: Any,
    ) -> Any:

        incoming = (
            extract_trace_context(
                request.headers
            )
        )

        request_id = (
            request.headers.get(
                "x-request-id"
            )
            or uuid4().hex
        )

        request_token = (
            _current_request_id.set(
                request_id
            )
        )

        span = tracer.start_span(
            (
                f"{request.method} "
                f"{request.url.path}"
            ),

            component=(
                TraceComponent.API
            ),

            kind=SpanKind.SERVER,

            trace_id=(
                incoming.trace_id
            ),

            parent_span_id=(
                incoming.parent_span_id
            ),

            attributes={
                "http.method":
                    request.method,

                "http.route":
                    request.url.path,
            },
        )

        try:
            async with span:
                response = await call_next(
                    request
                )

                span.set_attribute(
                    "http.status_code",
                    response.status_code,
                )

                response.headers[
                    "x-trace-id"
                ] = span.trace_id

                response.headers[
                    "x-request-id"
                ] = request_id

                return response

        finally:
            _current_request_id.reset(
                request_token
            )


# =========================================================
# Kafka helpers
# =========================================================

def kafka_trace_headers() -> dict[
    str,
    str,
]:
    return inject_trace_context(
        {}
    )


@contextmanager
def trace_kafka_producer(
    topic: str,
) -> Iterator[TraceSpan]:

    with tracer.start_span(
        "kafka.produce",

        component=(
            TraceComponent.KAFKA
        ),

        kind=SpanKind.PRODUCER,

        attributes={
            "messaging.system":
                "kafka",

            "messaging.destination":
                topic,
        },
    ) as span:
        yield span


@contextmanager
def trace_kafka_consumer(
    topic: str,
    *,
    headers: Mapping[
        str,
        str,
    ] | None = None,
) -> Iterator[TraceSpan]:

    context = (
        extract_trace_context(
            headers or {}
        )
    )

    with tracer.start_span(
        "kafka.consume",

        component=(
            TraceComponent.KAFKA
        ),

        kind=SpanKind.CONSUMER,

        trace_id=(
            context.trace_id
        ),

        parent_span_id=(
            context.parent_span_id
        ),

        attributes={
            "messaging.system":
                "kafka",

            "messaging.destination":
                topic,
        },
    ) as span:
        yield span


# =========================================================
# ML tracing
# =========================================================

@contextmanager
def trace_model_inference(
    *,
    model_name: str,
    model_version: str,
    model_type: str,
    asset_id: str | None = None,
) -> Iterator[TraceSpan]:

    attributes = {
        "ml.model.name":
            model_name,

        "ml.model.version":
            model_version,

        "ml.model.type":
            model_type,
    }

    if asset_id:
        attributes[
            "industrial.asset_id"
        ] = asset_id

    with tracer.start_span(
        "model.inference",

        component=(
            TraceComponent.ML
        ),

        attributes=attributes,
    ) as span:
        yield span


# =========================================================
# RAG tracing
# =========================================================

@contextmanager
def trace_retrieval(
    *,
    retriever: str,
    top_k: int,
) -> Iterator[TraceSpan]:

    with tracer.start_span(
        "rag.retrieval",

        component=(
            TraceComponent.RETRIEVAL
        ),

        attributes={
            "rag.retriever":
                retriever,

            "rag.top_k":
                top_k,
        },
    ) as span:
        yield span


@contextmanager
def trace_reranker(
    *,
    model_name: str,
    document_count: int,
) -> Iterator[TraceSpan]:

    with tracer.start_span(
        "rag.reranker",

        component=(
            TraceComponent.RERANKER
        ),

        attributes={
            "rag.reranker.model":
                model_name,

            "rag.document_count":
                document_count,
        },
    ) as span:
        yield span


# =========================================================
# LLM tracing
# =========================================================

@contextmanager
def trace_llm(
    *,
    provider: str,
    model: str,
    operation: str = "chat",
    streaming: bool = False,
) -> Iterator[TraceSpan]:

    with tracer.start_span(
        "llm.request",

        component=(
            TraceComponent.LLM
        ),

        kind=SpanKind.CLIENT,

        attributes={
            "llm.provider":
                provider,

            "llm.model":
                model,

            "llm.operation":
                operation,

            "llm.streaming":
                streaming,
        },
    ) as span:
        yield span


# =========================================================
# Agent tracing
# =========================================================

@contextmanager
def trace_agent(
    *,
    agent_name: str,
    agent_role: str | None = None,
) -> Iterator[TraceSpan]:

    attributes: dict[
        str,
        Any,
    ] = {
        "agent.name":
            agent_name,
    }

    if agent_role:
        attributes[
            "agent.role"
        ] = agent_role

    with tracer.start_span(
        "agent.run",

        component=(
            TraceComponent.AGENT
        ),

        attributes=attributes,
    ) as span:
        yield span


# =========================================================
# Tool tracing
# =========================================================

@contextmanager
def trace_tool(
    *,
    tool_name: str,
) -> Iterator[TraceSpan]:

    with tracer.start_span(
        "agent.tool",

        component=(
            TraceComponent.TOOL
        ),

        attributes={
            "tool.name":
                tool_name,
        },
    ) as span:
        yield span


# =========================================================
# Database tracing
# =========================================================

@contextmanager
def trace_database(
    *,
    operation: str,
    table: str | None = None,
) -> Iterator[TraceSpan]:

    attributes: dict[
        str,
        Any,
    ] = {
        "db.operation":
            operation,
    }

    if table:
        attributes[
            "db.table"
        ] = table

    with tracer.start_span(
        "database.operation",

        component=(
            TraceComponent.DATABASE
        ),

        kind=SpanKind.CLIENT,

        attributes=attributes,
    ) as span:
        yield span


# =========================================================
# Vector DB tracing
# =========================================================

@contextmanager
def trace_vector_search(
    *,
    provider: str,
    collection: str,
    top_k: int,
) -> Iterator[TraceSpan]:

    with tracer.start_span(
        "vector.search",

        component=(
            TraceComponent.VECTOR_DB
        ),

        kind=SpanKind.CLIENT,

        attributes={
            "vector.provider":
                provider,

            "vector.collection":
                collection,

            "vector.top_k":
                top_k,
        },
    ) as span:
        yield span


# =========================================================
# Safety tracing
# =========================================================

@contextmanager
def trace_safety_engine(
    *,
    engine_name: str,
    asset_id: str | None = None,
) -> Iterator[TraceSpan]:

    attributes: dict[
        str,
        Any,
    ] = {
        "safety.engine":
            engine_name,
    }

    if asset_id:
        attributes[
            "industrial.asset_id"
        ] = asset_id

    with tracer.start_span(
        "safety.evaluate",

        component=(
            TraceComponent.SAFETY
        ),

        attributes=attributes,
    ) as span:
        yield span


# =========================================================
# Trace queries
# =========================================================

def get_trace(
    trace_id: str,
) -> TraceSummary | None:

    spans = (
        memory_exporter
        .spans_for_trace(
            trace_id
        )
    )

    if not spans:
        return None

    spans = sorted(
        spans,
        key=lambda span:
            span.started_at,
    )

    roots = [
        span
        for span in spans
        if span.parent_span_id
        is None
    ]

    root = (
        roots[0]
        if roots
        else spans[0]
    )

    started_at = min(
        span.started_at
        for span in spans
    )

    ended_values = [
        span.ended_at
        for span in spans
        if span.ended_at
        is not None
    ]

    ended_at = (
        max(ended_values)
        if ended_values
        else None
    )

    total_duration = (
        (
            ended_at
            - started_at
        ).total_seconds()
        * 1000.0
        if ended_at
        else None
    )

    components = sorted(
        {
            span.component.value
            for span in spans
        }
    )

    errors = sum(
        span.status
        == SpanStatus.ERROR
        for span in spans
    )

    return TraceSummary(
        trace_id=trace_id,

        span_count=len(
            spans
        ),

        root_span=(
            root.name
            if root
            else None
        ),

        total_duration_ms=(
            total_duration
        ),

        error_count=errors,

        components=components,

        started_at=started_at,
        ended_at=ended_at,

        spans=spans,
    )


# =========================================================
# OpenTelemetry bridge
# =========================================================

class OpenTelemetrySpanExporter(
    SpanExporter
):
    """
    Optional bridge.

    This module does not require OpenTelemetry.
    Install it only when production exporting is needed:

        pip install \
            opentelemetry-api \
            opentelemetry-sdk \
            opentelemetry-exporter-otlp
    """

    def __init__(
        self,
        tracer_name: str = (
            "industrial-ai"
        ),
    ) -> None:
        try:
            from opentelemetry import trace

        except ImportError as exc:
            raise RuntimeError(
                "OpenTelemetry is not installed."
            ) from exc

        self._otel_trace = trace

        self._tracer = (
            trace.get_tracer(
                tracer_name
            )
        )

    def export(
        self,
        span: SpanRecord,
    ) -> None:
        # This adapter creates an OTEL span from the
        # internal representation. For full cross-service
        # context propagation, use native OTEL
        # instrumentation in production.
        with self._tracer.start_as_current_span(
            span.name
        ) as otel_span:

            for key, value in (
                span.attributes.items()
            ):
                try:
                    otel_span.set_attribute(
                        key,
                        value,
                    )
                except Exception:
                    continue

            otel_span.set_attribute(
                "industrial.trace_id",
                span.trace_id,
            )

            otel_span.set_attribute(
                "industrial.span_id",
                span.span_id,
            )

            otel_span.set_attribute(
                "industrial.component",
                span.component.value,
            )

            if span.duration_ms is not None:
                otel_span.set_attribute(
                    "industrial.duration_ms",
                    span.duration_ms,
                )

            for event in span.events:
                try:
                    otel_span.add_event(
                        event.name,
                        attributes=(
                            event.attributes
                        ),
                    )
                except Exception:
                    continue

            if (
                span.status
                == SpanStatus.ERROR
            ):
                try:
                    from opentelemetry.trace import (
                        Status,
                        StatusCode,
                    )

                    otel_span.set_status(
                        Status(
                            StatusCode.ERROR,
                            span.error_message,
                        )
                    )

                except Exception:
                    pass


# =========================================================
# FastAPI trace lookup endpoint
# =========================================================

def install_trace_endpoint(
    app: Any,
    *,
    path: str = "/api/v1/traces/{trace_id}",
) -> None:

    try:
        from fastapi import (
            HTTPException,
        )

    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is required."
        ) from exc

    @app.get(path)
    async def trace_endpoint(
        trace_id: str,
    ) -> dict[str, Any]:

        trace = get_trace(
            trace_id
        )

        if trace is None:
            raise HTTPException(
                status_code=404,
                detail="Trace not found.",
            )

        return trace.to_dict()


# =========================================================
# Example higher-level pipeline tracing
# =========================================================

@asynccontextmanager
async def trace_prediction_pipeline(
    *,
    asset_id: str,
    model_name: str,
) -> AsyncIterator[TraceSpan]:

    asset_token = (
        _current_asset_id.set(
            asset_id
        )
    )

    try:
        async with tracer.start_span(
            "predictive_maintenance.pipeline",

            component=(
                TraceComponent.ML
            ),

            attributes={
                "industrial.asset_id":
                    asset_id,

                "ml.model.name":
                    model_name,
            },
        ) as span:

            yield span

    finally:
        _current_asset_id.reset(
            asset_token
        )


# =========================================================
# Public API
# =========================================================

__all__ = [
    # enums
    "SpanKind",
    "SpanStatus",
    "TraceComponent",

    # records
    "SpanEvent",
    "SpanRecord",
    "TraceSummary",
    "ExtractedTraceContext",

    # exporters
    "SpanExporter",
    "InMemorySpanExporter",
    "LoggingSpanExporter",
    "CompositeSpanExporter",
    "OpenTelemetrySpanExporter",

    # tracing
    "TraceSpan",
    "Tracer",
    "tracer",
    "memory_exporter",

    # IDs/context
    "generate_trace_id",
    "generate_span_id",
    "current_trace_id",
    "current_span_id",
    "current_request_id",
    "current_asset_id",
    "current_site_id",
    "set_request_id",
    "set_asset_id",
    "set_site_id",

    # helpers
    "trace_span",
    "async_trace_span",
    "traced",

    # propagation
    "TRACE_ID_HEADER",
    "SPAN_ID_HEADER",
    "inject_trace_context",
    "extract_trace_context",

    # integrations
    "install_tracing_middleware",
    "install_trace_endpoint",

    "kafka_trace_headers",
    "trace_kafka_producer",
    "trace_kafka_consumer",

    "trace_model_inference",

    "trace_retrieval",
    "trace_reranker",

    "trace_llm",
    "trace_agent",
    "trace_tool",

    "trace_database",
    "trace_vector_search",

    "trace_safety_engine",

    "trace_prediction_pipeline",

    # query
    "get_trace",
]