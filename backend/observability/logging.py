# app/observability/logging.py

from __future__ import annotations

import contextvars
import json
import logging
import logging.config
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID, uuid4


# =========================================================
# Context variables
# =========================================================

_request_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("request_id", default=None)
)

_trace_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("trace_id", default=None)
)

_span_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("span_id", default=None)
)

_user_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("user_id", default=None)
)

_asset_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("asset_id", default=None)
)

_site_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("site_id", default=None)
)

_conversation_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("conversation_id", default=None)
)


# =========================================================
# Enums
# =========================================================

class LogEventType(str, Enum):
    APPLICATION = "application"

    HTTP_REQUEST = "http_request"
    HTTP_RESPONSE = "http_response"

    DATABASE = "database"
    REDIS = "redis"
    KAFKA = "kafka"
    MQTT = "mqtt"

    TELEMETRY = "telemetry"
    SENSOR = "sensor"

    MODEL_INFERENCE = "model_inference"
    MODEL_ERROR = "model_error"
    MODEL_DRIFT = "model_drift"
    DATA_DRIFT = "data_drift"

    ANOMALY = "anomaly"
    FAILURE_PREDICTION = "failure_prediction"
    RUL_PREDICTION = "rul_prediction"

    RAG = "rag"
    RETRIEVAL = "retrieval"
    LLM = "llm"
    AGENT = "agent"
    TOOL = "tool"

    SAFETY = "safety"
    SECURITY = "security"
    AUDIT = "audit"

    ALERT = "alert"
    INCIDENT = "incident"
    EMERGENCY = "emergency"

    MAINTENANCE = "maintenance"
    WORK_ORDER = "work_order"

    COST = "cost"

    SYSTEM = "system"


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"

    json_logs: bool = True

    console_enabled: bool = True
    file_enabled: bool = True

    log_directory: str = "logs"

    application_file: str = "application.log"
    error_file: str = "error.log"
    audit_file: str = "audit.log"
    security_file: str = "security.log"
    llm_file: str = "llm.log"

    max_file_bytes: int = 25 * 1024 * 1024
    backup_count: int = 10

    include_process: bool = True
    include_thread: bool = False

    service_name: str = (
        "industrial-predictive-maintenance-ai"
    )

    environment: str = "development"

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        return cls(
            level=os.getenv(
                "LOG_LEVEL",
                "INFO",
            ).upper(),

            json_logs=(
                os.getenv(
                    "LOG_FORMAT",
                    "json",
                ).lower()
                == "json"
            ),

            console_enabled=(
                os.getenv(
                    "LOG_CONSOLE",
                    "true",
                ).lower()
                == "true"
            ),

            file_enabled=(
                os.getenv(
                    "LOG_FILE_ENABLED",
                    "true",
                ).lower()
                == "true"
            ),

            log_directory=os.getenv(
                "LOG_DIR",
                "logs",
            ),

            service_name=os.getenv(
                "SERVICE_NAME",
                "industrial-predictive-maintenance-ai",
            ),

            environment=os.getenv(
                "ENVIRONMENT",
                "development",
            ),
        )


# =========================================================
# Sensitive-data protection
# =========================================================

SENSITIVE_KEYS = {
    "password",
    "passwd",
    "pwd",

    "secret",
    "secret_key",

    "api_key",
    "apikey",

    "authorization",
    "access_token",
    "refresh_token",
    "id_token",
    "token",

    "cookie",
    "set_cookie",
    "session",
    "session_id",

    "private_key",

    "credit_card",
    "card_number",
    "cvv",

    "ssn",

    "email",
    "phone",

    "prompt",
    "raw_prompt",
    "response",
    "raw_response",

    "document_text",
    "context_text",

    "tool_arguments",
    "tool_output",
}


BEARER_PATTERN = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*"
)

API_KEY_PATTERN = re.compile(
    r"(?i)"
    r"(api[_-]?key\s*[=:]\s*)"
    r"([^\s,;]+)"
)

PASSWORD_PATTERN = re.compile(
    r"(?i)"
    r"(password\s*[=:]\s*)"
    r"([^\s,;]+)"
)


def is_sensitive_key(
    key: str,
) -> bool:

    normalized = (
        key.lower()
        .replace("-", "_")
        .strip()
    )

    return (
        normalized in SENSITIVE_KEYS
        or normalized.endswith(
            "_password"
        )
        or normalized.endswith(
            "_secret"
        )
        or normalized.endswith(
            "_token"
        )
        or normalized.endswith(
            "_api_key"
        )
    )


def sanitize_string(
    value: str,
) -> str:

    value = BEARER_PATTERN.sub(
        "Bearer [REDACTED]",
        value,
    )

    value = API_KEY_PATTERN.sub(
        r"\1[REDACTED]",
        value,
    )

    value = PASSWORD_PATTERN.sub(
        r"\1[REDACTED]",
        value,
    )

    return value


def sanitize_data(
    value: Any,
    *,
    max_depth: int = 8,
    _depth: int = 0,
) -> Any:

    if _depth > max_depth:
        return "[MAX_DEPTH]"

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        if isinstance(value, str):
            return sanitize_string(value)

        return value

    if isinstance(
        value,
        (datetime, date),
    ):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return sanitize_data(
            asdict(value),
            max_depth=max_depth,
            _depth=_depth + 1,
        )

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}

        for key, item in value.items():
            key_string = str(key)

            if is_sensitive_key(
                key_string
            ):
                sanitized[
                    key_string
                ] = "[REDACTED]"

            else:
                sanitized[
                    key_string
                ] = sanitize_data(
                    item,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )

        return sanitized

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            sanitize_data(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if hasattr(
        value,
        "model_dump",
    ):
        try:
            return sanitize_data(
                value.model_dump(),
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        except Exception:
            pass

    return sanitize_string(
        str(value)
    )


# =========================================================
# Context helpers
# =========================================================

def set_request_id(
    value: str | None = None,
) -> str:

    value = value or str(uuid4())

    _request_id.set(value)

    return value


def set_trace_id(
    value: str | None = None,
) -> str:

    value = value or str(uuid4())

    _trace_id.set(value)

    return value


def set_span_id(
    value: str | None = None,
) -> str:

    value = value or str(uuid4())

    _span_id.set(value)

    return value


def set_user_context(
    user_id: str | None,
) -> None:
    _user_id.set(user_id)


def set_asset_context(
    asset_id: str | None,
) -> None:
    _asset_id.set(asset_id)


def set_site_context(
    site_id: str | None,
) -> None:
    _site_id.set(site_id)


def set_conversation_context(
    conversation_id: str | None,
) -> None:
    _conversation_id.set(
        conversation_id
    )


def get_logging_context() -> dict[str, Any]:

    return {
        "request_id":
            _request_id.get(),

        "trace_id":
            _trace_id.get(),

        "span_id":
            _span_id.get(),

        "user_id":
            _user_id.get(),

        "asset_id":
            _asset_id.get(),

        "site_id":
            _site_id.get(),

        "conversation_id":
            _conversation_id.get(),
    }


def clear_logging_context() -> None:

    _request_id.set(None)
    _trace_id.set(None)
    _span_id.set(None)

    _user_id.set(None)
    _asset_id.set(None)
    _site_id.set(None)

    _conversation_id.set(None)


# =========================================================
# Context filter
# =========================================================

class ContextFilter(logging.Filter):

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:

        context = get_logging_context()

        for key, value in context.items():
            if not hasattr(
                record,
                key,
            ):
                setattr(
                    record,
                    key,
                    value,
                )

        return True


# =========================================================
# JSON formatter
# =========================================================

class JSONFormatter(logging.Formatter):

    def __init__(
        self,
        *,
        service_name: str,
        environment: str,
        include_process: bool = True,
        include_thread: bool = False,
    ) -> None:

        super().__init__()

        self.service_name = service_name
        self.environment = environment

        self.include_process = (
            include_process
        )

        self.include_thread = (
            include_thread
        )

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        timestamp = datetime.fromtimestamp(
            record.created,
            tz=timezone.utc,
        ).isoformat()

        payload: dict[str, Any] = {
            "timestamp": timestamp,

            "level":
                record.levelname,

            "logger":
                record.name,

            "message":
                sanitize_string(
                    record.getMessage()
                ),

            "service":
                self.service_name,

            "environment":
                self.environment,

            "module":
                record.module,

            "function":
                record.funcName,

            "line":
                record.lineno,
        }

        # ---------------------------------------------
        # Distributed context
        # ---------------------------------------------

        for name in (
            "request_id",
            "trace_id",
            "span_id",
            "user_id",
            "asset_id",
            "site_id",
            "conversation_id",
        ):
            value = getattr(
                record,
                name,
                None,
            )

            if value is not None:
                payload[name] = value

        # ---------------------------------------------
        # Event type
        # ---------------------------------------------

        event_type = getattr(
            record,
            "event_type",
            None,
        )

        if event_type is not None:
            if isinstance(
                event_type,
                Enum,
            ):
                event_type = (
                    event_type.value
                )

            payload[
                "event_type"
            ] = event_type

        # ---------------------------------------------
        # Structured fields
        # ---------------------------------------------

        structured = getattr(
            record,
            "structured_data",
            None,
        )

        if structured is not None:
            payload["data"] = (
                sanitize_data(
                    structured
                )
            )

        # ---------------------------------------------
        # Process
        # ---------------------------------------------

        if self.include_process:
            payload[
                "process_id"
            ] = record.process

        if self.include_thread:
            payload[
                "thread_id"
            ] = record.thread

        # ---------------------------------------------
        # Exception
        # ---------------------------------------------

        if record.exc_info:
            exc_type = (
                record.exc_info[0]
            )

            exc_value = (
                record.exc_info[1]
            )

            payload["exception"] = {
                "type": (
                    exc_type.__name__
                    if exc_type
                    else None
                ),

                "message": sanitize_string(
                    str(exc_value)
                ),

                "stack_trace":
                    sanitize_string(
                        self.formatException(
                            record.exc_info
                        )
                    ),
            }

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )


# =========================================================
# Human-readable formatter
# =========================================================

class TextFormatter(logging.Formatter):

    def __init__(self) -> None:

        super().__init__(
            fmt=(
                "%(asctime)s "
                "%(levelname)s "
                "[%(name)s] "
                "%(message)s "
                "request=%(request_id)s "
                "trace=%(trace_id)s "
                "asset=%(asset_id)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )


# =========================================================
# Logger adapters
# =========================================================

class StructuredLogger:

    def __init__(
        self,
        logger_instance: logging.Logger,
    ) -> None:

        self.logger = (
            logger_instance
        )

    def log(
        self,
        level: int,
        message: str,
        *,
        event_type: (
            LogEventType | str
        ) = LogEventType.APPLICATION,
        data: Mapping[
            str,
            Any,
        ] | None = None,
        exc_info: bool = False,
    ) -> None:

        self.logger.log(
            level,
            message,
            exc_info=exc_info,
            extra={
                "event_type": (
                    event_type.value
                    if isinstance(
                        event_type,
                        Enum,
                    )
                    else event_type
                ),

                "structured_data": (
                    dict(data)
                    if data
                    else {}
                ),
            },
        )

    def debug(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:

        self.log(
            logging.DEBUG,
            message,
            **kwargs,
        )

    def info(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:

        self.log(
            logging.INFO,
            message,
            **kwargs,
        )

    def warning(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:

        self.log(
            logging.WARNING,
            message,
            **kwargs,
        )

    def error(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:

        self.log(
            logging.ERROR,
            message,
            **kwargs,
        )

    def critical(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:

        self.log(
            logging.CRITICAL,
            message,
            **kwargs,
        )

    def exception(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:

        kwargs["exc_info"] = True

        self.error(
            message,
            **kwargs,
        )


# =========================================================
# Specialized loggers
# =========================================================

def get_logger(
    name: str,
) -> StructuredLogger:

    return StructuredLogger(
        logging.getLogger(name)
    )


def get_audit_logger() -> StructuredLogger:

    return StructuredLogger(
        logging.getLogger(
            "industrial.audit"
        )
    )


def get_security_logger() -> StructuredLogger:

    return StructuredLogger(
        logging.getLogger(
            "industrial.security"
        )
    )


def get_llm_logger() -> StructuredLogger:

    return StructuredLogger(
        logging.getLogger(
            "industrial.llm"
        )
    )


# =========================================================
# Event helpers
# =========================================================

def log_sensor_event(
    *,
    sensor_id: str,
    sensor_type: str,
    value: float,
    unit: str,
    asset_id: str | None = None,
    status: str | None = None,
) -> None:

    get_logger(
        "industrial.sensor"
    ).info(
        "Sensor measurement received.",
        event_type=LogEventType.SENSOR,
        data={
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "asset_id": asset_id,
            "value": value,
            "unit": unit,
            "status": status,
        },
    )


def log_model_inference(
    *,
    model_name: str,
    model_version: str | None,
    prediction_type: str,
    latency_ms: float,
    asset_id: str | None = None,
    confidence: float | None = None,
    prediction_id: str | None = None,
) -> None:

    get_logger(
        "industrial.model"
    ).info(
        "Model inference completed.",
        event_type=(
            LogEventType.MODEL_INFERENCE
        ),
        data={
            "model_name":
                model_name,

            "model_version":
                model_version,

            "prediction_type":
                prediction_type,

            "prediction_id":
                prediction_id,

            "asset_id":
                asset_id,

            "confidence":
                confidence,

            "latency_ms":
                latency_ms,
        },
    )


def log_anomaly(
    *,
    asset_id: str,
    anomaly_type: str,
    anomaly_score: float,
    severity: str,
    prediction_id: str | None = None,
) -> None:

    get_logger(
        "industrial.anomaly"
    ).warning(
        "Industrial anomaly detected.",
        event_type=LogEventType.ANOMALY,
        data={
            "asset_id": asset_id,
            "anomaly_type": anomaly_type,
            "anomaly_score":
                anomaly_score,

            "severity": severity,
            "prediction_id":
                prediction_id,
        },
    )


def log_failure_prediction(
    *,
    asset_id: str,
    failure_mode: str,
    probability: float,
    model_name: str,
    prediction_id: str | None = None,
) -> None:

    get_logger(
        "industrial.failure"
    ).warning(
        "Failure risk predicted.",
        event_type=(
            LogEventType
            .FAILURE_PREDICTION
        ),
        data={
            "asset_id": asset_id,
            "failure_mode":
                failure_mode,

            "failure_probability":
                probability,

            "model_name":
                model_name,

            "prediction_id":
                prediction_id,
        },
    )


def log_rul_prediction(
    *,
    asset_id: str,
    predicted_rul: float,
    unit: str,
    model_name: str,
    confidence: float | None = None,
) -> None:

    get_logger(
        "industrial.rul"
    ).info(
        "RUL prediction generated.",
        event_type=(
            LogEventType.RUL_PREDICTION
        ),
        data={
            "asset_id": asset_id,
            "predicted_rul":
                predicted_rul,

            "unit": unit,
            "model_name":
                model_name,

            "confidence":
                confidence,
        },
    )


# =========================================================
# Drift logging
# =========================================================

def log_data_drift(
    *,
    model_name: str,
    model_version: str | None,
    drifted_feature_ratio: float,
    weighted_drift_score: float,
    affected_features: list[str],
    severity: str,
) -> None:

    level = (
        logging.ERROR
        if severity
        in {
            "critical",
            "high",
        }
        else logging.WARNING
    )

    StructuredLogger(
        logging.getLogger(
            "industrial.drift"
        )
    ).log(
        level,
        "Data drift evaluation completed.",
        event_type=(
            LogEventType.DATA_DRIFT
        ),
        data={
            "model_name":
                model_name,

            "model_version":
                model_version,

            "drifted_feature_ratio":
                drifted_feature_ratio,

            "weighted_drift_score":
                weighted_drift_score,

            "affected_features":
                affected_features,

            "severity":
                severity,
        },
    )


# =========================================================
# RAG logging
# =========================================================

def log_rag_request(
    *,
    query_id: str,
    retrieved_chunks: int,
    context_chunks_used: int,
    retrieval_latency_ms: float,
    reranker_latency_ms: float | None = None,
    groundedness: float | None = None,
    citation_count: int | None = None,
) -> None:
    """
    Deliberately avoids storing the raw user query.
    """

    get_llm_logger().info(
        "RAG request completed.",
        event_type=LogEventType.RAG,
        data={
            "query_id":
                query_id,

            "retrieved_chunks":
                retrieved_chunks,

            "context_chunks_used":
                context_chunks_used,

            "retrieval_latency_ms":
                retrieval_latency_ms,

            "reranker_latency_ms":
                reranker_latency_ms,

            "groundedness":
                groundedness,

            "citation_count":
                citation_count,
        },
    )


# =========================================================
# LLM logging
# =========================================================

def log_llm_request(
    *,
    request_id: str,
    provider: str,
    model: str,
    operation: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    status: str,
    cost: float | None = None,
    ttft_ms: float | None = None,
    groundedness: float | None = None,
    fallback_used: bool = False,
) -> None:
    """
    Raw prompts and generated answers are intentionally
    excluded from the production log.
    """

    get_llm_logger().info(
        "LLM request completed.",
        event_type=LogEventType.LLM,
        data={
            "llm_request_id":
                request_id,

            "provider":
                provider,

            "model":
                model,

            "operation":
                operation,

            "status":
                status,

            "latency_ms":
                latency_ms,

            "ttft_ms":
                ttft_ms,

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,

            "total_tokens":
                input_tokens
                + output_tokens,

            "cost":
                cost,

            "groundedness":
                groundedness,

            "fallback_used":
                fallback_used,
        },
    )


# =========================================================
# Agent / tool logging
# =========================================================

def log_agent_step(
    *,
    agent_name: str,
    step_name: str,
    status: str,
    latency_ms: float,
    tool_calls: int = 0,
    error_type: str | None = None,
) -> None:

    get_llm_logger().info(
        "Agent step completed.",
        event_type=LogEventType.AGENT,
        data={
            "agent_name":
                agent_name,

            "step_name":
                step_name,

            "status":
                status,

            "latency_ms":
                latency_ms,

            "tool_calls":
                tool_calls,

            "error_type":
                error_type,
        },
    )


def log_tool_call(
    *,
    tool_name: str,
    status: str,
    latency_ms: float,
    tool_call_id: str | None = None,
    error_type: str | None = None,
) -> None:

    get_llm_logger().info(
        "Agent tool call completed.",
        event_type=LogEventType.TOOL,
        data={
            "tool_call_id":
                tool_call_id,

            "tool_name":
                tool_name,

            "status":
                status,

            "latency_ms":
                latency_ms,

            "error_type":
                error_type,
        },
    )


# =========================================================
# Safety / incident logging
# =========================================================

def log_safety_incident(
    *,
    incident_id: str,
    incident_type: str,
    severity: str,
    site_id: str | None = None,
    asset_id: str | None = None,
    confidence: float | None = None,
) -> None:

    log = get_logger(
        "industrial.safety"
    )

    data = {
        "incident_id":
            incident_id,

        "incident_type":
            incident_type,

        "severity":
            severity,

        "site_id":
            site_id,

        "asset_id":
            asset_id,

        "confidence":
            confidence,
    }

    if severity.lower() in {
        "critical",
        "emergency",
    }:
        log.critical(
            "Critical industrial safety incident.",
            event_type=(
                LogEventType.SAFETY
            ),
            data=data,
        )

    else:
        log.warning(
            "Industrial safety incident.",
            event_type=(
                LogEventType.SAFETY
            ),
            data=data,
        )


# =========================================================
# Security / audit
# =========================================================

def log_security_event(
    *,
    event: str,
    severity: str,
    user_id: str | None = None,
    source_ip: str | None = None,
    metadata: Mapping[
        str,
        Any,
    ] | None = None,
) -> None:

    get_security_logger().warning(
        "Security event detected.",
        event_type=(
            LogEventType.SECURITY
        ),
        data={
            "security_event":
                event,

            "severity":
                severity,

            "user_id":
                user_id,

            "source_ip":
                source_ip,

            **dict(
                metadata or {}
            ),
        },
    )


def log_audit_event(
    *,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    user_id: str | None = None,
    outcome: str = "success",
    metadata: Mapping[
        str,
        Any,
    ] | None = None,
) -> None:

    get_audit_logger().info(
        "Audit event.",
        event_type=LogEventType.AUDIT,
        data={
            "action":
                action,

            "resource_type":
                resource_type,

            "resource_id":
                resource_id,

            "user_id":
                user_id,

            "outcome":
                outcome,

            **dict(
                metadata or {}
            ),
        },
    )


# =========================================================
# Alert logging
# =========================================================

def log_alert(
    *,
    alert_id: str,
    title: str,
    severity: str,
    category: str,
    status: str,
    asset_id: str | None = None,
    occurrence_count: int = 1,
) -> None:

    severity_lower = (
        severity.lower()
    )

    if severity_lower in {
        "critical",
        "emergency",
    }:
        level = logging.CRITICAL

    elif severity_lower == "high":
        level = logging.ERROR

    elif severity_lower == "warning":
        level = logging.WARNING

    else:
        level = logging.INFO

    StructuredLogger(
        logging.getLogger(
            "industrial.alert"
        )
    ).log(
        level,
        "Industrial alert lifecycle event.",
        event_type=LogEventType.ALERT,
        data={
            "alert_id":
                alert_id,

            "title":
                title,

            "severity":
                severity,

            "category":
                category,

            "status":
                status,

            "asset_id":
                asset_id,

            "occurrence_count":
                occurrence_count,
        },
    )


# =========================================================
# Error logging
# =========================================================

def log_exception(
    logger_name: str,
    message: str,
    exception: Exception,
    *,
    event_type: (
        LogEventType | str
    ) = LogEventType.APPLICATION,
    metadata: Mapping[
        str,
        Any,
    ] | None = None,
) -> None:

    get_logger(
        logger_name
    ).error(
        message,
        event_type=event_type,
        data={
            "exception_type":
                type(
                    exception
                ).__name__,

            "exception_message":
                str(exception),

            **dict(
                metadata or {}
            ),
        },
        exc_info=True,
    )


# =========================================================
# Filters for dedicated files
# =========================================================

class LoggerPrefixFilter(logging.Filter):

    def __init__(
        self,
        prefix: str,
    ) -> None:

        super().__init__()

        self.prefix = prefix

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:

        return record.name.startswith(
            self.prefix
        )


class MaximumLevelFilter(logging.Filter):

    def __init__(
        self,
        maximum: int,
    ) -> None:

        super().__init__()

        self.maximum = maximum

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:

        return (
            record.levelno
            <= self.maximum
        )


# =========================================================
# Setup
# =========================================================

def _create_rotating_handler(
    path: Path,
    *,
    level: int,
    formatter: logging.Formatter,
    config: LoggingConfig,
) -> RotatingFileHandler:

    handler = RotatingFileHandler(
        filename=path,
        maxBytes=config.max_file_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )

    handler.setLevel(level)
    handler.setFormatter(formatter)

    handler.addFilter(
        ContextFilter()
    )

    return handler


def setup_logging(
    config: LoggingConfig | None = None,
) -> None:

    config = (
        config
        or LoggingConfig.from_env()
    )

    level = getattr(
        logging,
        config.level.upper(),
        logging.INFO,
    )

    if config.json_logs:
        formatter: logging.Formatter = (
            JSONFormatter(
                service_name=(
                    config.service_name
                ),

                environment=(
                    config.environment
                ),

                include_process=(
                    config.include_process
                ),

                include_thread=(
                    config.include_thread
                ),
            )
        )

    else:
        formatter = TextFormatter()

    root = logging.getLogger()

    root.setLevel(level)

    # Prevent duplicate handlers when FastAPI tests
    # initialize the application multiple times.
    for handler in list(
        root.handlers
    ):
        root.removeHandler(
            handler
        )

        try:
            handler.close()
        except Exception:
            pass

    # -----------------------------------------------------
    # Console
    # -----------------------------------------------------

    if config.console_enabled:
        console = logging.StreamHandler(
            sys.stdout
        )

        console.setLevel(level)
        console.setFormatter(formatter)

        console.addFilter(
            ContextFilter()
        )

        root.addHandler(
            console
        )

    # -----------------------------------------------------
    # Files
    # -----------------------------------------------------

    if config.file_enabled:

        log_directory = Path(
            config.log_directory
        )

        log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        application_handler = (
            _create_rotating_handler(
                log_directory
                / config.application_file,

                level=level,
                formatter=formatter,
                config=config,
            )
        )

        root.addHandler(
            application_handler
        )

        error_handler = (
            _create_rotating_handler(
                log_directory
                / config.error_file,

                level=logging.ERROR,
                formatter=formatter,
                config=config,
            )
        )

        root.addHandler(
            error_handler
        )

        audit_handler = (
            _create_rotating_handler(
                log_directory
                / config.audit_file,

                level=logging.INFO,
                formatter=formatter,
                config=config,
            )
        )

        audit_handler.addFilter(
            LoggerPrefixFilter(
                "industrial.audit"
            )
        )

        root.addHandler(
            audit_handler
        )

        security_handler = (
            _create_rotating_handler(
                log_directory
                / config.security_file,

                level=logging.INFO,
                formatter=formatter,
                config=config,
            )
        )

        security_handler.addFilter(
            LoggerPrefixFilter(
                "industrial.security"
            )
        )

        root.addHandler(
            security_handler
        )

        llm_handler = (
            _create_rotating_handler(
                log_directory
                / config.llm_file,

                level=logging.INFO,
                formatter=formatter,
                config=config,
            )
        )

        llm_handler.addFilter(
            LoggerPrefixFilter(
                "industrial.llm"
            )
        )

        root.addHandler(
            llm_handler
        )

    # -----------------------------------------------------
    # Third-party logger noise
    # -----------------------------------------------------

    logging.getLogger(
        "httpx"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "httpcore"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "urllib3"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "asyncio"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "aiokafka"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "redis"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "sqlalchemy.engine"
    ).setLevel(
        logging.WARNING
    )

    logging.getLogger(
        "multipart"
    ).setLevel(logging.WARNING)

    logging.captureWarnings(
        True
    )


# =========================================================
# FastAPI middleware helper
# =========================================================

async def request_logging_middleware(
    request: Any,
    call_next: Any,
) -> Any:
    """
    Can be registered directly:

        app.middleware("http")(
            request_logging_middleware
        )
    """

    from time import perf_counter

    request_id = (
        request.headers.get(
            "X-Request-ID"
        )
        or str(uuid4())
    )

    trace_id = (
        request.headers.get(
            "X-Trace-ID"
        )
        or str(uuid4())
    )

    set_request_id(
        request_id
    )

    set_trace_id(
        trace_id
    )

    set_span_id()

    started = perf_counter()

    http_logger = get_logger(
        "industrial.http"
    )

    http_logger.info(
        "HTTP request started.",
        event_type=(
            LogEventType.HTTP_REQUEST
        ),
        data={
            "method":
                request.method,

            "path":
                request.url.path,

            "client_ip": (
                request.client.host
                if request.client
                else None
            ),
        },
    )

    try:
        response = await call_next(
            request
        )

        latency_ms = (
            perf_counter()
            - started
        ) * 1000.0

        http_logger.info(
            "HTTP request completed.",
            event_type=(
                LogEventType.HTTP_RESPONSE
            ),
            data={
                "method":
                    request.method,

                "path":
                    request.url.path,

                "status_code":
                    response.status_code,

                "latency_ms":
                    round(
                        latency_ms,
                        4,
                    ),
            },
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        response.headers[
            "X-Trace-ID"
        ] = trace_id

        return response

    except Exception as exc:

        latency_ms = (
            perf_counter()
            - started
        ) * 1000.0

        http_logger.error(
            "HTTP request failed.",
            event_type=(
                LogEventType.HTTP_RESPONSE
            ),
            data={
                "method":
                    request.method,

                "path":
                    request.url.path,

                "latency_ms":
                    round(
                        latency_ms,
                        4,
                    ),

                "error_type":
                    type(
                        exc
                    ).__name__,
            },
            exc_info=True,
        )

        raise

    finally:
        clear_logging_context()


# =========================================================
# Startup convenience
# =========================================================

_default_config = (
    LoggingConfig.from_env()
)


def configure_default_logging() -> None:

    setup_logging(
        _default_config
    )


__all__ = [
    "LogEventType",
    "LoggingConfig",
    "ContextFilter",
    "JSONFormatter",
    "TextFormatter",
    "StructuredLogger",
    "sanitize_string",
    "sanitize_data",
    "is_sensitive_key",
    "set_request_id",
    "set_trace_id",
    "set_span_id",
    "set_user_context",
    "set_asset_context",
    "set_site_context",
    "set_conversation_context",
    "get_logging_context",
    "clear_logging_context",
    "get_logger",
    "get_audit_logger",
    "get_security_logger",
    "get_llm_logger",
    "log_sensor_event",
    "log_model_inference",
    "log_anomaly",
    "log_failure_prediction",
    "log_rul_prediction",
    "log_data_drift",
    "log_rag_request",
    "log_llm_request",
    "log_agent_step",
    "log_tool_call",
    "log_safety_incident",
    "log_security_event",
    "log_audit_event",
    "log_alert",
    "log_exception",
    "setup_logging",
    "configure_default_logging",
    "request_logging_middleware",
]