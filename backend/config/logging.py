# app/config/logging.py

from __future__ import annotations

import contextvars
import json
import logging
import logging.config
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


# =========================================================
# Context variables
# =========================================================

request_id_ctx: contextvars.ContextVar[str] = (
    contextvars.ContextVar(
        "request_id",
        default="-",
    )
)

asset_id_ctx: contextvars.ContextVar[str] = (
    contextvars.ContextVar(
        "asset_id",
        default="-",
    )
)

site_id_ctx: contextvars.ContextVar[str] = (
    contextvars.ContextVar(
        "site_id",
        default="-",
    )
)

user_id_ctx: contextvars.ContextVar[str] = (
    contextvars.ContextVar(
        "user_id",
        default="-",
    )
)


# =========================================================
# Configuration
# =========================================================

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "json",
).lower()

LOG_DIR = Path(
    os.getenv(
        "LOG_DIR",
        "logs",
    )
)

LOG_FILE = os.getenv(
    "LOG_FILE",
    "industrial-ai.log",
)

ERROR_LOG_FILE = os.getenv(
    "ERROR_LOG_FILE",
    "error.log",
)

AUDIT_LOG_FILE = os.getenv(
    "AUDIT_LOG_FILE",
    "audit.log",
)

SECURITY_LOG_FILE = os.getenv(
    "SECURITY_LOG_FILE",
    "security.log",
)

MAX_LOG_BYTES = int(
    os.getenv(
        "MAX_LOG_BYTES",
        str(20 * 1024 * 1024),
    )
)

LOG_BACKUP_COUNT = int(
    os.getenv(
        "LOG_BACKUP_COUNT",
        "10",
    )
)


# =========================================================
# Sensitive-data masking
# =========================================================

SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "api_key",
    "apikey",
    "private_key",
    "client_secret",
    "cookie",
    "session",
}


def mask_sensitive(
    value: Any,
) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}

        for key, item in value.items():
            normalized = str(key).lower()

            if normalized in SENSITIVE_KEYS:
                result[key] = "***REDACTED***"
            else:
                result[key] = mask_sensitive(item)

        return result

    if isinstance(value, list):
        return [
            mask_sensitive(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            mask_sensitive(item)
            for item in value
        )

    return value


_BEARER_PATTERN = re.compile(
    r"Bearer\s+[A-Za-z0-9\-._~+/]+=*",
    flags=re.IGNORECASE,
)


def sanitize_message(
    message: str,
) -> str:
    return _BEARER_PATTERN.sub(
        "Bearer ***REDACTED***",
        message,
    )


# =========================================================
# Context filter
# =========================================================

class ContextFilter(logging.Filter):

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        record.request_id = request_id_ctx.get()
        record.asset_id = asset_id_ctx.get()
        record.site_id = site_id_ctx.get()
        record.user_id = user_id_ctx.get()

        return True


# =========================================================
# JSON formatter
# =========================================================

class JSONFormatter(logging.Formatter):

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
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_message(
                record.getMessage()
            ),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
            "request_id": getattr(
                record,
                "request_id",
                "-",
            ),
            "site_id": getattr(
                record,
                "site_id",
                "-",
            ),
            "asset_id": getattr(
                record,
                "asset_id",
                "-",
            ),
            "user_id": getattr(
                record,
                "user_id",
                "-",
            ),
        }

        # Optional structured metadata.
        extra_data = getattr(
            record,
            "data",
            None,
        )

        if extra_data is not None:
            payload["data"] = mask_sensitive(
                extra_data
            )

        event_type = getattr(
            record,
            "event_type",
            None,
        )

        if event_type:
            payload["event_type"] = event_type

        if record.exc_info:
            payload["exception"] = (
                self.formatException(
                    record.exc_info
                )
            )

        if record.stack_info:
            payload["stack"] = (
                self.formatStack(
                    record.stack_info
                )
            )

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


# =========================================================
# Human-readable formatter
# =========================================================

class TextFormatter(logging.Formatter):

    def __init__(self) -> None:
        super().__init__(
            fmt=(
                "%(asctime)s | "
                "%(levelname)-8s | "
                "%(name)s | "
                "request=%(request_id)s | "
                "site=%(site_id)s | "
                "asset=%(asset_id)s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        formatted = super().format(
            record
        )

        return sanitize_message(
            formatted
        )


# =========================================================
# Logging setup
# =========================================================

def setup_logging(
    *,
    level: str | None = None,
    log_format: str | None = None,
) -> None:

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_level = (
        level or LOG_LEVEL
    ).upper()

    selected_format = (
        log_format or LOG_FORMAT
    ).lower()

    formatter_name = (
        "json"
        if selected_format == "json"
        else "text"
    )

    config = {
        "version": 1,
        "disable_existing_loggers": False,

        "filters": {
            "context": {
                "()": ContextFilter,
            },
        },

        "formatters": {
            "json": {
                "()": JSONFormatter,
            },
            "text": {
                "()": TextFormatter,
            },
        },

        "handlers": {
            "console": {
                "class": (
                    "logging.StreamHandler"
                ),
                "level": selected_level,
                "formatter": formatter_name,
                "filters": ["context"],
                "stream": "ext://sys.stdout",
            },

            "application_file": {
                "class": (
                    "logging.handlers."
                    "RotatingFileHandler"
                ),
                "level": selected_level,
                "formatter": "json",
                "filters": ["context"],
                "filename": str(
                    LOG_DIR / LOG_FILE
                ),
                "maxBytes": MAX_LOG_BYTES,
                "backupCount": LOG_BACKUP_COUNT,
                "encoding": "utf-8",
            },

            "error_file": {
                "class": (
                    "logging.handlers."
                    "RotatingFileHandler"
                ),
                "level": "ERROR",
                "formatter": "json",
                "filters": ["context"],
                "filename": str(
                    LOG_DIR / ERROR_LOG_FILE
                ),
                "maxBytes": MAX_LOG_BYTES,
                "backupCount": LOG_BACKUP_COUNT,
                "encoding": "utf-8",
            },

            "audit_file": {
                "class": (
                    "logging.handlers."
                    "RotatingFileHandler"
                ),
                "level": "INFO",
                "formatter": "json",
                "filters": ["context"],
                "filename": str(
                    LOG_DIR / AUDIT_LOG_FILE
                ),
                "maxBytes": MAX_LOG_BYTES,
                "backupCount": LOG_BACKUP_COUNT,
                "encoding": "utf-8",
            },

            "security_file": {
                "class": (
                    "logging.handlers."
                    "RotatingFileHandler"
                ),
                "level": "WARNING",
                "formatter": "json",
                "filters": ["context"],
                "filename": str(
                    LOG_DIR
                    / SECURITY_LOG_FILE
                ),
                "maxBytes": MAX_LOG_BYTES,
                "backupCount": LOG_BACKUP_COUNT,
                "encoding": "utf-8",
            },
        },

        "loggers": {
            "industrial_ai": {
                "handlers": [
                    "console",
                    "application_file",
                    "error_file",
                ],
                "level": selected_level,
                "propagate": False,
            },

            "audit": {
                "handlers": [
                    "audit_file",
                ],
                "level": "INFO",
                "propagate": False,
            },

            "security": {
                "handlers": [
                    "console",
                    "security_file",
                ],
                "level": "WARNING",
                "propagate": False,
            },

            "uvicorn": {
                "handlers": [
                    "console",
                    "application_file",
                ],
                "level": selected_level,
                "propagate": False,
            },

            "uvicorn.error": {
                "handlers": [
                    "console",
                    "application_file",
                    "error_file",
                ],
                "level": selected_level,
                "propagate": False,
            },

            "uvicorn.access": {
                "handlers": [
                    "console",
                    "application_file",
                ],
                "level": selected_level,
                "propagate": False,
            },
        },

        "root": {
            "handlers": [
                "console",
                "application_file",
                "error_file",
            ],
            "level": selected_level,
        },
    }

    logging.config.dictConfig(
        config
    )

    logging.getLogger(
        "industrial_ai"
    ).info(
        "Logging initialized",
        extra={
            "event_type": (
                "logging_initialized"
            ),
            "data": {
                "level": selected_level,
                "format": selected_format,
                "directory": str(LOG_DIR),
            },
        },
    )


# =========================================================
# Context helpers
# =========================================================

def set_request_id(
    request_id: str | None = None,
) -> str:
    value = (
        request_id
        or str(uuid4())
    )

    request_id_ctx.set(value)

    return value


def set_asset_context(
    asset_id: str | None,
) -> None:
    asset_id_ctx.set(
        asset_id or "-"
    )


def set_site_context(
    site_id: str | None,
) -> None:
    site_id_ctx.set(
        site_id or "-"
    )


def set_user_context(
    user_id: str | None,
) -> None:
    user_id_ctx.set(
        user_id or "-"
    )


def clear_logging_context() -> None:
    request_id_ctx.set("-")
    asset_id_ctx.set("-")
    site_id_ctx.set("-")
    user_id_ctx.set("-")


# =========================================================
# Logger helpers
# =========================================================

def get_logger(
    name: str,
) -> logging.Logger:
    return logging.getLogger(
        f"industrial_ai.{name}"
    )


audit_logger = logging.getLogger(
    "audit"
)

security_logger = logging.getLogger(
    "security"
)


# =========================================================
# Structured event helpers
# =========================================================

def log_event(
    logger: logging.Logger,
    *,
    event_type: str,
    message: str,
    level: int = logging.INFO,
    data: dict[str, Any] | None = None,
) -> None:

    logger.log(
        level,
        message,
        extra={
            "event_type": event_type,
            "data": mask_sensitive(
                data or {}
            ),
        },
    )


def log_audit_event(
    *,
    action: str,
    resource: str,
    resource_id: str | None = None,
    result: str = "success",
    data: dict[str, Any] | None = None,
) -> None:

    log_event(
        audit_logger,
        event_type="audit",
        message=(
            f"{action} {resource}"
        ),
        data={
            "action": action,
            "resource": resource,
            "resource_id": resource_id,
            "result": result,
            **(data or {}),
        },
    )


def log_security_event(
    *,
    event_type: str,
    message: str,
    severity: str = "warning",
    data: dict[str, Any] | None = None,
) -> None:

    severity_map = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    log_event(
        security_logger,
        event_type=event_type,
        message=message,
        level=severity_map.get(
            severity.lower(),
            logging.WARNING,
        ),
        data=data,
    )


# =========================================================
# Industrial-specific logging helpers
# =========================================================

def log_sensor_event(
    *,
    sensor_type: str,
    sensor_id: str,
    asset_id: str,
    value: Any,
    status: str,
) -> None:

    logger = get_logger(
        "sensors"
    )

    log_event(
        logger,
        event_type="sensor_reading",
        message=(
            f"{sensor_type} sensor "
            f"{sensor_id}: {status}"
        ),
        data={
            "sensor_type": sensor_type,
            "sensor_id": sensor_id,
            "asset_id": asset_id,
            "value": value,
            "status": status,
        },
    )


def log_prediction(
    *,
    asset_id: str,
    model: str,
    prediction: Any,
    confidence: float | None = None,
) -> None:

    logger = get_logger(
        "ml.prediction"
    )

    log_event(
        logger,
        event_type="prediction",
        message=(
            f"Prediction generated "
            f"for asset {asset_id}"
        ),
        data={
            "asset_id": asset_id,
            "model": model,
            "prediction": prediction,
            "confidence": confidence,
        },
    )


def log_safety_incident(
    *,
    incident_type: str,
    asset_id: str | None,
    site_id: str,
    severity: str,
    details: dict[str, Any] | None = None,
) -> None:

    logger = get_logger(
        "safety"
    )

    level = (
        logging.CRITICAL
        if severity.lower()
        in {"critical", "emergency"}
        else logging.WARNING
    )

    log_event(
        logger,
        event_type="safety_incident",
        message=(
            f"Safety incident: "
            f"{incident_type}"
        ),
        level=level,
        data={
            "incident_type": incident_type,
            "asset_id": asset_id,
            "site_id": site_id,
            "severity": severity,
            "details": details or {},
        },
    )