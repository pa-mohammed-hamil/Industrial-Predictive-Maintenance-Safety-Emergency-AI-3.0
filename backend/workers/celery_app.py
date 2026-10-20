"""
Celery application configuration for the industrial AI platform.

Responsibilities:
- Create and configure the Celery application
- Configure Redis as broker/result backend
- Configure serialization and task execution
- Configure retries and task time limits
- Configure worker reliability settings
- Provide a shared Celery application instance
"""

from __future__ import annotations

import logging
import os
from typing import Any

from celery import Celery
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    worker_ready,
    worker_shutdown,
)

logger = logging.getLogger(__name__)


def _get_env(
    name: str,
    default: str,
) -> str:
    """Read an environment variable with a fallback."""

    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip()


CELERY_BROKER_URL = _get_env(
    "CELERY_BROKER_URL",
    "redis://localhost:6379/1",
)

CELERY_RESULT_BACKEND = _get_env(
    "CELERY_RESULT_BACKEND",
    "redis://localhost:6379/2",
)

CELERY_TIMEZONE = _get_env(
    "CELERY_TIMEZONE",
    "UTC",
)

CELERY_TASK_DEFAULT_QUEUE = _get_env(
    "CELERY_TASK_DEFAULT_QUEUE",
    "industrial.default",
)

CELERY_TASK_DEFAULT_EXCHANGE = _get_env(
    "CELERY_TASK_DEFAULT_EXCHANGE",
    "industrial",
)

CELERY_TASK_DEFAULT_ROUTING_KEY = _get_env(
    "CELERY_TASK_DEFAULT_ROUTING_KEY",
    "industrial.default",
)


celery_app = Celery(
    "industrial_ai",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)


celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=[
        "json",
    ],

    # Timezone
    timezone=CELERY_TIMEZONE,
    enable_utc=True,

    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Result handling
    result_expires=3600,

    # Task execution
    task_track_started=True,
    task_send_sent_event=True,

    # Default routing
    task_default_queue=CELERY_TASK_DEFAULT_QUEUE,
    task_default_exchange=CELERY_TASK_DEFAULT_EXCHANGE,
    task_default_exchange_type="direct",
    task_default_routing_key=CELERY_TASK_DEFAULT_ROUTING_KEY,

    # Retry / broker behavior
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,

    # Worker lifecycle
    worker_disable_rate_limits=False,

    # Beat scheduler
    beat_scheduler="celery.beat:PersistentScheduler",

    # Task discovery
    imports=(
        "app.workers.telemetry_tasks",
        "app.workers.prediction_tasks",
        "app.workers.image_tasks",
        "app.workers.cost_tasks",
        "app.workers.report_tasks",
        "app.workers.notification_tasks",
    ),
)


# ---------------------------------------------------------------------------
# Task routes
# ---------------------------------------------------------------------------

celery_app.conf.task_routes = {
    "app.workers.telemetry_tasks.*": {
        "queue": "industrial.telemetry",
    },
    "app.workers.prediction_tasks.*": {
        "queue": "industrial.prediction",
    },
    "app.workers.image_tasks.*": {
        "queue": "industrial.vision",
    },
    "app.workers.cost_tasks.*": {
        "queue": "industrial.cost",
    },
    "app.workers.report_tasks.*": {
        "queue": "industrial.reports",
    },
    "app.workers.notification_tasks.*": {
        "queue": "industrial.notifications",
    },
}


# ---------------------------------------------------------------------------
# Task annotations
# ---------------------------------------------------------------------------

celery_app.conf.task_annotations = {
    "*": {
        "rate_limit": "100/s",
    },
}


# ---------------------------------------------------------------------------
# Optional periodic tasks
# ---------------------------------------------------------------------------

celery_app.conf.beat_schedule = {
    "health-check-workers": {
        "task": "app.workers.telemetry_tasks.worker_health_check",
        "schedule": 60.0,
    },
}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

@worker_ready.connect
def handle_worker_ready(
    sender: Any = None,
    **kwargs: Any,
) -> None:
    """Log when a Celery worker becomes ready."""

    logger.info(
        "Industrial AI Celery worker ready",
        extra={
            "worker": str(sender),
        },
    )


@worker_shutdown.connect
def handle_worker_shutdown(
    sender: Any = None,
    **kwargs: Any,
) -> None:
    """Log worker shutdown."""

    logger.info(
        "Industrial AI Celery worker shutting down",
        extra={
            "worker": str(sender),
        },
    )


@task_prerun.connect
def handle_task_prerun(
    task_id: str,
    task: Any,
    **kwargs: Any,
) -> None:
    """Log task execution start."""

    logger.debug(
        "Celery task started",
        extra={
            "task_id": task_id,
            "task_name": task.name,
        },
    )


@task_postrun.connect
def handle_task_postrun(
    task_id: str,
    task: Any,
    retval: Any = None,
    state: str | None = None,
    **kwargs: Any,
) -> None:
    """Log task completion."""

    logger.debug(
        "Celery task completed",
        extra={
            "task_id": task_id,
            "task_name": task.name,
            "state": state,
        },
    )


@task_failure.connect
def handle_task_failure(
    task_id: str,
    exception: Exception,
    task: Any,
    **kwargs: Any,
) -> None:
    """Log failed Celery tasks."""

    logger.error(
        "Celery task failed",
        extra={
            "task_id": task_id,
            "task_name": task.name,
            "error": str(exception),
        },
        exc_info=True,
    )


# ---------------------------------------------------------------------------
# Health helper
# ---------------------------------------------------------------------------

def celery_health_check() -> dict[str, Any]:
    """
    Return basic Celery configuration health information.

    This does not perform a broker ping, making it safe to call
    from lightweight application health endpoints.
    """

    return {
        "status": "configured",
        "broker": CELERY_BROKER_URL.split("@")[-1],
        "backend": CELERY_RESULT_BACKEND.split("@")[-1],
        "timezone": CELERY_TIMEZONE,
        "default_queue": CELERY_TASK_DEFAULT_QUEUE,
    }


# ---------------------------------------------------------------------------
# Compatibility alias
# ---------------------------------------------------------------------------

app = celery_app


__all__ = [
    "celery_app",
    "app",
    "celery_health_check",
]