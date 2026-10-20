"""
Asynchronous telemetry processing tasks.

Responsibilities:
- Ingest telemetry batches.
- Validate and normalize telemetry.
- Persist telemetry.
- Trigger downstream telemetry processing.
- Aggregate telemetry windows.
- Detect data-quality problems.
- Publish normalized telemetry events.

ML inference belongs in app/ml/.
Business logic belongs in app/services/.
Streaming/event transport belongs in app/streaming/.
Safety decisions belong in app/safety/.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from celery import Task

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class TelemetryTask(Task):
    """Base Celery task for telemetry processing."""

    autoretry_for = (
        TimeoutError,
        ConnectionError,
    )

    retry_backoff = True
    retry_backoff_max = 300
    retry_jitter = True
    max_retries = 3

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.exception(
            "Telemetry task failed",
            extra={
                "task_id": task_id,
                "error": str(exc),
            },
        )


# ---------------------------------------------------------------------------
# Single telemetry point
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=TelemetryTask,
    name="telemetry.process",
    queue="telemetry",
)
def process_telemetry(
    self: TelemetryTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Process one telemetry event.

    Expected payload may contain:
        machine_id
        sensor_id
        timestamp
        temperature
        vibration
        pressure
        current
        voltage
        energy
        rpm
        operating_hours
    """

    if not isinstance(payload, dict):
        raise ValueError("Telemetry payload must be a dictionary")

    try:
        normalized = _normalize_telemetry(payload)

        _validate_telemetry(normalized)

        from app.services.telemetry_service import TelemetryService

        service = TelemetryService()

        result = service.process_telemetry(
            telemetry=normalized,
        )

        _publish_telemetry_event(
            telemetry=normalized,
            result=result,
        )

        return {
            "status": "processed",
            "task_id": self.request.id,
            "machine_id": normalized.get("machine_id"),
            "sensor_id": normalized.get("sensor_id"),
            "result": result,
            "processed_at": utc_now().isoformat(),
        }

    except ValueError:
        logger.exception(
            "Invalid telemetry payload",
            extra={
                "task_id": self.request.id,
            },
        )
        raise

    except Exception as exc:
        logger.exception(
            "Telemetry processing failed",
            extra={
                "task_id": self.request.id,
                "machine_id": payload.get("machine_id"),
            },
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Batch telemetry
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=TelemetryTask,
    name="telemetry.process_batch",
    queue="telemetry",
)
def process_telemetry_batch(
    self: TelemetryTask,
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Process a batch of telemetry records.

    Designed for:
    - MQTT batches
    - OPC-UA polling
    - PLC gateways
    - uploaded telemetry
    - Kafka consumers
    """

    if not isinstance(payloads, list):
        raise ValueError("Telemetry batch must be a list")

    processed = 0
    rejected = 0
    errors: list[dict[str, Any]] = []

    from app.services.telemetry_service import TelemetryService

    service = TelemetryService()

    normalized_records: list[dict[str, Any]] = []

    for index, payload in enumerate(payloads):
        try:
            if not isinstance(payload, dict):
                raise ValueError("Telemetry record must be a dictionary")

            normalized = _normalize_telemetry(payload)
            _validate_telemetry(normalized)

            normalized_records.append(normalized)

        except (TypeError, ValueError) as exc:
            rejected += 1

            errors.append(
                {
                    "index": index,
                    "error": str(exc),
                }
            )

    if normalized_records:
        try:
            result = service.process_batch(
                telemetry_records=normalized_records,
            )

            processed = len(normalized_records)

            _publish_batch_event(
                records=normalized_records,
                result=result,
            )

        except Exception as exc:
            logger.exception(
                "Telemetry batch processing failed",
                extra={
                    "task_id": self.request.id,
                    "records": len(normalized_records),
                },
            )

            raise self.retry(exc=exc)

    else:
        result = None

    return {
        "status": "completed",
        "task_id": self.request.id,
        "received": len(payloads),
        "processed": processed,
        "rejected": rejected,
        "errors": errors,
        "result": result,
        "processed_at": utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=TelemetryTask,
    name="telemetry.persist",
    queue="telemetry_persistence",
)
def persist_telemetry(
    self: TelemetryTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist normalized telemetry independently from downstream processing."""

    if not isinstance(payload, dict):
        raise ValueError("Telemetry payload must be a dictionary")

    normalized = _normalize_telemetry(payload)
    _validate_telemetry(normalized)

    try:
        from app.repositories.telemetry_repository import TelemetryRepository

        repository = TelemetryRepository()

        record = repository.create(
            telemetry=normalized,
        )

        return {
            "status": "persisted",
            "task_id": self.request.id,
            "machine_id": normalized.get("machine_id"),
            "sensor_id": normalized.get("sensor_id"),
            "record": record,
            "persisted_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Telemetry persistence failed",
            extra={
                "machine_id": normalized.get("machine_id"),
                "sensor_id": normalized.get("sensor_id"),
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=TelemetryTask,
    name="telemetry.aggregate",
    queue="telemetry_aggregation",
)
def aggregate_telemetry(
    self: TelemetryTask,
    machine_id: str,
    start_time: str,
    end_time: str,
    interval_seconds: int = 60,
) -> dict[str, Any]:
    """
    Aggregate telemetry into time windows.

    Typical aggregations:
    - mean
    - min
    - max
    - standard deviation
    - percentiles
    - rate of change
    - sample count
    """

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    try:
        from app.services.telemetry_service import TelemetryService

        service = TelemetryService()

        result = service.aggregate(
            machine_id=machine_id,
            start_time=start_time,
            end_time=end_time,
            interval_seconds=interval_seconds,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "machine_id": machine_id,
            "start_time": start_time,
            "end_time": end_time,
            "interval_seconds": interval_seconds,
            "result": result,
            "aggregated_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Telemetry aggregation failed",
            extra={
                "machine_id": machine_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=TelemetryTask,
    name="telemetry.quality_check",
    queue="telemetry_quality",
)
def check_telemetry_quality(
    self: TelemetryTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Check telemetry quality.

    Detects:
    - missing values
    - invalid numeric values
    - impossible readings
    - stale timestamps
    - duplicate events
    - sensor disconnect indicators
    """

    if not isinstance(payload, dict):
        raise ValueError("Telemetry payload must be a dictionary")

    try:
        from app.services.telemetry_service import TelemetryService

        service = TelemetryService()

        result = service.check_quality(
            telemetry=payload,
        )

        quality_status = _extract_quality_status(result)

        if quality_status in {"invalid", "critical", "failed"}:
            _publish_quality_event(
                telemetry=payload,
                result=result,
            )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "machine_id": payload.get("machine_id"),
            "sensor_id": payload.get("sensor_id"),
            "quality_status": quality_status,
            "result": result,
            "checked_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Telemetry quality check failed",
            extra={
                "machine_id": payload.get("machine_id"),
                "sensor_id": payload.get("sensor_id"),
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Machine telemetry pipeline
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=TelemetryTask,
    name="telemetry.machine_pipeline",
    queue="telemetry",
)
def run_machine_telemetry_pipeline(
    self: TelemetryTask,
    machine_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """
    Run the machine telemetry processing pipeline.

    Pipeline:

        fetch
          ↓
        quality check
          ↓
        normalization
          ↓
        persistence
          ↓
        aggregation
          ↓
        downstream prediction/anomaly triggers
    """

    try:
        from app.services.telemetry_service import TelemetryService

        service = TelemetryService()

        result = service.run_machine_pipeline(
            machine_id=machine_id,
            start_time=start_time,
            end_time=end_time,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "machine_id": machine_id,
            "result": result,
            "completed_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Machine telemetry pipeline failed",
            extra={
                "machine_id": machine_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Trigger downstream intelligence
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=TelemetryTask,
    name="telemetry.trigger_intelligence",
    queue="telemetry",
)
def trigger_telemetry_intelligence(
    self: TelemetryTask,
    machine_id: str,
) -> dict[str, Any]:
    """
    Trigger downstream ML processing.

    This task does NOT perform ML inference itself.

    It schedules:
    - failure prediction
    - RUL prediction
    - anomaly detection
    - degradation analysis
    """

    try:
        from app.workers.prediction_tasks import (
            predict_degradation,
            predict_failure,
            predict_rul,
        )

        from app.services.anomaly_service import AnomalyService

        jobs: list[dict[str, Any]] = []

        failure_task = predict_failure.delay(machine_id)
        rul_task = predict_rul.delay(machine_id)
        degradation_task = predict_degradation.delay(machine_id)

        jobs.extend(
            [
                {
                    "type": "failure_prediction",
                    "task_id": failure_task.id,
                },
                {
                    "type": "rul_prediction",
                    "task_id": rul_task.id,
                },
                {
                    "type": "degradation_prediction",
                    "task_id": degradation_task.id,
                },
            ]
        )

        anomaly_service = AnomalyService()

        anomaly_task = anomaly_service.enqueue_detection(
            machine_id=machine_id,
        )

        jobs.append(
            {
                "type": "anomaly_detection",
                "task_id": getattr(anomaly_task, "id", None),
            }
        )

        return {
            "status": "scheduled",
            "task_id": self.request.id,
            "machine_id": machine_id,
            "jobs": jobs,
            "scheduled_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Failed to trigger telemetry intelligence",
            extra={
                "machine_id": machine_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=TelemetryTask,
    name="telemetry.backfill",
    queue="telemetry_backfill",
)
def backfill_telemetry(
    self: TelemetryTask,
    machine_id: str,
    start_time: str,
    end_time: str,
) -> dict[str, Any]:
    """
    Reprocess historical telemetry.

    Useful for:
    - corrected sensor data
    - model retraining
    - incident investigation
    - feature backfills
    """

    try:
        from app.services.telemetry_service import TelemetryService

        service = TelemetryService()

        result = service.backfill(
            machine_id=machine_id,
            start_time=start_time,
            end_time=end_time,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "machine_id": machine_id,
            "start_time": start_time,
            "end_time": end_time,
            "result": result,
            "completed_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Telemetry backfill failed",
            extra={
                "machine_id": machine_id,
                "start_time": start_time,
                "end_time": end_time,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_telemetry(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Normalize common telemetry fields."""

    result = dict(payload)

    if "machine_id" in result:
        result["machine_id"] = str(result["machine_id"])

    if "sensor_id" in result:
        result["sensor_id"] = str(result["sensor_id"])

    if "timestamp" not in result:
        result["timestamp"] = utc_now().isoformat()

    numeric_fields = (
        "temperature",
        "vibration",
        "pressure",
        "current",
        "voltage",
        "energy",
        "rpm",
        "speed",
        "flow_rate",
        "humidity",
        "power",
        "load",
        "operating_hours",
    )

    for field in numeric_fields:
        if field not in result:
            continue

        value = result[field]

        if value is None:
            continue

        try:
            result[field] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Telemetry field '{field}' must be numeric"
            )

    return result


def _validate_telemetry(
    telemetry: dict[str, Any],
) -> None:
    """Perform basic structural validation."""

    machine_id = telemetry.get("machine_id")

    if not machine_id:
        raise ValueError("machine_id is required")

    timestamp = telemetry.get("timestamp")

    if timestamp is None:
        raise ValueError("timestamp is required")

    # Basic physical sanity checks.
    if telemetry.get("humidity") is not None:
        humidity = float(telemetry["humidity"])

        if humidity < 0 or humidity > 100:
            raise ValueError("humidity must be between 0 and 100")

    if telemetry.get("rpm") is not None:
        rpm = float(telemetry["rpm"])

        if rpm < 0:
            raise ValueError("rpm cannot be negative")

    if telemetry.get("voltage") is not None:
        voltage = float(telemetry["voltage"])

        if voltage < 0:
            raise ValueError("voltage cannot be negative")

    if telemetry.get("current") is not None:
        current = float(telemetry["current"])

        if current < 0:
            raise ValueError("current cannot be negative")


def _publish_telemetry_event(
    *,
    telemetry: dict[str, Any],
    result: Any,
) -> None:
    """Publish a normalized telemetry event to Kafka."""

    try:
        from app.streaming.kafka_producer import publish_event

        publish_event(
            topic="industrial.telemetry",
            event={
                "event_type": "telemetry.processed",
                "timestamp": utc_now().isoformat(),
                "machine_id": telemetry.get("machine_id"),
                "sensor_id": telemetry.get("sensor_id"),
                "telemetry": telemetry,
                "result": result,
            },
        )

    except Exception:
        # Kafka publication should not silently destroy already-processed
        # telemetry. The streaming layer should have its own retry/DLQ policy.
        logger.exception(
            "Failed to publish telemetry event",
            extra={
                "machine_id": telemetry.get("machine_id"),
                "sensor_id": telemetry.get("sensor_id"),
            },
        )


def _publish_batch_event(
    *,
    records: list[dict[str, Any]],
    result: Any,
) -> None:
    """Publish a batch-level telemetry event."""

    if not records:
        return

    try:
        from app.streaming.kafka_producer import publish_event

        machine_ids = sorted(
            {
                record["machine_id"]
                for record in records
                if record.get("machine_id")
            }
        )

        publish_event(
            topic="industrial.telemetry",
            event={
                "event_type": "telemetry.batch_processed",
                "timestamp": utc_now().isoformat(),
                "machine_ids": machine_ids,
                "record_count": len(records),
                "result": result,
            },
        )

    except Exception:
        logger.exception(
            "Failed to publish telemetry batch event"
        )


def _publish_quality_event(
    *,
    telemetry: dict[str, Any],
    result: Any,
) -> None:
    """Publish telemetry data-quality event."""

    try:
        from app.streaming.kafka_producer import publish_event

        publish_event(
            topic="industrial.telemetry.quality",
            event={
                "event_type": "telemetry.quality_issue",
                "timestamp": utc_now().isoformat(),
                "machine_id": telemetry.get("machine_id"),
                "sensor_id": telemetry.get("sensor_id"),
                "result": result,
            },
        )

    except Exception:
        logger.exception(
            "Failed to publish telemetry quality event"
        )


def _extract_quality_status(
    result: Any,
) -> str:
    """Safely extract a normalized quality status."""

    if not isinstance(result, dict):
        return "unknown"

    value = result.get("status")

    if value is None:
        value = result.get("quality_status")

    if value is None:
        return "unknown"

    return str(value).strip().lower()