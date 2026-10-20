"""
Asynchronous predictive-maintenance tasks.

Responsibilities:
- Run failure-probability inference.
- Run RUL prediction.
- Run degradation estimation.
- Run maintenance forecasting.
- Persist prediction results through services.
- Publish important prediction events.
- Trigger downstream maintenance workflows.

ML implementations remain inside app/ml/.
Business logic remains inside app/services/.
Kafka/event handling remains inside app/streaming/.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from celery import Task

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class PredictionTask(Task):
    """Base task for predictive-maintenance workers."""

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
            "Prediction task failed",
            extra={
                "task_id": task_id,
                "error": str(exc),
            },
        )


# ---------------------------------------------------------------------------
# Failure prediction
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=PredictionTask,
    name="prediction.failure",
    queue="ml_inference",
)
def predict_failure(
    self: PredictionTask,
    machine_id: str,
    *,
    horizon_hours: int = 24,
) -> dict[str, Any]:
    """
    Predict probability of machine/component failure.

    The task delegates feature preparation and inference to the
    predictive-maintenance ML layer.
    """

    logger.info(
        "Starting failure prediction",
        extra={
            "task_id": self.request.id,
            "machine_id": machine_id,
            "horizon_hours": horizon_hours,
        },
    )

    try:
        from app.ml.predictive_maintenance.failure_prediction import (
            FailurePredictionModel,
        )

        model = FailurePredictionModel()

        prediction = model.predict(
            machine_id=machine_id,
            horizon_hours=horizon_hours,
        )

        result = _normalize_prediction(
            prediction,
            task_id=self.request.id,
            machine_id=machine_id,
            prediction_type="failure_probability",
        )

        _persist_prediction(
            machine_id=machine_id,
            prediction_type="failure_probability",
            result=result,
        )

        if _is_critical_failure_prediction(prediction):
            _publish_prediction_event(
                event_type="failure.predicted",
                machine_id=machine_id,
                result=result,
            )

        return result

    except Exception as exc:
        logger.exception(
            "Failure prediction failed",
            extra={"machine_id": machine_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# RUL prediction
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=PredictionTask,
    name="prediction.rul",
    queue="ml_inference",
)
def predict_rul(
    self: PredictionTask,
    machine_id: str,
) -> dict[str, Any]:
    """
    Predict remaining useful life of a machine/component.
    """

    logger.info(
        "Starting RUL prediction",
        extra={
            "task_id": self.request.id,
            "machine_id": machine_id,
        },
    )

    try:
        from app.ml.predictive_maintenance.rul_prediction import (
            RULPredictionModel,
        )

        model = RULPredictionModel()

        prediction = model.predict(
            machine_id=machine_id,
        )

        result = _normalize_prediction(
            prediction,
            task_id=self.request.id,
            machine_id=machine_id,
            prediction_type="rul",
        )

        _persist_prediction(
            machine_id=machine_id,
            prediction_type="rul",
            result=result,
        )

        if _is_critical_rul_prediction(prediction):
            _publish_prediction_event(
                event_type="rul.critical",
                machine_id=machine_id,
                result=result,
            )

        return result

    except Exception as exc:
        logger.exception(
            "RUL prediction failed",
            extra={"machine_id": machine_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Degradation prediction
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=PredictionTask,
    name="prediction.degradation",
    queue="ml_inference",
)
def predict_degradation(
    self: PredictionTask,
    machine_id: str,
) -> dict[str, Any]:
    """Estimate machine/component degradation."""

    try:
        from app.ml.predictive_maintenance.degradation_model import (
            DegradationModel,
        )

        model = DegradationModel()

        prediction = model.predict(
            machine_id=machine_id,
        )

        result = _normalize_prediction(
            prediction,
            task_id=self.request.id,
            machine_id=machine_id,
            prediction_type="degradation",
        )

        _persist_prediction(
            machine_id=machine_id,
            prediction_type="degradation",
            result=result,
        )

        return result

    except Exception as exc:
        logger.exception(
            "Degradation prediction failed",
            extra={"machine_id": machine_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Maintenance forecasting
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=PredictionTask,
    name="prediction.maintenance_forecast",
    queue="ml_inference",
)
def forecast_maintenance(
    self: PredictionTask,
    machine_id: str,
    *,
    horizon_days: int = 30,
) -> dict[str, Any]:
    """
    Forecast likely maintenance requirements.
    """

    try:
        from app.ml.predictive_maintenance.maintenance_forecasting import (
            MaintenanceForecastModel,
        )

        model = MaintenanceForecastModel()

        prediction = model.predict(
            machine_id=machine_id,
            horizon_days=horizon_days,
        )

        result = _normalize_prediction(
            prediction,
            task_id=self.request.id,
            machine_id=machine_id,
            prediction_type="maintenance_forecast",
        )

        _persist_prediction(
            machine_id=machine_id,
            prediction_type="maintenance_forecast",
            result=result,
        )

        return result

    except Exception as exc:
        logger.exception(
            "Maintenance forecast failed",
            extra={"machine_id": machine_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Complete predictive-maintenance pipeline
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=PredictionTask,
    name="prediction.run_pipeline",
    queue="ml_orchestration",
)
def run_prediction_pipeline(
    self: PredictionTask,
    machine_id: str,
) -> dict[str, Any]:
    """
    Run the main predictive-maintenance pipeline.

    This provides a single orchestration entry point for:
    - failure prediction
    - RUL
    - degradation
    - maintenance forecasting
    """

    results: dict[str, Any] = {}

    tasks = {
        "failure": predict_failure,
        "rul": predict_rul,
        "degradation": predict_degradation,
        "maintenance_forecast": forecast_maintenance,
    }

    for name, task in tasks.items():
        try:
            result = task.apply(
                kwargs={
                    "machine_id": machine_id,
                }
            ).get()

            results[name] = result

        except Exception as exc:
            logger.exception(
                "Prediction pipeline component failed",
                extra={
                    "machine_id": machine_id,
                    "prediction_type": name,
                },
            )

            results[name] = {
                "status": "failed",
                "error": str(exc),
            }

    critical = _contains_critical_prediction(results)

    if critical:
        _publish_prediction_event(
            event_type="machine.prediction.critical",
            machine_id=machine_id,
            result={
                "predictions": results,
                "critical": True,
            },
        )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "machine_id": machine_id,
        "critical": critical,
        "predictions": results,
        "completed_at": utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=PredictionTask,
    name="prediction.batch",
    queue="ml_batch",
)
def batch_predict(
    self: PredictionTask,
    machine_ids: list[str],
) -> dict[str, Any]:
    """
    Run predictive inference for multiple machines.

    Suitable for scheduled fleet-wide inference.
    """

    results: dict[str, Any] = {}

    for machine_id in machine_ids:
        try:
            results[machine_id] = predict_failure.apply(
                kwargs={
                    "machine_id": machine_id,
                }
            ).get()

        except Exception as exc:
            logger.exception(
                "Batch prediction failed",
                extra={"machine_id": machine_id},
            )

            results[machine_id] = {
                "status": "failed",
                "error": str(exc),
            }

    return {
        "status": "completed",
        "task_id": self.request.id,
        "machine_count": len(machine_ids),
        "results": results,
        "completed_at": utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_prediction(
    prediction: Any,
    *,
    task_id: str,
    machine_id: str,
    prediction_type: str,
) -> dict[str, Any]:
    """Convert model output into a consistent worker response."""

    if isinstance(prediction, dict):
        data = dict(prediction)
    else:
        data = {
            "prediction": prediction,
        }

    return {
        "status": "completed",
        "task_id": task_id,
        "machine_id": machine_id,
        "prediction_type": prediction_type,
        "prediction": data,
        "predicted_at": utc_now().isoformat(),
    }


def _persist_prediction(
    *,
    machine_id: str,
    prediction_type: str,
    result: dict[str, Any],
) -> None:
    """Persist prediction through the repository/service layer."""

    try:
        from app.repositories.prediction_repository import (
            PredictionRepository,
        )

        repository = PredictionRepository()

        repository.create(
            machine_id=machine_id,
            prediction_type=prediction_type,
            result=result,
        )

    except Exception:
        logger.exception(
            "Failed to persist prediction",
            extra={
                "machine_id": machine_id,
                "prediction_type": prediction_type,
            },
        )


def _publish_prediction_event(
    *,
    event_type: str,
    machine_id: str,
    result: dict[str, Any],
) -> None:
    """Publish prediction event to Kafka."""

    try:
        from app.streaming.kafka_producer import publish_event

        publish_event(
            topic="industrial.predictions",
            event={
                "event_type": event_type,
                "machine_id": machine_id,
                "result": result,
                "timestamp": utc_now().isoformat(),
            },
        )

    except Exception:
        logger.exception(
            "Failed to publish prediction event",
            extra={
                "event_type": event_type,
                "machine_id": machine_id,
            },
        )


def _is_critical_failure_prediction(prediction: Any) -> bool:
    """Determine whether failure probability requires attention."""

    if not isinstance(prediction, dict):
        return False

    probability = prediction.get("failure_probability")

    if probability is None:
        return False

    try:
        return float(probability) >= 0.80
    except (TypeError, ValueError):
        return False


def _is_critical_rul_prediction(prediction: Any) -> bool:
    """Determine whether predicted RUL is critically low."""

    if not isinstance(prediction, dict):
        return False

    rul_hours = prediction.get("rul_hours")

    if rul_hours is None:
        return False

    try:
        return float(rul_hours) <= 24
    except (TypeError, ValueError):
        return False


def _contains_critical_prediction(
    results: dict[str, Any],
) -> bool:
    """
    Determine whether any prediction requires downstream action.
    """

    for result in results.values():
        if not isinstance(result, dict):
            continue

        if result.get("critical") is True:
            return True

        prediction = result.get("prediction")

        if isinstance(prediction, dict):
            if prediction.get("critical") is True:
                return True

            if prediction.get("severity") == "critical":
                return True

    return False