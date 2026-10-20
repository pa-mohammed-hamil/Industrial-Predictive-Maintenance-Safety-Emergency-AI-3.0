"""
Asynchronous image-processing tasks.

Responsibilities:
- Process uploaded industrial images asynchronously.
- Run defect / damage / corrosion / crack / leakage detection.
- Run fire / smoke / overheating / motion / person detection.
- Persist inference results.
- Trigger downstream safety/incident workflows when required.
- Track task status and failures.
- Avoid embedding ML implementation directly in Celery tasks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from celery import Task

from app.config.logging import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__) if "get_logger" in globals() else logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class ImageTask(Task):
    """Base task for image processing."""

    autoretry_for = (TimeoutError, ConnectionError)
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
            "Image task failed",
            extra={
                "task_id": task_id,
                "error": str(exc),
            },
        )


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.process",
    queue="image_processing",
)
def process_image(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
    inspection_id: str | None = None,
) -> dict[str, Any]:
    """
    Process an industrial image.

    The task intentionally delegates ML inference to the CV layer.
    """

    started_at = utc_now()

    logger.info(
        "Starting image processing",
        extra={
            "task_id": self.request.id,
            "image_id": image_id,
            "asset_id": asset_id,
        },
    )

    try:
        from app.services.inspection_service import InspectionService

        service = InspectionService()

        result = service.process_image(
            image_id=image_id,
            asset_id=asset_id,
            inspection_id=inspection_id,
        )

        completed_at = utc_now()

        return {
            "status": "completed",
            "task_id": self.request.id,
            "image_id": image_id,
            "asset_id": asset_id,
            "inspection_id": inspection_id,
            "result": result,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Image processing failed",
            extra={
                "task_id": self.request.id,
                "image_id": image_id,
            },
        )
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.detect_defects",
    queue="cv_inference",
)
def detect_defects(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
) -> dict[str, Any]:
    """Run general industrial defect detection."""

    from app.ml.computer_vision.defect_detection import DefectDetector

    detector = DefectDetector()

    detections = detector.predict(
        image_id=image_id,
        asset_id=asset_id,
    )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "detections": detections,
        "processed_at": utc_now().isoformat(),
    }


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.detect_corrosion",
    queue="cv_inference",
)
def detect_corrosion(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
) -> dict[str, Any]:
    """Detect corrosion in industrial equipment."""

    from app.ml.computer_vision.corrosion_detection import CorrosionDetector

    detector = CorrosionDetector()

    detections = detector.predict(
        image_id=image_id,
        asset_id=asset_id,
    )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "detections": detections,
        "processed_at": utc_now().isoformat(),
    }


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.detect_cracks",
    queue="cv_inference",
)
def detect_cracks(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
) -> dict[str, Any]:
    """Detect cracks and structural defects."""

    from app.ml.computer_vision.crack_detection import CrackDetector

    detector = CrackDetector()

    detections = detector.predict(
        image_id=image_id,
        asset_id=asset_id,
    )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "detections": detections,
        "processed_at": utc_now().isoformat(),
    }


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.detect_leakage",
    queue="cv_inference",
)
def detect_leakage(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
) -> dict[str, Any]:
    """Detect fluid/gas leakage indicators."""

    from app.ml.computer_vision.leakage_detection import LeakageDetector

    detector = LeakageDetector()

    detections = detector.predict(
        image_id=image_id,
        asset_id=asset_id,
    )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "detections": detections,
        "processed_at": utc_now().isoformat(),
    }


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.detect_fire",
    queue="safety_cv",
)
def detect_fire(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
    site_id: str | None = None,
) -> dict[str, Any]:
    """
    Detect fire and trigger the safety pipeline when required.
    """

    from app.ml.computer_vision.fire_detection import FireDetector

    detector = FireDetector()

    result = detector.predict(
        image_id=image_id,
        asset_id=asset_id,
    )

    detected = bool(result.get("detected", False))

    if detected:
        logger.warning(
            "Fire detected",
            extra={
                "image_id": image_id,
                "asset_id": asset_id,
                "site_id": site_id,
            },
        )

        _trigger_safety_event(
            event_type="fire.detected",
            image_id=image_id,
            asset_id=asset_id,
            site_id=site_id,
            result=result,
        )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "site_id": site_id,
        "detected": detected,
        "result": result,
        "processed_at": utc_now().isoformat(),
    }


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.detect_smoke",
    queue="safety_cv",
)
def detect_smoke(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
    site_id: str | None = None,
) -> dict[str, Any]:
    """Detect smoke and trigger safety processing."""

    from app.ml.computer_vision.smoke_detection import SmokeDetector

    detector = SmokeDetector()

    result = detector.predict(
        image_id=image_id,
        asset_id=asset_id,
    )

    detected = bool(result.get("detected", False))

    if detected:
        _trigger_safety_event(
            event_type="smoke.detected",
            image_id=image_id,
            asset_id=asset_id,
            site_id=site_id,
            result=result,
        )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "site_id": site_id,
        "detected": detected,
        "result": result,
        "processed_at": utc_now().isoformat(),
    }


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.detect_person",
    queue="safety_cv",
)
def detect_person(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
    zone_id: str | None = None,
) -> dict[str, Any]:
    """Detect people in industrial imagery."""

    from app.ml.computer_vision.person_detection import PersonDetector

    detector = PersonDetector()

    result = detector.predict(
        image_id=image_id,
        asset_id=asset_id,
        zone_id=zone_id,
    )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "zone_id": zone_id,
        "result": result,
        "processed_at": utc_now().isoformat(),
    }


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.detect_restricted_zone",
    queue="safety_cv",
)
def detect_restricted_zone(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
    zone_id: str | None = None,
) -> dict[str, Any]:
    """Detect restricted-zone violations."""

    from app.ml.computer_vision.restricted_zone_detection import (
        RestrictedZoneDetector,
    )

    detector = RestrictedZoneDetector()

    result = detector.predict(
        image_id=image_id,
        asset_id=asset_id,
        zone_id=zone_id,
    )

    violation = bool(result.get("violation", False))

    if violation:
        _trigger_safety_event(
            event_type="restricted_zone.violation",
            image_id=image_id,
            asset_id=asset_id,
            site_id=None,
            result=result,
        )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "zone_id": zone_id,
        "violation": violation,
        "result": result,
        "processed_at": utc_now().isoformat(),
    }


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.detect_overheating",
    queue="safety_cv",
)
def detect_overheating(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
) -> dict[str, Any]:
    """Detect visual/thermal overheating indicators."""

    from app.ml.computer_vision.overheating_detection import (
        OverheatingDetector,
    )

    detector = OverheatingDetector()

    result = detector.predict(
        image_id=image_id,
        asset_id=asset_id,
    )

    detected = bool(result.get("detected", False))

    if detected:
        _trigger_safety_event(
            event_type="overheating.detected",
            image_id=image_id,
            asset_id=asset_id,
            site_id=None,
            result=result,
        )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "detected": detected,
        "result": result,
        "processed_at": utc_now().isoformat(),
    }


@celery_app.task(
    bind=True,
    base=ImageTask,
    name="image.run_safety_pipeline",
    queue="safety_cv",
)
def run_safety_pipeline(
    self: ImageTask,
    image_id: str,
    asset_id: str | None = None,
    site_id: str | None = None,
) -> dict[str, Any]:
    """
    Run multiple safety detectors against the same image.

    This provides a single orchestration entry point for inspection
    images requiring comprehensive safety analysis.
    """

    results: dict[str, Any] = {}

    detectors = (
        ("fire", detect_fire),
        ("smoke", detect_smoke),
        ("overheating", detect_overheating),
        ("person", detect_person),
        ("restricted_zone", detect_restricted_zone),
    )

    for name, task in detectors:
        try:
            if name == "fire":
                result = task.apply(
                    kwargs={
                        "image_id": image_id,
                        "asset_id": asset_id,
                        "site_id": site_id,
                    }
                ).get()
            elif name == "smoke":
                result = task.apply(
                    kwargs={
                        "image_id": image_id,
                        "asset_id": asset_id,
                        "site_id": site_id,
                    }
                ).get()
            elif name == "person":
                result = task.apply(
                    kwargs={
                        "image_id": image_id,
                        "asset_id": asset_id,
                    }
                ).get()
            elif name == "restricted_zone":
                result = task.apply(
                    kwargs={
                        "image_id": image_id,
                        "asset_id": asset_id,
                    }
                ).get()
            else:
                result = task.apply(
                    kwargs={
                        "image_id": image_id,
                        "asset_id": asset_id,
                    }
                ).get()

            results[name] = result

        except Exception as exc:
            logger.exception(
                "Safety detector failed",
                extra={
                    "detector": name,
                    "image_id": image_id,
                    "error": str(exc),
                },
            )

            results[name] = {
                "status": "failed",
                "error": str(exc),
            }

    critical = _contains_critical_detection(results)

    return {
        "status": "completed",
        "task_id": self.request.id,
        "image_id": image_id,
        "asset_id": asset_id,
        "site_id": site_id,
        "critical": critical,
        "results": results,
        "processed_at": utc_now().isoformat(),
    }


def _trigger_safety_event(
    *,
    event_type: str,
    image_id: str,
    asset_id: str | None,
    site_id: str | None,
    result: dict[str, Any],
) -> None:
    """
    Send a safety event into the application's event pipeline.

    In production this should normally publish to Kafka rather than
    performing expensive incident processing directly in the Celery worker.
    """

    try:
        from app.streaming.kafka_producer import publish_event

        publish_event(
            topic="industrial.safety.events",
            event={
                "event_type": event_type,
                "image_id": image_id,
                "asset_id": asset_id,
                "site_id": site_id,
                "result": result,
                "timestamp": utc_now().isoformat(),
            },
        )

    except Exception:
        logger.exception(
            "Failed to publish safety event",
            extra={
                "event_type": event_type,
                "image_id": image_id,
            },
        )


def _contains_critical_detection(results: dict[str, Any]) -> bool:
    """Determine whether any detector produced a critical result."""

    critical_keys = {
        "fire",
        "smoke",
        "overheating",
        "restricted_zone",
    }

    for detector_name, result in results.items():
        if detector_name not in critical_keys:
            continue

        if not isinstance(result, dict):
            continue

        if result.get("detected") is True:
            return True

        if result.get("violation") is True:
            return True

        nested = result.get("result")

        if isinstance(nested, dict):
            if nested.get("severity") == "critical":
                return True

            if nested.get("critical") is True:
                return True

    return False