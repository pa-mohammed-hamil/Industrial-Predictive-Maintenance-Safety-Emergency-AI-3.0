"""
Asynchronous reporting tasks.

Responsibilities:
- Generate asset-health reports.
- Generate predictive-maintenance reports.
- Generate safety/incident reports.
- Generate cost/ROI reports.
- Generate executive reports.
- Export reports to PDF/JSON/CSV.
- Store generated reports.
- Optionally notify users when reports are ready.

Business calculations remain in services/cost_engine/.
ML inference remains in app/ml/.
Safety decisions remain in app/safety/.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from celery import Task

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class ReportTask(Task):
    """Base Celery task for report generation."""

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
            "Report task failed",
            extra={
                "task_id": task_id,
                "error": str(exc),
            },
        )


# ---------------------------------------------------------------------------
# Generic report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.generate",
    queue="reports",
)
def generate_report(
    self: ReportTask,
    report_id: str,
) -> dict[str, Any]:
    """
    Generate a persisted report definition.

    The report service determines:
    - report type
    - data sources
    - date range
    - tenant/site/asset scope
    - requested format
    """

    logger.info(
        "Starting report generation",
        extra={
            "task_id": self.request.id,
            "report_id": report_id,
        },
    )

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        result = service.generate(
            report_id=report_id,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "report_id": report_id,
            "result": result,
            "completed_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Report generation failed",
            extra={
                "report_id": report_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Asset health report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.asset_health",
    queue="reports",
)
def generate_asset_health_report(
    self: ReportTask,
    asset_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """Generate a comprehensive asset-health report."""

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        report = service.generate_asset_health_report(
            asset_id=asset_id,
            start_date=start_date,
            end_date=end_date,
            output_format=output_format,
        )

        return _report_result(
            task_id=self.request.id,
            report_type="asset_health",
            report=report,
        )

    except Exception as exc:
        logger.exception(
            "Asset health report failed",
            extra={"asset_id": asset_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Predictive maintenance report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.predictive_maintenance",
    queue="reports",
)
def generate_predictive_maintenance_report(
    self: ReportTask,
    asset_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """
    Generate predictive-maintenance report.

    May include:
    - failure probabilities
    - RUL
    - degradation
    - anomalies
    - maintenance recommendations
    - historical failures
    """

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        report = service.generate_predictive_maintenance_report(
            asset_id=asset_id,
            start_date=start_date,
            end_date=end_date,
            output_format=output_format,
        )

        return _report_result(
            task_id=self.request.id,
            report_type="predictive_maintenance",
            report=report,
        )

    except Exception as exc:
        logger.exception(
            "Predictive-maintenance report failed",
            extra={"asset_id": asset_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Safety report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.safety",
    queue="reports",
)
def generate_safety_report(
    self: ReportTask,
    site_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """
    Generate industrial safety report.

    May include:
    - fire events
    - smoke events
    - overheating
    - restricted-zone violations
    - unsafe proximity
    - incidents
    - emergency responses
    - severity distribution
    """

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        report = service.generate_safety_report(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            output_format=output_format,
        )

        return _report_result(
            task_id=self.request.id,
            report_type="safety",
            report=report,
        )

    except Exception as exc:
        logger.exception(
            "Safety report failed",
            extra={"site_id": site_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Incident report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.incident",
    queue="reports",
)
def generate_incident_report(
    self: ReportTask,
    incident_id: str,
    *,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """Generate a detailed incident investigation report."""

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        report = service.generate_incident_report(
            incident_id=incident_id,
            output_format=output_format,
        )

        return _report_result(
            task_id=self.request.id,
            report_type="incident",
            report=report,
        )

    except Exception as exc:
        logger.exception(
            "Incident report failed",
            extra={"incident_id": incident_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Cost report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.cost",
    queue="reports",
)
def generate_cost_report(
    self: ReportTask,
    asset_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """
    Generate cost-intelligence report.

    May include:
    - repair cost
    - replacement cost
    - downtime cost
    - production loss
    - emergency cost
    - energy cost
    - expected failure cost
    - avoided cost
    - TCO
    - ROI
    """

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        report = service.generate_cost_report(
            asset_id=asset_id,
            start_date=start_date,
            end_date=end_date,
            output_format=output_format,
        )

        return _report_result(
            task_id=self.request.id,
            report_type="cost",
            report=report,
        )

    except Exception as exc:
        logger.exception(
            "Cost report failed",
            extra={"asset_id": asset_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# ROI report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.roi",
    queue="reports",
)
def generate_roi_report(
    self: ReportTask,
    asset_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """Generate maintenance/safety ROI report."""

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        report = service.generate_roi_report(
            asset_id=asset_id,
            start_date=start_date,
            end_date=end_date,
            output_format=output_format,
        )

        return _report_result(
            task_id=self.request.id,
            report_type="roi",
            report=report,
        )

    except Exception as exc:
        logger.exception(
            "ROI report failed",
            extra={"asset_id": asset_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Executive report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.executive",
    queue="reports",
)
def generate_executive_report(
    self: ReportTask,
    company_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """
    Generate executive-level industrial intelligence report.
    """

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        report = service.generate_executive_report(
            company_id=company_id,
            start_date=start_date,
            end_date=end_date,
            output_format=output_format,
        )

        return _report_result(
            task_id=self.request.id,
            report_type="executive",
            report=report,
        )

    except Exception as exc:
        logger.exception(
            "Executive report failed",
            extra={"company_id": company_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Fleet report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.fleet",
    queue="reports",
)
def generate_fleet_report(
    self: ReportTask,
    company_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """Generate fleet-wide asset intelligence report."""

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        report = service.generate_fleet_report(
            company_id=company_id,
            start_date=start_date,
            end_date=end_date,
            output_format=output_format,
        )

        return _report_result(
            task_id=self.request.id,
            report_type="fleet",
            report=report,
        )

    except Exception as exc:
        logger.exception(
            "Fleet report failed",
            extra={"company_id": company_id},
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.export",
    queue="report_exports",
)
def export_report(
    self: ReportTask,
    report_id: str,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """
    Export an already-generated report.

    Supported formats can include:
    PDF / CSV / JSON / XLSX.
    """

    supported_formats = {
        "pdf",
        "csv",
        "json",
        "xlsx",
    }

    if output_format.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported report format: {output_format}"
        )

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        result = service.export(
            report_id=report_id,
            output_format=output_format.lower(),
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "report_id": report_id,
            "format": output_format.lower(),
            "result": result,
            "exported_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Report export failed",
            extra={
                "report_id": report_id,
                "format": output_format,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Report delivery
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.deliver",
    queue="report_delivery",
)
def deliver_report(
    self: ReportTask,
    report_id: str,
    recipient: str,
    channel: str = "email",
) -> dict[str, Any]:
    """
    Deliver a completed report.

    Actual channel implementation belongs to notification services.
    """

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        result = service.deliver(
            report_id=report_id,
            recipient=recipient,
            channel=channel,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "report_id": report_id,
            "recipient": recipient,
            "channel": channel,
            "result": result,
            "delivered_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Report delivery failed",
            extra={
                "report_id": report_id,
                "recipient": recipient,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Scheduled report
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=ReportTask,
    name="report.scheduled",
    queue="reports",
)
def generate_scheduled_report(
    self: ReportTask,
    schedule_id: str,
) -> dict[str, Any]:
    """
    Generate a report according to a persisted report schedule.

    Celery Beat can invoke this task periodically.
    """

    try:
        from app.services.report_service import ReportService

        service = ReportService()

        result = service.generate_scheduled_report(
            schedule_id=schedule_id,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "schedule_id": schedule_id,
            "result": result,
            "generated_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Scheduled report failed",
            extra={
                "schedule_id": schedule_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report_result(
    *,
    task_id: str,
    report_type: str,
    report: Any,
) -> dict[str, Any]:
    """Normalize report-service output."""

    if isinstance(report, dict):
        result = dict(report)
    else:
        result = {
            "report": report,
        }

    return {
        "status": "completed",
        "task_id": task_id,
        "report_type": report_type,
        "result": result,
        "generated_at": utc_now().isoformat(),
    }