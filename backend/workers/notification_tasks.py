"""
Asynchronous notification and escalation tasks.

Responsibilities:
- Send email, SMS, webhook, and push notifications.
- Dispatch notifications asynchronously through Celery.
- Handle retries and transient provider failures.
- Support severity-based escalation.
- Track delivery attempts and status.
- Prevent duplicate notifications.
- Keep safety/business decisions outside the worker.
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


class NotificationTask(Task):
    """Base Celery task for notification operations."""

    autoretry_for = (
        TimeoutError,
        ConnectionError,
    )

    retry_backoff = True
    retry_backoff_max = 300
    retry_jitter = True
    max_retries = 5

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.exception(
            "Notification task failed",
            extra={
                "task_id": task_id,
                "error": str(exc),
            },
        )


# ---------------------------------------------------------------------------
# Generic notification
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.send",
    queue="notifications",
)
def send_notification(
    self: NotificationTask,
    notification_id: str,
) -> dict[str, Any]:
    """
    Process a persisted notification.

    The notification service determines:
    - recipient
    - channel
    - template
    - severity
    - tenant/site
    - deduplication
    """

    logger.info(
        "Processing notification",
        extra={
            "task_id": self.request.id,
            "notification_id": notification_id,
        },
    )

    try:
        from app.services.notification_service import NotificationService

        service = NotificationService()

        result = service.send(
            notification_id=notification_id,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "notification_id": notification_id,
            "result": result,
            "completed_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Notification delivery failed",
            extra={
                "notification_id": notification_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.send_email",
    queue="email",
)
def send_email(
    self: NotificationTask,
    recipient: str,
    subject: str,
    body: str,
    *,
    notification_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send an email notification."""

    from app.integrations.notifications.email import EmailClient

    client = EmailClient()

    try:
        result = client.send(
            recipient=recipient,
            subject=subject,
            body=body,
            metadata=metadata or {},
        )

        return {
            "status": "sent",
            "channel": "email",
            "task_id": self.request.id,
            "notification_id": notification_id,
            "recipient": recipient,
            "provider_result": result,
            "sent_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Email delivery failed",
            extra={
                "recipient": recipient,
                "notification_id": notification_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# SMS
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.send_sms",
    queue="sms",
)
def send_sms(
    self: NotificationTask,
    phone_number: str,
    message: str,
    *,
    notification_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send an SMS notification."""

    from app.integrations.notifications.sms import SMSClient

    client = SMSClient()

    try:
        result = client.send(
            phone_number=phone_number,
            message=message,
            metadata=metadata or {},
        )

        return {
            "status": "sent",
            "channel": "sms",
            "task_id": self.request.id,
            "notification_id": notification_id,
            "recipient": phone_number,
            "provider_result": result,
            "sent_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "SMS delivery failed",
            extra={
                "notification_id": notification_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.send_webhook",
    queue="webhooks",
)
def send_webhook(
    self: NotificationTask,
    webhook_url: str,
    payload: dict[str, Any],
    *,
    notification_id: str | None = None,
) -> dict[str, Any]:
    """Send an event to an external webhook."""

    from app.integrations.notifications.webhook import WebhookClient

    client = WebhookClient()

    try:
        result = client.send(
            url=webhook_url,
            payload=payload,
        )

        return {
            "status": "sent",
            "channel": "webhook",
            "task_id": self.request.id,
            "notification_id": notification_id,
            "provider_result": result,
            "sent_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Webhook delivery failed",
            extra={
                "notification_id": notification_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Push notification
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.send_push",
    queue="push",
)
def send_push(
    self: NotificationTask,
    device_token: str,
    title: str,
    body: str,
    *,
    notification_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a push notification."""

    from app.integrations.notifications.push import PushClient

    client = PushClient()

    try:
        result = client.send(
            device_token=device_token,
            title=title,
            body=body,
            data=data or {},
        )

        return {
            "status": "sent",
            "channel": "push",
            "task_id": self.request.id,
            "notification_id": notification_id,
            "provider_result": result,
            "sent_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Push notification failed",
            extra={
                "notification_id": notification_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Safety / emergency notifications
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.send_safety_alert",
    queue="critical_notifications",
)
def send_safety_alert(
    self: NotificationTask,
    incident_id: str,
    severity: str,
) -> dict[str, Any]:
    """
    Dispatch a safety alert.

    Safety severity is calculated upstream by the safety/incident
    domain layer. This task only performs notification delivery.
    """

    from app.services.notification_service import NotificationService

    service = NotificationService()

    try:
        result = service.send_safety_alert(
            incident_id=incident_id,
            severity=severity,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "incident_id": incident_id,
            "severity": severity,
            "result": result,
            "completed_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Safety alert delivery failed",
            extra={
                "incident_id": incident_id,
                "severity": severity,
            },
        )

        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.send_emergency_alert",
    queue="critical_notifications",
)
def send_emergency_alert(
    self: NotificationTask,
    incident_id: str,
) -> dict[str, Any]:
    """
    Send emergency notifications using the emergency response policy.

    The emergency service determines the actual recipients/channels.
    """

    from app.services.emergency_service import EmergencyService

    service = EmergencyService()

    try:
        result = service.notify_emergency(
            incident_id=incident_id,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "incident_id": incident_id,
            "result": result,
            "completed_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Emergency notification failed",
            extra={
                "incident_id": incident_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.escalate",
    queue="critical_notifications",
)
def escalate_notification(
    self: NotificationTask,
    incident_id: str,
    escalation_level: int = 1,
) -> dict[str, Any]:
    """
    Execute the next notification escalation level.

    Example:

        Level 1 → operator
        Level 2 → maintenance supervisor
        Level 3 → safety manager
        Level 4 → emergency response team
    """

    from app.services.notification_service import NotificationService

    service = NotificationService()

    try:
        result = service.escalate(
            incident_id=incident_id,
            escalation_level=escalation_level,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "incident_id": incident_id,
            "escalation_level": escalation_level,
            "result": result,
            "completed_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Notification escalation failed",
            extra={
                "incident_id": incident_id,
                "escalation_level": escalation_level,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Notification orchestration
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.dispatch_incident",
    queue="critical_notifications",
)
def dispatch_incident_notifications(
    self: NotificationTask,
    incident_id: str,
) -> dict[str, Any]:
    """
    Dispatch all notifications required for an incident.

    Channel selection remains inside NotificationService.
    """

    from app.services.notification_service import NotificationService

    service = NotificationService()

    try:
        result = service.dispatch_incident(
            incident_id=incident_id,
        )

        return {
            "status": "completed",
            "task_id": self.request.id,
            "incident_id": incident_id,
            "result": result,
            "completed_at": utc_now().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Incident notification dispatch failed",
            extra={
                "incident_id": incident_id,
            },
        )

        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Delivery tracking
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.record_delivery",
    queue="notification_tracking",
)
def record_delivery(
    self: NotificationTask,
    notification_id: str,
    channel: str,
    status: str,
    provider_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist notification delivery status."""

    from app.repositories.notification_repository import NotificationRepository

    repository = NotificationRepository()

    result = repository.record_delivery(
        notification_id=notification_id,
        channel=channel,
        status=status,
        provider_response=provider_response or {},
        delivered_at=utc_now() if status == "delivered" else None,
    )

    return {
        "status": "recorded",
        "task_id": self.request.id,
        "notification_id": notification_id,
        "channel": channel,
        "delivery_status": status,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Scheduled retry
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=NotificationTask,
    name="notification.retry_delivery",
    queue="notifications",
)
def retry_delivery(
    self: NotificationTask,
    notification_id: str,
) -> dict[str, Any]:
    """Retry a failed notification through the notification service."""

    from app.services.notification_service import NotificationService

    service = NotificationService()

    result = service.retry_delivery(
        notification_id=notification_id,
    )

    return {
        "status": "completed",
        "task_id": self.request.id,
        "notification_id": notification_id,
        "result": result,
        "retried_at": utc_now().isoformat(),
    }