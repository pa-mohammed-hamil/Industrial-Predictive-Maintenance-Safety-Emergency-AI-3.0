"""
app/services/notification_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
========================================================

Enterprise Notification Service

Features:
    - In-app notifications
    - Email notifications
    - SMS notifications
    - Push notifications
    - Webhook notifications
    - Incident and emergency alerts
    - Maintenance notifications
    - Failure prediction alerts
    - Machine health alerts
    - Inspection notifications
    - Notification priority and severity
    - Notification templates
    - Multi-channel delivery
    - Retry and delivery tracking
    - Notification preferences
    - Read/unread management
    - Notification history
    - Batch notifications
    - Optional repository persistence
    - Optional audit logging

Safety:
    Notifications are supplementary communication.

    Critical industrial alarms must also use
    independently engineered, certified safety
    and emergency communication systems.

    Production deployment requires durable queues,
    transactional outbox, provider idempotency,
    rate limiting, RBAC, tenant isolation, and
    verified notification delivery.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class NotificationServiceError(Exception):
    """Base notification exception."""


class NotificationValidationError(NotificationServiceError):
    """Invalid notification input."""


class NotificationNotFoundError(NotificationServiceError):
    """Notification does not exist."""


class NotificationDeliveryError(NotificationServiceError):
    """Notification delivery failed."""


class NotificationConfigurationError(NotificationServiceError):
    """Required notification configuration is missing."""


# ============================================================
# ENUMS
# ============================================================

class NotificationChannel(str, Enum):
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NotificationType(str, Enum):
    GENERAL = "general"
    INCIDENT = "incident"
    EMERGENCY = "emergency"
    SAFETY = "safety"
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    MACHINE_FAILURE = "machine_failure"
    FAILURE_PREDICTION = "failure_prediction"
    ANOMALY = "anomaly"
    MACHINE_HEALTH = "machine_health"
    MAINTENANCE = "maintenance"
    INSPECTION = "inspection"
    WORK_ORDER = "work_order"
    RUL_WARNING = "rul_warning"
    COST_ALERT = "cost_alert"
    SYSTEM = "system"


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def required_text(
    value: Any,
    name: str,
    max_length: int = 1000,
) -> str:

    if not isinstance(value, str):
        raise NotificationValidationError(
            f"{name} must be a string"
        )

    cleaned = value.strip()

    if not cleaned:
        raise NotificationValidationError(
            f"{name} cannot be empty"
        )

    if len(cleaned) > max_length:
        raise NotificationValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return cleaned


def optional_text(
    value: Optional[str],
    name: str,
    max_length: int = 1000,
) -> Optional[str]:

    if value is None:
        return None

    return required_text(
        value,
        name,
        max_length,
    )


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:

    try:
        return enum_class(value)
    except (ValueError, TypeError) as exc:
        raise NotificationValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def serialize(value: Any) -> Any:

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialize(item)
            for item in value
        ]

    return value


async def invoke(
    method: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:

    result = method(*args, **kwargs)

    if inspect.isawaitable(result):
        return await result

    return result


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class NotificationRecipient:
    user_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    webhook_url: Optional[str] = None


@dataclass
class DeliveryAttempt:
    attempt_id: str
    channel: str
    status: str
    attempted_at: datetime
    provider_message_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class NotificationRecord:
    notification_id: str
    company_id: str
    notification_type: str
    priority: str
    title: str
    message: str
    recipient_id: str
    channels: list[str]
    status: str

    site_id: Optional[str] = None
    asset_id: Optional[str] = None
    incident_id: Optional[str] = None
    emergency_id: Optional[str] = None
    maintenance_id: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    delivery_attempts: list[DeliveryAttempt] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None

    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class NotificationPreference:
    user_id: str
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = True
    in_app_enabled: bool = True
    webhook_enabled: bool = False
    minimum_priority: str = (
        NotificationPriority.LOW.value
    )


@dataclass
class NotificationTemplate:
    template_id: str
    notification_type: str
    title_template: str
    message_template: str
    default_priority: str = (
        NotificationPriority.NORMAL.value
    )


@dataclass
class DeliveryResult:
    notification_id: str
    channel: str
    status: str
    provider_message_id: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# DEFAULT TEMPLATES
# ============================================================

DEFAULT_TEMPLATES = {
    NotificationType.INCIDENT.value: NotificationTemplate(
        template_id="incident_alert",
        notification_type=NotificationType.INCIDENT.value,
        title_template="Industrial Incident Alert",
        message_template=(
            "Incident {incident_id}: {description}. "
            "Severity: {severity}."
        ),
        default_priority=NotificationPriority.HIGH.value,
    ),
    NotificationType.EMERGENCY.value: NotificationTemplate(
        template_id="emergency_alert",
        notification_type=NotificationType.EMERGENCY.value,
        title_template="Emergency Alert",
        message_template=(
            "Emergency {emergency_id}: {description}. "
            "Severity: {severity}."
        ),
        default_priority=NotificationPriority.CRITICAL.value,
    ),
    NotificationType.MAINTENANCE.value: NotificationTemplate(
        template_id="maintenance_alert",
        notification_type=NotificationType.MAINTENANCE.value,
        title_template="Maintenance Notification",
        message_template=(
            "Maintenance {maintenance_id}: {description}."
        ),
        default_priority=NotificationPriority.NORMAL.value,
    ),
    NotificationType.FAILURE_PREDICTION.value: (
        NotificationTemplate(
            template_id="failure_prediction_alert",
            notification_type=(
                NotificationType.FAILURE_PREDICTION.value
            ),
            title_template="Predicted Equipment Failure",
            message_template=(
                "Asset {asset_id} has elevated predicted "
                "failure risk: {risk_level}."
            ),
            default_priority=NotificationPriority.HIGH.value,
        )
    ),
}


# ============================================================
# NOTIFICATION SERVICE
# ============================================================

class NotificationService:
    """
    Multi-channel notification orchestration.

    Expected repository methods:

        async create_notification(data)
        async get_notification(notification_id)
        async update_notification(notification_id, updates)
        async list_notifications(filters)

    Expected optional channel adapters:

        async send_email(recipient, title, message)
        async send_sms(recipient, message)
        async send_push(recipient, title, message)
        async send_webhook(recipient, payload)

    Provider adapters should return:

        {
            "status": "sent" | "delivered",
            "message_id": "provider-id"
        }

    An adapter response indicates only the status
    reported by that provider, not independent proof
    of human receipt or acknowledgment.
    """

    PRIORITY_ORDER = {
        NotificationPriority.LOW.value: 0,
        NotificationPriority.NORMAL.value: 1,
        NotificationPriority.HIGH.value: 2,
        NotificationPriority.CRITICAL.value: 3,
    }

    def __init__(
        self,
        notification_repository: Any,
        email_adapter: Any = None,
        sms_adapter: Any = None,
        push_adapter: Any = None,
        webhook_adapter: Any = None,
        audit_service: Any = None,
        max_retries: int = 3,
    ):

        if notification_repository is None:
            raise NotificationConfigurationError(
                "notification_repository is required"
            )

        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 10
        ):
            raise NotificationValidationError(
                "max_retries must be between 0 and 10"
            )

        self.repository = notification_repository
        self.email_adapter = email_adapter
        self.sms_adapter = sms_adapter
        self.push_adapter = push_adapter
        self.webhook_adapter = webhook_adapter
        self.audit = audit_service
        self.max_retries = max_retries

        self.templates = dict(DEFAULT_TEMPLATES)

    @staticmethod
    def _field(
        record: Any,
        name: str,
        default: Any = None,
    ) -> Any:

        if isinstance(record, Mapping):
            return record.get(name, default)

        return getattr(record, name, default)

    # ========================================================
    # AUDIT
    # ========================================================

    async def _audit(
        self,
        notification_id: str,
        action: str,
        actor_id: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:

        if self.audit is None:
            return

        method = getattr(
            self.audit,
            "record_event",
            None,
        )

        if method is None:
            return

        try:
            await invoke(
                method,
                {
                    "event_id": str(uuid4()),
                    "entity_type": "notification",
                    "entity_id": notification_id,
                    "action": action,
                    "actor_id": actor_id,
                    "details": details or {},
                    "timestamp": utc_now().isoformat(),
                },
            )
        except Exception:
            logger.exception(
                "Notification audit failed: %s",
                notification_id,
            )

    # ========================================================
    # CREATE NOTIFICATION
    # ========================================================

    async def create_notification(
        self,
        *,
        company_id: str,
        recipient_id: str,
        title: str,
        message: str,
        notification_type: str | NotificationType = (
            NotificationType.GENERAL
        ),
        priority: str | NotificationPriority = (
            NotificationPriority.NORMAL
        ),
        channels: Optional[
            Sequence[str | NotificationChannel]
        ] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        incident_id: Optional[str] = None,
        emergency_id: Optional[str] = None,
        maintenance_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:

        kind = parse_enum(
            NotificationType,
            notification_type,
            "notification_type",
        )

        selected_priority = parse_enum(
            NotificationPriority,
            priority,
            "priority",
        )

        selected_channels = []

        for channel in (
            channels or [NotificationChannel.IN_APP]
        ):
            parsed = parse_enum(
                NotificationChannel,
                channel,
                "channel",
            )

            if parsed.value not in selected_channels:
                selected_channels.append(parsed.value)

        record = NotificationRecord(
            notification_id=str(uuid4()),
            company_id=required_text(
                company_id,
                "company_id",
                100,
            ),
            notification_type=kind.value,
            priority=selected_priority.value,
            title=required_text(
                title,
                "title",
                255,
            ),
            message=required_text(
                message,
                "message",
                10000,
            ),
            recipient_id=required_text(
                recipient_id,
                "recipient_id",
                100,
            ),
            channels=selected_channels,
            status=NotificationStatus.PENDING.value,
            site_id=site_id,
            asset_id=asset_id,
            incident_id=incident_id,
            emergency_id=emergency_id,
            maintenance_id=maintenance_id,
            metadata=dict(metadata or {}),
        )

        created = await invoke(
            self.repository.create_notification,
            record.to_dict(),
        )

        await self._audit(
            record.notification_id,
            "notification_created",
            "system",
            {
                "type": kind.value,
                "priority": selected_priority.value,
                "channels": selected_channels,
            },
        )

        return created

    # ========================================================
    # GET NOTIFICATION
    # ========================================================

    async def get_notification(
        self,
        notification_id: str,
    ) -> Any:

        notification_id = required_text(
            notification_id,
            "notification_id",
            100,
        )

        record = await invoke(
            self.repository.get_notification,
            notification_id,
        )

        if record is None:
            raise NotificationNotFoundError(
                f"Notification not found: {notification_id}"
            )

        return record

    # ========================================================
    # TEMPLATE MANAGEMENT
    # ========================================================

    def register_template(
        self,
        template: NotificationTemplate,
    ) -> None:

        if not isinstance(
            template,
            NotificationTemplate,
        ):
            raise NotificationValidationError(
                "Invalid notification template"
            )

        self.templates[
            template.notification_type
        ] = template

    def render_template(
        self,
        *,
        notification_type: str | NotificationType,
        context: Mapping[str, Any],
    ) -> dict[str, str]:

        kind = parse_enum(
            NotificationType,
            notification_type,
            "notification_type",
        )

        template = self.templates.get(
            kind.value
        )

        if template is None:
            raise NotificationValidationError(
                f"No template for {kind.value}"
            )

        try:
            title = template.title_template.format(
                **context
            )

            message = template.message_template.format(
                **context
            )

        except KeyError as exc:
            raise NotificationValidationError(
                f"Missing template variable: {exc}"
            ) from exc

        return {
            "title": title,
            "message": message,
            "priority": template.default_priority,
        }

    # ========================================================
    # CHANNEL DELIVERY
    # ========================================================

    async def _deliver_channel(
        self,
        *,
        notification: Any,
        recipient: NotificationRecipient,
        channel: NotificationChannel,
    ) -> DeliveryResult:

        notification_id = self._field(
            notification,
            "notification_id",
        )

        title = self._field(
            notification,
            "title",
        )

        message = self._field(
            notification,
            "message",
        )

        if channel == NotificationChannel.IN_APP:
            return DeliveryResult(
                notification_id=notification_id,
                channel=channel.value,
                status=DeliveryStatus.DELIVERED.value,
            )

        adapter = None
        method_name = None
        arguments = ()

        if channel == NotificationChannel.EMAIL:
            adapter = self.email_adapter
            method_name = "send_email"

            if not recipient.email:
                raise NotificationDeliveryError(
                    "Recipient email missing"
                )

            arguments = (
                recipient.email,
                title,
                message,
            )

        elif channel == NotificationChannel.SMS:
            adapter = self.sms_adapter
            method_name = "send_sms"

            if not recipient.phone:
                raise NotificationDeliveryError(
                    "Recipient phone missing"
                )

            arguments = (
                recipient.phone,
                message,
            )

        elif channel == NotificationChannel.PUSH:
            adapter = self.push_adapter
            method_name = "send_push"

            if not recipient.push_token:
                raise NotificationDeliveryError(
                    "Recipient push token missing"
                )

            arguments = (
                recipient.push_token,
                title,
                message,
            )

        elif channel == NotificationChannel.WEBHOOK:
            adapter = self.webhook_adapter
            method_name = "send_webhook"

            if not recipient.webhook_url:
                raise NotificationDeliveryError(
                    "Recipient webhook URL missing"
                )

            arguments = (
                recipient.webhook_url,
                {
                    "notification_id": notification_id,
                    "title": title,
                    "message": message,
                    "type": self._field(
                        notification,
                        "notification_type",
                    ),
                    "priority": self._field(
                        notification,
                        "priority",
                    ),
                },
            )

        if adapter is None:
            raise NotificationConfigurationError(
                f"No adapter configured for {channel.value}"
            )

        method = getattr(
            adapter,
            method_name,
            None,
        )

        if method is None:
            raise NotificationConfigurationError(
                f"Adapter missing method: {method_name}"
            )

        response = await invoke(
            method,
            *arguments,
        )

        if not isinstance(response, Mapping):
            raise NotificationDeliveryError(
                "Adapter must return a delivery mapping"
            )

        status = response.get(
            "status",
            DeliveryStatus.SENT.value,
        )

        if status not in {
            DeliveryStatus.SENT.value,
            DeliveryStatus.DELIVERED.value,
        }:
            raise NotificationDeliveryError(
                f"Provider returned unsuccessful status: "
                f"{status}"
            )

        return DeliveryResult(
            notification_id=notification_id,
            channel=channel.value,
            status=status,
            provider_message_id=response.get(
                "message_id"
            ),
        )

    # ========================================================
    # SEND NOTIFICATION
    # ========================================================

    async def send_notification(
        self,
        *,
        notification_id: str,
        recipient: NotificationRecipient,
    ) -> dict[str, Any]:

        notification = await self.get_notification(
            notification_id
        )

        current_status = self._field(
            notification,
            "status",
        )

        if current_status not in {
            NotificationStatus.PENDING.value,
            NotificationStatus.FAILED.value,
        }:
            raise NotificationValidationError(
                "Notification cannot be sent "
                "in its current status"
            )

        if recipient.user_id != self._field(
            notification,
            "recipient_id",
        ):
            raise NotificationValidationError(
                "Recipient does not match notification"
            )

        await invoke(
            self.repository.update_notification,
            notification_id,
            {
                "status": (
                    NotificationStatus.PROCESSING.value
                ),
                "updated_at": utc_now().isoformat(),
            },
        )

        channels = self._field(
            notification,
            "channels",
            [],
        )

        attempts = [
            serialize(item)
            for item in self._field(
                notification,
                "delivery_attempts",
                [],
            )
        ]

        results = []

        for channel_name in channels:
            channel = parse_enum(
                NotificationChannel,
                channel_name,
                "channel",
            )

            try:
                result = await self._deliver_channel(
                    notification=notification,
                    recipient=recipient,
                    channel=channel,
                )

            except Exception as exc:
                logger.exception(
                    "Notification delivery failed "
                    "id=%s channel=%s",
                    notification_id,
                    channel.value,
                )

                result = DeliveryResult(
                    notification_id=notification_id,
                    channel=channel.value,
                    status=DeliveryStatus.FAILED.value,
                    error_message=str(exc),
                )

            results.append(result)

            attempts.append(
                serialize(
                    DeliveryAttempt(
                        attempt_id=str(uuid4()),
                        channel=channel.value,
                        status=result.status,
                        attempted_at=utc_now(),
                        provider_message_id=(
                            result.provider_message_id
                        ),
                        error_message=(
                            result.error_message
                        ),
                    )
                )
            )

        success_count = sum(
            result.status in {
                DeliveryStatus.SENT.value,
                DeliveryStatus.DELIVERED.value,
            }
            for result in results
        )

        all_delivered = (
            len(results) > 0
            and all(
                result.status
                == DeliveryStatus.DELIVERED.value
                for result in results
            )
        )

        if all_delivered:
            final_status = NotificationStatus.DELIVERED
        elif success_count == len(results) and results:
            final_status = NotificationStatus.SENT
        elif success_count > 0:
            final_status = NotificationStatus.SENT
        else:
            final_status = NotificationStatus.FAILED

        now = utc_now().isoformat()

        updates: dict[str, Any] = {
            "status": final_status.value,
            "delivery_attempts": attempts,
            "updated_at": now,
        }

        if success_count > 0:
            updates["sent_at"] = now

        if all_delivered:
            updates["delivered_at"] = now

        if final_status == NotificationStatus.FAILED:
            updates["retry_count"] = (
                int(
                    self._field(
                        notification,
                        "retry_count",
                        0,
                    )
                )
                + 1
            )

        await invoke(
            self.repository.update_notification,
            notification_id,
            updates,
        )

        await self._audit(
            notification_id,
            "notification_delivery_attempted",
            "system",
            {
                "status": final_status.value,
                "successful_channels": success_count,
                "total_channels": len(results),
            },
        )

        return {
            "notification_id": notification_id,
            "status": final_status.value,
            "results": [
                result.to_dict()
                for result in results
            ],
            "successful_channels": success_count,
            "total_channels": len(results),
        }

    # ========================================================
    # RETRY FAILED NOTIFICATION
    # ========================================================

    async def retry_notification(
        self,
        *,
        notification_id: str,
        recipient: NotificationRecipient,
    ) -> dict[str, Any]:

        notification = await self.get_notification(
            notification_id
        )

        status = self._field(
            notification,
            "status",
        )

        if status != NotificationStatus.FAILED.value:
            raise NotificationValidationError(
                "Only failed notifications can be retried"
            )

        retry_count = int(
            self._field(
                notification,
                "retry_count",
                0,
            )
        )

        if retry_count > self.max_retries:
            raise NotificationDeliveryError(
                "Maximum retries exceeded"
            )

        return await self.send_notification(
            notification_id=notification_id,
            recipient=recipient,
        )

    # ========================================================
    # MARK AS READ
    # ========================================================

    async def mark_as_read(
        self,
        *,
        notification_id: str,
        user_id: str,
    ) -> Any:

        notification = await self.get_notification(
            notification_id
        )

        user_id = required_text(
            user_id,
            "user_id",
            100,
        )

        if self._field(
            notification,
            "recipient_id",
        ) != user_id:
            raise NotificationValidationError(
                "Notification does not belong to user"
            )

        updated = await invoke(
            self.repository.update_notification,
            notification_id,
            {
                "read_at": utc_now().isoformat(),
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            notification_id,
            "notification_read",
            user_id,
        )

        return updated

    # ========================================================
    # LIST NOTIFICATIONS
    # ========================================================

    async def list_notifications(
        self,
        *,
        company_id: Optional[str] = None,
        recipient_id: Optional[str] = None,
        notification_type: Optional[str] = None,
        priority: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Any:

        filters = {}

        if company_id is not None:
            filters["company_id"] = company_id

        if recipient_id is not None:
            filters["recipient_id"] = recipient_id

        if notification_type is not None:
            filters["notification_type"] = parse_enum(
                NotificationType,
                notification_type,
                "notification_type",
            ).value

        if priority is not None:
            filters["priority"] = parse_enum(
                NotificationPriority,
                priority,
                "priority",
            ).value

        if status is not None:
            filters["status"] = parse_enum(
                NotificationStatus,
                status,
                "status",
            ).value

        return await invoke(
            self.repository.list_notifications,
            filters,
        )

    # ========================================================
    # UNREAD NOTIFICATIONS
    # ========================================================

    async def get_unread_notifications(
        self,
        *,
        recipient_id: str,
        company_id: Optional[str] = None,
    ) -> list[Any]:

        records = await self.list_notifications(
            company_id=company_id,
            recipient_id=recipient_id,
        )

        return [
            record
            for record in records
            if self._field(
                record,
                "read_at",
            ) is None
        ]

    # ========================================================
    # BATCH NOTIFICATIONS
    # ========================================================

    async def create_batch_notifications(
        self,
        *,
        company_id: str,
        recipient_ids: Sequence[str],
        title: str,
        message: str,
        notification_type: str = "general",
        priority: str = "normal",
        channels: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> list[Any]:

        if not recipient_ids:
            raise NotificationValidationError(
                "At least one recipient is required"
            )

        results = []

        for recipient_id in dict.fromkeys(recipient_ids):
            notification = await self.create_notification(
                company_id=company_id,
                recipient_id=recipient_id,
                title=title,
                message=message,
                notification_type=notification_type,
                priority=priority,
                channels=channels,
                metadata=metadata,
            )

            results.append(notification)

        return results

    # ========================================================
    # INCIDENT ALERT
    # ========================================================

    async def send_incident_alert(
        self,
        data: Mapping[str, Any],
    ) -> Any:
        """
        Creates a notification for an incident.

        Expected fields:
            company_id
            recipient_id
            incident_id
            severity
            title

        This method creates a notification record.
        Actual external delivery requires a dispatcher
        calling send_notification with recipient details.
        """

        severity = str(
            data.get("severity", "high")
        ).lower()

        priority = (
            NotificationPriority.CRITICAL
            if severity == "critical"
            else NotificationPriority.HIGH
        )

        return await self.create_notification(
            company_id=required_text(
                data.get("company_id"),
                "company_id",
            ),
            recipient_id=required_text(
                data.get("recipient_id"),
                "recipient_id",
            ),
            title="Industrial Incident Alert",
            message=(
                f"Incident {data.get('incident_id')}: "
                f"{data.get('title', 'Incident reported')}. "
                f"Severity: {severity}."
            ),
            notification_type=NotificationType.INCIDENT,
            priority=priority,
            channels=[
                NotificationChannel.IN_APP,
            ],
            site_id=data.get("site_id"),
            incident_id=data.get("incident_id"),
            metadata=dict(data),
        )

    # ========================================================
    # EMERGENCY ALERT
    # ========================================================

    async def send_emergency_alert(
        self,
        data: Mapping[str, Any],
    ) -> Any:

        return await self.create_notification(
            company_id=required_text(
                data.get("company_id"),
                "company_id",
            ),
            recipient_id=required_text(
                data.get("recipient_id"),
                "recipient_id",
            ),
            title="Critical Industrial Emergency",
            message=(
                f"Emergency {data.get('emergency_id')}: "
                f"{data.get('description', 'Emergency detected')}"
            ),
            notification_type=NotificationType.EMERGENCY,
            priority=NotificationPriority.CRITICAL,
            channels=[
                NotificationChannel.IN_APP,
            ],
            site_id=data.get("site_id"),
            asset_id=data.get("asset_id"),
            emergency_id=data.get("emergency_id"),
            metadata=dict(data),
        )

    # ========================================================
    # MAINTENANCE NOTIFICATION
    # ========================================================

    async def send_maintenance_notification(
        self,
        data: Mapping[str, Any],
    ) -> Any:

        return await self.create_notification(
            company_id=required_text(
                data.get("company_id"),
                "company_id",
            ),
            recipient_id=required_text(
                data.get("recipient_id"),
                "recipient_id",
            ),
            title="Maintenance Update",
            message=(
                f"Maintenance {data.get('maintenance_id')}: "
                f"{data.get('action', 'updated')}."
            ),
            notification_type=NotificationType.MAINTENANCE,
            priority=NotificationPriority.NORMAL,
            channels=[
                NotificationChannel.IN_APP,
            ],
            maintenance_id=data.get("maintenance_id"),
            metadata=dict(data),
        )

    # ========================================================
    # FAILURE PREDICTION ALERT
    # ========================================================

    async def send_failure_prediction_alert(
        self,
        *,
        company_id: str,
        recipient_id: str,
        asset_id: str,
        failure_probability: float,
        risk_level: str,
    ) -> Any:

        if isinstance(failure_probability, bool):
            raise NotificationValidationError(
                "Invalid failure probability"
            )

        try:
            probability = float(failure_probability)
        except (TypeError, ValueError) as exc:
            raise NotificationValidationError(
                "Invalid failure probability"
            ) from exc

        if not (
            0 <= probability <= 1
        ):
            raise NotificationValidationError(
                "Failure probability must be between 0 and 1"
            )

        priority = (
            NotificationPriority.CRITICAL
            if risk_level == "critical"
            else NotificationPriority.HIGH
        )

        return await self.create_notification(
            company_id=company_id,
            recipient_id=recipient_id,
            asset_id=asset_id,
            title="Equipment Failure Prediction Alert",
            message=(
                f"Asset {asset_id} has a predicted "
                f"failure risk of {risk_level}. "
                f"Model score: {probability:.2%}. "
                "Review the equipment condition."
            ),
            notification_type=(
                NotificationType.FAILURE_PREDICTION
            ),
            priority=priority,
            channels=[
                NotificationChannel.IN_APP,
            ],
            metadata={
                "failure_probability": probability,
                "risk_level": risk_level,
            },
        )

    # ========================================================
    # NOTIFICATION STATISTICS
    # ========================================================

    async def get_notification_statistics(
        self,
        *,
        company_id: Optional[str] = None,
        recipient_id: Optional[str] = None,
    ) -> dict[str, Any]:

        records = await self.list_notifications(
            company_id=company_id,
            recipient_id=recipient_id,
        )

        status_counts = {
            item.value: 0
            for item in NotificationStatus
        }

        priority_counts = {
            item.value: 0
            for item in NotificationPriority
        }

        unread_count = 0

        for record in records:
            status = self._field(
                record,
                "status",
            )

            priority = self._field(
                record,
                "priority",
            )

            if status in status_counts:
                status_counts[status] += 1

            if priority in priority_counts:
                priority_counts[priority] += 1

            if self._field(
                record,
                "read_at",
            ) is None:
                unread_count += 1

        return {
            "total_notifications": len(records),
            "unread_notifications": unread_count,
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# IN-MEMORY DEMONSTRATION REPOSITORY
# ============================================================

class InMemoryNotificationRepository:
    """
    Development-only repository.

    Not durable, concurrency-safe or suitable
    for safety-critical notification delivery.
    """

    def __init__(self):
        self.records: dict[str, dict] = {}

    async def create_notification(
        self,
        data: dict,
    ) -> dict:

        import copy

        record = copy.deepcopy(data)

        self.records[
            record["notification_id"]
        ] = record

        return copy.deepcopy(record)

    async def get_notification(
        self,
        notification_id: str,
    ) -> Optional[dict]:

        import copy

        record = self.records.get(
            notification_id
        )

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def update_notification(
        self,
        notification_id: str,
        updates: dict,
    ) -> dict:

        import copy

        if notification_id not in self.records:
            raise NotificationNotFoundError(
                "Notification not found"
            )

        self.records[notification_id].update(
            copy.deepcopy(updates)
        )

        return copy.deepcopy(
            self.records[notification_id]
        )

    async def list_notifications(
        self,
        filters: dict,
    ) -> list[dict]:

        import copy

        return [
            copy.deepcopy(record)
            for record in self.records.values()
            if all(
                record.get(key) == value
                for key, value in filters.items()
            )
        ]


# ============================================================
# DEMONSTRATION ADAPTERS
# ============================================================

class DemoEmailAdapter:

    async def send_email(
        self,
        recipient: str,
        title: str,
        message: str,
    ) -> dict[str, str]:

        logger.info(
            "DEMO EMAIL recipient=%s title=%s",
            recipient,
            title,
        )

        return {
            "status": "sent",
            "message_id": str(uuid4()),
        }


class DemoSMSAdapter:

    async def send_sms(
        self,
        recipient: str,
        message: str,
    ) -> dict[str, str]:

        logger.info(
            "DEMO SMS recipient=%s",
            recipient,
        )

        return {
            "status": "sent",
            "message_id": str(uuid4()),
        }


class DemoPushAdapter:

    async def send_push(
        self,
        recipient: str,
        title: str,
        message: str,
    ) -> dict[str, str]:

        logger.info(
            "DEMO PUSH token=%s title=%s",
            recipient[:8],
            title,
        )

        return {
            "status": "sent",
            "message_id": str(uuid4()),
        }


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    import json

    async def main():

        logging.basicConfig(
            level=logging.INFO
        )

        repository = (
            InMemoryNotificationRepository()
        )

        service = NotificationService(
            notification_repository=repository,
            email_adapter=DemoEmailAdapter(),
            sms_adapter=DemoSMSAdapter(),
            push_adapter=DemoPushAdapter(),
        )

        recipient = NotificationRecipient(
            user_id="ENGINEER-001",
            email="engineer@example.com",
            phone="+910000000000",
            push_token="demo-device-token",
        )

        notification = await service.create_notification(
            company_id="COMPANY-001",
            recipient_id=recipient.user_id,
            title="Motor Overheating Alert",
            message=(
                "Motor temperature exceeded "
                "the configured warning threshold."
            ),
            notification_type="overheating",
            priority="high",
            channels=[
                "in_app",
                "email",
                "sms",
                "push",
            ],
            site_id="SITE-001",
            asset_id="MACHINE-001",
        )

        notification_id = notification[
            "notification_id"
        ]

        delivery = await service.send_notification(
            notification_id=notification_id,
            recipient=recipient,
        )

        print(
            "DELIVERY RESULT:"
        )

        print(
            json.dumps(
                delivery,
                indent=2,
            )
        )

        await service.mark_as_read(
            notification_id=notification_id,
            user_id=recipient.user_id,
        )

        statistics = (
            await service.get_notification_statistics(
                company_id="COMPANY-001"
            )
        )

        print(
            "\nNOTIFICATION STATISTICS:"
        )

        print(
            json.dumps(
                statistics,
                indent=2,
            )
        )

    asyncio.run(main())