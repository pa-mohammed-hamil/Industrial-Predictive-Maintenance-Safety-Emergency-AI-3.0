"""
app/ai/ml/models/notification.py

Industrial AI Notification Models
=================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant notifications
- Email, SMS, push, dashboard and webhook channels
- Predictive maintenance alerts
- Equipment failure and anomaly notifications
- Safety incidents and emergency notifications
- Recipient and delivery tracking
- Retry and escalation metadata
- Notification acknowledgment
- Scheduling and expiration
- Template references
- Create, update and response schemas

This module defines validated notification records.

Actual sending belongs in notification services
and channel adapters. Delivery status must be
updated from trusted provider responses.
"""

from __future__ import annotations

import json

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_aware(
    value: datetime | None,
) -> datetime | None:
    if value is not None:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "Datetime must be timezone-aware"
            )

    return value


# ============================================================
# ENUMS
# ============================================================

class NotificationType(str, Enum):
    ANOMALY = "anomaly"
    FAILURE_PREDICTION = "failure_prediction"
    RUL_WARNING = "rul_warning"
    MACHINE_HEALTH = "machine_health"

    MAINTENANCE_DUE = "maintenance_due"
    MAINTENANCE_OVERDUE = "maintenance_overdue"
    MAINTENANCE_ASSIGNED = "maintenance_assigned"
    MAINTENANCE_COMPLETED = "maintenance_completed"

    INSPECTION_DUE = "inspection_due"
    INSPECTION_FINDING = "inspection_finding"

    SAFETY_ALERT = "safety_alert"
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    GAS_LEAK = "gas_leak"
    WORKER_SAFETY = "worker_safety"
    EMERGENCY = "emergency"

    INCIDENT = "incident"
    EQUIPMENT_FAILURE = "equipment_failure"

    COST_ALERT = "cost_alert"
    BUDGET_ALERT = "budget_alert"

    SYSTEM = "system"
    SECURITY = "security"
    REPORT_READY = "report_ready"
    OTHER = "other"


class NotificationChannel(str, Enum):
    DASHBOARD = "dashboard"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    CONTROL_ROOM = "control_room"


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class NotificationStatus(str, Enum):
    DRAFT = "draft"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    PROCESSING = "processing"
    SENT = "sent"
    PARTIALLY_DELIVERED = "partially_delivered"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    SKIPPED = "skipped"


class NotificationSource(str, Enum):
    SENSOR = "sensor"
    ML_MODEL = "ml_model"
    AI_AGENT = "ai_agent"
    COMPUTER_VISION = "computer_vision"
    RULE_ENGINE = "rule_engine"
    MAINTENANCE_SYSTEM = "maintenance_system"
    INSPECTION_SYSTEM = "inspection_system"
    INCIDENT_SYSTEM = "incident_system"
    EMERGENCY_SYSTEM = "emergency_system"
    HUMAN = "human"
    SYSTEM = "system"


class AcknowledgmentStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"


class EscalationStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


# ============================================================
# BASE MODEL
# ============================================================

class NotificationBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ============================================================
# CONTEXT
# ============================================================

class NotificationContext(NotificationBaseModel):
    company_id: str = Field(min_length=1)

    site_id: str | None = None
    asset_id: str | None = None
    machine_id: str | None = None
    equipment_id: str | None = None
    component_id: str | None = None

    production_line_id: str | None = None

    prediction_id: str | None = None
    anomaly_id: str | None = None
    failure_id: str | None = None

    maintenance_id: str | None = None
    inspection_id: str | None = None
    incident_id: str | None = None
    emergency_id: str | None = None

    correlation_id: str | None = None


# ============================================================
# RECIPIENT
# ============================================================

class NotificationRecipient(NotificationBaseModel):
    recipient_id: str = Field(min_length=1)

    user_id: str | None = None

    display_name: str | None = None

    role: str | None = None

    email: str | None = None
    phone_number: str | None = None

    push_token_reference: str | None = None
    webhook_endpoint_id: str | None = None

    enabled_channels: list[
        NotificationChannel
    ] = Field(default_factory=list)

    @field_validator("enabled_channels")
    @classmethod
    def unique_channels(
        cls,
        values: list[str],
    ) -> list[str]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def validate_destinations(
        self,
    ) -> "NotificationRecipient":
        if (
            NotificationChannel.EMAIL.value
            in self.enabled_channels
            and not self.email
        ):
            raise ValueError(
                "Email channel requires email address"
            )

        if (
            NotificationChannel.SMS.value
            in self.enabled_channels
            and not self.phone_number
        ):
            raise ValueError(
                "SMS channel requires phone number"
            )

        if (
            NotificationChannel.PUSH.value
            in self.enabled_channels
            and not self.push_token_reference
        ):
            raise ValueError(
                "Push channel requires token reference"
            )

        if (
            NotificationChannel.WEBHOOK.value
            in self.enabled_channels
            and not self.webhook_endpoint_id
        ):
            raise ValueError(
                "Webhook channel requires endpoint ID"
            )

        return self


# ============================================================
# CONTENT
# ============================================================

class NotificationContent(NotificationBaseModel):
    title: str = Field(
        min_length=1,
        max_length=250,
    )

    message: str = Field(
        min_length=1,
        max_length=10000,
    )

    short_message: str | None = Field(
        default=None,
        max_length=500,
    )

    template_id: str | None = None

    template_version: str | None = None

    template_variables: dict[
        str, Any
    ] = Field(default_factory=dict)

    action_label: str | None = None

    action_reference: str | None = None

    @field_validator("template_variables")
    @classmethod
    def validate_variables(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        json.dumps(
            value,
            allow_nan=False,
        )
        return value


# ============================================================
# DELIVERY ATTEMPT
# ============================================================

class NotificationDeliveryAttempt(
    NotificationBaseModel
):
    attempt_id: UUID = Field(
        default_factory=uuid4
    )

    attempt_number: int = Field(ge=1)

    attempted_at: datetime = Field(
        default_factory=utc_now
    )

    success: bool

    provider_message_id: str | None = None

    error_code: str | None = None
    error_message: str | None = None

    @field_validator("attempted_at")
    @classmethod
    def validate_attempted_at(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_aware(value)


# ============================================================
# CHANNEL DELIVERY
# ============================================================

class NotificationDelivery(NotificationBaseModel):
    delivery_id: UUID = Field(
        default_factory=uuid4
    )

    recipient_id: str = Field(min_length=1)

    channel: NotificationChannel

    status: DeliveryStatus = (
        DeliveryStatus.PENDING
    )

    provider: str | None = None

    provider_message_id: str | None = None

    attempts: list[
        NotificationDeliveryAttempt
    ] = Field(default_factory=list)

    sent_at: datetime | None = None
    delivered_at: datetime | None = None

    next_retry_at: datetime | None = None

    error_message: str | None = None

    @field_validator(
        "sent_at",
        "delivered_at",
        "next_retry_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_delivery(
        self,
    ) -> "NotificationDelivery":
        if (
            self.status == DeliveryStatus.SENT.value
            and self.sent_at is None
        ):
            raise ValueError(
                "Sent delivery requires sent_at"
            )

        if (
            self.status == DeliveryStatus.DELIVERED.value
            and self.delivered_at is None
        ):
            raise ValueError(
                "Delivered status requires delivered_at"
            )

        if (
            self.sent_at is not None
            and self.delivered_at is not None
            and self.delivered_at < self.sent_at
        ):
            raise ValueError(
                "Delivery cannot precede sending"
            )

        numbers = [
            item.attempt_number
            for item in self.attempts
        ]

        if len(numbers) != len(set(numbers)):
            raise ValueError(
                "Duplicate delivery attempt numbers"
            )

        return self


# ============================================================
# ACKNOWLEDGMENT
# ============================================================

class NotificationAcknowledgment(
    NotificationBaseModel
):
    status: AcknowledgmentStatus = (
        AcknowledgmentStatus.PENDING
    )

    acknowledged_by: str | None = None

    acknowledged_at: datetime | None = None

    notes: str | None = None

    @model_validator(mode="after")
    def validate_acknowledgment(
        self,
    ) -> "NotificationAcknowledgment":
        validate_aware(self.acknowledged_at)

        if (
            self.status
            == AcknowledgmentStatus.ACKNOWLEDGED.value
            and not (
                self.acknowledged_by
                and self.acknowledged_at
            )
        ):
            raise ValueError(
                "Acknowledgment requires actor and time"
            )

        return self


# ============================================================
# ESCALATION
# ============================================================

class NotificationEscalation(
    NotificationBaseModel
):
    status: EscalationStatus = (
        EscalationStatus.NOT_REQUIRED
    )

    escalation_level: int = Field(
        default=0,
        ge=0,
    )

    escalate_after_minutes: int | None = Field(
        default=None,
        gt=0,
    )

    escalation_recipient_ids: list[str] = Field(
        default_factory=list
    )

    escalated_at: datetime | None = None

    resolved_at: datetime | None = None

    @field_validator(
        "escalated_at",
        "resolved_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)


# ============================================================
# SCHEDULE
# ============================================================

class NotificationSchedule(
    NotificationBaseModel
):
    scheduled_at: datetime | None = None

    expires_at: datetime | None = None

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=20,
    )

    retry_delay_seconds: int = Field(
        default=60,
        ge=1,
    )

    @field_validator(
        "scheduled_at",
        "expires_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_schedule(
        self,
    ) -> "NotificationSchedule":
        if (
            self.scheduled_at is not None
            and self.expires_at is not None
            and self.expires_at <= self.scheduled_at
        ):
            raise ValueError(
                "Expiration must follow scheduled time"
            )

        return self


# ============================================================
# MAIN NOTIFICATION
# ============================================================

class IndustrialNotification(
    NotificationBaseModel
):
    notification_id: UUID = Field(
        default_factory=uuid4
    )

    notification_code: str = Field(
        min_length=1,
        max_length=100,
    )

    context: NotificationContext

    notification_type: NotificationType

    source: NotificationSource

    priority: NotificationPriority = (
        NotificationPriority.NORMAL
    )

    status: NotificationStatus = (
        NotificationStatus.DRAFT
    )

    content: NotificationContent

    recipients: list[
        NotificationRecipient
    ] = Field(min_length=1)

    channels: list[
        NotificationChannel
    ] = Field(min_length=1)

    deliveries: list[
        NotificationDelivery
    ] = Field(default_factory=list)

    schedule: NotificationSchedule = Field(
        default_factory=NotificationSchedule
    )

    acknowledgment: (
        NotificationAcknowledgment
    ) = Field(
        default_factory=NotificationAcknowledgment
    )

    escalation: NotificationEscalation = Field(
        default_factory=NotificationEscalation
    )

    deduplication_key: str | None = None

    requires_human_review: bool = False

    created_by: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    queued_at: datetime | None = None

    completed_at: datetime | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator(
        "created_at",
        "updated_at",
        "queued_at",
        "completed_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @field_validator("channels")
    @classmethod
    def unique_channels(
        cls,
        values: list[str],
    ) -> list[str]:
        return list(dict.fromkeys(values))

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        json.dumps(value, allow_nan=False)
        return value

    @model_validator(mode="after")
    def validate_notification(
        self,
    ) -> "IndustrialNotification":
        recipient_ids = [
            item.recipient_id
            for item in self.recipients
        ]

        if len(recipient_ids) != len(set(recipient_ids)):
            raise ValueError(
                "Duplicate notification recipients"
            )

        for channel in self.channels:
            if not any(
                channel in recipient.enabled_channels
                for recipient in self.recipients
            ):
                raise ValueError(
                    f"No eligible recipient for {channel}"
                )

        valid_pairs = {
            (
                recipient.recipient_id,
                channel,
            )
            for recipient in self.recipients
            for channel in recipient.enabled_channels
            if channel in self.channels
        }

        delivery_pairs = [
            (
                delivery.recipient_id,
                delivery.channel,
            )
            for delivery in self.deliveries
        ]

        if len(delivery_pairs) != len(
            set(delivery_pairs)
        ):
            raise ValueError(
                "Duplicate recipient/channel deliveries"
            )

        if any(
            pair not in valid_pairs
            for pair in delivery_pairs
        ):
            raise ValueError(
                "Delivery references an ineligible "
                "recipient/channel pair"
            )

        return self

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = utc_now()

    # --------------------------------------------------------
    # DELIVERY COUNTS
    # --------------------------------------------------------

    @property
    def delivered_count(self) -> int:
        return sum(
            delivery.status
            == DeliveryStatus.DELIVERED.value
            for delivery in self.deliveries
        )

    @property
    def failed_count(self) -> int:
        return sum(
            delivery.status
            == DeliveryStatus.FAILED.value
            for delivery in self.deliveries
        )

    @property
    def pending_count(self) -> int:
        return sum(
            delivery.status in {
                DeliveryStatus.PENDING.value,
                DeliveryStatus.SENDING.value,
            }
            for delivery in self.deliveries
        )

    # --------------------------------------------------------
    # CREATE DELIVERY RECORDS
    # --------------------------------------------------------

    def initialize_deliveries(self) -> None:
        if self.status != NotificationStatus.DRAFT.value:
            raise ValueError(
                "Deliveries can only be initialized "
                "for draft notifications"
            )

        if self.deliveries:
            raise ValueError(
                "Deliveries already initialized"
            )

        self.deliveries = [
            NotificationDelivery(
                recipient_id=recipient.recipient_id,
                channel=channel,
            )
            for recipient in self.recipients
            for channel in self.channels
            if channel in recipient.enabled_channels
        ]

        self._touch()

    # --------------------------------------------------------
    # QUEUE
    # --------------------------------------------------------

    def queue(self) -> None:
        if self.status != NotificationStatus.DRAFT.value:
            raise ValueError(
                "Only draft notifications can be queued"
            )

        if not self.deliveries:
            self.initialize_deliveries()

        now = utc_now()

        if (
            self.schedule.expires_at is not None
            and now >= self.schedule.expires_at
        ):
            self.status = NotificationStatus.EXPIRED
            self._touch()
            return

        if (
            self.schedule.scheduled_at is not None
            and self.schedule.scheduled_at > now
        ):
            self.status = NotificationStatus.SCHEDULED
        else:
            self.status = NotificationStatus.QUEUED

        self.queued_at = now
        self._touch()

    # --------------------------------------------------------
    # RECORD DELIVERY ATTEMPT
    # --------------------------------------------------------

    def record_delivery_attempt(
        self,
        delivery_id: UUID,
        *,
        success: bool,
        provider_message_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        delivery = next(
            (
                item
                for item in self.deliveries
                if item.delivery_id == delivery_id
            ),
            None,
        )

        if delivery is None:
            raise ValueError(
                "Delivery record not found"
            )

        if self.status not in {
            NotificationStatus.QUEUED.value,
            NotificationStatus.PROCESSING.value,
            NotificationStatus.PARTIALLY_DELIVERED.value,
        }:
            raise ValueError(
                "Notification is not dispatchable"
            )

        if delivery.status in {
            DeliveryStatus.DELIVERED.value,
            DeliveryStatus.SKIPPED.value,
        }:
            raise ValueError(
                "Delivery is already finalized"
            )

        attempt_number = len(delivery.attempts) + 1

        if attempt_number > self.schedule.max_attempts:
            raise ValueError(
                "Maximum delivery attempts exceeded"
            )

        attempt = NotificationDeliveryAttempt(
            attempt_number=attempt_number,
            success=success,
            provider_message_id=provider_message_id,
            error_code=error_code,
            error_message=error_message,
        )

        delivery.attempts.append(attempt)

        if success:
            delivery.status = DeliveryStatus.SENT
            delivery.sent_at = utc_now()
            delivery.provider_message_id = (
                provider_message_id
            )
            delivery.error_message = None
            delivery.next_retry_at = None
        else:
            delivery.status = DeliveryStatus.FAILED
            delivery.error_message = error_message

        self.status = NotificationStatus.PROCESSING
        self._touch()

    # --------------------------------------------------------
    # CONFIRM DELIVERY
    # --------------------------------------------------------

    def confirm_delivery(
        self,
        delivery_id: UUID,
        delivered_at: datetime | None = None,
    ) -> None:
        delivery = next(
            (
                item
                for item in self.deliveries
                if item.delivery_id == delivery_id
            ),
            None,
        )

        if delivery is None:
            raise ValueError(
                "Delivery record not found"
            )

        if delivery.status not in {
            DeliveryStatus.SENT.value,
            DeliveryStatus.DELIVERED.value,
        }:
            raise ValueError(
                "Delivery must first be sent"
            )

        timestamp = delivered_at or utc_now()
        validate_aware(timestamp)

        if (
            delivery.sent_at is not None
            and timestamp < delivery.sent_at
        ):
            raise ValueError(
                "Delivery timestamp precedes sent time"
            )

        delivery.status = DeliveryStatus.DELIVERED
        delivery.delivered_at = timestamp

        self.refresh_status()

    # --------------------------------------------------------
    # REFRESH STATUS
    # --------------------------------------------------------

    def refresh_status(self) -> None:
        if not self.deliveries:
            return

        statuses = {
            item.status
            for item in self.deliveries
        }

        if statuses == {
            DeliveryStatus.DELIVERED.value
        }:
            self.status = NotificationStatus.DELIVERED
            self.completed_at = utc_now()

        elif (
            DeliveryStatus.DELIVERED.value
            in statuses
        ):
            self.status = (
                NotificationStatus.PARTIALLY_DELIVERED
            )

        elif statuses.issubset({
            DeliveryStatus.FAILED.value,
            DeliveryStatus.SKIPPED.value,
        }):
            self.status = NotificationStatus.FAILED
            self.completed_at = utc_now()

        elif DeliveryStatus.SENT.value in statuses:
            self.status = NotificationStatus.SENT

        else:
            self.status = NotificationStatus.PROCESSING

        self._touch()

    # --------------------------------------------------------
    # ACKNOWLEDGE
    # --------------------------------------------------------

    def acknowledge(
        self,
        user_id: str,
        notes: str | None = None,
    ) -> None:
        if not user_id.strip():
            raise ValueError(
                "user_id is required"
            )

        if (
            self.acknowledgment.status
            == AcknowledgmentStatus.NOT_REQUIRED.value
        ):
            raise ValueError(
                "Acknowledgment is not required"
            )

        self.acknowledgment = (
            NotificationAcknowledgment(
                status=(
                    AcknowledgmentStatus.ACKNOWLEDGED
                ),
                acknowledged_by=user_id,
                acknowledged_at=utc_now(),
                notes=notes,
            )
        )

        self._touch()

    # --------------------------------------------------------
    # ESCALATE
    # --------------------------------------------------------

    def escalate(self) -> None:
        if (
            self.escalation.status
            == EscalationStatus.NOT_REQUIRED.value
        ):
            raise ValueError(
                "Escalation is not configured"
            )

        if not self.escalation.escalation_recipient_ids:
            raise ValueError(
                "No escalation recipients configured"
            )

        self.escalation.status = (
            EscalationStatus.ESCALATED
        )

        self.escalation.escalation_level += 1
        self.escalation.escalated_at = utc_now()

        self._touch()

    # --------------------------------------------------------
    # EXPIRATION
    # --------------------------------------------------------

    def is_expired(
        self,
        reference_time: datetime | None = None,
    ) -> bool:
        if self.schedule.expires_at is None:
            return False

        now = reference_time or utc_now()
        validate_aware(now)

        return now >= self.schedule.expires_at

    # --------------------------------------------------------
    # SAFETY PRIORITY
    # --------------------------------------------------------

    def is_critical(self) -> bool:
        return self.priority == (
            NotificationPriority.CRITICAL.value
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class NotificationCreate(NotificationBaseModel):
    notification_code: str = Field(min_length=1)

    context: NotificationContext

    notification_type: NotificationType
    source: NotificationSource

    priority: NotificationPriority = (
        NotificationPriority.NORMAL
    )

    content: NotificationContent

    recipients: list[
        NotificationRecipient
    ] = Field(min_length=1)

    channels: list[
        NotificationChannel
    ] = Field(min_length=1)

    schedule: NotificationSchedule = Field(
        default_factory=NotificationSchedule
    )

    deduplication_key: str | None = None


# ============================================================
# UPDATE SCHEMA
# ============================================================

class NotificationUpdate(NotificationBaseModel):
    priority: NotificationPriority | None = None

    content: NotificationContent | None = None

    schedule: NotificationSchedule | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class NotificationResponse(IndustrialNotification):
    pass


# ============================================================
# SUMMARY
# ============================================================

class NotificationSummary(NotificationBaseModel):
    notification_id: UUID

    notification_code: str

    company_id: str

    notification_type: NotificationType

    priority: NotificationPriority
    status: NotificationStatus

    title: str

    recipient_count: int
    delivery_count: int

    delivered_count: int
    failed_count: int

    acknowledged: bool
    escalated: bool

    created_at: datetime

    @classmethod
    def from_notification(
        cls,
        notification: IndustrialNotification,
    ) -> "NotificationSummary":
        return cls(
            notification_id=notification.notification_id,
            notification_code=notification.notification_code,
            company_id=notification.context.company_id,
            notification_type=notification.notification_type,
            priority=notification.priority,
            status=notification.status,
            title=notification.content.title,
            recipient_count=len(notification.recipients),
            delivery_count=len(notification.deliveries),
            delivered_count=notification.delivered_count,
            failed_count=notification.failed_count,
            acknowledged=(
                notification.acknowledgment.status
                == AcknowledgmentStatus.ACKNOWLEDGED.value
            ),
            escalated=(
                notification.escalation.status
                == EscalationStatus.ESCALATED.value
            ),
            created_at=notification.created_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    notification = IndustrialNotification(
        notification_code="NOTIF-2026-001",
        context=NotificationContext(
            company_id="company-001",
            site_id="SITE-01",
            asset_id="ASSET-101",
            machine_id="PUMP-101",
            prediction_id="PRED-101",
        ),
        notification_type=(
            NotificationType.FAILURE_PREDICTION
        ),
        source=NotificationSource.ML_MODEL,
        priority=NotificationPriority.HIGH,
        content=NotificationContent(
            title="Pump Bearing Failure Risk",
            message=(
                "The predictive maintenance model "
                "identified elevated bearing failure "
                "risk within the next 168 hours. "
                "Engineering review is recommended."
            ),
            short_message=(
                "High bearing failure risk: PUMP-101"
            ),
        ),
        recipients=[
            NotificationRecipient(
                recipient_id="USER-001",
                user_id="USER-001",
                display_name="Maintenance Engineer",
                email="engineer@example.com",
                enabled_channels=[
                    NotificationChannel.EMAIL,
                    NotificationChannel.DASHBOARD,
                ],
            )
        ],
        channels=[
            NotificationChannel.EMAIL,
            NotificationChannel.DASHBOARD,
        ],
        acknowledgment=NotificationAcknowledgment(
            status=AcknowledgmentStatus.PENDING
        ),
        escalation=NotificationEscalation(
            status=EscalationStatus.PENDING,
            escalate_after_minutes=30,
            escalation_recipient_ids=[
                "SUPERVISOR-001"
            ],
        ),
        deduplication_key=(
            "company-001:PUMP-101:"
            "failure_prediction:high"
        ),
    )

    notification.queue()

    print(
        json.dumps(
            NotificationSummary.from_notification(
                notification
            ).to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()