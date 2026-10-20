# app/observability/alerting.py

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4


logger = logging.getLogger(__name__)


# =========================================================
# Enums
# =========================================================

class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(str, Enum):
    TELEMETRY = "telemetry"
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    ENERGY = "energy"

    ANOMALY = "anomaly"
    FAILURE_PREDICTION = "failure_prediction"
    RUL = "rul"

    MAINTENANCE = "maintenance"

    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    MOTION = "motion"
    PPE = "ppe"
    RESTRICTED_ZONE = "restricted_zone"
    WORKER_SAFETY = "worker_safety"

    EMERGENCY = "emergency"

    MODEL = "model"
    DATA_QUALITY = "data_quality"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    COST = "cost"

    OTHER = "other"


class AlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertChannel(str, Enum):
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    KAFKA = "kafka"
    WEBSOCKET = "websocket"


# =========================================================
# Helpers
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_datetime(
    value: datetime | None,
) -> str | None:
    return value.isoformat() if value else None


# =========================================================
# Alert model
# =========================================================

@dataclass(slots=True)
class Alert:
    title: str
    message: str

    severity: AlertSeverity
    category: AlertCategory

    alert_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    status: AlertStatus = AlertStatus.OPEN

    asset_id: str | None = None
    site_id: str | None = None
    sensor_id: str | None = None

    incident_id: str | None = None
    prediction_id: str | None = None
    work_order_id: str | None = None

    metric_name: str | None = None
    metric_value: float | None = None
    threshold: float | None = None
    unit: str | None = None

    source: str | None = None

    deduplication_key: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None

    escalated_at: datetime | None = None

    resolved_at: datetime | None = None
    resolved_by: str | None = None

    occurrence_count: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["severity"] = self.severity.value
        data["category"] = self.category.value
        data["status"] = self.status.value

        data["created_at"] = _serialize_datetime(
            self.created_at
        )
        data["updated_at"] = _serialize_datetime(
            self.updated_at
        )
        data["acknowledged_at"] = _serialize_datetime(
            self.acknowledged_at
        )
        data["escalated_at"] = _serialize_datetime(
            self.escalated_at
        )
        data["resolved_at"] = _serialize_datetime(
            self.resolved_at
        )

        return data


# =========================================================
# Alert rule
# =========================================================

@dataclass(slots=True)
class AlertRule:
    rule_id: str
    name: str

    category: AlertCategory
    severity: AlertSeverity

    metric_name: str

    operator: str
    threshold: float

    enabled: bool = True

    cooldown_seconds: int = 300

    channels: tuple[AlertChannel, ...] = (
        AlertChannel.LOG,
    )

    message_template: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def matches(
        self,
        value: float,
    ) -> bool:

        operators: dict[
            str,
            Callable[[float, float], bool],
        ] = {
            ">": lambda x, y: x > y,
            ">=": lambda x, y: x >= y,
            "<": lambda x, y: x < y,
            "<=": lambda x, y: x <= y,
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y,
        }

        operation = operators.get(
            self.operator
        )

        if operation is None:
            raise ValueError(
                f"Unsupported operator: "
                f"{self.operator}"
            )

        return operation(
            value,
            self.threshold,
        )


# =========================================================
# Escalation
# =========================================================

@dataclass(slots=True)
class EscalationPolicy:
    severity: AlertSeverity

    escalate_after_seconds: int

    channels: tuple[AlertChannel, ...]

    target_severity: (
        AlertSeverity | None
    ) = None


DEFAULT_ESCALATION_POLICIES = {
    AlertSeverity.WARNING: EscalationPolicy(
        severity=AlertSeverity.WARNING,
        escalate_after_seconds=1800,
        channels=(
            AlertChannel.EMAIL,
            AlertChannel.PUSH,
        ),
        target_severity=AlertSeverity.HIGH,
    ),

    AlertSeverity.HIGH: EscalationPolicy(
        severity=AlertSeverity.HIGH,
        escalate_after_seconds=600,
        channels=(
            AlertChannel.EMAIL,
            AlertChannel.SMS,
            AlertChannel.PUSH,
        ),
        target_severity=(
            AlertSeverity.CRITICAL
        ),
    ),

    AlertSeverity.CRITICAL: EscalationPolicy(
        severity=AlertSeverity.CRITICAL,
        escalate_after_seconds=120,
        channels=(
            AlertChannel.EMAIL,
            AlertChannel.SMS,
            AlertChannel.PUSH,
            AlertChannel.WEBHOOK,
        ),
        target_severity=(
            AlertSeverity.EMERGENCY
        ),
    ),
}


# =========================================================
# Handler protocol
# =========================================================

class AlertHandler(Protocol):

    async def __call__(
        self,
        alert: Alert,
    ) -> None:
        ...


AlertCallback = Callable[
    [Alert],
    Awaitable[None] | None,
]


# =========================================================
# Alert store
# =========================================================

class InMemoryAlertStore:
    """
    Development/default alert repository.

    Replace with PostgreSQL/Redis implementation for
    distributed production deployments.
    """

    def __init__(self) -> None:
        self._alerts: dict[
            str,
            Alert,
        ] = {}

        self._lock = asyncio.Lock()

    async def save(
        self,
        alert: Alert,
    ) -> Alert:

        async with self._lock:
            self._alerts[
                alert.alert_id
            ] = alert

        return alert

    async def get(
        self,
        alert_id: str,
    ) -> Alert | None:

        async with self._lock:
            return self._alerts.get(
                alert_id
            )

    async def list(
        self,
        *,
        status: AlertStatus | None = None,
        severity: AlertSeverity | None = None,
        category: AlertCategory | None = None,
        asset_id: str | None = None,
    ) -> list[Alert]:

        async with self._lock:
            alerts = list(
                self._alerts.values()
            )

        if status is not None:
            alerts = [
                alert
                for alert in alerts
                if alert.status == status
            ]

        if severity is not None:
            alerts = [
                alert
                for alert in alerts
                if alert.severity == severity
            ]

        if category is not None:
            alerts = [
                alert
                for alert in alerts
                if alert.category == category
            ]

        if asset_id is not None:
            alerts = [
                alert
                for alert in alerts
                if alert.asset_id == asset_id
            ]

        return sorted(
            alerts,
            key=lambda alert: alert.created_at,
            reverse=True,
        )


# =========================================================
# Alert manager
# =========================================================

class AlertManager:

    def __init__(
        self,
        *,
        store: InMemoryAlertStore | None = None,
        escalation_policies: dict[
            AlertSeverity,
            EscalationPolicy,
        ] | None = None,
    ) -> None:

        self.store = (
            store
            or InMemoryAlertStore()
        )

        self.rules: dict[
            str,
            AlertRule,
        ] = {}

        self.handlers: dict[
            AlertChannel,
            list[AlertCallback],
        ] = {
            channel: []
            for channel in AlertChannel
        }

        self.escalation_policies = (
            escalation_policies
            or DEFAULT_ESCALATION_POLICIES
        )

        self._deduplication: dict[
            str,
            tuple[str, datetime],
        ] = {}

        self._escalation_tasks: dict[
            str,
            asyncio.Task[None],
        ] = {}

        self._lock = asyncio.Lock()

    # -----------------------------------------------------
    # Rules
    # -----------------------------------------------------

    def register_rule(
        self,
        rule: AlertRule,
    ) -> None:

        self.rules[
            rule.rule_id
        ] = rule

    def remove_rule(
        self,
        rule_id: str,
    ) -> None:

        self.rules.pop(
            rule_id,
            None,
        )

    # -----------------------------------------------------
    # Handlers
    # -----------------------------------------------------

    def register_handler(
        self,
        channel: AlertChannel,
        handler: AlertCallback,
    ) -> None:

        self.handlers[
            channel
        ].append(
            handler
        )

    # -----------------------------------------------------
    # Metric evaluation
    # -----------------------------------------------------

    async def evaluate_metric(
        self,
        *,
        metric_name: str,
        value: float,
        asset_id: str | None = None,
        site_id: str | None = None,
        sensor_id: str | None = None,
        unit: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Alert]:

        alerts: list[Alert] = []

        for rule in self.rules.values():

            if (
                not rule.enabled
                or rule.metric_name
                != metric_name
            ):
                continue

            if not rule.matches(value):
                continue

            message = (
                rule.message_template
                or (
                    f"{metric_name}={value} "
                    f"triggered rule "
                    f"{rule.name}: "
                    f"{rule.operator} "
                    f"{rule.threshold}"
                )
            )

            dedupe_key = (
                f"{rule.rule_id}:"
                f"{site_id or '-'}:"
                f"{asset_id or '-'}:"
                f"{sensor_id or '-'}"
            )

            alert = Alert(
                title=rule.name,
                message=message,
                severity=rule.severity,
                category=rule.category,

                asset_id=asset_id,
                site_id=site_id,
                sensor_id=sensor_id,

                metric_name=metric_name,
                metric_value=value,
                threshold=rule.threshold,
                unit=unit,

                source=(
                    f"rule:{rule.rule_id}"
                ),

                deduplication_key=(
                    dedupe_key
                ),

                metadata={
                    **rule.metadata,
                    **(metadata or {}),
                },
            )

            emitted = await self.emit(
                alert,
                channels=rule.channels,
                cooldown_seconds=(
                    rule.cooldown_seconds
                ),
            )

            alerts.append(
                emitted
            )

        return alerts

    # -----------------------------------------------------
    # Emit alert
    # -----------------------------------------------------

    async def emit(
        self,
        alert: Alert,
        *,
        channels: tuple[
            AlertChannel,
            ...
        ] | None = None,
        cooldown_seconds: int = 300,
    ) -> Alert:

        if alert.deduplication_key:
            existing = (
                await self._handle_deduplication(
                    alert,
                    cooldown_seconds,
                )
            )

            if existing is not None:
                return existing

        await self.store.save(
            alert
        )

        await self._dispatch(
            alert,
            channels
            or self._default_channels(
                alert.severity
            ),
        )

        self._schedule_escalation(
            alert
        )

        return alert

    # -----------------------------------------------------
    # Deduplication
    # -----------------------------------------------------

    async def _handle_deduplication(
        self,
        alert: Alert,
        cooldown_seconds: int,
    ) -> Alert | None:

        key = alert.deduplication_key

        if key is None:
            return None

        async with self._lock:

            existing_data = (
                self._deduplication.get(
                    key
                )
            )

            now = utc_now()

            if existing_data:
                (
                    alert_id,
                    last_seen,
                ) = existing_data

                if (
                    now - last_seen
                    < timedelta(
                        seconds=(
                            cooldown_seconds
                        )
                    )
                ):
                    existing = (
                        await self.store.get(
                            alert_id
                        )
                    )

                    if (
                        existing is not None
                        and existing.status
                        != AlertStatus.RESOLVED
                    ):
                        existing.occurrence_count += 1
                        existing.updated_at = now

                        existing.metric_value = (
                            alert.metric_value
                        )

                        existing.message = (
                            alert.message
                        )

                        existing.metadata.update(
                            alert.metadata
                        )

                        await self.store.save(
                            existing
                        )

                        self._deduplication[
                            key
                        ] = (
                            alert_id,
                            now,
                        )

                        return existing

            self._deduplication[
                key
            ] = (
                alert.alert_id,
                now,
            )

        return None

    # -----------------------------------------------------
    # Dispatch
    # -----------------------------------------------------

    async def _dispatch(
        self,
        alert: Alert,
        channels: tuple[
            AlertChannel,
            ...
        ],
    ) -> None:

        for channel in channels:

            if channel == AlertChannel.LOG:
                self._log_alert(
                    alert
                )

            callbacks = self.handlers.get(
                channel,
                [],
            )

            for callback in callbacks:
                try:
                    result = callback(
                        alert
                    )

                    if inspect.isawaitable(
                        result
                    ):
                        await result

                except Exception:
                    logger.exception(
                        "Alert handler failed: "
                        "channel=%s alert=%s",
                        channel.value,
                        alert.alert_id,
                    )

    # -----------------------------------------------------
    # Default channels
    # -----------------------------------------------------

    @staticmethod
    def _default_channels(
        severity: AlertSeverity,
    ) -> tuple[
        AlertChannel,
        ...
    ]:

        if severity in {
            AlertSeverity.EMERGENCY,
            AlertSeverity.CRITICAL,
        }:
            return (
                AlertChannel.LOG,
                AlertChannel.EMAIL,
                AlertChannel.SMS,
                AlertChannel.PUSH,
                AlertChannel.WEBHOOK,
                AlertChannel.KAFKA,
                AlertChannel.WEBSOCKET,
            )

        if severity == AlertSeverity.HIGH:
            return (
                AlertChannel.LOG,
                AlertChannel.EMAIL,
                AlertChannel.PUSH,
                AlertChannel.KAFKA,
                AlertChannel.WEBSOCKET,
            )

        if severity == AlertSeverity.WARNING:
            return (
                AlertChannel.LOG,
                AlertChannel.PUSH,
                AlertChannel.WEBSOCKET,
            )

        return (
            AlertChannel.LOG,
            AlertChannel.WEBSOCKET,
        )

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    @staticmethod
    def _log_alert(
        alert: Alert,
    ) -> None:

        message = (
            "industrial_alert "
            "id=%s severity=%s "
            "category=%s asset=%s "
            "title=%s message=%s"
        )

        args = (
            alert.alert_id,
            alert.severity.value,
            alert.category.value,
            alert.asset_id,
            alert.title,
            alert.message,
        )

        if alert.severity in {
            AlertSeverity.EMERGENCY,
            AlertSeverity.CRITICAL,
        }:
            logger.critical(
                message,
                *args,
            )

        elif alert.severity == AlertSeverity.HIGH:
            logger.error(
                message,
                *args,
            )

        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(
                message,
                *args,
            )

        else:
            logger.info(
                message,
                *args,
            )

    # -----------------------------------------------------
    # Acknowledge
    # -----------------------------------------------------

    async def acknowledge(
        self,
        alert_id: str,
        *,
        user_id: str,
    ) -> Alert:

        alert = await self._require_alert(
            alert_id
        )

        if alert.status == AlertStatus.RESOLVED:
            raise ValueError(
                "Resolved alert cannot be "
                "acknowledged."
            )

        alert.status = (
            AlertStatus.ACKNOWLEDGED
        )

        alert.acknowledged_at = utc_now()
        alert.acknowledged_by = user_id
        alert.updated_at = utc_now()

        await self.store.save(
            alert
        )

        self._cancel_escalation(
            alert_id
        )

        return alert

    # -----------------------------------------------------
    # Resolve
    # -----------------------------------------------------

    async def resolve(
        self,
        alert_id: str,
        *,
        user_id: str,
        resolution: str | None = None,
    ) -> Alert:

        alert = await self._require_alert(
            alert_id
        )

        alert.status = (
            AlertStatus.RESOLVED
        )

        alert.resolved_at = utc_now()
        alert.resolved_by = user_id
        alert.updated_at = utc_now()

        if resolution:
            alert.metadata[
                "resolution"
            ] = resolution

        await self.store.save(
            alert
        )

        self._cancel_escalation(
            alert_id
        )

        if alert.deduplication_key:
            async with self._lock:
                current = (
                    self._deduplication.get(
                        alert.deduplication_key
                    )
                )

                if (
                    current is not None
                    and current[0]
                    == alert.alert_id
                ):
                    self._deduplication.pop(
                        alert.deduplication_key,
                        None,
                    )

        return alert

    # -----------------------------------------------------
    # Manual escalation
    # -----------------------------------------------------

    async def escalate(
        self,
        alert_id: str,
        *,
        severity: AlertSeverity | None = None,
    ) -> Alert:

        alert = await self._require_alert(
            alert_id
        )

        if alert.status == AlertStatus.RESOLVED:
            return alert

        policy = (
            self.escalation_policies.get(
                alert.severity
            )
        )

        if severity is not None:
            alert.severity = severity

        elif (
            policy is not None
            and policy.target_severity
            is not None
        ):
            alert.severity = (
                policy.target_severity
            )

        alert.status = (
            AlertStatus.ESCALATED
        )

        alert.escalated_at = utc_now()
        alert.updated_at = utc_now()

        await self.store.save(
            alert
        )

        channels = (
            policy.channels
            if policy is not None
            else self._default_channels(
                alert.severity
            )
        )

        await self._dispatch(
            alert,
            channels,
        )

        return alert

    # -----------------------------------------------------
    # Escalation scheduler
    # -----------------------------------------------------

    def _schedule_escalation(
        self,
        alert: Alert,
    ) -> None:

        policy = (
            self.escalation_policies.get(
                alert.severity
            )
        )

        if policy is None:
            return

        self._cancel_escalation(
            alert.alert_id
        )

        task = asyncio.create_task(
            self._escalate_after_delay(
                alert.alert_id,
                policy,
            )
        )

        self._escalation_tasks[
            alert.alert_id
        ] = task

    async def _escalate_after_delay(
        self,
        alert_id: str,
        policy: EscalationPolicy,
    ) -> None:

        try:
            await asyncio.sleep(
                policy.escalate_after_seconds
            )

            alert = await self.store.get(
                alert_id
            )

            if alert is None:
                return

            if alert.status in {
                AlertStatus.ACKNOWLEDGED,
                AlertStatus.RESOLVED,
                AlertStatus.SUPPRESSED,
            }:
                return

            await self.escalate(
                alert_id
            )

        except asyncio.CancelledError:
            return

        except Exception:
            logger.exception(
                "Alert escalation failed: %s",
                alert_id,
            )

        finally:
            self._escalation_tasks.pop(
                alert_id,
                None,
            )

    def _cancel_escalation(
        self,
        alert_id: str,
    ) -> None:

        task = (
            self._escalation_tasks.pop(
                alert_id,
                None,
            )
        )

        if (
            task is not None
            and not task.done()
        ):
            task.cancel()

    # -----------------------------------------------------
    # Query
    # -----------------------------------------------------

    async def get(
        self,
        alert_id: str,
    ) -> Alert | None:

        return await self.store.get(
            alert_id
        )

    async def active_alerts(
        self,
        *,
        asset_id: str | None = None,
    ) -> list[Alert]:

        alerts = await self.store.list(
            asset_id=asset_id
        )

        return [
            alert
            for alert in alerts
            if alert.status
            not in {
                AlertStatus.RESOLVED,
                AlertStatus.SUPPRESSED,
            }
        ]

    async def critical_alerts(
        self,
    ) -> list[Alert]:

        alerts = await self.active_alerts()

        return [
            alert
            for alert in alerts
            if alert.severity
            in {
                AlertSeverity.CRITICAL,
                AlertSeverity.EMERGENCY,
            }
        ]

    async def _require_alert(
        self,
        alert_id: str,
    ) -> Alert:

        alert = await self.store.get(
            alert_id
        )

        if alert is None:
            raise KeyError(
                f"Alert not found: {alert_id}"
            )

        return alert

    # -----------------------------------------------------
    # Shutdown
    # -----------------------------------------------------

    async def shutdown(
        self,
    ) -> None:

        tasks = list(
            self._escalation_tasks.values()
        )

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        self._escalation_tasks.clear()


# =========================================================
# Industrial alert helpers
# =========================================================

async def create_anomaly_alert(
    manager: AlertManager,
    *,
    asset_id: str,
    anomaly_score: float,
    anomaly_type: str,
    prediction_id: str | None = None,
) -> Alert:

    severity = (
        AlertSeverity.CRITICAL
        if anomaly_score >= 0.95
        else AlertSeverity.HIGH
        if anomaly_score >= 0.80
        else AlertSeverity.WARNING
    )

    return await manager.emit(
        Alert(
            title="Industrial anomaly detected",

            message=(
                f"{anomaly_type} anomaly detected "
                f"for asset {asset_id} "
                f"(score={anomaly_score:.3f})."
            ),

            severity=severity,
            category=AlertCategory.ANOMALY,

            asset_id=asset_id,
            prediction_id=prediction_id,

            metric_name="anomaly_score",
            metric_value=anomaly_score,

            source="anomaly_engine",

            deduplication_key=(
                f"anomaly:{asset_id}:"
                f"{anomaly_type}"
            ),

            metadata={
                "anomaly_type":
                    anomaly_type,
            },
        )
    )


async def create_failure_alert(
    manager: AlertManager,
    *,
    asset_id: str,
    failure_probability: float,
    failure_mode: str,
    prediction_id: str | None = None,
) -> Alert:

    severity = (
        AlertSeverity.CRITICAL
        if failure_probability >= 0.90
        else AlertSeverity.HIGH
        if failure_probability >= 0.70
        else AlertSeverity.WARNING
    )

    return await manager.emit(
        Alert(
            title="Failure risk detected",

            message=(
                f"Predicted failure mode "
                f"'{failure_mode}' for "
                f"{asset_id}; probability="
                f"{failure_probability:.1%}."
            ),

            severity=severity,

            category=(
                AlertCategory
                .FAILURE_PREDICTION
            ),

            asset_id=asset_id,
            prediction_id=prediction_id,

            metric_name=(
                "failure_probability"
            ),

            metric_value=(
                failure_probability
            ),

            source="failure_prediction",

            deduplication_key=(
                f"failure:{asset_id}:"
                f"{failure_mode}"
            ),

            metadata={
                "failure_mode":
                    failure_mode,
            },
        )
    )


async def create_rul_alert(
    manager: AlertManager,
    *,
    asset_id: str,
    rul_hours: float,
    prediction_id: str | None = None,
) -> Alert:

    severity = (
        AlertSeverity.CRITICAL
        if rul_hours <= 24
        else AlertSeverity.HIGH
        if rul_hours <= 72
        else AlertSeverity.WARNING
    )

    return await manager.emit(
        Alert(
            title="Low remaining useful life",

            message=(
                f"Asset {asset_id} has "
                f"estimated RUL of "
                f"{rul_hours:.1f} hours."
            ),

            severity=severity,
            category=AlertCategory.RUL,

            asset_id=asset_id,
            prediction_id=prediction_id,

            metric_name="rul_hours",
            metric_value=rul_hours,
            unit="hours",

            source="rul_prediction",

            deduplication_key=(
                f"rul:{asset_id}"
            ),
        )
    )


async def create_safety_alert(
    manager: AlertManager,
    *,
    asset_id: str | None,
    category: AlertCategory,
    message: str,
    incident_id: str | None = None,
    site_id: str | None = None,
    emergency: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Alert:

    return await manager.emit(
        Alert(
            title=(
                "Emergency safety event"
                if emergency
                else "Industrial safety event"
            ),

            message=message,

            severity=(
                AlertSeverity.EMERGENCY
                if emergency
                else AlertSeverity.CRITICAL
            ),

            category=category,

            asset_id=asset_id,
            site_id=site_id,
            incident_id=incident_id,

            source="safety_engine",

            deduplication_key=(
                f"safety:{site_id}:"
                f"{asset_id}:"
                f"{category.value}"
            ),

            metadata=(
                metadata or {}
            ),
        ),
        cooldown_seconds=30,
    )


# =========================================================
# Default rules
# =========================================================

def register_default_rules(
    manager: AlertManager,
) -> None:

    rules = [
        AlertRule(
            rule_id="TEMP-HIGH",
            name="High temperature",
            category=(
                AlertCategory.TEMPERATURE
            ),
            severity=(
                AlertSeverity.HIGH
            ),
            metric_name="temperature_c",
            operator=">=",
            threshold=85.0,
            cooldown_seconds=300,
        ),

        AlertRule(
            rule_id="TEMP-CRITICAL",
            name="Critical temperature",
            category=(
                AlertCategory.OVERHEATING
            ),
            severity=(
                AlertSeverity.CRITICAL
            ),
            metric_name="temperature_c",
            operator=">=",
            threshold=105.0,
            cooldown_seconds=60,
            channels=(
                AlertChannel.LOG,
                AlertChannel.EMAIL,
                AlertChannel.SMS,
                AlertChannel.PUSH,
                AlertChannel.KAFKA,
                AlertChannel.WEBSOCKET,
            ),
        ),

        AlertRule(
            rule_id="VIBRATION-HIGH",
            name="High vibration",
            category=(
                AlertCategory.VIBRATION
            ),
            severity=(
                AlertSeverity.HIGH
            ),
            metric_name="vibration_rms",
            operator=">=",
            threshold=7.1,
            cooldown_seconds=300,
        ),

        AlertRule(
            rule_id="RUL-CRITICAL",
            name="Critical RUL",
            category=AlertCategory.RUL,
            severity=(
                AlertSeverity.CRITICAL
            ),
            metric_name="rul_hours",
            operator="<=",
            threshold=24.0,
            cooldown_seconds=1800,
        ),

        AlertRule(
            rule_id="FAILURE-RISK",
            name="High failure probability",
            category=(
                AlertCategory
                .FAILURE_PREDICTION
            ),
            severity=(
                AlertSeverity.CRITICAL
            ),
            metric_name=(
                "failure_probability"
            ),
            operator=">=",
            threshold=0.90,
            cooldown_seconds=300,
        ),

        AlertRule(
            rule_id="ANOMALY-HIGH",
            name="Critical anomaly score",
            category=(
                AlertCategory.ANOMALY
            ),
            severity=(
                AlertSeverity.CRITICAL
            ),
            metric_name="anomaly_score",
            operator=">=",
            threshold=0.95,
            cooldown_seconds=120,
        ),
    ]

    for rule in rules:
        manager.register_rule(
            rule
        )


# =========================================================
# Global manager
# =========================================================

alert_manager = AlertManager()

register_default_rules(
    alert_manager
)


__all__ = [
    "AlertSeverity",
    "AlertCategory",
    "AlertStatus",
    "AlertChannel",
    "Alert",
    "AlertRule",
    "EscalationPolicy",
    "DEFAULT_ESCALATION_POLICIES",
    "AlertHandler",
    "InMemoryAlertStore",
    "AlertManager",
    "create_anomaly_alert",
    "create_failure_alert",
    "create_rul_alert",
    "create_safety_alert",
    "register_default_rules",
    "alert_manager",
]