# app/integrations/notifications/push.py

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
from urllib import error, request

logger = logging.getLogger(__name__)


class PushPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class PushCategory(str, Enum):
    GENERAL = "general"
    SAFETY = "safety"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    FAILURE = "failure"
    ANOMALY = "anomaly"
    WORK_ORDER = "work_order"
    COST = "cost"


@dataclass(slots=True)
class PushMessage:
    title: str
    body: str

    category: PushCategory = PushCategory.GENERAL
    priority: PushPriority = PushPriority.NORMAL

    data: dict[str, Any] = field(default_factory=dict)

    image_url: str | None = None
    action_url: str | None = None


@dataclass(slots=True)
class PushConfig:
    endpoint: str

    api_key: str | None = None

    timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0

    enabled: bool = True


class PushProvider(Protocol):
    async def send(
        self,
        *,
        tokens: list[str],
        message: PushMessage,
    ) -> bool:
        ...


class HTTPPushProvider:
    """
    Generic HTTP push provider.

    Can sit behind:
    - Firebase gateway
    - OneSignal gateway
    - Internal notification service
    - Mobile push microservice
    """

    def __init__(self, config: PushConfig) -> None:
        self.config = config

    async def send(
        self,
        *,
        tokens: list[str],
        message: PushMessage,
    ) -> bool:

        if not self.config.enabled:
            logger.debug("Push notifications are disabled")
            return False

        if not tokens:
            logger.warning(
                "Push notification skipped: no device tokens"
            )
            return False

        payload = {
            "tokens": tokens,
            "notification": {
                "title": message.title,
                "body": message.body,
                "image_url": message.image_url,
            },
            "category": message.category.value,
            "priority": message.priority.value,
            "data": {
                **message.data,
                "action_url": message.action_url,
            },
        }

        return await self._send_with_retry(payload)

    async def _send_with_retry(
        self,
        payload: dict[str, Any],
    ) -> bool:

        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                await asyncio.to_thread(
                    self._send_sync,
                    payload,
                )

                return True

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Push attempt %s/%s failed: %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )

                if attempt < self.config.max_retries:
                    await asyncio.sleep(
                        self.config.retry_delay * attempt
                    )

        logger.error(
            "Push notification failed after %s attempts: %s",
            self.config.max_retries,
            last_error,
        )

        return False

    def _send_sync(
        self,
        payload: dict[str, Any],
    ) -> None:

        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = (
                f"Bearer {self.config.api_key}"
            )

        req = request.Request(
            self.config.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(
                req,
                timeout=self.config.timeout,
            ) as response:

                status = response.getcode()

                if not 200 <= status < 300:
                    raise RuntimeError(
                        f"Push provider returned HTTP {status}"
                    )

        except error.HTTPError as exc:
            raise RuntimeError(
                f"Push provider HTTP error: {exc.code}"
            ) from exc

        except error.URLError as exc:
            raise ConnectionError(
                f"Push provider unavailable: {exc.reason}"
            ) from exc


class PushNotificationClient:
    """
    Industrial push notification service.

    Handles:
    - safety events
    - emergency incidents
    - predictive maintenance
    - anomaly detection
    - failure prediction
    - work orders
    - cost alerts
    """

    def __init__(
        self,
        provider: PushProvider,
    ) -> None:
        self.provider = provider

    async def send(
        self,
        *,
        tokens: list[str],
        title: str,
        body: str,
        category: PushCategory = PushCategory.GENERAL,
        priority: PushPriority = PushPriority.NORMAL,
        data: dict[str, Any] | None = None,
        action_url: str | None = None,
        image_url: str | None = None,
    ) -> bool:

        message = PushMessage(
            title=title,
            body=body,
            category=category,
            priority=priority,
            data=data or {},
            action_url=action_url,
            image_url=image_url,
        )

        return await self.provider.send(
            tokens=tokens,
            message=message,
        )

    # -----------------------------------------------------
    # Safety
    # -----------------------------------------------------

    async def send_safety_alert(
        self,
        *,
        tokens: list[str],
        site_id: str,
        asset_id: str,
        event_type: str,
        severity: str,
        description: str,
    ) -> bool:

        priority = self._severity_priority(severity)

        return await self.send(
            tokens=tokens,
            title=f"Safety Alert: {event_type}",
            body=description,
            category=PushCategory.SAFETY,
            priority=priority,
            data={
                "site_id": site_id,
                "asset_id": asset_id,
                "event_type": event_type,
                "severity": severity,
            },
            action_url=f"/safety/{asset_id}",
        )

    # -----------------------------------------------------
    # Emergency
    # -----------------------------------------------------

    async def send_emergency_alert(
        self,
        *,
        tokens: list[str],
        incident_id: str,
        site_id: str,
        asset_id: str,
        emergency_type: str,
        description: str,
    ) -> bool:

        return await self.send(
            tokens=tokens,
            title=f"EMERGENCY: {emergency_type}",
            body=description,
            category=PushCategory.EMERGENCY,
            priority=PushPriority.CRITICAL,
            data={
                "incident_id": incident_id,
                "site_id": site_id,
                "asset_id": asset_id,
                "emergency_type": emergency_type,
            },
            action_url=f"/emergency/{incident_id}",
        )

    # -----------------------------------------------------
    # Predictive maintenance
    # -----------------------------------------------------

    async def send_maintenance_alert(
        self,
        *,
        tokens: list[str],
        site_id: str,
        asset_id: str,
        failure_probability: float,
        rul_hours: float | None,
        recommendation: str,
    ) -> bool:

        probability = max(
            0.0,
            min(float(failure_probability), 1.0),
        )

        if probability >= 0.85:
            priority = PushPriority.CRITICAL
        elif probability >= 0.65:
            priority = PushPriority.HIGH
        else:
            priority = PushPriority.NORMAL

        rul = (
            round(rul_hours, 2)
            if rul_hours is not None
            else None
        )

        return await self.send(
            tokens=tokens,
            title=f"Maintenance Alert: {asset_id}",
            body=recommendation,
            category=PushCategory.MAINTENANCE,
            priority=priority,
            data={
                "site_id": site_id,
                "asset_id": asset_id,
                "failure_probability": probability,
                "rul_hours": rul,
            },
            action_url=f"/assets/{asset_id}/maintenance",
        )

    # -----------------------------------------------------
    # Failure prediction
    # -----------------------------------------------------

    async def send_failure_prediction(
        self,
        *,
        tokens: list[str],
        asset_id: str,
        failure_probability: float,
        predicted_failure: str,
    ) -> bool:

        probability = max(
            0.0,
            min(float(failure_probability), 1.0),
        )

        return await self.send(
            tokens=tokens,
            title="Failure Risk Detected",
            body=(
                f"{asset_id}: {predicted_failure} "
                f"({probability:.1%} probability)"
            ),
            category=PushCategory.FAILURE,
            priority=(
                PushPriority.CRITICAL
                if probability >= 0.85
                else PushPriority.HIGH
            ),
            data={
                "asset_id": asset_id,
                "failure_probability": probability,
                "predicted_failure": predicted_failure,
            },
            action_url=f"/assets/{asset_id}",
        )

    # -----------------------------------------------------
    # Anomaly
    # -----------------------------------------------------

    async def send_anomaly_alert(
        self,
        *,
        tokens: list[str],
        asset_id: str,
        sensor_type: str,
        anomaly_score: float,
    ) -> bool:

        score = max(
            0.0,
            min(float(anomaly_score), 1.0),
        )

        return await self.send(
            tokens=tokens,
            title="Sensor Anomaly Detected",
            body=(
                f"{sensor_type} anomaly detected "
                f"on {asset_id}."
            ),
            category=PushCategory.ANOMALY,
            priority=(
                PushPriority.HIGH
                if score >= 0.8
                else PushPriority.NORMAL
            ),
            data={
                "asset_id": asset_id,
                "sensor_type": sensor_type,
                "anomaly_score": score,
            },
            action_url=f"/assets/{asset_id}/anomalies",
        )

    # -----------------------------------------------------
    # Work order
    # -----------------------------------------------------

    async def send_work_order(
        self,
        *,
        tokens: list[str],
        work_order_id: str,
        asset_id: str,
        description: str,
        urgent: bool = False,
    ) -> bool:

        return await self.send(
            tokens=tokens,
            title=f"Work Order {work_order_id}",
            body=description,
            category=PushCategory.WORK_ORDER,
            priority=(
                PushPriority.HIGH
                if urgent
                else PushPriority.NORMAL
            ),
            data={
                "work_order_id": work_order_id,
                "asset_id": asset_id,
                "urgent": urgent,
            },
            action_url=f"/work-orders/{work_order_id}",
        )

    # -----------------------------------------------------
    # Cost intelligence
    # -----------------------------------------------------

    async def send_cost_alert(
        self,
        *,
        tokens: list[str],
        asset_id: str,
        estimated_cost: float,
        currency: str = "USD",
        recommendation: str,
    ) -> bool:

        return await self.send(
            tokens=tokens,
            title=f"Cost Alert: {asset_id}",
            body=recommendation,
            category=PushCategory.COST,
            priority=PushPriority.NORMAL,
            data={
                "asset_id": asset_id,
                "estimated_cost": round(
                    float(estimated_cost),
                    2,
                ),
                "currency": currency,
            },
            action_url=f"/assets/{asset_id}/costs",
        )

    @staticmethod
    def _severity_priority(
        severity: str,
    ) -> PushPriority:

        severity = severity.strip().lower()

        mapping = {
            "critical": PushPriority.CRITICAL,
            "emergency": PushPriority.CRITICAL,
            "high": PushPriority.HIGH,
            "warning": PushPriority.HIGH,
            "medium": PushPriority.NORMAL,
            "low": PushPriority.LOW,
            "info": PushPriority.LOW,
        }

        return mapping.get(
            severity,
            PushPriority.NORMAL,
        )


def create_push_client(
    *,
    endpoint: str,
    api_key: str | None = None,
    timeout: float = 10.0,
    max_retries: int = 3,
    enabled: bool = True,
) -> PushNotificationClient:

    config = PushConfig(
        endpoint=endpoint,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        enabled=enabled,
    )

    provider = HTTPPushProvider(config)

    return PushNotificationClient(provider)