# app/integrations/notifications/sms.py

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol
from urllib import error, request

logger = logging.getLogger(__name__)


class SMSPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SMSCategory(str, Enum):
    GENERAL = "general"
    SAFETY = "safety"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    FAILURE = "failure"
    ANOMALY = "anomaly"
    WORK_ORDER = "work_order"


@dataclass(slots=True)
class SMSConfig:
    endpoint: str
    api_key: str | None = None
    sender_id: str = "INDUSTRIAL_AI"

    timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0

    enabled: bool = True


@dataclass(slots=True)
class SMSMessage:
    body: str
    category: SMSCategory = SMSCategory.GENERAL
    priority: SMSPriority = SMSPriority.NORMAL
    metadata: dict[str, Any] | None = None


class SMSProvider(Protocol):

    async def send(
        self,
        *,
        phone_numbers: list[str],
        message: SMSMessage,
    ) -> bool:
        ...


class HTTPSMSProvider:
    """
    Generic HTTP SMS gateway.

    Can be replaced by adapters for:
    - Twilio
    - AWS SNS
    - Vonage
    - MessageBird
    - MSG91
    - Internal enterprise SMS gateway
    """

    def __init__(self, config: SMSConfig) -> None:
        self.config = config

    async def send(
        self,
        *,
        phone_numbers: list[str],
        message: SMSMessage,
    ) -> bool:

        if not self.config.enabled:
            logger.debug("SMS notifications disabled")
            return False

        numbers = [
            self._normalize_phone_number(number)
            for number in phone_numbers
            if number.strip()
        ]

        if not numbers:
            logger.warning(
                "SMS notification skipped: no recipients"
            )
            return False

        payload = {
            "recipients": numbers,
            "sender_id": self.config.sender_id,
            "message": message.body,
            "priority": message.priority.value,
            "category": message.category.value,
            "metadata": message.metadata or {},
        }

        return await self._send_with_retry(payload)

    async def _send_with_retry(
        self,
        payload: dict[str, Any],
    ) -> bool:

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.config.max_retries + 1,
        ):
            try:
                await asyncio.to_thread(
                    self._send_sync,
                    payload,
                )

                logger.info(
                    "SMS delivered to %s recipient(s)",
                    len(payload["recipients"]),
                )

                return True

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "SMS attempt %s/%s failed: %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )

                if attempt < self.config.max_retries:
                    await asyncio.sleep(
                        self.config.retry_delay * attempt
                    )

        logger.error(
            "SMS delivery failed after %s attempts: %s",
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
                        f"SMS gateway returned HTTP {status}"
                    )

        except error.HTTPError as exc:
            raise RuntimeError(
                f"SMS gateway HTTP error: {exc.code}"
            ) from exc

        except error.URLError as exc:
            raise ConnectionError(
                f"SMS gateway unavailable: {exc.reason}"
            ) from exc

    @staticmethod
    def _normalize_phone_number(
        number: str,
    ) -> str:

        number = number.strip()

        allowed = {
            character
            for character in number
            if character.isdigit() or character == "+"
        }

        normalized = "".join(
            character
            for character in number
            if character in allowed
        )

        if not normalized:
            raise ValueError(
                f"Invalid phone number: {number!r}"
            )

        return normalized


class SMSNotificationClient:
    """
    Industrial SMS notification service.

    Handles:
    - fire/smoke alerts
    - emergency incidents
    - overheating
    - safety violations
    - failure prediction
    - RUL warnings
    - anomalies
    - maintenance
    - work orders
    """

    def __init__(
        self,
        provider: SMSProvider,
    ) -> None:
        self.provider = provider

    async def send(
        self,
        *,
        phone_numbers: list[str],
        body: str,
        category: SMSCategory = SMSCategory.GENERAL,
        priority: SMSPriority = SMSPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> bool:

        if not body.strip():
            raise ValueError(
                "SMS message cannot be empty"
            )

        message = SMSMessage(
            body=body.strip(),
            category=category,
            priority=priority,
            metadata=metadata,
        )

        return await self.provider.send(
            phone_numbers=phone_numbers,
            message=message,
        )

    async def send_safety_alert(
        self,
        *,
        phone_numbers: list[str],
        site_id: str,
        asset_id: str,
        event_type: str,
        severity: str,
    ) -> bool:

        priority = self._severity_priority(severity)

        body = (
            f"SAFETY ALERT [{severity.upper()}] "
            f"{event_type} detected at site {site_id}, "
            f"asset {asset_id}. Check Safety Center immediately."
        )

        return await self.send(
            phone_numbers=phone_numbers,
            body=body,
            category=SMSCategory.SAFETY,
            priority=priority,
            metadata={
                "site_id": site_id,
                "asset_id": asset_id,
                "event_type": event_type,
                "severity": severity,
            },
        )

    async def send_fire_alert(
        self,
        *,
        phone_numbers: list[str],
        site_id: str,
        asset_id: str,
    ) -> bool:

        return await self.send(
            phone_numbers=phone_numbers,
            body=(
                f"CRITICAL FIRE ALERT: Fire detected at "
                f"{site_id}/{asset_id}. "
                "Follow emergency procedures immediately."
            ),
            category=SMSCategory.EMERGENCY,
            priority=SMSPriority.CRITICAL,
            metadata={
                "site_id": site_id,
                "asset_id": asset_id,
                "event_type": "fire",
            },
        )

    async def send_smoke_alert(
        self,
        *,
        phone_numbers: list[str],
        site_id: str,
        asset_id: str,
    ) -> bool:

        return await self.send(
            phone_numbers=phone_numbers,
            body=(
                f"SMOKE ALERT: Smoke detected at "
                f"{site_id}/{asset_id}. "
                "Inspect area immediately."
            ),
            category=SMSCategory.SAFETY,
            priority=SMSPriority.HIGH,
            metadata={
                "site_id": site_id,
                "asset_id": asset_id,
                "event_type": "smoke",
            },
        )

    async def send_overheating_alert(
        self,
        *,
        phone_numbers: list[str],
        site_id: str,
        asset_id: str,
        temperature: float,
        threshold: float,
    ) -> bool:

        return await self.send(
            phone_numbers=phone_numbers,
            body=(
                f"OVERHEATING: {asset_id} at {site_id} "
                f"reached {temperature:.1f}°C "
                f"(limit {threshold:.1f}°C)."
            ),
            category=SMSCategory.SAFETY,
            priority=SMSPriority.HIGH,
            metadata={
                "site_id": site_id,
                "asset_id": asset_id,
                "temperature": temperature,
                "threshold": threshold,
            },
        )

    async def send_emergency_alert(
        self,
        *,
        phone_numbers: list[str],
        incident_id: str,
        site_id: str,
        asset_id: str,
        emergency_type: str,
    ) -> bool:

        return await self.send(
            phone_numbers=phone_numbers,
            body=(
                f"EMERGENCY [{emergency_type.upper()}] "
                f"Incident {incident_id} at "
                f"{site_id}/{asset_id}. "
                "Immediate response required."
            ),
            category=SMSCategory.EMERGENCY,
            priority=SMSPriority.CRITICAL,
            metadata={
                "incident_id": incident_id,
                "site_id": site_id,
                "asset_id": asset_id,
                "emergency_type": emergency_type,
            },
        )

    async def send_failure_prediction(
        self,
        *,
        phone_numbers: list[str],
        asset_id: str,
        failure_probability: float,
        predicted_failure: str,
    ) -> bool:

        probability = self._normalize_probability(
            failure_probability
        )

        priority = (
            SMSPriority.CRITICAL
            if probability >= 0.90
            else SMSPriority.HIGH
        )

        return await self.send(
            phone_numbers=phone_numbers,
            body=(
                f"FAILURE RISK: {asset_id} may experience "
                f"{predicted_failure}. "
                f"Probability {probability:.0%}."
            ),
            category=SMSCategory.FAILURE,
            priority=priority,
            metadata={
                "asset_id": asset_id,
                "failure_probability": probability,
                "predicted_failure": predicted_failure,
            },
        )

    async def send_rul_warning(
        self,
        *,
        phone_numbers: list[str],
        asset_id: str,
        rul_hours: float,
    ) -> bool:

        rul = max(0.0, float(rul_hours))

        priority = (
            SMSPriority.CRITICAL
            if rul <= 24
            else SMSPriority.HIGH
            if rul <= 72
            else SMSPriority.NORMAL
        )

        return await self.send(
            phone_numbers=phone_numbers,
            body=(
                f"RUL WARNING: {asset_id} estimated "
                f"remaining useful life is {rul:.1f} hours."
            ),
            category=SMSCategory.MAINTENANCE,
            priority=priority,
            metadata={
                "asset_id": asset_id,
                "rul_hours": rul,
            },
        )

    async def send_anomaly_alert(
        self,
        *,
        phone_numbers: list[str],
        asset_id: str,
        sensor_type: str,
        anomaly_score: float,
    ) -> bool:

        score = self._normalize_probability(
            anomaly_score
        )

        return await self.send(
            phone_numbers=phone_numbers,
            body=(
                f"ANOMALY: {sensor_type} abnormality "
                f"detected on {asset_id}. "
                f"Risk score {score:.0%}."
            ),
            category=SMSCategory.ANOMALY,
            priority=(
                SMSPriority.HIGH
                if score >= 0.80
                else SMSPriority.NORMAL
            ),
            metadata={
                "asset_id": asset_id,
                "sensor_type": sensor_type,
                "anomaly_score": score,
            },
        )

    async def send_maintenance_alert(
        self,
        *,
        phone_numbers: list[str],
        asset_id: str,
        recommendation: str,
    ) -> bool:

        return await self.send(
            phone_numbers=phone_numbers,
            body=(
                f"MAINTENANCE: {asset_id}. "
                f"{recommendation}"
            ),
            category=SMSCategory.MAINTENANCE,
            priority=SMSPriority.NORMAL,
            metadata={
                "asset_id": asset_id,
            },
        )

    async def send_work_order(
        self,
        *,
        phone_numbers: list[str],
        work_order_id: str,
        asset_id: str,
        description: str,
        urgent: bool = False,
    ) -> bool:

        return await self.send(
            phone_numbers=phone_numbers,
            body=(
                f"WORK ORDER {work_order_id}: "
                f"{asset_id}. {description}"
            ),
            category=SMSCategory.WORK_ORDER,
            priority=(
                SMSPriority.HIGH
                if urgent
                else SMSPriority.NORMAL
            ),
            metadata={
                "work_order_id": work_order_id,
                "asset_id": asset_id,
                "urgent": urgent,
            },
        )

    @staticmethod
    def _normalize_probability(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(float(value), 1.0),
        )

    @staticmethod
    def _severity_priority(
        severity: str,
    ) -> SMSPriority:

        mapping = {
            "critical": SMSPriority.CRITICAL,
            "emergency": SMSPriority.CRITICAL,
            "high": SMSPriority.HIGH,
            "warning": SMSPriority.HIGH,
            "medium": SMSPriority.NORMAL,
            "low": SMSPriority.LOW,
            "info": SMSPriority.LOW,
        }

        return mapping.get(
            severity.strip().lower(),
            SMSPriority.NORMAL,
        )


def create_sms_client(
    *,
    endpoint: str,
    api_key: str | None = None,
    sender_id: str = "INDUSTRIAL_AI",
    timeout: float = 10.0,
    max_retries: int = 3,
    enabled: bool = True,
) -> SMSNotificationClient:
    """Create the configured application SMS client."""

    config = SMSConfig(
        endpoint=endpoint,
        api_key=api_key,
        sender_id=sender_id,
        timeout=timeout,
        max_retries=max_retries,
        enabled=enabled,
    )

    provider = HTTPSMSProvider(config)

    return SMSNotificationClient(provider)