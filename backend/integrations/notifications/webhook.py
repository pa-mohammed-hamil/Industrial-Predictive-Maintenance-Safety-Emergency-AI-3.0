# app/integrations/notifications/webhook.py

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from urllib import error, request

logger = logging.getLogger(__name__)


class WebhookEventType(str, Enum):
    TELEMETRY = "telemetry"
    ANOMALY = "anomaly"
    FAILURE_PREDICTION = "failure_prediction"
    RUL_WARNING = "rul_warning"

    SAFETY = "safety"
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    MOTION = "motion"
    RESTRICTED_ZONE = "restricted_zone"

    INCIDENT = "incident"
    EMERGENCY = "emergency"

    MAINTENANCE = "maintenance"
    WORK_ORDER = "work_order"

    COST_ALERT = "cost_alert"
    REPORT_READY = "report_ready"

    SYSTEM = "system"


class WebhookPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class WebhookConfig:
    url: str

    secret: str | None = None
    api_key: str | None = None

    timeout: float = 10.0

    max_retries: int = 3
    retry_delay: float = 1.0

    enabled: bool = True

    user_agent: str = "industrial-ai-webhook/1.0"

    extra_headers: dict[str, str] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class WebhookEvent:
    event_type: WebhookEventType

    data: dict[str, Any]

    priority: WebhookPriority = WebhookPriority.NORMAL

    event_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    timestamp: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    source: str = "industrial-ai"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class WebhookResponse:
    success: bool

    status_code: int | None = None
    response_body: str | None = None

    attempts: int = 0

    error: str | None = None


class WebhookClient:
    """
    Reliable outbound webhook client.

    Supports:
    - JSON webhook delivery
    - retries
    - exponential backoff
    - HMAC-SHA256 signatures
    - API-key authentication
    - custom headers
    - idempotency/event IDs
    - safety/emergency helpers
    - maintenance events
    - prediction/anomaly events
    """

    def __init__(
        self,
        config: WebhookConfig,
    ) -> None:

        self.config = config

        if not config.url:
            raise ValueError(
                "Webhook URL cannot be empty"
            )

    async def send_event(
        self,
        event: WebhookEvent,
    ) -> WebhookResponse:

        if not self.config.enabled:
            logger.debug(
                "Webhook disabled event=%s",
                event.event_type.value,
            )

            return WebhookResponse(
                success=False,
                error="Webhook disabled",
            )

        payload = self._serialize_event(event)

        return await self._send_with_retry(
            payload=payload,
            event=event,
        )

    async def emit(
        self,
        *,
        event_type: WebhookEventType,
        data: Mapping[str, Any],
        priority: WebhookPriority = WebhookPriority.NORMAL,
        metadata: Mapping[str, Any] | None = None,
    ) -> WebhookResponse:

        event = WebhookEvent(
            event_type=event_type,
            data=dict(data),
            priority=priority,
            metadata=dict(metadata or {}),
        )

        return await self.send_event(event)

    async def _send_with_retry(
        self,
        *,
        payload: bytes,
        event: WebhookEvent,
    ) -> WebhookResponse:

        last_error: str | None = None

        for attempt in range(
            1,
            self.config.max_retries + 1,
        ):

            try:

                status, response_body = (
                    await asyncio.to_thread(
                        self._send_sync,
                        payload,
                        event,
                    )
                )

                logger.info(
                    "Webhook delivered "
                    "event=%s id=%s status=%s",
                    event.event_type.value,
                    event.event_id,
                    status,
                )

                return WebhookResponse(
                    success=True,
                    status_code=status,
                    response_body=response_body,
                    attempts=attempt,
                )

            except Exception as exc:

                last_error = str(exc)

                logger.warning(
                    "Webhook attempt %s/%s failed "
                    "event=%s id=%s error=%s",
                    attempt,
                    self.config.max_retries,
                    event.event_type.value,
                    event.event_id,
                    exc,
                )

                if attempt < self.config.max_retries:

                    delay = (
                        self.config.retry_delay
                        * (2 ** (attempt - 1))
                    )

                    await asyncio.sleep(delay)

        logger.error(
            "Webhook delivery failed "
            "event=%s id=%s attempts=%s",
            event.event_type.value,
            event.event_id,
            self.config.max_retries,
        )

        return WebhookResponse(
            success=False,
            attempts=self.config.max_retries,
            error=last_error,
        )

    def _send_sync(
        self,
        payload: bytes,
        event: WebhookEvent,
    ) -> tuple[int, str]:

        timestamp = str(int(time.time()))

        headers = self._build_headers(
            payload=payload,
            timestamp=timestamp,
            event=event,
        )

        req = request.Request(
            self.config.url,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:

            with request.urlopen(
                req,
                timeout=self.config.timeout,
            ) as response:

                status = response.getcode()

                response_body = (
                    response.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

                if not 200 <= status < 300:
                    raise RuntimeError(
                        f"Webhook returned HTTP {status}"
                    )

                return status, response_body

        except error.HTTPError as exc:

            body = ""

            try:
                body = (
                    exc.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )
            except Exception:
                pass

            raise RuntimeError(
                f"Webhook HTTP error "
                f"{exc.code}: {body}"
            ) from exc

        except error.URLError as exc:

            raise ConnectionError(
                f"Webhook unavailable: {exc.reason}"
            ) from exc

    def _build_headers(
        self,
        *,
        payload: bytes,
        timestamp: str,
        event: WebhookEvent,
    ) -> dict[str, str]:

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.config.user_agent,
            "X-Webhook-Event": event.event_type.value,
            "X-Webhook-Event-ID": event.event_id,
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-Priority": event.priority.value,
        }

        if self.config.api_key:
            headers["Authorization"] = (
                f"Bearer {self.config.api_key}"
            )

        if self.config.secret:

            signature = self._generate_signature(
                payload=payload,
                timestamp=timestamp,
            )

            headers["X-Webhook-Signature"] = (
                f"sha256={signature}"
            )

        headers.update(
            self.config.extra_headers
        )

        return headers

    def _generate_signature(
        self,
        *,
        payload: bytes,
        timestamp: str,
    ) -> str:

        if not self.config.secret:
            raise ValueError(
                "Webhook secret not configured"
            )

        signed_payload = (
            timestamp.encode("utf-8")
            + b"."
            + payload
        )

        return hmac.new(
            self.config.secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _serialize_event(
        event: WebhookEvent,
    ) -> bytes:

        payload = {
            "id": event.event_id,
            "type": event.event_type.value,
            "source": event.source,
            "timestamp": event.timestamp,
            "priority": event.priority.value,
            "data": event.data,
            "metadata": event.metadata,
        }

        return json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")

    # --------------------------------------------------
    # Safety
    # --------------------------------------------------

    async def send_safety_alert(
        self,
        *,
        site_id: str,
        asset_id: str,
        event_type: str,
        severity: str,
        description: str,
        confidence: float | None = None,
    ) -> WebhookResponse:

        priority = self._severity_priority(
            severity
        )

        return await self.emit(
            event_type=WebhookEventType.SAFETY,
            priority=priority,
            data={
                "site_id": site_id,
                "asset_id": asset_id,
                "event_type": event_type,
                "severity": severity,
                "description": description,
                "confidence": confidence,
            },
        )

    async def send_fire_alert(
        self,
        *,
        site_id: str,
        asset_id: str,
        confidence: float,
        camera_id: str | None = None,
    ) -> WebhookResponse:

        return await self.emit(
            event_type=WebhookEventType.FIRE,
            priority=WebhookPriority.CRITICAL,
            data={
                "site_id": site_id,
                "asset_id": asset_id,
                "confidence": self._probability(
                    confidence
                ),
                "camera_id": camera_id,
            },
        )

    async def send_smoke_alert(
        self,
        *,
        site_id: str,
        asset_id: str,
        confidence: float,
    ) -> WebhookResponse:

        return await self.emit(
            event_type=WebhookEventType.SMOKE,
            priority=WebhookPriority.HIGH,
            data={
                "site_id": site_id,
                "asset_id": asset_id,
                "confidence": self._probability(
                    confidence
                ),
            },
        )

    async def send_overheating_alert(
        self,
        *,
        site_id: str,
        asset_id: str,
        temperature: float,
        threshold: float,
    ) -> WebhookResponse:

        return await self.emit(
            event_type=WebhookEventType.OVERHEATING,
            priority=WebhookPriority.HIGH,
            data={
                "site_id": site_id,
                "asset_id": asset_id,
                "temperature": temperature,
                "threshold": threshold,
            },
        )

    # --------------------------------------------------
    # Emergency
    # --------------------------------------------------

    async def send_emergency(
        self,
        *,
        incident_id: str,
        site_id: str,
        asset_id: str,
        emergency_type: str,
        severity: str,
        description: str,
    ) -> WebhookResponse:

        return await self.emit(
            event_type=WebhookEventType.EMERGENCY,
            priority=WebhookPriority.CRITICAL,
            data={
                "incident_id": incident_id,
                "site_id": site_id,
                "asset_id": asset_id,
                "emergency_type": emergency_type,
                "severity": severity,
                "description": description,
            },
        )

    # --------------------------------------------------
    # Failure prediction
    # --------------------------------------------------

    async def send_failure_prediction(
        self,
        *,
        asset_id: str,
        failure_probability: float,
        predicted_failure: str,
        model_version: str | None = None,
    ) -> WebhookResponse:

        probability = self._probability(
            failure_probability
        )

        priority = (
            WebhookPriority.CRITICAL
            if probability >= 0.90
            else WebhookPriority.HIGH
            if probability >= 0.70
            else WebhookPriority.NORMAL
        )

        return await self.emit(
            event_type=(
                WebhookEventType.FAILURE_PREDICTION
            ),
            priority=priority,
            data={
                "asset_id": asset_id,
                "failure_probability": probability,
                "predicted_failure": predicted_failure,
                "model_version": model_version,
            },
        )

    # --------------------------------------------------
    # RUL
    # --------------------------------------------------

    async def send_rul_warning(
        self,
        *,
        asset_id: str,
        rul_hours: float,
        confidence: float | None = None,
    ) -> WebhookResponse:

        rul = max(
            0.0,
            float(rul_hours),
        )

        if rul <= 24:
            priority = WebhookPriority.CRITICAL
        elif rul <= 72:
            priority = WebhookPriority.HIGH
        else:
            priority = WebhookPriority.NORMAL

        return await self.emit(
            event_type=WebhookEventType.RUL_WARNING,
            priority=priority,
            data={
                "asset_id": asset_id,
                "rul_hours": rul,
                "confidence": (
                    self._probability(confidence)
                    if confidence is not None
                    else None
                ),
            },
        )

    # --------------------------------------------------
    # Anomaly
    # --------------------------------------------------

    async def send_anomaly(
        self,
        *,
        asset_id: str,
        sensor_type: str,
        anomaly_score: float,
        value: float | None = None,
    ) -> WebhookResponse:

        score = self._probability(
            anomaly_score
        )

        priority = (
            WebhookPriority.CRITICAL
            if score >= 0.95
            else WebhookPriority.HIGH
            if score >= 0.80
            else WebhookPriority.NORMAL
        )

        return await self.emit(
            event_type=WebhookEventType.ANOMALY,
            priority=priority,
            data={
                "asset_id": asset_id,
                "sensor_type": sensor_type,
                "anomaly_score": score,
                "value": value,
            },
        )

    # --------------------------------------------------
    # Maintenance
    # --------------------------------------------------

    async def send_maintenance_event(
        self,
        *,
        asset_id: str,
        maintenance_type: str,
        recommendation: str,
        scheduled_at: str | None = None,
    ) -> WebhookResponse:

        return await self.emit(
            event_type=WebhookEventType.MAINTENANCE,
            data={
                "asset_id": asset_id,
                "maintenance_type": maintenance_type,
                "recommendation": recommendation,
                "scheduled_at": scheduled_at,
            },
        )

    # --------------------------------------------------
    # Work order
    # --------------------------------------------------

    async def send_work_order(
        self,
        *,
        work_order_id: str,
        asset_id: str,
        status: str,
        description: str,
        urgent: bool = False,
    ) -> WebhookResponse:

        return await self.emit(
            event_type=WebhookEventType.WORK_ORDER,
            priority=(
                WebhookPriority.HIGH
                if urgent
                else WebhookPriority.NORMAL
            ),
            data={
                "work_order_id": work_order_id,
                "asset_id": asset_id,
                "status": status,
                "description": description,
                "urgent": urgent,
            },
        )

    # --------------------------------------------------
    # Cost intelligence
    # --------------------------------------------------

    async def send_cost_alert(
        self,
        *,
        asset_id: str,
        estimated_cost: float,
        currency: str,
        cost_type: str,
        recommendation: str | None = None,
    ) -> WebhookResponse:

        return await self.emit(
            event_type=WebhookEventType.COST_ALERT,
            data={
                "asset_id": asset_id,
                "estimated_cost": round(
                    max(0.0, float(estimated_cost)),
                    2,
                ),
                "currency": currency,
                "cost_type": cost_type,
                "recommendation": recommendation,
            },
        )

    # --------------------------------------------------
    # Incident
    # --------------------------------------------------

    async def send_incident(
        self,
        *,
        incident_id: str,
        site_id: str,
        asset_id: str,
        incident_type: str,
        severity: str,
    ) -> WebhookResponse:

        return await self.emit(
            event_type=WebhookEventType.INCIDENT,
            priority=self._severity_priority(
                severity
            ),
            data={
                "incident_id": incident_id,
                "site_id": site_id,
                "asset_id": asset_id,
                "incident_type": incident_type,
                "severity": severity,
            },
        )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    @staticmethod
    def _probability(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(float(value), 1.0),
        )

    @staticmethod
    def _severity_priority(
        severity: str,
    ) -> WebhookPriority:

        mapping = {
            "critical": WebhookPriority.CRITICAL,
            "emergency": WebhookPriority.CRITICAL,
            "high": WebhookPriority.HIGH,
            "warning": WebhookPriority.HIGH,
            "medium": WebhookPriority.NORMAL,
            "low": WebhookPriority.LOW,
            "info": WebhookPriority.LOW,
        }

        return mapping.get(
            severity.strip().lower(),
            WebhookPriority.NORMAL,
        )


def verify_webhook_signature(
    *,
    payload: bytes,
    timestamp: str,
    signature: str,
    secret: str,
) -> bool:
    """
    Verify an HMAC-SHA256 webhook signature.

    Expected header:

        X-Webhook-Signature: sha256=<digest>
    """

    if not payload:
        return False

    if not timestamp:
        return False

    if not signature:
        return False

    if not secret:
        return False

    if signature.startswith("sha256="):
        signature = signature[7:]

    signed_payload = (
        timestamp.encode("utf-8")
        + b"."
        + payload
    )

    expected = hmac.new(
        secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(
        expected,
        signature,
    )


def create_webhook_client(
    *,
    url: str,
    secret: str | None = None,
    api_key: str | None = None,
    timeout: float = 10.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    enabled: bool = True,
    extra_headers: dict[str, str] | None = None,
) -> WebhookClient:

    config = WebhookConfig(
        url=url,
        secret=secret,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        enabled=enabled,
        extra_headers=extra_headers or {},
    )

    return WebhookClient(config)