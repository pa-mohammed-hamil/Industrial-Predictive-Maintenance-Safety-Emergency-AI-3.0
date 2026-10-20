
"""
app/ai/ml/models/audit.py

Industrial AI Audit Models
==========================

Python 3.11+
Pydantic v2

Dependencies:
    pip install "pydantic>=2,<3"

Features:
- ML training and inference audit records
- Model registration and deployment auditing
- Asset and equipment change tracking
- Safety and emergency decision auditing
- User and service actor attribution
- Request, trace and correlation IDs
- Before/after change records
- Model version and prediction references
- Audit outcome and severity classification
- PII-aware metadata sanitization
- Hash-chained audit records
- JSON-compatible Pydantic schemas

SECURITY:
Hash chaining provides tamper-evidence only
when record order and chain anchors are
protected by trusted infrastructure.

It is not a substitute for an append-only
audit store, access controls, signatures,
retention policies or external verification.

Do not store raw passwords, API keys,
access tokens or sensitive personal data.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# ENUMS
# ============================================================

class AuditAction(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    MODEL_TRAIN = "model_train"
    MODEL_EVALUATE = "model_evaluate"
    MODEL_REGISTER = "model_register"
    MODEL_DEPLOY = "model_deploy"
    MODEL_ROLLBACK = "model_rollback"

    INFERENCE = "inference"
    ANOMALY_DETECTION = "anomaly_detection"
    FAILURE_PREDICTION = "failure_prediction"
    RUL_PREDICTION = "rul_prediction"

    ASSET_HEALTH_UPDATE = "asset_health_update"
    TELEMETRY_INGEST = "telemetry_ingest"

    SAFETY_ALERT = "safety_alert"
    EMERGENCY_EVENT = "emergency_event"

    INSPECTION = "inspection"
    MAINTENANCE_ACTION = "maintenance_action"

    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_DENIED = "access_denied"

    CONFIG_CHANGE = "config_change"
    PERMISSION_CHANGE = "permission_change"


class AuditEntityType(str, Enum):
    MODEL = "model"
    MODEL_VERSION = "model_version"
    PREDICTION = "prediction"
    ASSET = "asset"
    MACHINE = "machine"
    SENSOR = "sensor"
    TELEMETRY = "telemetry"
    ANOMALY = "anomaly"
    FAILURE = "failure"
    INSPECTION = "inspection"
    INCIDENT = "incident"
    MAINTENANCE = "maintenance"
    WORK_ORDER = "work_order"
    USER = "user"
    COMPANY = "company"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


class AuditActorType(str, Enum):
    USER = "user"
    SERVICE = "service"
    AGENT = "agent"
    SYSTEM = "system"
    SCHEDULER = "scheduler"


class AuditOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    PARTIAL = "partial"


class AuditSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# BASE MODEL
# ============================================================

class AuditBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# SANITIZATION
# ============================================================

class AuditSanitizer:
    """
    Basic recursive metadata redaction.

    This is defense in depth, not complete
    PII detection or DLP enforcement.
    """

    SENSITIVE_KEYS = {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "private_key",
        "client_secret",
        "session_id",
    }

    @classmethod
    def _normalize_key(
        cls,
        key: str,
    ) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            "_",
            key.lower(),
        ).strip("_")

    @classmethod
    def sanitize(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, Mapping):
            result = {}

            for key, item in value.items():
                name = str(key)
                normalized = cls._normalize_key(
                    name
                )

                if (
                    normalized in cls.SENSITIVE_KEYS
                    or normalized.endswith("_password")
                    or normalized.endswith("_secret")
                    or normalized.endswith("_token")
                    or normalized.endswith("_api_key")
                ):
                    result[name] = "[REDACTED]"
                else:
                    result[name] = cls.sanitize(
                        item
                    )

            return result

        if isinstance(value, (list, tuple)):
            return [
                cls.sanitize(item)
                for item in value
            ]

        return value


# ============================================================
# ACTOR
# ============================================================

class AuditActor(AuditBaseModel):
    actor_id: str = Field(
        min_length=1,
        max_length=200,
    )

    actor_type: AuditActorType

    display_name: str | None = None

    role: str | None = None

    service_name: str | None = None

    agent_name: str | None = None


# ============================================================
# REQUEST CONTEXT
# ============================================================

class AuditRequestContext(AuditBaseModel):
    request_id: str | None = None
    trace_id: str | None = None
    correlation_id: str | None = None

    source_service: str | None = None

    endpoint: str | None = None

    http_method: str | None = None

    ip_address: str | None = None

    user_agent: str | None = None


# ============================================================
# MODEL REFERENCE
# ============================================================

class AuditModelReference(AuditBaseModel):
    model_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None

    model_run_id: str | None = None
    prediction_id: str | None = None

    dataset_version: str | None = None

    feature_schema_version: str | None = None


# ============================================================
# CHANGE RECORD
# ============================================================

class AuditChange(AuditBaseModel):
    field_name: str = Field(
        min_length=1,
        max_length=300,
    )

    old_value: Any = None
    new_value: Any = None

    @model_validator(mode="after")
    def sanitize_values(self) -> "AuditChange":
        self.old_value = AuditSanitizer.sanitize(
            self.old_value
        )

        self.new_value = AuditSanitizer.sanitize(
            self.new_value
        )

        return self


# ============================================================
# AUDIT EVENT
# ============================================================

class AuditEvent(AuditBaseModel):
    audit_id: uuid.UUID = Field(
        default_factory=uuid.uuid4
    )

    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    action: AuditAction
    entity_type: AuditEntityType

    entity_id: str | None = None

    actor: AuditActor

    outcome: AuditOutcome = AuditOutcome.SUCCESS

    severity: AuditSeverity = AuditSeverity.INFO

    description: str = Field(
        min_length=1,
        max_length=2000,
    )

    asset_id: str | None = None
    equipment_id: str | None = None

    model_reference: (
        AuditModelReference | None
    ) = None

    request_context: (
        AuditRequestContext | None
    ) = None

    changes: list[AuditChange] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    previous_hash: str | None = None
    record_hash: str | None = None

    @field_validator("occurred_at")
    @classmethod
    def validate_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "occurred_at must be timezone-aware"
            )

        return value

    @field_validator("metadata")
    @classmethod
    def sanitize_metadata(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        return AuditSanitizer.sanitize(
            value
        )

    @field_validator("previous_hash", "record_hash")
    @classmethod
    def validate_hash(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        if not re.fullmatch(
            r"[a-f0-9]{64}",
            value,
        ):
            raise ValueError(
                "Expected SHA-256 hexadecimal hash"
            )

        return value

    def canonical_payload(
        self,
    ) -> str:
        """
        Deterministic serialization excluding
        record_hash itself.
        """
        data = self.model_dump(
            mode="json",
            exclude={"record_hash"},
        )

        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def compute_hash(
        self,
    ) -> str:
        return hashlib.sha256(
            self.canonical_payload().encode(
                "utf-8"
            )
        ).hexdigest()

    def seal(
        self,
        previous_hash: str | None = None,
    ) -> None:
        """
        Seal a record against its predecessor.

        The application must serialize chain
        writes transactionally to prevent
        concurrent forks.
        """
        self.previous_hash = previous_hash

        self.record_hash = self.compute_hash()

    def verify_hash(self) -> bool:
        if self.record_hash is None:
            return False

        return (
            self.compute_hash()
            == self.record_hash
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json"
        )


# ============================================================
# AUDIT EVENT FACTORY
# ============================================================

class AuditEventFactory:
    """
    Convenience factory for consistent
    application audit events.
    """

    @staticmethod
    def create(
        *,
        company_id: str,
        action: AuditAction,
        entity_type: AuditEntityType,
        actor_id: str,
        actor_type: AuditActorType,
        description: str,
        entity_id: str | None = None,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        outcome: AuditOutcome = (
            AuditOutcome.SUCCESS
        ),
        severity: AuditSeverity = (
            AuditSeverity.INFO
        ),
        model_reference: (
            AuditModelReference | None
        ) = None,
        request_context: (
            AuditRequestContext | None
        ) = None,
        changes: list[AuditChange] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        return AuditEvent(
            company_id=company_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            actor=AuditActor(
                actor_id=actor_id,
                actor_type=actor_type,
            ),
            description=description,
            asset_id=asset_id,
            equipment_id=equipment_id,
            outcome=outcome,
            severity=severity,
            model_reference=model_reference,
            request_context=request_context,
            changes=changes or [],
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def model_inference(
        *,
        company_id: str,
        actor_id: str,
        model_id: str,
        model_version: str,
        prediction_id: str,
        equipment_id: str,
        is_anomaly: bool,
        trace_id: str | None = None,
    ) -> AuditEvent:
        return AuditEventFactory.create(
            company_id=company_id,
            action=AuditAction.INFERENCE,
            entity_type=AuditEntityType.PREDICTION,
            entity_id=prediction_id,
            actor_id=actor_id,
            actor_type=AuditActorType.SERVICE,
            description=(
                "Industrial ML inference completed"
            ),
            equipment_id=equipment_id,
            severity=(
                AuditSeverity.HIGH
                if is_anomaly
                else AuditSeverity.INFO
            ),
            model_reference=AuditModelReference(
                model_id=model_id,
                model_version=model_version,
                prediction_id=prediction_id,
            ),
            request_context=AuditRequestContext(
                trace_id=trace_id
            ),
            metadata={
                "is_anomaly": is_anomaly
            },
        )


# ============================================================
# HASH CHAIN VERIFIER
# ============================================================

class AuditChainVerifier:
    """
    Verify a sequence of audit events.

    Events must be supplied in committed
    chain order, not arbitrary timestamp order.

    For production, compare the final hash
    against a trusted external anchor.
    """

    @staticmethod
    def verify(
        events: list[AuditEvent],
        *,
        expected_previous_hash: str | None = None,
    ) -> dict[str, Any]:
        previous_hash = expected_previous_hash

        for index, event in enumerate(events):
            if (
                event.previous_hash
                != previous_hash
            ):
                return {
                    "valid": False,
                    "checked_records": index,
                    "failed_index": index,
                    "reason": (
                        "Previous hash mismatch"
                    ),
                }

            if not event.verify_hash():
                return {
                    "valid": False,
                    "checked_records": index,
                    "failed_index": index,
                    "reason": (
                        "Record hash mismatch"
                    ),
                }

            previous_hash = event.record_hash

        return {
            "valid": True,
            "checked_records": len(events),
            "failed_index": None,
            "reason": None,
            "last_hash": previous_hash,
        }


# ============================================================
# CREATE / QUERY SCHEMAS
# ============================================================

class AuditEventCreate(AuditBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    action: AuditAction
    entity_type: AuditEntityType

    entity_id: str | None = None

    actor: AuditActor

    description: str = Field(
        min_length=1,
        max_length=2000,
    )

    outcome: AuditOutcome = AuditOutcome.SUCCESS
    severity: AuditSeverity = AuditSeverity.INFO

    asset_id: str | None = None
    equipment_id: str | None = None

    model_reference: (
        AuditModelReference | None
    ) = None

    request_context: (
        AuditRequestContext | None
    ) = None

    changes: list[AuditChange] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class AuditQuery(AuditBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    actor_id: str | None = None
    entity_id: str | None = None

    action: AuditAction | None = None
    entity_type: AuditEntityType | None = None

    severity: AuditSeverity | None = None
    outcome: AuditOutcome | None = None

    start_time: datetime | None = None
    end_time: datetime | None = None

    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
    )

    offset: int = Field(
        default=0,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "AuditQuery":
        if (
            self.start_time is not None
            and self.start_time.tzinfo is None
        ):
            raise ValueError(
                "start_time must be timezone-aware"
            )

        if (
            self.end_time is not None
            and self.end_time.tzinfo is None
        ):
            raise ValueError(
                "end_time must be timezone-aware"
            )

        if (
            self.start_time
            and self.end_time
            and self.end_time < self.start_time
        ):
            raise ValueError(
                "end_time cannot precede start_time"
            )

        return self


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    event1 = AuditEventFactory.model_inference(
        company_id="company-001",
        actor_id="anomaly-service",
        model_id="anomaly-model-001",
        model_version="1.0.0",
        prediction_id="prediction-101",
        equipment_id="PUMP-101",
        is_anomaly=True,
        trace_id="trace-001",
    )

    event1.seal()

    event2 = AuditEventFactory.create(
        company_id="company-001",
        action=AuditAction.ASSET_HEALTH_UPDATE,
        entity_type=AuditEntityType.ASSET,
        entity_id="ASSET-101",
        actor_id="asset-service",
        actor_type=AuditActorType.SERVICE,
        description="Asset health score updated",
        asset_id="ASSET-101",
        equipment_id="PUMP-101",
        changes=[
            AuditChange(
                field_name="health_score",
                old_value=88.0,
                new_value=72.0,
            )
        ],
        metadata={
            "source": "ml_pipeline",
            "api_key": "sensitive-example",
        },
    )

    event2.seal(
        previous_hash=event1.record_hash
    )

    result = AuditChainVerifier.verify(
        [event1, event2]
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        json.dumps(
            event2.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()
