"""
app/services/telemetry_service.py

Industrial Predictive Maintenance, Safety & Emergency AI

Features:
- Real-time industrial telemetry ingestion
- Temperature, vibration, pressure, current, voltage, energy
- Multi-sensor and batch ingestion
- Sensor data validation and normalization
- Sensor quality assessment
- Configurable threshold monitoring
- Telemetry anomaly detection
- Rolling-window statistics
- Sensor trend analysis
- Asset telemetry health summaries
- Latest sensor readings
- Historical telemetry queries
- Optional anomaly service integration
- Optional safety service integration
- Optional notification service integration
- Optional WebSocket broadcasting
- Optional Redis caching
- Optional audit logging
- In-memory development repository

Repository interface:
    create_telemetry(data)
    create_telemetry_batch(data)
    get_telemetry(telemetry_id)
    get_latest_telemetry(asset_id, sensor_type=None)
    list_telemetry(filters)

This service does not directly operate machinery.
Industrial safety interlocks and emergency controls must
remain independent of this application.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import math
import statistics

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class TelemetryServiceError(Exception):
    pass


class TelemetryValidationError(TelemetryServiceError):
    pass


class TelemetryNotFoundError(TelemetryServiceError):
    pass


class TelemetryRepositoryError(TelemetryServiceError):
    pass


class TelemetryIntegrationError(TelemetryServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    VOLTAGE = "voltage"
    POWER = "power"
    ENERGY = "energy"
    HUMIDITY = "humidity"
    RPM = "rpm"
    FLOW_RATE = "flow_rate"
    TORQUE = "torque"
    SOUND = "sound"
    FREQUENCY = "frequency"
    ACCELERATION = "acceleration"
    DISPLACEMENT = "displacement"
    OIL_LEVEL = "oil_level"
    OIL_TEMPERATURE = "oil_temperature"
    BATTERY_SOC = "battery_soc"
    BATTERY_SOH = "battery_soh"
    GAS_CONCENTRATION = "gas_concentration"
    SMOKE_LEVEL = "smoke_level"
    OTHER = "other"


class TelemetrySource(str, Enum):
    MQTT = "mqtt"
    KAFKA = "kafka"
    OPC_UA = "opc_ua"
    MODBUS = "modbus"
    PLC = "plc"
    SCADA = "scada"
    HTTP = "http"
    WEBSOCKET = "websocket"
    SIMULATOR = "simulator"
    MANUAL = "manual"
    OTHER = "other"


class TelemetryQuality(str, Enum):
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    MISSING = "missing"


class TelemetrySeverity(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class TelemetryTrend(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def required_text(
    value: Any,
    name: str,
    max_length: int = 255,
) -> str:
    if not isinstance(value, str):
        raise TelemetryValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not result:
        raise TelemetryValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > max_length:
        raise TelemetryValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def optional_text(
    value: Optional[str],
    name: str,
) -> Optional[str]:
    if value is None:
        return None

    return required_text(value, name)


def finite_float(
    value: Any,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool):
        raise TelemetryValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TelemetryValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise TelemetryValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise TelemetryValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise TelemetryValidationError(
            f"{name} must be <= {maximum}"
        )

    return result


def positive_integer(
    value: Any,
    name: str,
    maximum: int = 100000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise TelemetryValidationError(
            f"{name} must be between 1 and {maximum}"
        )

    return value


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:
    try:
        return enum_class(value)
    except (TypeError, ValueError) as exc:
        raise TelemetryValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def parse_datetime(
    value: Any,
    name: str,
) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise TelemetryValidationError(
                f"Invalid {name}"
            ) from exc

    if not isinstance(value, datetime):
        raise TelemetryValidationError(
            f"{name} must be datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise TelemetryValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


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

    if isinstance(value, (list, tuple, deque)):
        return [serialize(item) for item in value]

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

@dataclass(frozen=True)
class TelemetryThreshold:
    sensor_type: str
    warning_min: Optional[float] = None
    warning_max: Optional[float] = None
    critical_min: Optional[float] = None
    critical_max: Optional[float] = None
    unit: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class TelemetryRecord:
    telemetry_id: str
    company_id: str
    site_id: str
    asset_id: str
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    source: str
    quality: str
    severity: str
    recorded_at: datetime
    received_at: datetime

    machine_id: Optional[str] = None
    component_id: Optional[str] = None

    threshold_triggered: bool = False
    anomaly_detected: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class TelemetryStatistics:
    asset_id: str
    sensor_type: str
    count: int
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    standard_deviation: Optional[float]
    latest_value: Optional[float]
    latest_timestamp: Optional[datetime]

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class TelemetryTrendResult:
    asset_id: str
    sensor_type: str
    trend: str
    count: int
    first_value: Optional[float]
    latest_value: Optional[float]
    change: Optional[float]
    slope_per_hour: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class TelemetryIngestionResult:
    telemetry_id: str
    asset_id: str
    sensor_id: str
    sensor_type: str
    value: float
    quality: str
    severity: str
    threshold_triggered: bool
    anomaly_detected: bool
    stored: bool
    notification_attempted: bool
    broadcast_attempted: bool

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# TELEMETRY SERVICE
# ============================================================

class TelemetryService:
    """
    Industrial telemetry ingestion and analysis.

    Thresholds are asset-specific configuration values.
    No universal safety thresholds are assumed.

    The rolling cache is local to the service instance.
    Use Redis or a database for distributed deployments.
    """

    DEFAULT_UNITS = {
        SensorType.TEMPERATURE: "°C",
        SensorType.VIBRATION: "mm/s",
        SensorType.PRESSURE: "bar",
        SensorType.CURRENT: "A",
        SensorType.VOLTAGE: "V",
        SensorType.POWER: "kW",
        SensorType.ENERGY: "kWh",
        SensorType.HUMIDITY: "%",
        SensorType.RPM: "rpm",
        SensorType.FLOW_RATE: "L/min",
        SensorType.TORQUE: "Nm",
        SensorType.SOUND: "dB",
        SensorType.FREQUENCY: "Hz",
        SensorType.ACCELERATION: "m/s²",
        SensorType.DISPLACEMENT: "mm",
        SensorType.OIL_LEVEL: "%",
        SensorType.OIL_TEMPERATURE: "°C",
        SensorType.BATTERY_SOC: "%",
        SensorType.BATTERY_SOH: "%",
        SensorType.GAS_CONCENTRATION: "ppm",
        SensorType.SMOKE_LEVEL: "ppm",
        SensorType.OTHER: "unit",
    }

    def __init__(
        self,
        telemetry_repository: Any,
        anomaly_service: Any = None,
        safety_service: Any = None,
        notification_service: Any = None,
        websocket_manager: Any = None,
        redis_client: Any = None,
        audit_service: Any = None,
        rolling_window_size: int = 1000,
        enable_anomaly_detection: bool = False,
        enable_safety_events: bool = False,
        enable_notifications: bool = False,
        enable_websocket: bool = False,
    ):
        if telemetry_repository is None:
            raise TelemetryValidationError(
                "telemetry_repository is required"
            )

        self.repository = telemetry_repository
        self.anomaly_service = anomaly_service
        self.safety_service = safety_service
        self.notification_service = notification_service
        self.websocket_manager = websocket_manager
        self.redis_client = redis_client
        self.audit_service = audit_service

        self.rolling_window_size = positive_integer(
            rolling_window_size,
            "rolling_window_size",
        )

        self.enable_anomaly_detection = bool(
            enable_anomaly_detection
        )

        self.enable_safety_events = bool(
            enable_safety_events
        )

        self.enable_notifications = bool(
            enable_notifications
        )

        self.enable_websocket = bool(
            enable_websocket
        )

        self.thresholds: dict[
            tuple[str, str, str],
            TelemetryThreshold,
        ] = {}

        self.rolling_windows: dict[
            tuple[str, str, str],
            deque[TelemetryRecord],
        ] = defaultdict(
            lambda: deque(
                maxlen=self.rolling_window_size
            )
        )

    # ========================================================
    # CONFIGURE THRESHOLD
    # ========================================================

    def configure_threshold(
        self,
        *,
        company_id: str,
        asset_id: str,
        sensor_type: str | SensorType,
        warning_min: Optional[float] = None,
        warning_max: Optional[float] = None,
        critical_min: Optional[float] = None,
        critical_max: Optional[float] = None,
        unit: Optional[str] = None,
    ) -> TelemetryThreshold:
        company = required_text(
            company_id,
            "company_id",
        )

        asset = required_text(
            asset_id,
            "asset_id",
        )

        selected_type = parse_enum(
            SensorType,
            sensor_type,
            "sensor_type",
        )

        values = {}

        for name, value in {
            "warning_min": warning_min,
            "warning_max": warning_max,
            "critical_min": critical_min,
            "critical_max": critical_max,
        }.items():
            values[name] = (
                finite_float(value, name)
                if value is not None
                else None
            )

        if (
            values["warning_min"] is not None
            and values["warning_max"] is not None
            and values["warning_min"]
            > values["warning_max"]
        ):
            raise TelemetryValidationError(
                "warning_min cannot exceed warning_max"
            )

        if (
            values["critical_min"] is not None
            and values["warning_min"] is not None
            and values["critical_min"]
            > values["warning_min"]
        ):
            raise TelemetryValidationError(
                "critical_min must not exceed warning_min"
            )

        if (
            values["critical_max"] is not None
            and values["warning_max"] is not None
            and values["critical_max"]
            < values["warning_max"]
        ):
            raise TelemetryValidationError(
                "critical_max must not be below warning_max"
            )

        threshold = TelemetryThreshold(
            sensor_type=selected_type.value,
            warning_min=values["warning_min"],
            warning_max=values["warning_max"],
            critical_min=values["critical_min"],
            critical_max=values["critical_max"],
            unit=unit or self.DEFAULT_UNITS[
                selected_type
            ],
        )

        key = (
            company,
            asset,
            selected_type.value,
        )

        self.thresholds[key] = threshold

        return threshold

    # ========================================================
    # EVALUATE THRESHOLD
    # ========================================================

    def evaluate_threshold(
        self,
        *,
        company_id: str,
        asset_id: str,
        sensor_type: str | SensorType,
        value: float,
        unit: Optional[str] = None,
    ) -> TelemetrySeverity:
        selected_type = parse_enum(
            SensorType,
            sensor_type,
            "sensor_type",
        )

        reading = finite_float(
            value,
            "value",
        )

        threshold = self.thresholds.get(
            (
                company_id,
                asset_id,
                selected_type.value,
            )
        )

        if threshold is None:
            return TelemetrySeverity.NORMAL

        if unit is not None and unit != threshold.unit:
            raise TelemetryValidationError(
                "Telemetry unit does not match configured "
                "threshold unit"
            )

        if (
            threshold.critical_min is not None
            and reading < threshold.critical_min
        ):
            return TelemetrySeverity.CRITICAL

        if (
            threshold.critical_max is not None
            and reading > threshold.critical_max
        ):
            return TelemetrySeverity.CRITICAL

        if (
            threshold.warning_min is not None
            and reading < threshold.warning_min
        ):
            return TelemetrySeverity.WARNING

        if (
            threshold.warning_max is not None
            and reading > threshold.warning_max
        ):
            return TelemetrySeverity.WARNING

        return TelemetrySeverity.NORMAL

    # ========================================================
    # SENSOR QUALITY
    # ========================================================

    @staticmethod
    def assess_quality(
        *,
        value: Any,
        recorded_at: datetime,
        maximum_age_seconds: float = 300,
        future_tolerance_seconds: float = 30,
    ) -> TelemetryQuality:
        try:
            finite_float(
                value,
                "value",
            )
        except TelemetryValidationError:
            return TelemetryQuality.BAD

        timestamp = parse_datetime(
            recorded_at,
            "recorded_at",
        )

        now = utc_now()

        age_seconds = (
            now - timestamp
        ).total_seconds()

        if age_seconds > maximum_age_seconds:
            return TelemetryQuality.SUSPECT

        if age_seconds < -future_tolerance_seconds:
            return TelemetryQuality.SUSPECT

        return TelemetryQuality.GOOD

    # ========================================================
    # INGEST TELEMETRY
    # ========================================================

    async def ingest_telemetry(
        self,
        *,
        company_id: str,
        site_id: str,
        asset_id: str,
        sensor_id: str,
        sensor_type: str | SensorType,
        value: float,
        source: str | TelemetrySource,
        unit: Optional[str] = None,
        recorded_at: Optional[datetime] = None,
        machine_id: Optional[str] = None,
        component_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TelemetryIngestionResult:
        company = required_text(
            company_id,
            "company_id",
        )

        site = required_text(
            site_id,
            "site_id",
        )

        asset = required_text(
            asset_id,
            "asset_id",
        )

        sensor = required_text(
            sensor_id,
            "sensor_id",
        )

        selected_type = parse_enum(
            SensorType,
            sensor_type,
            "sensor_type",
        )

        selected_source = parse_enum(
            TelemetrySource,
            source,
            "source",
        )

        reading = finite_float(
            value,
            "value",
        )

        timestamp = (
            parse_datetime(
                recorded_at,
                "recorded_at",
            )
            if recorded_at is not None
            else utc_now()
        )

        selected_unit = (
            required_text(
                unit,
                "unit",
                32,
            )
            if unit is not None
            else self.DEFAULT_UNITS[selected_type]
        )

        quality = self.assess_quality(
            value=reading,
            recorded_at=timestamp,
        )

        severity = self.evaluate_threshold(
            company_id=company,
            asset_id=asset,
            sensor_type=selected_type,
            value=reading,
            unit=selected_unit,
        )

        if quality != TelemetryQuality.GOOD:
            # Stale or future-dated telemetry must not
            # independently trigger a live safety event.
            effective_severity = TelemetrySeverity.NORMAL
        else:
            effective_severity = severity

        threshold_triggered = (
            effective_severity
            != TelemetrySeverity.NORMAL
        )

        anomaly_detected = False

        if (
            self.enable_anomaly_detection
            and self.anomaly_service is not None
            and quality == TelemetryQuality.GOOD
        ):
            anomaly_detected = (
                await self._detect_anomaly(
                    company_id=company,
                    site_id=site,
                    asset_id=asset,
                    sensor_id=sensor,
                    sensor_type=selected_type.value,
                    value=reading,
                    recorded_at=timestamp,
                )
            )

        record = TelemetryRecord(
            telemetry_id=str(uuid4()),
            company_id=company,
            site_id=site,
            asset_id=asset,
            sensor_id=sensor,
            sensor_type=selected_type.value,
            value=reading,
            unit=selected_unit,
            source=selected_source.value,
            quality=quality.value,
            severity=severity.value,
            recorded_at=timestamp,
            received_at=utc_now(),
            machine_id=optional_text(
                machine_id,
                "machine_id",
            ),
            component_id=optional_text(
                component_id,
                "component_id",
            ),
            threshold_triggered=threshold_triggered,
            anomaly_detected=anomaly_detected,
            metadata=dict(metadata or {}),
        )

        try:
            await invoke(
                self.repository.create_telemetry,
                record.to_dict(),
            )
        except Exception as exc:
            raise TelemetryRepositoryError(
                "Unable to persist telemetry"
            ) from exc

        window_key = (
            company,
            asset,
            sensor,
        )

        self.rolling_windows[
            window_key
        ].append(record)

        await self._cache_latest(record)

        if (
            self.enable_safety_events
            and threshold_triggered
        ):
            await self._create_safety_event(record)

        notification_attempted = False

        if (
            self.enable_notifications
            and threshold_triggered
        ):
            notification_attempted = True
            await self._send_alert(record)

        broadcast_attempted = False

        if self.enable_websocket:
            broadcast_attempted = True
            await self._broadcast(record)

        return TelemetryIngestionResult(
            telemetry_id=record.telemetry_id,
            asset_id=record.asset_id,
            sensor_id=record.sensor_id,
            sensor_type=record.sensor_type,
            value=record.value,
            quality=record.quality,
            severity=record.severity,
            threshold_triggered=threshold_triggered,
            anomaly_detected=anomaly_detected,
            stored=True,
            notification_attempted=(
                notification_attempted
            ),
            broadcast_attempted=broadcast_attempted,
        )

    # ========================================================
    # BATCH INGESTION
    # ========================================================

    async def ingest_batch(
        self,
        telemetry_records: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[TelemetryIngestionResult]:
        if not telemetry_records:
            return []

        results = []

        for record in telemetry_records:
            if not isinstance(record, Mapping):
                raise TelemetryValidationError(
                    "Each telemetry record must be a mapping"
                )

            result = await self.ingest_telemetry(
                **dict(record)
            )

            results.append(result)

        return results

    # ========================================================
    # ANOMALY INTEGRATION
    # ========================================================

    async def _detect_anomaly(
        self,
        **payload: Any,
    ) -> bool:
        method = getattr(
            self.anomaly_service,
            "detect_anomaly",
            None,
        )

        if method is None:
            logger.warning(
                "Anomaly service does not implement "
                "detect_anomaly"
            )
            return False

        try:
            result = await invoke(
                method,
                **payload,
            )

            if isinstance(result, bool):
                return result

            if isinstance(result, Mapping):
                return bool(
                    result.get(
                        "is_anomaly",
                        result.get(
                            "anomaly_detected",
                            False,
                        ),
                    )
                )

            return bool(
                getattr(
                    result,
                    "is_anomaly",
                    False,
                )
            )

        except Exception:
            logger.exception(
                "Telemetry anomaly detection failed"
            )
            return False

    # ========================================================
    # SAFETY INTEGRATION
    # ========================================================

    async def _create_safety_event(
        self,
        record: TelemetryRecord,
    ) -> None:
        if self.safety_service is None:
            return

        method = getattr(
            self.safety_service,
            "create_safety_event",
            None,
        )

        if method is None:
            return

        event_type_mapping = {
            SensorType.TEMPERATURE.value: "overheating",
            SensorType.OIL_TEMPERATURE.value: "overheating",
            SensorType.VIBRATION.value: "vibration_hazard",
            SensorType.PRESSURE.value: "pressure_hazard",
            SensorType.CURRENT.value: "electrical_hazard",
            SensorType.VOLTAGE.value: "electrical_hazard",
            SensorType.GAS_CONCENTRATION.value: "gas_leak",
            SensorType.SMOKE_LEVEL.value: "smoke",
        }

        event_type = event_type_mapping.get(
            record.sensor_type
        )

        if event_type is None:
            return

        severity = (
            "critical"
            if record.severity
            == TelemetrySeverity.CRITICAL.value
            else "high"
        )

        try:
            await invoke(
                method,
                company_id=record.company_id,
                site_id=record.site_id,
                asset_id=record.asset_id,
                machine_id=record.machine_id,
                event_type=event_type,
                title=(
                    f"{record.sensor_type.replace('_', ' ').title()} "
                    "Threshold Alert"
                ),
                description=(
                    f"Sensor {record.sensor_id} reported "
                    f"{record.value} {record.unit}, "
                    f"exceeding a configured threshold."
                ),
                source="telemetry",
                severity=severity,
                metadata={
                    "telemetry_id": record.telemetry_id,
                    "sensor_id": record.sensor_id,
                    "value": record.value,
                    "unit": record.unit,
                    "threshold_severity": record.severity,
                    "requires_independent_verification": True,
                },
            )

        except Exception:
            logger.exception(
                "Unable to create safety event "
                "for telemetry %s",
                record.telemetry_id,
            )

    # ========================================================
    # NOTIFICATION INTEGRATION
    # ========================================================

    async def _send_alert(
        self,
        record: TelemetryRecord,
    ) -> None:
        if self.notification_service is None:
            return

        method = getattr(
            self.notification_service,
            "send_telemetry_alert",
            None,
        )

        if method is None:
            logger.warning(
                "Notification service does not implement "
                "send_telemetry_alert"
            )
            return

        try:
            await invoke(
                method,
                record.to_dict(),
            )

        except Exception:
            logger.exception(
                "Telemetry notification failed: %s",
                record.telemetry_id,
            )

    # ========================================================
    # REDIS CACHE
    # ========================================================

    async def _cache_latest(
        self,
        record: TelemetryRecord,
    ) -> None:
        if self.redis_client is None:
            return

        key = (
            f"telemetry:latest:"
            f"{record.company_id}:"
            f"{record.asset_id}:"
            f"{record.sensor_id}"
        )

        try:
            await invoke(
                self.redis_client.set,
                key,
                json.dumps(
                    record.to_dict()
                ),
            )

        except Exception:
            logger.exception(
                "Telemetry Redis cache failed"
            )

    # ========================================================
    # WEBSOCKET BROADCAST
    # ========================================================

    async def _broadcast(
        self,
        record: TelemetryRecord,
    ) -> None:
        if self.websocket_manager is None:
            return

        method = getattr(
            self.websocket_manager,
            "broadcast",
            None,
        )

        if method is None:
            return

        try:
            await invoke(
                method,
                {
                    "event": "telemetry_update",
                    "company_id": record.company_id,
                    "site_id": record.site_id,
                    "asset_id": record.asset_id,
                    "data": record.to_dict(),
                },
            )

        except Exception:
            logger.exception(
                "Telemetry WebSocket broadcast failed"
            )

    # ========================================================
    # GET TELEMETRY
    # ========================================================

    async def get_telemetry(
        self,
        telemetry_id: str,
    ) -> dict[str, Any]:
        identifier = required_text(
            telemetry_id,
            "telemetry_id",
        )

        record = await invoke(
            self.repository.get_telemetry,
            identifier,
        )

        if record is None:
            raise TelemetryNotFoundError(
                f"Telemetry not found: {identifier}"
            )

        return record

    # ========================================================
    # LATEST TELEMETRY
    # ========================================================

    async def get_latest_telemetry(
        self,
        *,
        asset_id: str,
        sensor_type: Optional[
            str | SensorType
        ] = None,
    ) -> Any:
        asset = required_text(
            asset_id,
            "asset_id",
        )

        selected_type = (
            parse_enum(
                SensorType,
                sensor_type,
                "sensor_type",
            ).value
            if sensor_type is not None
            else None
        )

        return await invoke(
            self.repository.get_latest_telemetry,
            asset,
            selected_type,
        )

    # ========================================================
    # LIST TELEMETRY
    # ========================================================

    async def list_telemetry(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {}

        for name, value in {
            "company_id": company_id,
            "site_id": site_id,
            "asset_id": asset_id,
            "sensor_id": sensor_id,
        }.items():
            if value is not None:
                filters[name] = required_text(
                    value,
                    name,
                )

        if sensor_type is not None:
            filters["sensor_type"] = parse_enum(
                SensorType,
                sensor_type,
                "sensor_type",
            ).value

        if start_time is not None:
            filters["start_time"] = parse_datetime(
                start_time,
                "start_time",
            ).isoformat()

        if end_time is not None:
            filters["end_time"] = parse_datetime(
                end_time,
                "end_time",
            ).isoformat()

        if (
            start_time is not None
            and end_time is not None
            and parse_datetime(start_time, "start_time")
            > parse_datetime(end_time, "end_time")
        ):
            raise TelemetryValidationError(
                "start_time cannot exceed end_time"
            )

        filters["limit"] = positive_integer(
            limit,
            "limit",
            100000,
        )

        return await invoke(
            self.repository.list_telemetry,
            filters,
        )

    # ========================================================
    # ROLLING STATISTICS
    # ========================================================

    def get_rolling_statistics(
        self,
        *,
        company_id: str,
        asset_id: str,
        sensor_id: str,
    ) -> TelemetryStatistics:
        key = (
            required_text(company_id, "company_id"),
            required_text(asset_id, "asset_id"),
            required_text(sensor_id, "sensor_id"),
        )

        records = list(
            self.rolling_windows.get(key, [])
        )

        if not records:
            return TelemetryStatistics(
                asset_id=asset_id,
                sensor_type="unknown",
                count=0,
                minimum=None,
                maximum=None,
                mean=None,
                median=None,
                standard_deviation=None,
                latest_value=None,
                latest_timestamp=None,
            )

        values = [
            record.value
            for record in records
        ]

        latest = max(
            records,
            key=lambda record: record.recorded_at,
        )

        return TelemetryStatistics(
            asset_id=asset_id,
            sensor_type=latest.sensor_type,
            count=len(values),
            minimum=min(values),
            maximum=max(values),
            mean=round(
                statistics.fmean(values),
                4,
            ),
            median=round(
                statistics.median(values),
                4,
            ),
            standard_deviation=(
                round(
                    statistics.pstdev(values),
                    4,
                )
                if len(values) > 1
                else 0.0
            ),
            latest_value=latest.value,
            latest_timestamp=latest.recorded_at,
        )

    # ========================================================
    # SENSOR TREND
    # ========================================================

    def analyze_sensor_trend(
        self,
        *,
        company_id: str,
        asset_id: str,
        sensor_id: str,
        stable_tolerance_per_hour: float = 0.01,
    ) -> TelemetryTrendResult:
        key = (
            required_text(company_id, "company_id"),
            required_text(asset_id, "asset_id"),
            required_text(sensor_id, "sensor_id"),
        )

        records = sorted(
            self.rolling_windows.get(key, []),
            key=lambda record: record.recorded_at,
        )

        if len(records) < 2:
            return TelemetryTrendResult(
                asset_id=asset_id,
                sensor_type=(
                    records[0].sensor_type
                    if records
                    else "unknown"
                ),
                trend=TelemetryTrend.INSUFFICIENT_DATA.value,
                count=len(records),
                first_value=(
                    records[0].value
                    if records
                    else None
                ),
                latest_value=(
                    records[-1].value
                    if records
                    else None
                ),
                change=None,
                slope_per_hour=None,
            )

        first = records[0]
        latest = records[-1]

        elapsed_hours = (
            latest.recorded_at - first.recorded_at
        ).total_seconds() / 3600.0

        if elapsed_hours <= 0:
            raise TelemetryValidationError(
                "Distinct timestamps are required "
                "for trend analysis"
            )

        change = latest.value - first.value
        slope = change / elapsed_hours

        tolerance = finite_float(
            stable_tolerance_per_hour,
            "stable_tolerance_per_hour",
            0,
        )

        if abs(slope) <= tolerance:
            trend = TelemetryTrend.STABLE
        elif slope > 0:
            trend = TelemetryTrend.INCREASING
        else:
            trend = TelemetryTrend.DECREASING

        return TelemetryTrendResult(
            asset_id=asset_id,
            sensor_type=latest.sensor_type,
            trend=trend.value,
            count=len(records),
            first_value=first.value,
            latest_value=latest.value,
            change=round(change, 4),
            slope_per_hour=round(slope, 4),
        )

    # ========================================================
    # ASSET TELEMETRY SUMMARY
    # ========================================================

    async def get_asset_telemetry_summary(
        self,
        *,
        company_id: str,
        asset_id: str,
        limit: int = 1000,
    ) -> dict[str, Any]:
        records = await self.list_telemetry(
            company_id=company_id,
            asset_id=asset_id,
            limit=limit,
        )

        grouped: dict[
            str,
            list[dict[str, Any]],
        ] = defaultdict(list)

        for record in records:
            grouped[
                record["sensor_type"]
            ].append(record)

        sensor_summaries = {}

        for sensor_type, items in grouped.items():
            values = [
                finite_float(
                    item["value"],
                    "value",
                )
                for item in items
            ]

            latest = max(
                items,
                key=lambda item: parse_datetime(
                    item["recorded_at"],
                    "recorded_at",
                ),
            )

            sensor_summaries[sensor_type] = {
                "count": len(items),
                "minimum": min(values),
                "maximum": max(values),
                "mean": round(
                    statistics.fmean(values),
                    4,
                ),
                "latest_value": latest["value"],
                "latest_unit": latest["unit"],
                "latest_timestamp": (
                    latest["recorded_at"]
                ),
            }

        return {
            "company_id": company_id,
            "asset_id": asset_id,
            "total_records": len(records),
            "sensor_summaries": sensor_summaries,
            "warning_records": sum(
                record.get("severity")
                == TelemetrySeverity.WARNING.value
                for record in records
            ),
            "critical_records": sum(
                record.get("severity")
                == TelemetrySeverity.CRITICAL.value
                for record in records
            ),
            "anomaly_records": sum(
                bool(record.get("anomaly_detected"))
                for record in records
            ),
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# IN-MEMORY REPOSITORY
# ============================================================

class InMemoryTelemetryRepository:
    """
    Development-only telemetry repository.

    Not durable or concurrency-safe.
    """

    def __init__(self):
        self.records: dict[
            str,
            dict[str, Any],
        ] = {}

    async def create_telemetry(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        record = copy.deepcopy(data)

        telemetry_id = record["telemetry_id"]

        if telemetry_id in self.records:
            raise TelemetryRepositoryError(
                "Duplicate telemetry ID"
            )

        self.records[telemetry_id] = record

        return copy.deepcopy(record)

    async def create_telemetry_batch(
        self,
        data: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        results = []

        for record in data:
            results.append(
                await self.create_telemetry(record)
            )

        return results

    async def get_telemetry(
        self,
        telemetry_id: str,
    ) -> Optional[dict[str, Any]]:
        record = self.records.get(
            telemetry_id
        )

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def get_latest_telemetry(
        self,
        asset_id: str,
        sensor_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        records = [
            record
            for record in self.records.values()
            if record["asset_id"] == asset_id
            and (
                sensor_type is None
                or record["sensor_type"] == sensor_type
            )
        ]

        if not records:
            return None

        latest = max(
            records,
            key=lambda record: record["recorded_at"],
        )

        return copy.deepcopy(latest)

    async def list_telemetry(
        self,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records = []

        start_time = filters.get(
            "start_time"
        )

        end_time = filters.get(
            "end_time"
        )

        limit = filters.get(
            "limit",
            1000,
        )

        for record in self.records.values():
            matched = True

            for key, value in filters.items():
                if key in {
                    "start_time",
                    "end_time",
                    "limit",
                }:
                    continue

                if record.get(key) != value:
                    matched = False
                    break

            if not matched:
                continue

            timestamp = parse_datetime(
                record["recorded_at"],
                "recorded_at",
            )

            if (
                start_time is not None
                and timestamp < parse_datetime(
                    start_time,
                    "start_time",
                )
            ):
                continue

            if (
                end_time is not None
                and timestamp > parse_datetime(
                    end_time,
                    "end_time",
                )
            ):
                continue

            records.append(
                copy.deepcopy(record)
            )

        records.sort(
            key=lambda record: record["recorded_at"],
            reverse=True,
        )

        return records[:limit]


# ============================================================
# DEMONSTRATION
# ============================================================

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO
    )

    repository = InMemoryTelemetryRepository()

    service = TelemetryService(
        telemetry_repository=repository,
        rolling_window_size=100,
    )

    # Demonstration thresholds only.
    service.configure_threshold(
        company_id="COMPANY-001",
        asset_id="MACHINE-001",
        sensor_type="temperature",
        warning_max=85.0,
        critical_max=100.0,
        unit="°C",
    )

    service.configure_threshold(
        company_id="COMPANY-001",
        asset_id="MACHINE-001",
        sensor_type="vibration",
        warning_max=7.0,
        critical_max=12.0,
        unit="mm/s",
    )

    now = utc_now()

    telemetry_records = [
        {
            "company_id": "COMPANY-001",
            "site_id": "SITE-001",
            "asset_id": "MACHINE-001",
            "sensor_id": "TEMP-001",
            "sensor_type": "temperature",
            "value": 78.0,
            "source": "simulator",
            "recorded_at": now - timedelta(seconds=20),
        },
        {
            "company_id": "COMPANY-001",
            "site_id": "SITE-001",
            "asset_id": "MACHINE-001",
            "sensor_id": "TEMP-001",
            "sensor_type": "temperature",
            "value": 88.5,
            "source": "simulator",
            "recorded_at": now - timedelta(seconds=10),
        },
        {
            "company_id": "COMPANY-001",
            "site_id": "SITE-001",
            "asset_id": "MACHINE-001",
            "sensor_id": "TEMP-001",
            "sensor_type": "temperature",
            "value": 104.2,
            "source": "simulator",
            "recorded_at": now,
        },
        {
            "company_id": "COMPANY-001",
            "site_id": "SITE-001",
            "asset_id": "MACHINE-001",
            "sensor_id": "VIB-001",
            "sensor_type": "vibration",
            "value": 8.4,
            "source": "simulator",
            "recorded_at": now,
        },
    ]

    results = await service.ingest_batch(
        telemetry_records
    )

    print("\nINGESTION RESULTS")

    print(
        json.dumps(
            [
                result.to_dict()
                for result in results
            ],
            indent=2,
        )
    )

    print("\nROLLING STATISTICS")

    stats = service.get_rolling_statistics(
        company_id="COMPANY-001",
        asset_id="MACHINE-001",
        sensor_id="TEMP-001",
    )

    print(
        json.dumps(
            stats.to_dict(),
            indent=2,
        )
    )

    print("\nTEMPERATURE TREND")

    trend = service.analyze_sensor_trend(
        company_id="COMPANY-001",
        asset_id="MACHINE-001",
        sensor_id="TEMP-001",
    )

    print(
        json.dumps(
            trend.to_dict(),
            indent=2,
        )
    )

    print("\nASSET TELEMETRY SUMMARY")

    summary = (
        await service.get_asset_telemetry_summary(
            company_id="COMPANY-001",
            asset_id="MACHINE-001",
        )
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())