# app/agents/tools/asset_tools.py

from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence
from uuid import UUID


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, float(value)))


def _finite(
    value: Any,
    *,
    default: float | None = None,
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def _non_negative(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    result = _finite(value)

    if result is None:
        return default

    return max(0.0, result)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if hasattr(value, "model_dump"):
        try:
            return _json_safe(
                value.model_dump(mode="json")
            )
        except Exception:
            return _json_safe(
                value.model_dump()
            )

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item)
            for item in value
        ]

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None

    return value


def _value(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    if source is None:
        return default

    if isinstance(source, Mapping):
        return source.get(key, default)

    return getattr(source, key, default)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


# =========================================================
# Enums
# =========================================================

class AssetStatus(str, Enum):
    UNKNOWN = "unknown"
    OFFLINE = "offline"
    IDLE = "idle"
    RUNNING = "running"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    EMERGENCY = "emergency"


class AssetCondition(str, Enum):
    UNKNOWN = "unknown"
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class AssetCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataQuality(str, Enum):
    UNKNOWN = "unknown"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class AssetToolError(str, Enum):
    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    REPOSITORY_ERROR = "repository_error"
    TELEMETRY_ERROR = "telemetry_error"
    HISTORY_ERROR = "history_error"


# =========================================================
# Asset model
# =========================================================

@dataclass(slots=True)
class AssetRecord:
    asset_id: str

    name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    zone_id: str | None = None
    location: str | None = None

    parent_asset_id: str | None = None

    manufacturer: str | None = None
    model_number: str | None = None
    serial_number: str | None = None

    install_date: datetime | None = None

    expected_life_years: float | None = None

    operating_hours: float | None = None

    status: AssetStatus = AssetStatus.UNKNOWN

    criticality_score: float = 0.5

    safety_critical: bool = False
    production_critical: bool = False

    replacement_cost: float | None = None

    currency: str = "INR"

    tags: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.criticality_score = _clamp(
            self.criticality_score
        )

        if self.expected_life_years is not None:
            self.expected_life_years = (
                _non_negative(
                    self.expected_life_years
                )
            )

        if self.operating_hours is not None:
            self.operating_hours = (
                _non_negative(
                    self.operating_hours
                )
            )

        if self.replacement_cost is not None:
            self.replacement_cost = (
                _non_negative(
                    self.replacement_cost
                )
            )


# =========================================================
# Telemetry
# =========================================================

@dataclass(slots=True)
class AssetTelemetry:
    asset_id: str

    timestamp: datetime

    temperature_c: float | None = None
    vibration_mm_s: float | None = None
    pressure_bar: float | None = None
    current_a: float | None = None
    power_kw: float | None = None

    load_percent: float | None = None

    speed_rpm: float | None = None

    energy_kwh: float | None = None

    operating: bool | None = None

    emergency_stop_active: bool = False

    sensor_count: int = 0
    valid_sensor_count: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def quality_score(self) -> float:
        if self.sensor_count <= 0:
            return 0.0

        return _clamp(
            self.valid_sensor_count
            / self.sensor_count
        )


# =========================================================
# Maintenance summary
# =========================================================

@dataclass(slots=True)
class MaintenanceSummary:
    total_work_orders: int = 0

    completed_work_orders: int = 0
    open_work_orders: int = 0

    failure_count: int = 0

    repeat_failure_count: int = 0

    last_maintenance_at: datetime | None = None
    last_failure_at: datetime | None = None

    total_maintenance_cost: float = 0.0

    average_repair_hours: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Health context
# =========================================================

@dataclass(slots=True)
class AssetHealthContext:
    health_score: float | None = None

    condition: AssetCondition = (
        AssetCondition.UNKNOWN
    )

    failure_probability: float | None = None

    anomaly_score: float | None = None

    predicted_rul_hours: float | None = None

    degradation_score: float | None = None

    risk_score: float | None = None

    confidence: float = 0.0

    model_drift_detected: bool = False
    data_drift_detected: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Asset snapshot
# =========================================================

@dataclass(slots=True)
class AssetSnapshot:
    asset: AssetRecord

    telemetry: AssetTelemetry | None

    health: AssetHealthContext

    maintenance: MaintenanceSummary

    age_years: float | None

    life_consumed_ratio: float | None

    criticality: AssetCriticality

    data_quality: DataQuality
    data_quality_score: float

    stale_telemetry: bool

    generated_at: datetime

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


# =========================================================
# Tool result
# =========================================================

@dataclass(slots=True)
class AssetToolResult:
    success: bool

    operation: str

    asset_id: str | None = None

    data: Any = None

    error_code: AssetToolError | None = None
    error: str | None = None

    generated_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


# =========================================================
# Repository contracts
# =========================================================

class AssetRepository(Protocol):

    async def get_by_id(
        self,
        asset_id: str,
    ) -> Any | None:
        ...

    async def list_assets(
        self,
        *,
        site_id: str | None = None,
        zone_id: str | None = None,
        asset_type: str | None = None,
        limit: int = 100,
    ) -> Sequence[Any]:
        ...


class TelemetryRepository(Protocol):

    async def get_latest(
        self,
        asset_id: str,
    ) -> Any | None:
        ...


class MaintenanceRepository(Protocol):

    async def get_asset_summary(
        self,
        asset_id: str,
    ) -> Any | None:
        ...


class HealthRepository(Protocol):

    async def get_latest_health(
        self,
        asset_id: str,
    ) -> Any | None:
        ...


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class AssetToolsConfig:
    telemetry_stale_seconds: float = 300.0

    criticality_high_threshold: float = 0.70
    criticality_critical_threshold: float = 0.90

    health_good_threshold: float = 70.0
    health_fair_threshold: float = 50.0
    health_poor_threshold: float = 30.0

    quality_fair_threshold: float = 0.50
    quality_good_threshold: float = 0.75
    quality_excellent_threshold: float = 0.95

    maximum_list_size: int = 500


# =========================================================
# Asset tools
# =========================================================

class AssetTools:

    def __init__(
        self,
        *,
        asset_repository: AssetRepository | None = None,
        telemetry_repository: (
            TelemetryRepository | None
        ) = None,
        maintenance_repository: (
            MaintenanceRepository | None
        ) = None,
        health_repository: (
            HealthRepository | None
        ) = None,
        config: AssetToolsConfig | None = None,
    ) -> None:

        self.asset_repository = asset_repository
        self.telemetry_repository = (
            telemetry_repository
        )
        self.maintenance_repository = (
            maintenance_repository
        )
        self.health_repository = health_repository

        self.config = (
            config
            or AssetToolsConfig()
        )

    # =====================================================
    # Get asset
    # =====================================================

    async def get_asset(
        self,
        asset_id: str,
    ) -> AssetToolResult:

        asset_id = asset_id.strip()

        if not asset_id:
            return self._error(
                operation="get_asset",
                code=AssetToolError.INVALID_INPUT,
                message="asset_id is required.",
            )

        if self.asset_repository is None:
            return self._error(
                operation="get_asset",
                asset_id=asset_id,
                code=(
                    AssetToolError.REPOSITORY_ERROR
                ),
                message=(
                    "Asset repository is not "
                    "configured."
                ),
            )

        try:
            raw = await _maybe_await(
                self.asset_repository.get_by_id(
                    asset_id
                )
            )

        except Exception as exc:
            return self._error(
                operation="get_asset",
                asset_id=asset_id,
                code=(
                    AssetToolError.REPOSITORY_ERROR
                ),
                message=str(exc),
            )

        if raw is None:
            return self._error(
                operation="get_asset",
                asset_id=asset_id,
                code=AssetToolError.NOT_FOUND,
                message=(
                    f"Asset {asset_id!r} "
                    "was not found."
                ),
            )

        asset = self._normalize_asset(raw)

        return AssetToolResult(
            success=True,
            operation="get_asset",
            asset_id=asset.asset_id,
            data=asset,
        )

    # =====================================================
    # List assets
    # =====================================================

    async def list_assets(
        self,
        *,
        site_id: str | None = None,
        zone_id: str | None = None,
        asset_type: str | None = None,
        limit: int = 100,
    ) -> AssetToolResult:

        if self.asset_repository is None:
            return self._error(
                operation="list_assets",
                code=(
                    AssetToolError.REPOSITORY_ERROR
                ),
                message=(
                    "Asset repository is not "
                    "configured."
                ),
            )

        limit = max(
            1,
            min(
                int(limit),
                self.config.maximum_list_size,
            ),
        )

        try:
            raw_assets = await _maybe_await(
                self.asset_repository.list_assets(
                    site_id=site_id,
                    zone_id=zone_id,
                    asset_type=asset_type,
                    limit=limit,
                )
            )

            assets = [
                self._normalize_asset(item)
                for item in raw_assets
            ]

            return AssetToolResult(
                success=True,
                operation="list_assets",
                data=assets,
                metadata={
                    "count": len(assets),
                    "site_id": site_id,
                    "zone_id": zone_id,
                    "asset_type": asset_type,
                },
            )

        except Exception as exc:
            return self._error(
                operation="list_assets",
                code=(
                    AssetToolError.REPOSITORY_ERROR
                ),
                message=str(exc),
            )

    # =====================================================
    # Latest telemetry
    # =====================================================

    async def get_latest_telemetry(
        self,
        asset_id: str,
    ) -> AssetToolResult:

        if self.telemetry_repository is None:
            return self._error(
                operation="get_latest_telemetry",
                asset_id=asset_id,
                code=AssetToolError.TELEMETRY_ERROR,
                message=(
                    "Telemetry repository is not "
                    "configured."
                ),
            )

        try:
            raw = await _maybe_await(
                self.telemetry_repository.get_latest(
                    asset_id
                )
            )

        except Exception as exc:
            return self._error(
                operation="get_latest_telemetry",
                asset_id=asset_id,
                code=AssetToolError.TELEMETRY_ERROR,
                message=str(exc),
            )

        if raw is None:
            return self._error(
                operation="get_latest_telemetry",
                asset_id=asset_id,
                code=AssetToolError.NOT_FOUND,
                message=(
                    "No telemetry is available for "
                    f"{asset_id!r}."
                ),
            )

        telemetry = self._normalize_telemetry(
            asset_id,
            raw,
        )

        return AssetToolResult(
            success=True,
            operation="get_latest_telemetry",
            asset_id=asset_id,
            data=telemetry,
        )

    # =====================================================
    # Maintenance history summary
    # =====================================================

    async def get_maintenance_summary(
        self,
        asset_id: str,
    ) -> AssetToolResult:

        if self.maintenance_repository is None:
            return self._error(
                operation=(
                    "get_maintenance_summary"
                ),
                asset_id=asset_id,
                code=AssetToolError.HISTORY_ERROR,
                message=(
                    "Maintenance repository is not "
                    "configured."
                ),
            )

        try:
            raw = await _maybe_await(
                self.maintenance_repository
                .get_asset_summary(asset_id)
            )

        except Exception as exc:
            return self._error(
                operation=(
                    "get_maintenance_summary"
                ),
                asset_id=asset_id,
                code=AssetToolError.HISTORY_ERROR,
                message=str(exc),
            )

        summary = self._normalize_maintenance(
            raw
        )

        return AssetToolResult(
            success=True,
            operation="get_maintenance_summary",
            asset_id=asset_id,
            data=summary,
        )

    # =====================================================
    # Health
    # =====================================================

    async def get_health_context(
        self,
        asset_id: str,
    ) -> AssetToolResult:

        if self.health_repository is None:
            return AssetToolResult(
                success=True,
                operation="get_health_context",
                asset_id=asset_id,
                data=AssetHealthContext(),
                metadata={
                    "source": "no_health_repository",
                },
            )

        try:
            raw = await _maybe_await(
                self.health_repository
                .get_latest_health(asset_id)
            )

        except Exception as exc:
            return self._error(
                operation="get_health_context",
                asset_id=asset_id,
                code=(
                    AssetToolError.REPOSITORY_ERROR
                ),
                message=str(exc),
            )

        return AssetToolResult(
            success=True,
            operation="get_health_context",
            asset_id=asset_id,
            data=self._normalize_health(raw),
        )

    # =====================================================
    # Complete snapshot
    # =====================================================

    async def get_asset_snapshot(
        self,
        asset_id: str,
    ) -> AssetToolResult:

        asset_result = await self.get_asset(
            asset_id
        )

        if not asset_result.success:
            return asset_result

        asset: AssetRecord = asset_result.data

        telemetry_task = asyncio_create_task_safe(
            self.get_latest_telemetry(
                asset.asset_id
            )
        )

        maintenance_task = (
            asyncio_create_task_safe(
                self.get_maintenance_summary(
                    asset.asset_id
                )
            )
        )

        health_task = asyncio_create_task_safe(
            self.get_health_context(
                asset.asset_id
            )
        )

        telemetry_result = await telemetry_task
        maintenance_result = (
            await maintenance_task
        )
        health_result = await health_task

        telemetry = (
            telemetry_result.data
            if telemetry_result.success
            else None
        )

        maintenance = (
            maintenance_result.data
            if maintenance_result.success
            else MaintenanceSummary()
        )

        health = (
            health_result.data
            if health_result.success
            else AssetHealthContext()
        )

        age_years = self.calculate_age_years(
            asset.install_date
        )

        life_consumed = (
            self.calculate_life_consumed_ratio(
                age_years=age_years,
                expected_life_years=(
                    asset.expected_life_years
                ),
            )
        )

        stale = self.is_telemetry_stale(
            telemetry
        )

        quality_score = (
            self.calculate_data_quality_score(
                telemetry=telemetry,
                stale=stale,
            )
        )

        warnings: list[str] = []

        if telemetry is None:
            warnings.append(
                "Latest telemetry is unavailable."
            )

        elif stale:
            warnings.append(
                "Latest telemetry is stale."
            )

        if not maintenance_result.success:
            warnings.append(
                "Maintenance history is "
                "unavailable."
            )

        if not health_result.success:
            warnings.append(
                "Health context is unavailable."
            )

        if health.model_drift_detected:
            warnings.append(
                "Model drift is active."
            )

        if health.data_drift_detected:
            warnings.append(
                "Data drift is active."
            )

        snapshot = AssetSnapshot(
            asset=asset,

            telemetry=telemetry,

            health=health,

            maintenance=maintenance,

            age_years=age_years,

            life_consumed_ratio=life_consumed,

            criticality=(
                self.classify_criticality(
                    asset
                )
            ),

            data_quality=(
                self.classify_data_quality(
                    quality_score
                )
            ),

            data_quality_score=quality_score,

            stale_telemetry=stale,

            generated_at=utc_now(),

            warnings=warnings,
        )

        self._record_snapshot_metrics(
            snapshot
        )

        return AssetToolResult(
            success=True,
            operation="get_asset_snapshot",
            asset_id=asset.asset_id,
            data=snapshot,
        )

    # =====================================================
    # Age
    # =====================================================

    @staticmethod
    def calculate_age_years(
        install_date: datetime | None,
        *,
        reference_time: datetime | None = None,
    ) -> float | None:

        if install_date is None:
            return None

        reference = (
            reference_time
            or utc_now()
        )

        if install_date.tzinfo is None:
            install_date = install_date.replace(
                tzinfo=timezone.utc
            )

        if reference.tzinfo is None:
            reference = reference.replace(
                tzinfo=timezone.utc
            )

        seconds = (
            reference - install_date
        ).total_seconds()

        if seconds < 0:
            return 0.0

        return round(
            seconds
            / (365.2425 * 24 * 3600),
            4,
        )

    # =====================================================
    # Life consumed
    # =====================================================

    @staticmethod
    def calculate_life_consumed_ratio(
        *,
        age_years: float | None,
        expected_life_years: float | None,
    ) -> float | None:

        if (
            age_years is None
            or expected_life_years is None
            or expected_life_years <= 0
        ):
            return None

        return round(
            max(
                0.0,
                age_years
                / expected_life_years,
            ),
            4,
        )

    # =====================================================
    # Criticality
    # =====================================================

    def classify_criticality(
        self,
        asset: AssetRecord,
    ) -> AssetCriticality:

        score = asset.criticality_score

        if (
            asset.safety_critical
            and asset.production_critical
        ):
            score = max(score, 0.95)

        elif asset.safety_critical:
            score = max(score, 0.90)

        elif asset.production_critical:
            score = max(score, 0.75)

        if (
            score
            >= self.config
            .criticality_critical_threshold
        ):
            return AssetCriticality.CRITICAL

        if (
            score
            >= self.config
            .criticality_high_threshold
        ):
            return AssetCriticality.HIGH

        if score >= 0.40:
            return AssetCriticality.MEDIUM

        return AssetCriticality.LOW

    # =====================================================
    # Telemetry freshness
    # =====================================================

    def is_telemetry_stale(
        self,
        telemetry: AssetTelemetry | None,
        *,
        reference_time: datetime | None = None,
    ) -> bool:

        if telemetry is None:
            return True

        reference = (
            reference_time
            or utc_now()
        )

        timestamp = telemetry.timestamp

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        age = (
            reference - timestamp
        ).total_seconds()

        return (
            age
            > self.config.telemetry_stale_seconds
        )

    # =====================================================
    # Data quality
    # =====================================================

    @staticmethod
    def calculate_data_quality_score(
        *,
        telemetry: AssetTelemetry | None,
        stale: bool,
    ) -> float:

        if telemetry is None:
            return 0.0

        score = telemetry.quality_score

        if stale:
            score *= 0.50

        if telemetry.timestamp > (
            utc_now()
        ):
            score *= 0.75

        return round(
            _clamp(score),
            4,
        )

    def classify_data_quality(
        self,
        score: float,
    ) -> DataQuality:

        if (
            score
            >= self.config
            .quality_excellent_threshold
        ):
            return DataQuality.EXCELLENT

        if (
            score
            >= self.config
            .quality_good_threshold
        ):
            return DataQuality.GOOD

        if (
            score
            >= self.config
            .quality_fair_threshold
        ):
            return DataQuality.FAIR

        if score > 0:
            return DataQuality.POOR

        return DataQuality.UNKNOWN

    # =====================================================
    # Operating state
    # =====================================================

    @staticmethod
    def determine_operating_status(
        *,
        asset: AssetRecord,
        telemetry: AssetTelemetry | None,
        health: AssetHealthContext | None = None,
    ) -> AssetStatus:

        if (
            telemetry
            and telemetry.emergency_stop_active
        ):
            return AssetStatus.EMERGENCY

        if (
            health
            and health.condition
            == AssetCondition.CRITICAL
        ):
            return AssetStatus.FAULT

        if (
            health
            and health.condition
            == AssetCondition.POOR
        ):
            return AssetStatus.DEGRADED

        if (
            telemetry
            and telemetry.operating is True
        ):
            return AssetStatus.RUNNING

        if (
            telemetry
            and telemetry.operating is False
        ):
            return AssetStatus.IDLE

        return asset.status

    # =====================================================
    # Risk context
    # =====================================================

    @staticmethod
    def build_risk_context(
        snapshot: AssetSnapshot,
    ) -> dict[str, Any]:

        return {
            "asset_id":
                snapshot.asset.asset_id,

            "site_id":
                snapshot.asset.site_id,

            "zone_id":
                snapshot.asset.zone_id,

            "safety_critical":
                snapshot.asset.safety_critical,

            "production_critical":
                snapshot.asset
                .production_critical,

            "criticality":
                snapshot.criticality.value,

            "criticality_score":
                snapshot.asset
                .criticality_score,

            "health_score":
                snapshot.health.health_score,

            "failure_probability":
                snapshot.health
                .failure_probability,

            "anomaly_score":
                snapshot.health.anomaly_score,

            "predicted_rul_hours":
                snapshot.health
                .predicted_rul_hours,

            "degradation_score":
                snapshot.health
                .degradation_score,

            "risk_score":
                snapshot.health.risk_score,

            "life_consumed_ratio":
                snapshot.life_consumed_ratio,

            "repeat_failure_count":
                snapshot.maintenance
                .repeat_failure_count,

            "data_quality":
                snapshot.data_quality.value,

            "data_quality_score":
                snapshot.data_quality_score,

            "stale_telemetry":
                snapshot.stale_telemetry,

            "model_drift_detected":
                snapshot.health
                .model_drift_detected,

            "data_drift_detected":
                snapshot.health
                .data_drift_detected,
        }

    # =====================================================
    # Agent context
    # =====================================================

    @staticmethod
    def build_agent_context(
        snapshot: AssetSnapshot,
    ) -> dict[str, Any]:

        return {
            "asset":
                _json_safe(snapshot.asset),

            "telemetry":
                _json_safe(snapshot.telemetry),

            "health":
                _json_safe(snapshot.health),

            "maintenance":
                _json_safe(snapshot.maintenance),

            "age_years":
                snapshot.age_years,

            "life_consumed_ratio":
                snapshot.life_consumed_ratio,

            "criticality":
                snapshot.criticality.value,

            "data_quality":
                snapshot.data_quality.value,

            "data_quality_score":
                snapshot.data_quality_score,

            "stale_telemetry":
                snapshot.stale_telemetry,

            "warnings":
                list(snapshot.warnings),
        }

    # =====================================================
    # Normalization
    # =====================================================

    def _normalize_asset(
        self,
        raw: Any,
    ) -> AssetRecord:

        if isinstance(raw, AssetRecord):
            return raw

        install_date = self._datetime(
            _value(
                raw,
                "install_date",
                _value(
                    raw,
                    "commissioned_at",
                    None,
                ),
            )
        )

        status_value = str(
            _value(
                raw,
                "status",
                "unknown",
            )
        ).lower()

        try:
            status = AssetStatus(status_value)
        except ValueError:
            status = AssetStatus.UNKNOWN

        return AssetRecord(
            asset_id=str(
                _value(raw, "asset_id")
                or _value(raw, "id")
            ),

            name=_value(
                raw,
                "name",
                _value(
                    raw,
                    "asset_name",
                    None,
                ),
            ),

            asset_type=_value(
                raw,
                "asset_type",
                _value(raw, "type", None),
            ),

            site_id=_value(
                raw,
                "site_id",
            ),

            zone_id=_value(
                raw,
                "zone_id",
            ),

            location=_value(
                raw,
                "location",
            ),

            parent_asset_id=_value(
                raw,
                "parent_asset_id",
            ),

            manufacturer=_value(
                raw,
                "manufacturer",
            ),

            model_number=_value(
                raw,
                "model_number",
            ),

            serial_number=_value(
                raw,
                "serial_number",
            ),

            install_date=install_date,

            expected_life_years=_finite(
                _value(
                    raw,
                    "expected_life_years",
                    None,
                )
            ),

            operating_hours=_finite(
                _value(
                    raw,
                    "operating_hours",
                    None,
                )
            ),

            status=status,

            criticality_score=_clamp(
                _finite(
                    _value(
                        raw,
                        "criticality_score",
                        0.5,
                    ),
                    default=0.5,
                )
                or 0.5
            ),

            safety_critical=bool(
                _value(
                    raw,
                    "safety_critical",
                    False,
                )
            ),

            production_critical=bool(
                _value(
                    raw,
                    "production_critical",
                    False,
                )
            ),

            replacement_cost=_finite(
                _value(
                    raw,
                    "replacement_cost",
                    None,
                )
            ),

            currency=str(
                _value(
                    raw,
                    "currency",
                    "INR",
                )
            ),

            tags=list(
                _value(
                    raw,
                    "tags",
                    [],
                )
                or []
            ),

            metadata=dict(
                _value(
                    raw,
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    def _normalize_telemetry(
        self,
        asset_id: str,
        raw: Any,
    ) -> AssetTelemetry:

        timestamp = self._datetime(
            _value(
                raw,
                "timestamp",
                _value(
                    raw,
                    "recorded_at",
                    utc_now(),
                ),
            )
        ) or utc_now()

        fields = [
            "temperature_c",
            "vibration_mm_s",
            "pressure_bar",
            "current_a",
            "power_kw",
            "load_percent",
            "speed_rpm",
            "energy_kwh",
        ]

        values = {
            field_name: _finite(
                _value(
                    raw,
                    field_name,
                    None,
                )
            )
            for field_name in fields
        }

        supplied = [
            value
            for value in values.values()
            if value is not None
        ]

        sensor_count = int(
            _non_negative(
                _value(
                    raw,
                    "sensor_count",
                    len(fields),
                )
            )
        )

        valid_sensor_count = int(
            _non_negative(
                _value(
                    raw,
                    "valid_sensor_count",
                    len(supplied),
                )
            )
        )

        return AssetTelemetry(
            asset_id=asset_id,

            timestamp=timestamp,

            temperature_c=values[
                "temperature_c"
            ],

            vibration_mm_s=values[
                "vibration_mm_s"
            ],

            pressure_bar=values[
                "pressure_bar"
            ],

            current_a=values[
                "current_a"
            ],

            power_kw=values[
                "power_kw"
            ],

            load_percent=values[
                "load_percent"
            ],

            speed_rpm=values[
                "speed_rpm"
            ],

            energy_kwh=values[
                "energy_kwh"
            ],

            operating=_value(
                raw,
                "operating",
                None,
            ),

            emergency_stop_active=bool(
                _value(
                    raw,
                    "emergency_stop_active",
                    False,
                )
            ),

            sensor_count=sensor_count,

            valid_sensor_count=min(
                valid_sensor_count,
                sensor_count,
            ),

            metadata=dict(
                _value(
                    raw,
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    def _normalize_maintenance(
        self,
        raw: Any,
    ) -> MaintenanceSummary:

        if raw is None:
            return MaintenanceSummary()

        if isinstance(
            raw,
            MaintenanceSummary,
        ):
            return raw

        return MaintenanceSummary(
            total_work_orders=int(
                _non_negative(
                    _value(
                        raw,
                        "total_work_orders",
                        0,
                    )
                )
            ),

            completed_work_orders=int(
                _non_negative(
                    _value(
                        raw,
                        "completed_work_orders",
                        0,
                    )
                )
            ),

            open_work_orders=int(
                _non_negative(
                    _value(
                        raw,
                        "open_work_orders",
                        0,
                    )
                )
            ),

            failure_count=int(
                _non_negative(
                    _value(
                        raw,
                        "failure_count",
                        0,
                    )
                )
            ),

            repeat_failure_count=int(
                _non_negative(
                    _value(
                        raw,
                        "repeat_failure_count",
                        0,
                    )
                )
            ),

            last_maintenance_at=(
                self._datetime(
                    _value(
                        raw,
                        "last_maintenance_at",
                        None,
                    )
                )
            ),

            last_failure_at=(
                self._datetime(
                    _value(
                        raw,
                        "last_failure_at",
                        None,
                    )
                )
            ),

            total_maintenance_cost=(
                _non_negative(
                    _value(
                        raw,
                        "total_maintenance_cost",
                        0.0,
                    )
                )
            ),

            average_repair_hours=(
                _finite(
                    _value(
                        raw,
                        "average_repair_hours",
                        None,
                    )
                )
            ),

            metadata=dict(
                _value(
                    raw,
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    def _normalize_health(
        self,
        raw: Any,
    ) -> AssetHealthContext:

        if raw is None:
            return AssetHealthContext()

        if isinstance(
            raw,
            AssetHealthContext,
        ):
            return raw

        health_score = _finite(
            _value(
                raw,
                "health_score",
                None,
            )
        )

        if health_score is not None:
            health_score = max(
                0.0,
                min(100.0, health_score),
            )

        condition = self._health_condition(
            health_score
        )

        return AssetHealthContext(
            health_score=health_score,

            condition=condition,

            failure_probability=(
                self._optional_probability(
                    _value(
                        raw,
                        "failure_probability",
                        None,
                    )
                )
            ),

            anomaly_score=(
                self._optional_probability(
                    _value(
                        raw,
                        "anomaly_score",
                        None,
                    )
                )
            ),

            predicted_rul_hours=(
                _finite(
                    _value(
                        raw,
                        "predicted_rul_hours",
                        None,
                    )
                )
            ),

            degradation_score=(
                self._optional_probability(
                    _value(
                        raw,
                        "degradation_score",
                        None,
                    )
                )
            ),

            risk_score=(
                self._optional_probability(
                    _value(
                        raw,
                        "risk_score",
                        None,
                    )
                )
            ),

            confidence=(
                self._optional_probability(
                    _value(
                        raw,
                        "confidence",
                        0.0,
                    )
                )
                or 0.0
            ),

            model_drift_detected=bool(
                _value(
                    raw,
                    "model_drift_detected",
                    False,
                )
            ),

            data_drift_detected=bool(
                _value(
                    raw,
                    "data_drift_detected",
                    False,
                )
            ),

            metadata=dict(
                _value(
                    raw,
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    # =====================================================
    # Helpers
    # =====================================================

    def _health_condition(
        self,
        score: float | None,
    ) -> AssetCondition:

        if score is None:
            return AssetCondition.UNKNOWN

        if score >= 85:
            return AssetCondition.EXCELLENT

        if (
            score
            >= self.config.health_good_threshold
        ):
            return AssetCondition.GOOD

        if (
            score
            >= self.config.health_fair_threshold
        ):
            return AssetCondition.FAIR

        if (
            score
            >= self.config.health_poor_threshold
        ):
            return AssetCondition.POOR

        return AssetCondition.CRITICAL

    @staticmethod
    def _optional_probability(
        value: Any,
    ) -> float | None:

        result = _finite(value)

        if result is None:
            return None

        return _clamp(result)

    @staticmethod
    def _datetime(
        value: Any,
    ) -> datetime | None:

        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(
                    tzinfo=timezone.utc
                )

            return value

        if isinstance(value, date):
            return datetime(
                value.year,
                value.month,
                value.day,
                tzinfo=timezone.utc,
            )

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return None

            if text.endswith("Z"):
                text = (
                    text[:-1]
                    + "+00:00"
                )

            try:
                result = (
                    datetime.fromisoformat(
                        text
                    )
                )

                if result.tzinfo is None:
                    result = result.replace(
                        tzinfo=timezone.utc
                    )

                return result

            except ValueError:
                return None

        return None

    @staticmethod
    def _error(
        *,
        operation: str,
        code: AssetToolError,
        message: str,
        asset_id: str | None = None,
    ) -> AssetToolResult:

        return AssetToolResult(
            success=False,
            operation=operation,
            asset_id=asset_id,
            error_code=code,
            error=message,
        )

    # =====================================================
    # Observability
    # =====================================================

    @staticmethod
    def _record_snapshot_metrics(
        snapshot: AssetSnapshot,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "asset_tool_snapshots_total"
            ).inc(
                labels={
                    "criticality":
                        snapshot
                        .criticality.value,

                    "data_quality":
                        snapshot
                        .data_quality.value,
                }
            )

            registry.histogram(
                "asset_tool_data_quality_score"
            ).observe(
                snapshot.data_quality_score
            )

            if snapshot.health.health_score is not None:
                registry.histogram(
                    "asset_tool_health_score"
                ).observe(
                    snapshot
                    .health
                    .health_score
                )

        except Exception:
            pass


# =========================================================
# Async helper
# =========================================================

def asyncio_create_task_safe(
    awaitable: Any,
):
    """
    Creates a task when running inside an event loop.

    Kept as a small wrapper so agent tools can be tested
    or instrumented centrally later.
    """
    import asyncio

    return asyncio.create_task(awaitable)


# =========================================================
# Tool-style convenience functions
# =========================================================

_default_asset_tools: AssetTools | None = None


def configure_asset_tools(
    tools: AssetTools,
) -> None:
    global _default_asset_tools
    _default_asset_tools = tools


def get_asset_tools() -> AssetTools:
    if _default_asset_tools is None:
        raise RuntimeError(
            "AssetTools has not been configured. "
            "Call configure_asset_tools() during "
            "application startup."
        )

    return _default_asset_tools


async def get_asset(
    asset_id: str,
) -> dict[str, Any]:
    result = await get_asset_tools().get_asset(
        asset_id
    )

    return result.to_dict()


async def get_asset_snapshot(
    asset_id: str,
) -> dict[str, Any]:
    result = (
        await get_asset_tools()
        .get_asset_snapshot(asset_id)
    )

    return result.to_dict()


async def get_latest_asset_telemetry(
    asset_id: str,
) -> dict[str, Any]:
    result = (
        await get_asset_tools()
        .get_latest_telemetry(asset_id)
    )

    return result.to_dict()


async def get_asset_maintenance_summary(
    asset_id: str,
) -> dict[str, Any]:
    result = (
        await get_asset_tools()
        .get_maintenance_summary(asset_id)
    )

    return result.to_dict()


async def get_asset_health_context(
    asset_id: str,
) -> dict[str, Any]:
    result = (
        await get_asset_tools()
        .get_health_context(asset_id)
    )

    return result.to_dict()


# =========================================================
# Exports
# =========================================================

__all__ = [
    "AssetStatus",
    "AssetCondition",
    "AssetCriticality",
    "DataQuality",
    "AssetToolError",

    "AssetRecord",
    "AssetTelemetry",
    "MaintenanceSummary",
    "AssetHealthContext",
    "AssetSnapshot",
    "AssetToolResult",

    "AssetRepository",
    "TelemetryRepository",
    "MaintenanceRepository",
    "HealthRepository",

    "AssetToolsConfig",
    "AssetTools",

    "configure_asset_tools",
    "get_asset_tools",

    "get_asset",
    "get_asset_snapshot",
    "get_latest_asset_telemetry",
    "get_asset_maintenance_summary",
    "get_asset_health_context",
]