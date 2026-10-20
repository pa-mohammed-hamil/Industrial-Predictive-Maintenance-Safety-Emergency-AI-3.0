
"""
app/services/anomaly_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
Anomaly Detection Service

Features:
- Temperature, vibration, pressure, current and voltage monitoring
- Statistical anomaly detection using Z-scores
- Multivariate anomaly detection
- Configurable sensor thresholds
- Severity and risk scoring
- Batch telemetry analysis
- Optional anomaly repository persistence
- Structured output for FastAPI and WebSocket integrations

No hardcoded production safety actions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean, pstdev
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

logger = logging.getLogger(__name__)


class AnomalySeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SensorThreshold:
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    warning_minimum: Optional[float] = None
    warning_maximum: Optional[float] = None


@dataclass
class AnomalyResult:
    anomaly_id: str
    asset_id: str
    sensor_name: str
    value: float
    is_anomaly: bool
    severity: str
    risk_score: float
    anomaly_type: str
    message: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_THRESHOLDS = {
    "temperature": SensorThreshold(
        minimum=-20,
        maximum=100,
        warning_maximum=80,
    ),
    "vibration": SensorThreshold(
        minimum=0,
        maximum=12,
        warning_maximum=7,
    ),
    "pressure": SensorThreshold(
        minimum=0,
        maximum=250,
        warning_maximum=200,
    ),
    "current": SensorThreshold(
        minimum=0,
        maximum=100,
        warning_maximum=80,
    ),
    "voltage": SensorThreshold(
        minimum=180,
        maximum=260,
        warning_minimum=200,
        warning_maximum=240,
    ),
}


class AnomalyService:
    """
    Application service for industrial telemetry anomalies.

    Thresholds are illustrative demo defaults only.
    Production thresholds must be configured per asset,
    sensor, unit, operating mode and safety requirements.
    """

    def __init__(
        self,
        anomaly_repository: Any = None,
        thresholds: Optional[
            Mapping[str, SensorThreshold]
        ] = None,
        z_score_threshold: float = 3.0,
        minimum_history: int = 10,
    ):
        if z_score_threshold <= 0:
            raise ValueError(
                "z_score_threshold must be positive"
            )

        if minimum_history < 2:
            raise ValueError(
                "minimum_history must be at least 2"
            )

        self.repository = anomaly_repository
        self.thresholds = dict(
            thresholds
            if thresholds is not None
            else DEFAULT_THRESHOLDS
        )
        self.z_score_threshold = z_score_threshold
        self.minimum_history = minimum_history

    @staticmethod
    def _validate_number(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("Boolean is not a sensor value")

        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid sensor value: {value}"
            ) from exc

        if not math.isfinite(number):
            raise ValueError(
                "Sensor value must be finite"
            )

        return number

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    def _threshold_detection(
        self,
        sensor_name: str,
        value: float,
    ) -> tuple[bool, float, str, str]:

        threshold = self.thresholds.get(
            sensor_name.lower()
        )

        if threshold is None:
            return (
                False,
                0.0,
                "none",
                "No configured threshold",
            )

        if (
            threshold.minimum is not None
            and value < threshold.minimum
        ):
            return (
                True,
                90.0,
                "threshold_low",
                "Below minimum threshold",
            )

        if (
            threshold.maximum is not None
            and value > threshold.maximum
        ):
            return (
                True,
                90.0,
                "threshold_high",
                "Above maximum threshold",
            )

        if (
            threshold.warning_minimum is not None
            and value < threshold.warning_minimum
        ):
            return (
                True,
                55.0,
                "warning_low",
                "Below warning threshold",
            )

        if (
            threshold.warning_maximum is not None
            and value > threshold.warning_maximum
        ):
            return (
                True,
                55.0,
                "warning_high",
                "Above warning threshold",
            )

        return (
            False,
            0.0,
            "none",
            "Within configured thresholds",
        )

    def _statistical_detection(
        self,
        value: float,
        history: Sequence[float],
    ) -> tuple[bool, float, float]:

        if len(history) < self.minimum_history:
            return False, 0.0, 0.0

        values = [
            self._validate_number(v)
            for v in history
        ]

        average = mean(values)
        deviation = pstdev(values)

        if deviation < 1e-9:
            if abs(value - average) < 1e-9:
                return False, 0.0, 0.0

            return True, 100.0, float("inf")

        z_score = abs(
            (value - average) / deviation
        )

        is_anomaly = (
            z_score >= self.z_score_threshold
        )

        risk_score = min(
            100.0,
            (z_score / self.z_score_threshold) * 65.0,
        ) if is_anomaly else 0.0

        return (
            is_anomaly,
            round(risk_score, 2),
            round(z_score, 4),
        )

    @staticmethod
    def _severity(
        risk_score: float,
    ) -> AnomalySeverity:

        if risk_score >= 85:
            return AnomalySeverity.CRITICAL

        if risk_score >= 65:
            return AnomalySeverity.HIGH

        if risk_score >= 40:
            return AnomalySeverity.MEDIUM

        if risk_score > 0:
            return AnomalySeverity.LOW

        return AnomalySeverity.NORMAL

    def detect_anomaly(
        self,
        asset_id: str,
        sensor_name: str,
        value: float,
        history: Optional[Sequence[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AnomalyResult:

        if not asset_id or not sensor_name:
            raise ValueError(
                "asset_id and sensor_name are required"
            )

        numeric_value = self._validate_number(
            value
        )

        (
            threshold_anomaly,
            threshold_risk,
            threshold_type,
            threshold_message,
        ) = self._threshold_detection(
            sensor_name,
            numeric_value,
        )

        statistical_anomaly = False
        statistical_risk = 0.0
        z_score = 0.0

        if history is not None:
            (
                statistical_anomaly,
                statistical_risk,
                z_score,
            ) = self._statistical_detection(
                numeric_value,
                history,
            )

        is_anomaly = (
            threshold_anomaly
            or statistical_anomaly
        )

        risk_score = max(
            threshold_risk,
            statistical_risk,
        )

        if threshold_anomaly and statistical_anomaly:
            anomaly_type = "combined"
            message = (
                "Threshold and statistical "
                "anomalies detected"
            )

        elif threshold_anomaly:
            anomaly_type = threshold_type
            message = threshold_message

        elif statistical_anomaly:
            anomaly_type = "statistical"
            message = (
                "Unusual deviation from "
                "historical sensor readings"
            )

        else:
            anomaly_type = "none"
            message = "No anomaly detected"

        result_metadata = dict(metadata or {})
        result_metadata["z_score"] = (
            z_score if math.isfinite(z_score)
            else None
        )
        result_metadata["threshold_anomaly"] = (
            threshold_anomaly
        )
        result_metadata["statistical_anomaly"] = (
            statistical_anomaly
        )
        result_metadata["history_size"] = len(
            history or []
        )

        result = AnomalyResult(
            anomaly_id=str(uuid4()),
            asset_id=asset_id,
            sensor_name=sensor_name,
            value=numeric_value,
            is_anomaly=is_anomaly,
            severity=self._severity(
                risk_score
            ).value,
            risk_score=round(risk_score, 2),
            anomaly_type=anomaly_type,
            message=message,
            timestamp=self._now(),
            metadata=result_metadata,
        )

        logger.info(
            "Anomaly analysis asset=%s sensor=%s "
            "anomaly=%s severity=%s",
            asset_id,
            sensor_name,
            result.is_anomaly,
            result.severity,
        )

        return result

    def detect_multivariate_anomalies(
        self,
        asset_id: str,
        telemetry: Mapping[str, float],
        histories: Optional[
            Mapping[str, Sequence[float]]
        ] = None,
    ) -> dict[str, Any]:

        if not telemetry:
            raise ValueError(
                "Telemetry cannot be empty"
            )

        results = []

        for sensor_name, value in telemetry.items():
            history = (
                histories.get(sensor_name)
                if histories
                else None
            )

            result = self.detect_anomaly(
                asset_id=asset_id,
                sensor_name=sensor_name,
                value=value,
                history=history,
            )

            results.append(result)

        anomalies = [
            result
            for result in results
            if result.is_anomaly
        ]

        max_risk = max(
            (result.risk_score for result in results),
            default=0.0,
        )

        average_risk = (
            sum(r.risk_score for r in results)
            / len(results)
        )

        return {
            "asset_id": asset_id,
            "timestamp": self._now(),
            "total_sensors": len(results),
            "anomaly_count": len(anomalies),
            "has_anomaly": bool(anomalies),
            "max_risk_score": round(max_risk, 2),
            "average_risk_score": round(
                average_risk, 2
            ),
            "overall_severity": self._severity(
                max_risk
            ).value,
            "sensor_results": [
                result.to_dict()
                for result in results
            ],
        }

    def analyze_batch(
        self,
        asset_id: str,
        telemetry_records: Sequence[
            Mapping[str, float]
        ],
    ) -> list[dict[str, Any]]:

        output = []

        histories: dict[str, list[float]] = {}

        for record in telemetry_records:
            analysis = (
                self.detect_multivariate_anomalies(
                    asset_id=asset_id,
                    telemetry=record,
                    histories=histories,
                )
            )

            output.append(analysis)

            for sensor, value in record.items():
                histories.setdefault(
                    sensor, []
                ).append(
                    self._validate_number(value)
                )

        return output

    async def save_anomaly(
        self,
        result: AnomalyResult,
    ) -> Any:
        """
        Optional persistence adapter.

        Expects an injected repository exposing
        async create_anomaly(data: dict).
        """

        if self.repository is None:
            raise RuntimeError(
                "Anomaly repository not configured"
            )

        if not result.is_anomaly:
            return None

        return await self.repository.create_anomaly(
            result.to_dict()
        )

    async def analyze_and_save(
        self,
        asset_id: str,
        sensor_name: str,
        value: float,
        history: Optional[
            Sequence[float]
        ] = None,
    ) -> AnomalyResult:

        result = self.detect_anomaly(
            asset_id=asset_id,
            sensor_name=sensor_name,
            value=value,
            history=history,
        )

        if result.is_anomaly:
            await self.save_anomaly(result)

        return result


if __name__ == "__main__":

    service = AnomalyService()

    sample = service.detect_multivariate_anomalies(
        asset_id="MACHINE-001",
        telemetry={
            "temperature": 105.0,
            "vibration": 9.5,
            "pressure": 180.0,
            "current": 65.0,
            "voltage": 230.0,
        },
    )

    import json

    print(
        json.dumps(
            sample,
            indent=2,
        )
    )
