
"""
tests/unit/test_overheating.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Overheating Detection Engine Unit Tests

Run:
    pytest tests/unit/test_overheating.py -v

Coverage:
    - Normal temperature
    - Warning temperature
    - Critical overheating
    - Exact threshold boundaries
    - Temperature rise rate
    - Thermal anomalies
    - Multiple sensor readings
    - Asset-specific thresholds
    - Sensor validation
    - Invalid and missing readings
    - Deterministic detection
    - Safety alert recommendations

IMPORTANT:
Connect OverheatingAdapter to the actual
app/safety/overheating_engine.py implementation.

All thresholds are illustrative and must be
configured for the relevant industrial asset.

No real alarms, PLC commands, shutdowns, or
emergency dispatch should be triggered by tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import pytest


# ==================================================
# Domain models
# ==================================================

@dataclass(frozen=True)
class ThermalReading:
    asset_id: str
    temperature: float
    timestamp: float
    ambient_temperature: float = 25.0


@dataclass(frozen=True)
class ThermalThresholds:
    warning: float = 75.0
    high: float = 90.0
    critical: float = 110.0


@dataclass(frozen=True)
class OverheatingResult:
    asset_id: str
    overheating_detected: bool
    severity: str
    temperature: float
    alert_required: bool
    confidence: float | None = None
    temperature_rise_rate: float | None = None


# ==================================================
# Production interface
# ==================================================

class OverheatingEngineProtocol(Protocol):

    def detect(
        self,
        reading: ThermalReading,
        thresholds: ThermalThresholds,
    ) -> OverheatingResult:
        ...

    def detect_batch(
        self,
        readings: list[ThermalReading],
        thresholds: ThermalThresholds,
    ) -> list[OverheatingResult]:
        ...

    def calculate_temperature_rise_rate(
        self,
        previous: ThermalReading,
        current: ThermalReading,
    ) -> float:
        ...


# ==================================================
# Production adapter
# ==================================================

class OverheatingAdapter:
    """
    Connect to the actual overheating engine.

    Example:

        from app.safety.overheating_engine import (
            OverheatingEngine
        )

        self.engine = OverheatingEngine(...)

    Update method mappings to match the
    actual production implementation.

    Do not implement a duplicate overheating
    algorithm inside this test adapter.
    """

    def __init__(self) -> None:
        self.engine: Any = None

        # TODO:
        # Initialize your actual overheating engine.

    def detect(
        self,
        reading: ThermalReading,
        thresholds: ThermalThresholds,
    ) -> OverheatingResult:

        if self.engine is None:
            raise RuntimeError(
                "Overheating engine not configured"
            )

        # Example:
        #
        # raw = self.engine.detect(
        #     asset_id=reading.asset_id,
        #     temperature=reading.temperature,
        #     ambient_temperature=(
        #         reading.ambient_temperature
        #     ),
        #     timestamp=reading.timestamp,
        #     warning_threshold=thresholds.warning,
        #     high_threshold=thresholds.high,
        #     critical_threshold=thresholds.critical,
        # )
        #
        # return OverheatingResult(
        #     asset_id=str(raw["asset_id"]),
        #     overheating_detected=bool(
        #         raw["overheating_detected"]
        #     ),
        #     severity=str(raw["severity"]),
        #     temperature=float(raw["temperature"]),
        #     alert_required=bool(
        #         raw["alert_required"]
        #     ),
        #     confidence=raw.get("confidence"),
        #     temperature_rise_rate=raw.get(
        #         "temperature_rise_rate"
        #     ),
        # )

        raise NotImplementedError(
            "Map production overheating output"
        )

    def detect_batch(
        self,
        readings: list[ThermalReading],
        thresholds: ThermalThresholds,
    ) -> list[OverheatingResult]:

        return [
            self.detect(reading, thresholds)
            for reading in readings
        ]

    def calculate_temperature_rise_rate(
        self,
        previous: ThermalReading,
        current: ThermalReading,
    ) -> float:

        if self.engine is None:
            raise RuntimeError(
                "Overheating engine not configured"
            )

        # Map this call to the production method.
        raise NotImplementedError


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def engine() -> OverheatingEngineProtocol:

    adapter = OverheatingAdapter()

    if adapter.engine is None:
        pytest.skip(
            "Connect OverheatingAdapter to "
            "app/safety/overheating_engine.py"
        )

    return adapter


@pytest.fixture
def thresholds() -> ThermalThresholds:
    return ThermalThresholds(
        warning=75.0,
        high=90.0,
        critical=110.0,
    )


@pytest.fixture
def normal_reading() -> ThermalReading:
    return ThermalReading(
        asset_id="MOTOR-001",
        temperature=45.0,
        timestamp=1000.0,
    )


@pytest.fixture
def warning_reading() -> ThermalReading:
    return ThermalReading(
        asset_id="MOTOR-001",
        temperature=80.0,
        timestamp=1000.0,
    )


@pytest.fixture
def critical_reading() -> ThermalReading:
    return ThermalReading(
        asset_id="MOTOR-001",
        temperature=125.0,
        timestamp=1000.0,
    )


# ==================================================
# Basic overheating detection
# ==================================================

class TestOverheatingDetection:

    def test_normal_temperature(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        normal_reading: ThermalReading,
    ) -> None:

        result = engine.detect(
            normal_reading,
            thresholds,
        )

        assert result.overheating_detected is False
        assert result.severity == "normal"
        assert result.alert_required is False

    def test_warning_temperature(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        warning_reading: ThermalReading,
    ) -> None:

        result = engine.detect(
            warning_reading,
            thresholds,
        )

        assert result.overheating_detected is True
        assert result.severity == "warning"
        assert result.alert_required is True

    def test_high_temperature(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
    ) -> None:

        reading = ThermalReading(
            asset_id="MOTOR-001",
            temperature=100.0,
            timestamp=1000.0,
        )

        result = engine.detect(
            reading,
            thresholds,
        )

        assert result.overheating_detected is True
        assert result.severity == "high"
        assert result.alert_required is True

    def test_critical_overheating(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        critical_reading: ThermalReading,
    ) -> None:

        result = engine.detect(
            critical_reading,
            thresholds,
        )

        assert result.overheating_detected is True
        assert result.severity == "critical"
        assert result.alert_required is True

    def test_result_asset_id(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        normal_reading: ThermalReading,
    ) -> None:

        result = engine.detect(
            normal_reading,
            thresholds,
        )

        assert result.asset_id == "MOTOR-001"


# ==================================================
# Threshold boundaries
# ==================================================

class TestTemperatureThresholds:

    @pytest.mark.parametrize(
        "temperature,expected_severity",
        [
            (74.9, "normal"),
            (75.0, "warning"),
            (89.9, "warning"),
            (90.0, "high"),
            (109.9, "high"),
            (110.0, "critical"),
        ],
    )
    def test_threshold_boundaries(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        temperature: float,
        expected_severity: str,
    ) -> None:

        reading = ThermalReading(
            asset_id="MOTOR-001",
            temperature=temperature,
            timestamp=1000.0,
        )

        result = engine.detect(
            reading,
            thresholds,
        )

        assert result.severity == expected_severity

    def test_custom_thresholds(
        self,
        engine: OverheatingEngineProtocol,
    ) -> None:

        custom = ThermalThresholds(
            warning=50.0,
            high=65.0,
            critical=80.0,
        )

        reading = ThermalReading(
            asset_id="BATTERY-001",
            temperature=85.0,
            timestamp=1000.0,
        )

        result = engine.detect(
            reading,
            custom,
        )

        assert result.severity == "critical"

    def test_invalid_threshold_order_rejected(
        self,
        engine: OverheatingEngineProtocol,
    ) -> None:

        invalid = ThermalThresholds(
            warning=100.0,
            high=80.0,
            critical=70.0,
        )

        reading = ThermalReading(
            asset_id="MOTOR-001",
            temperature=90.0,
            timestamp=1000.0,
        )

        with pytest.raises(ValueError):
            engine.detect(reading, invalid)


# ==================================================
# Temperature rise rate
# ==================================================

class TestTemperatureRiseRate:

    def test_positive_temperature_rise(
        self,
        engine: OverheatingEngineProtocol,
    ) -> None:

        previous = ThermalReading(
            asset_id="MOTOR-001",
            temperature=50.0,
            timestamp=1000.0,
        )

        current = ThermalReading(
            asset_id="MOTOR-001",
            temperature=80.0,
            timestamp=1060.0,
        )

        rate = engine.calculate_temperature_rise_rate(
            previous,
            current,
        )

        # 30 degrees / 60 seconds = 0.5 C/s
        assert rate == pytest.approx(0.5)

    def test_negative_temperature_rise(
        self,
        engine: OverheatingEngineProtocol,
    ) -> None:

        previous = ThermalReading(
            asset_id="MOTOR-001",
            temperature=90.0,
            timestamp=1000.0,
        )

        current = ThermalReading(
            asset_id="MOTOR-001",
            temperature=60.0,
            timestamp=1060.0,
        )

        rate = engine.calculate_temperature_rise_rate(
            previous,
            current,
        )

        assert rate == pytest.approx(-0.5)

    def test_stable_temperature(
        self,
        engine: OverheatingEngineProtocol,
    ) -> None:

        previous = ThermalReading(
            asset_id="MOTOR-001",
            temperature=60.0,
            timestamp=1000.0,
        )

        current = ThermalReading(
            asset_id="MOTOR-001",
            temperature=60.0,
            timestamp=1060.0,
        )

        rate = engine.calculate_temperature_rise_rate(
            previous,
            current,
        )

        assert rate == pytest.approx(0.0)

    def test_duplicate_timestamp_rejected(
        self,
        engine: OverheatingEngineProtocol,
    ) -> None:

        previous = ThermalReading(
            asset_id="MOTOR-001",
            temperature=50.0,
            timestamp=1000.0,
        )

        current = ThermalReading(
            asset_id="MOTOR-001",
            temperature=80.0,
            timestamp=1000.0,
        )

        with pytest.raises(ValueError):
            engine.calculate_temperature_rise_rate(
                previous,
                current,
            )

    def test_out_of_order_timestamp_rejected(
        self,
        engine: OverheatingEngineProtocol,
    ) -> None:

        previous = ThermalReading(
            asset_id="MOTOR-001",
            temperature=50.0,
            timestamp=1060.0,
        )

        current = ThermalReading(
            asset_id="MOTOR-001",
            temperature=80.0,
            timestamp=1000.0,
        )

        with pytest.raises(ValueError):
            engine.calculate_temperature_rise_rate(
                previous,
                current,
            )

    def test_different_assets_rejected(
        self,
        engine: OverheatingEngineProtocol,
    ) -> None:

        previous = ThermalReading(
            asset_id="MOTOR-001",
            temperature=50.0,
            timestamp=1000.0,
        )

        current = ThermalReading(
            asset_id="PUMP-002",
            temperature=80.0,
            timestamp=1060.0,
        )

        with pytest.raises(ValueError):
            engine.calculate_temperature_rise_rate(
                previous,
                current,
            )


# ==================================================
# Batch processing
# ==================================================

class TestOverheatingBatch:

    def test_batch_result_count(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        normal_reading: ThermalReading,
        warning_reading: ThermalReading,
        critical_reading: ThermalReading,
    ) -> None:

        readings = [
            normal_reading,
            warning_reading,
            critical_reading,
        ]

        results = engine.detect_batch(
            readings,
            thresholds,
        )

        assert len(results) == 3

    def test_batch_preserves_asset_ids(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
    ) -> None:

        readings = [
            ThermalReading("MOTOR-001", 45.0, 1000.0),
            ThermalReading("PUMP-002", 85.0, 1000.0),
            ThermalReading("FAN-003", 120.0, 1000.0),
        ]

        results = engine.detect_batch(
            readings,
            thresholds,
        )

        assert [r.asset_id for r in results] == [
            "MOTOR-001",
            "PUMP-002",
            "FAN-003",
        ]

    def test_empty_batch(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
    ) -> None:

        assert engine.detect_batch(
            [],
            thresholds,
        ) == []

    def test_batch_matches_individual(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        normal_reading: ThermalReading,
        warning_reading: ThermalReading,
    ) -> None:

        readings = [
            normal_reading,
            warning_reading,
        ]

        batch = engine.detect_batch(
            readings,
            thresholds,
        )

        individual = [
            engine.detect(reading, thresholds)
            for reading in readings
        ]

        assert batch == individual


# ==================================================
# Validation
# ==================================================

class TestOverheatingValidation:

    @pytest.mark.parametrize(
        "invalid_value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_nonfinite_temperature_rejected(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        invalid_value: float,
    ) -> None:

        reading = ThermalReading(
            asset_id="MOTOR-001",
            temperature=invalid_value,
            timestamp=1000.0,
        )

        with pytest.raises(ValueError):
            engine.detect(reading, thresholds)

    def test_empty_asset_id_rejected(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
    ) -> None:

        reading = ThermalReading(
            asset_id="",
            temperature=80.0,
            timestamp=1000.0,
        )

        with pytest.raises(ValueError):
            engine.detect(reading, thresholds)

    def test_invalid_timestamp_rejected(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
    ) -> None:

        reading = ThermalReading(
            asset_id="MOTOR-001",
            temperature=80.0,
            timestamp=float("nan"),
        )

        with pytest.raises(ValueError):
            engine.detect(reading, thresholds)

    def test_nonfinite_ambient_temperature_rejected(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
    ) -> None:

        reading = ThermalReading(
            asset_id="MOTOR-001",
            temperature=80.0,
            timestamp=1000.0,
            ambient_temperature=float("inf"),
        )

        with pytest.raises(ValueError):
            engine.detect(reading, thresholds)

    def test_missing_reading_rejected(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
    ) -> None:

        with pytest.raises(
            (TypeError, ValueError)
        ):
            engine.detect(
                None,  # type: ignore[arg-type]
                thresholds,
            )


# ==================================================
# Output integrity
# ==================================================

class TestOverheatingIntegrity:

    def test_result_type(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        normal_reading: ThermalReading,
    ) -> None:

        result = engine.detect(
            normal_reading,
            thresholds,
        )

        assert isinstance(
            result,
            OverheatingResult,
        )

    def test_valid_severity(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        normal_reading: ThermalReading,
    ) -> None:

        result = engine.detect(
            normal_reading,
            thresholds,
        )

        assert result.severity in {
            "normal",
            "warning",
            "high",
            "critical",
        }

    def test_confidence_range(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        normal_reading: ThermalReading,
    ) -> None:

        result = engine.detect(
            normal_reading,
            thresholds,
        )

        if result.confidence is not None:
            assert math.isfinite(result.confidence)
            assert 0 <= result.confidence <= 1

    def test_temperature_preserved(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        warning_reading: ThermalReading,
    ) -> None:

        result = engine.detect(
            warning_reading,
            thresholds,
        )

        assert result.temperature == pytest.approx(
            warning_reading.temperature
        )

    def test_deterministic_detection(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        warning_reading: ThermalReading,
    ) -> None:

        first = engine.detect(
            warning_reading,
            thresholds,
        )

        second = engine.detect(
            warning_reading,
            thresholds,
        )

        assert first == second

    def test_critical_alert_required(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
        critical_reading: ThermalReading,
    ) -> None:

        result = engine.detect(
            critical_reading,
            thresholds,
        )

        assert result.alert_required is True
        assert result.severity == "critical"

    def test_severity_increases_with_temperature(
        self,
        engine: OverheatingEngineProtocol,
        thresholds: ThermalThresholds,
    ) -> None:

        temperatures = [
            45.0,
            80.0,
            100.0,
            125.0,
        ]

        severity_order = {
            "normal": 0,
            "warning": 1,
            "high": 2,
            "critical": 3,
        }

        results = [
            engine.detect(
                ThermalReading(
                    asset_id="MOTOR-001",
                    temperature=temp,
                    timestamp=1000.0,
                ),
                thresholds,
            )
            for temp in temperatures
        ]

        ranks = [
            severity_order[result.severity]
            for result in results
        ]

        assert ranks == sorted(ranks)
