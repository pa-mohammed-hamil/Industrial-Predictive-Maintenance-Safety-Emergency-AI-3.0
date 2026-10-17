"""
scripts/generate_telemetry.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Synthetic Industrial IoT Telemetry Generator
============================================

Python 3.11+

Dependencies:
    pip install numpy pandas

Outputs:
    data/synthetic/telemetry/
        assets.csv
        sensors.csv
        telemetry.csv
        anomalies.csv
        telemetry_alerts.csv
        asset_health.csv
        telemetry_summary.csv
        dataset_metadata.json

Features:
    - Multi-company, multi-site equipment simulation
    - Multiple industrial equipment types
    - Timestamped IoT sensor readings
    - Correlated temperature, vibration, current and load
    - Voltage, power and cumulative energy
    - Pressure and flow-rate measurements
    - Humidity and acoustic measurements
    - Progressive equipment degradation
    - Anomaly injection and ground-truth labels
    - Sensor threshold alerts
    - Equipment health scoring
    - Sensor quality and missing measurements
    - Reproducible random generation
    - CSV and JSON exports
    - Dataset validation
    - Command-line interface

All data is synthetic and intended for development,
ML experimentation, integration tests and demos.

This simulator is not calibrated to actual equipment
or certified industrial safety requirements.
"""

from __future__ import annotations

import argparse
import json
import math

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TelemetryConfig:
    output_dir: str = "data/synthetic/telemetry"

    num_assets: int = 100
    num_companies: int = 5
    sites_per_company: int = 3

    days: int = 30
    readings_per_day: int = 24

    seed: int = 42
    start_date: str = "2026-01-01"

    anomaly_probability: float = 0.025
    missing_probability: float = 0.005

    min_operating_hours_per_day: float = 8.0
    max_operating_hours_per_day: float = 24.0

    currency: str = "INR"


# ============================================================
# EQUIPMENT PROFILES
# ============================================================

EQUIPMENT_PROFILES: dict[str, dict[str, Any]] = {
    "pump": {
        "temperature": (40, 75),
        "vibration": (0.4, 3.5),
        "pressure": (2, 12),
        "current": (10, 90),
        "voltage": (380, 440),
        "rpm": (900, 3600),
        "flow_rate": (20, 500),
        "acoustic": (55, 85),
    },
    "motor": {
        "temperature": (35, 85),
        "vibration": (0.3, 3.0),
        "pressure": (0, 0.1),
        "current": (8, 180),
        "voltage": (380, 440),
        "rpm": (700, 3600),
        "flow_rate": (0, 0.1),
        "acoustic": (50, 85),
    },
    "compressor": {
        "temperature": (45, 105),
        "vibration": (0.6, 4.5),
        "pressure": (5, 25),
        "current": (20, 250),
        "voltage": (380, 440),
        "rpm": (900, 3000),
        "flow_rate": (50, 900),
        "acoustic": (65, 100),
    },
    "conveyor": {
        "temperature": (30, 70),
        "vibration": (0.3, 3.5),
        "pressure": (0, 0.1),
        "current": (5, 110),
        "voltage": (380, 440),
        "rpm": (50, 1200),
        "flow_rate": (0, 0.1),
        "acoustic": (50, 85),
    },
    "turbine": {
        "temperature": (65, 150),
        "vibration": (0.4, 4.0),
        "pressure": (5, 40),
        "current": (50, 500),
        "voltage": (380, 690),
        "rpm": (1500, 6000),
        "flow_rate": (100, 1500),
        "acoustic": (70, 105),
    },
    "generator": {
        "temperature": (40, 95),
        "vibration": (0.3, 3.5),
        "pressure": (0, 0.1),
        "current": (50, 600),
        "voltage": (380, 690),
        "rpm": (750, 3600),
        "flow_rate": (0, 0.1),
        "acoustic": (60, 95),
    },
    "robotic_arm": {
        "temperature": (30, 80),
        "vibration": (0.2, 2.5),
        "pressure": (0, 8),
        "current": (5, 100),
        "voltage": (200, 480),
        "rpm": (20, 600),
        "flow_rate": (0, 0.1),
        "acoustic": (45, 80),
    },
    "cnc_machine": {
        "temperature": (30, 85),
        "vibration": (0.3, 4.0),
        "pressure": (0, 10),
        "current": (15, 220),
        "voltage": (380, 480),
        "rpm": (100, 12000),
        "flow_rate": (0, 100),
        "acoustic": (55, 95),
    },
    "boiler": {
        "temperature": (80, 220),
        "vibration": (0.2, 3.0),
        "pressure": (5, 35),
        "current": (10, 150),
        "voltage": (380, 440),
        "rpm": (0, 500),
        "flow_rate": (50, 1000),
        "acoustic": (50, 90),
    },
    "hvac": {
        "temperature": (20, 65),
        "vibration": (0.2, 3.0),
        "pressure": (1, 12),
        "current": (5, 120),
        "voltage": (220, 440),
        "rpm": (400, 2400),
        "flow_rate": (20, 800),
        "acoustic": (40, 80),
    },
}


SENSOR_DEFINITIONS = {
    "temperature_c": {
        "sensor_type": "temperature",
        "unit": "C",
    },
    "vibration_mm_s": {
        "sensor_type": "vibration",
        "unit": "mm/s",
    },
    "pressure_bar": {
        "sensor_type": "pressure",
        "unit": "bar",
    },
    "current_a": {
        "sensor_type": "current",
        "unit": "A",
    },
    "voltage_v": {
        "sensor_type": "voltage",
        "unit": "V",
    },
    "rpm": {
        "sensor_type": "speed",
        "unit": "rpm",
    },
    "power_kw": {
        "sensor_type": "power",
        "unit": "kW",
    },
    "energy_kwh": {
        "sensor_type": "energy",
        "unit": "kWh",
    },
    "flow_rate_l_min": {
        "sensor_type": "flow",
        "unit": "L/min",
    },
    "humidity_percent": {
        "sensor_type": "humidity",
        "unit": "%",
    },
    "acoustic_db": {
        "sensor_type": "acoustic",
        "unit": "dB",
    },
}


ANOMALY_TYPES = [
    "temperature_spike",
    "vibration_spike",
    "pressure_drop",
    "current_surge",
    "voltage_drop",
    "rpm_instability",
    "flow_reduction",
    "acoustic_spike",
    "multi_sensor_anomaly",
]


# ============================================================
# GENERATOR
# ============================================================

class IndustrialTelemetryGenerator:
    def __init__(
        self,
        config: TelemetryConfig,
    ) -> None:
        self.config = config

        self.rng = np.random.default_rng(
            config.seed
        )

        self.output_dir = Path(
            config.output_dir
        )

        self.start_date = pd.Timestamp(
            config.start_date,
            tz="UTC",
        )

        self.total_steps = (
            config.days
            * config.readings_per_day
        )

        self.step_hours = (
            24.0 / config.readings_per_day
        )

        self._validate_config()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def _validate_config(self) -> None:
        positive_fields = {
            "num_assets": self.config.num_assets,
            "num_companies": self.config.num_companies,
            "sites_per_company": (
                self.config.sites_per_company
            ),
            "days": self.config.days,
            "readings_per_day": (
                self.config.readings_per_day
            ),
        }

        for name, value in positive_fields.items():
            if value < 1:
                raise ValueError(
                    f"{name} must be positive"
                )

        for name, probability in {
            "anomaly_probability": (
                self.config.anomaly_probability
            ),
            "missing_probability": (
                self.config.missing_probability
            ),
        }.items():
            if not 0 <= probability <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        if not (
            0
            < self.config.min_operating_hours_per_day
            <= self.config.max_operating_hours_per_day
            <= 24
        ):
            raise ValueError(
                "Operating hours must be between 0 and 24"
            )

    # --------------------------------------------------------
    # RANDOM HELPERS
    # --------------------------------------------------------

    def uniform(
        self,
        low: float,
        high: float,
    ) -> float:
        return float(
            self.rng.uniform(low, high)
        )

    def integer(
        self,
        low: int,
        high: int,
    ) -> int:
        return int(
            self.rng.integers(
                low,
                high + 1,
            )
        )

    def choice(
        self,
        values: list[str],
    ) -> str:
        return str(
            self.rng.choice(values)
        )

    # --------------------------------------------------------
    # GENERATE ASSETS
    # --------------------------------------------------------

    def generate_assets(self) -> pd.DataFrame:
        rows = []

        equipment_types = list(
            EQUIPMENT_PROFILES.keys()
        )

        for index in range(
            1,
            self.config.num_assets + 1,
        ):
            equipment_type = self.choice(
                equipment_types
            )

            company_number = self.integer(
                1,
                self.config.num_companies,
            )

            site_number = self.integer(
                1,
                self.config.sites_per_company,
            )

            company_id = (
                f"COMPANY-{company_number:03d}"
            )

            site_id = (
                f"{company_id}"
                f"-SITE-{site_number:03d}"
            )

            age_years = self.uniform(
                0.5,
                18,
            )

            initial_health = float(
                np.clip(
                    100
                    - age_years * 1.5
                    + self.rng.normal(0, 6),
                    45,
                    100,
                )
            )

            operating_hours_per_day = (
                self.uniform(
                    self.config
                    .min_operating_hours_per_day,
                    self.config
                    .max_operating_hours_per_day,
                )
            )

            rows.append({
                "asset_id": (
                    f"ASSET-{index:05d}"
                ),
                "company_id": company_id,
                "site_id": site_id,
                "equipment_type": equipment_type,
                "age_years": round(
                    age_years,
                    2,
                ),
                "initial_health_score": round(
                    initial_health,
                    3,
                ),
                "operating_hours_per_day": round(
                    operating_hours_per_day,
                    2,
                ),
                "criticality": self.choice([
                    "low",
                    "medium",
                    "high",
                    "critical",
                ]),
                "status": "active",
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # GENERATE SENSOR REGISTRY
    # --------------------------------------------------------

    def generate_sensors(
        self,
        assets: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []

        for asset in assets.to_dict(
            orient="records"
        ):
            for metric, definition in (
                SENSOR_DEFINITIONS.items()
            ):
                rows.append({
                    "sensor_id": (
                        f"{asset['asset_id']}"
                        f"-{definition['sensor_type']}"
                    ),
                    "asset_id": (
                        asset["asset_id"]
                    ),
                    "company_id": (
                        asset["company_id"]
                    ),
                    "site_id": (
                        asset["site_id"]
                    ),
                    "sensor_type": (
                        definition["sensor_type"]
                    ),
                    "metric_name": metric,
                    "unit": definition["unit"],
                    "protocol": self.choice([
                        "MQTT",
                        "OPC-UA",
                        "Modbus",
                        "HTTP",
                    ]),
                    "status": "active",
                })

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # BASELINE VALUES
    # --------------------------------------------------------

    def generate_baseline(
        self,
        equipment_type: str,
    ) -> dict[str, float]:
        profile = EQUIPMENT_PROFILES[
            equipment_type
        ]

        return {
            key: self.uniform(
                *value
            )
            for key, value in profile.items()
        }

    # --------------------------------------------------------
    # NORMAL SENSOR VALUES
    # --------------------------------------------------------

    def generate_normal_reading(
        self,
        baseline: dict[str, float],
        degradation: float,
        load_factor: float,
        ambient_temperature: float,
    ) -> dict[str, float]:
        temperature = (
            baseline["temperature"]
            + (load_factor - 0.7) * 20
            + degradation * 25
            + (ambient_temperature - 25) * 0.2
            + self.rng.normal(0, 1.5)
        )

        vibration = (
            baseline["vibration"]
            * (0.7 + load_factor * 0.4)
            + degradation * 4
            + self.rng.normal(0, 0.15)
        )

        pressure = (
            baseline["pressure"]
            * (0.6 + load_factor * 0.5)
            * (1 - degradation * 0.15)
        )

        current = (
            baseline["current"]
            * (0.5 + load_factor * 0.7)
            * (1 + degradation * 0.15)
        )

        voltage = (
            baseline["voltage"]
            + self.rng.normal(0, 2)
        )

        rpm = (
            baseline["rpm"]
            * (0.5 + load_factor * 0.6)
            * (1 - degradation * 0.08)
        )

        flow_rate = (
            baseline["flow_rate"]
            * (0.5 + load_factor * 0.6)
            * (1 - degradation * 0.20)
        )

        humidity = self.uniform(
            25,
            80,
        )

        acoustic = (
            baseline["acoustic"]
            + degradation * 12
            + load_factor * 3
            + self.rng.normal(0, 1.5)
        )

        power_factor = self.uniform(
            0.75,
            0.98,
        )

        # Illustrative three-phase electrical
        # power calculation.
        power_kw = (
            math.sqrt(3)
            * voltage
            * current
            * power_factor
            / 1000
        )

        return {
            "temperature_c": max(
                0,
                temperature,
            ),
            "vibration_mm_s": max(
                0,
                vibration,
            ),
            "pressure_bar": max(
                0,
                pressure,
            ),
            "current_a": max(
                0,
                current,
            ),
            "voltage_v": max(
                0,
                voltage,
            ),
            "rpm": max(
                0,
                rpm,
            ),
            "power_kw": max(
                0,
                power_kw,
            ),
            "flow_rate_l_min": max(
                0,
                flow_rate,
            ),
            "humidity_percent": float(
                np.clip(
                    humidity,
                    0,
                    100,
                )
            ),
            "acoustic_db": max(
                0,
                acoustic,
            ),
        }

    # --------------------------------------------------------
    # ANOMALY INJECTION
    # --------------------------------------------------------

    def inject_anomaly(
        self,
        reading: dict[str, float],
    ) -> tuple[
        dict[str, float],
        str,
    ]:
        anomaly_type = self.choice(
            ANOMALY_TYPES
        )

        modified = reading.copy()

        if anomaly_type == "temperature_spike":
            modified["temperature_c"] += (
                self.uniform(20, 80)
            )

        elif anomaly_type == "vibration_spike":
            modified["vibration_mm_s"] *= (
                self.uniform(2, 5)
            )

        elif anomaly_type == "pressure_drop":
            modified["pressure_bar"] *= (
                self.uniform(0.1, 0.5)
            )

        elif anomaly_type == "current_surge":
            modified["current_a"] *= (
                self.uniform(1.5, 3)
            )

        elif anomaly_type == "voltage_drop":
            modified["voltage_v"] *= (
                self.uniform(0.65, 0.90)
            )

        elif anomaly_type == "rpm_instability":
            modified["rpm"] *= (
                self.uniform(0.3, 1.7)
            )

        elif anomaly_type == "flow_reduction":
            modified["flow_rate_l_min"] *= (
                self.uniform(0.1, 0.5)
            )

        elif anomaly_type == "acoustic_spike":
            modified["acoustic_db"] += (
                self.uniform(10, 35)
            )

        elif anomaly_type == "multi_sensor_anomaly":
            modified["temperature_c"] += (
                self.uniform(15, 50)
            )

            modified["vibration_mm_s"] *= (
                self.uniform(1.5, 3)
            )

            modified["current_a"] *= (
                self.uniform(1.2, 2)
            )

        return modified, anomaly_type

    # --------------------------------------------------------
    # ALERT RULES
    # --------------------------------------------------------

    @staticmethod
    def evaluate_alerts(
        reading: dict[str, float],
        baseline: dict[str, float],
    ) -> list[dict[str, Any]]:
        alerts = []

        rules = {
            "temperature_c": (
                baseline["temperature"] + 35
            ),
            "vibration_mm_s": (
                baseline["vibration"] + 5
            ),
            "current_a": (
                baseline["current"] * 2.0
            ),
            "acoustic_db": (
                baseline["acoustic"] + 20
            ),
        }

        for metric, threshold in rules.items():
            value = reading.get(metric)

            if value is None:
                continue

            if value > threshold:
                alerts.append({
                    "metric": metric,
                    "value": round(
                        value,
                        3,
                    ),
                    "threshold": round(
                        threshold,
                        3,
                    ),
                    "operator": ">",
                    "severity": (
                        "critical"
                        if value > threshold * 1.3
                        else "high"
                    ),
                })

        return alerts

    # --------------------------------------------------------
    # GENERATE TELEMETRY
    # --------------------------------------------------------

    def generate_telemetry(
        self,
        assets: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        telemetry_rows = []
        anomaly_rows = []
        alert_rows = []
        health_rows = []

        anomaly_counter = 0
        alert_counter = 0

        for asset in assets.to_dict(
            orient="records"
        ):
            asset_id = asset["asset_id"]

            baseline = self.generate_baseline(
                asset["equipment_type"]
            )

            health = float(
                asset["initial_health_score"]
            )

            degradation = (
                100 - health
            ) / 100

            cumulative_energy = 0.0
            operating_hours = 0.0

            for step in range(
                self.total_steps
            ):
                timestamp = (
                    self.start_date
                    + pd.Timedelta(
                        hours=step * self.step_hours
                    )
                )

                load_factor = self.uniform(
                    0.35,
                    1.0,
                )

                ambient_temperature = (
                    25
                    + 7 * math.sin(
                        2
                        * math.pi
                        * (
                            step
                            % self.config.readings_per_day
                        )
                        / self.config.readings_per_day
                    )
                    + self.rng.normal(0, 1.5)
                )

                degradation += self.uniform(
                    0.0001,
                    0.0008,
                )

                degradation = float(
                    np.clip(
                        degradation,
                        0,
                        1,
                    )
                )

                health = float(
                    np.clip(
                        100 * (1 - degradation),
                        0,
                        100,
                    )
                )

                reading = self.generate_normal_reading(
                    baseline=baseline,
                    degradation=degradation,
                    load_factor=load_factor,
                    ambient_temperature=(
                        ambient_temperature
                    ),
                )

                is_anomaly = bool(
                    self.rng.random()
                    < self.config.anomaly_probability
                )

                anomaly_type = None

                if is_anomaly:
                    (
                        reading,
                        anomaly_type,
                    ) = self.inject_anomaly(
                        reading
                    )

                    anomaly_counter += 1

                    anomaly_rows.append({
                        "anomaly_id": (
                            f"ANOM-{anomaly_counter:07d}"
                        ),
                        "reading_id": (
                            f"{asset_id}-R-{step:07d}"
                        ),
                        "timestamp": (
                            timestamp.isoformat()
                        ),
                        "asset_id": asset_id,
                        "company_id": (
                            asset["company_id"]
                        ),
                        "site_id": (
                            asset["site_id"]
                        ),
                        "anomaly_type": (
                            anomaly_type
                        ),
                        "ground_truth": 1,
                        "source": "synthetic_injection",
                    })

                # Electrical consistency:
                # recompute power after current or
                # voltage anomalies.
                power_factor = 0.9

                reading["power_kw"] = (
                    math.sqrt(3)
                    * reading["voltage_v"]
                    * reading["current_a"]
                    * power_factor
                    / 1000
                )

                # Illustrative energy integration.
                cumulative_energy += (
                    reading["power_kw"]
                    * self.step_hours
                )

                operating_hours += (
                    self.step_hours
                    * (
                        asset[
                            "operating_hours_per_day"
                        ] / 24
                    )
                )

                reading["energy_kwh"] = (
                    cumulative_energy
                )

                alerts = self.evaluate_alerts(
                    reading,
                    baseline,
                )

                for alert in alerts:
                    alert_counter += 1

                    alert_rows.append({
                        "alert_id": (
                            f"ALERT-{alert_counter:07d}"
                        ),
                        "reading_id": (
                            f"{asset_id}-R-{step:07d}"
                        ),
                        "timestamp": (
                            timestamp.isoformat()
                        ),
                        "asset_id": asset_id,
                        "company_id": (
                            asset["company_id"]
                        ),
                        "site_id": (
                            asset["site_id"]
                        ),
                        **alert,
                        "status": "open",
                    })

                # Simulated sensor dropout.
                missing_metrics = []

                for metric in list(
                    reading.keys()
                ):
                    if (
                        self.rng.random()
                        < self.config.missing_probability
                    ):
                        reading[metric] = None

                        missing_metrics.append(
                            metric
                        )

                sensor_quality = (
                    "partial"
                    if missing_metrics
                    else "good"
                )

                telemetry_rows.append({
                    "reading_id": (
                        f"{asset_id}-R-{step:07d}"
                    ),
                    "timestamp": (
                        timestamp.isoformat()
                    ),
                    "asset_id": asset_id,
                    "company_id": (
                        asset["company_id"]
                    ),
                    "site_id": (
                        asset["site_id"]
                    ),
                    "equipment_type": (
                        asset["equipment_type"]
                    ),
                    "operating_hours": round(
                        operating_hours,
                        3,
                    ),
                    "load_factor": round(
                        load_factor,
                        5,
                    ),
                    "ambient_temperature_c": round(
                        ambient_temperature,
                        3,
                    ),
                    **{
                        key: (
                            round(value, 4)
                            if value is not None
                            else None
                        )
                        for key, value in reading.items()
                    },
                    "health_score": round(
                        health,
                        3,
                    ),
                    "degradation": round(
                        degradation,
                        6,
                    ),
                    "anomaly_label": int(
                        is_anomaly
                    ),
                    "anomaly_type": (
                        anomaly_type
                    ),
                    "alert_count": len(alerts),
                    "sensor_quality": (
                        sensor_quality
                    ),
                    "missing_metrics": (
                        ",".join(
                            missing_metrics
                        )
                    ),
                })

                health_rows.append({
                    "timestamp": (
                        timestamp.isoformat()
                    ),
                    "asset_id": asset_id,
                    "company_id": (
                        asset["company_id"]
                    ),
                    "site_id": (
                        asset["site_id"]
                    ),
                    "health_score": round(
                        health,
                        3,
                    ),
                    "degradation": round(
                        degradation,
                        6,
                    ),
                    "operating_hours": round(
                        operating_hours,
                        3,
                    ),
                })

        return (
            pd.DataFrame(telemetry_rows),
            pd.DataFrame(anomaly_rows),
            pd.DataFrame(alert_rows),
            pd.DataFrame(health_rows),
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def generate_summary(
        self,
        telemetry: pd.DataFrame,
    ) -> pd.DataFrame:
        summary = telemetry.groupby(
            [
                "company_id",
                "site_id",
                "asset_id",
                "equipment_type",
            ],
            as_index=False,
        ).agg(
            reading_count=(
                "reading_id",
                "count",
            ),
            avg_temperature_c=(
                "temperature_c",
                "mean",
            ),
            avg_vibration_mm_s=(
                "vibration_mm_s",
                "mean",
            ),
            avg_current_a=(
                "current_a",
                "mean",
            ),
            avg_power_kw=(
                "power_kw",
                "mean",
            ),
            anomaly_count=(
                "anomaly_label",
                "sum",
            ),
            alert_count=(
                "alert_count",
                "sum",
            ),
            latest_health_score=(
                "health_score",
                "last",
            ),
            total_energy_kwh=(
                "energy_kwh",
                "max",
            ),
        )

        return summary.round(3)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def validate_data(
        assets: pd.DataFrame,
        sensors: pd.DataFrame,
        telemetry: pd.DataFrame,
        anomalies: pd.DataFrame,
        alerts: pd.DataFrame,
    ) -> None:
        if assets.empty:
            raise ValueError(
                "Assets dataset is empty"
            )

        if telemetry.empty:
            raise ValueError(
                "Telemetry dataset is empty"
            )

        if assets["asset_id"].duplicated().any():
            raise ValueError(
                "Duplicate asset IDs"
            )

        if sensors["sensor_id"].duplicated().any():
            raise ValueError(
                "Duplicate sensor IDs"
            )

        if telemetry[
            "reading_id"
        ].duplicated().any():
            raise ValueError(
                "Duplicate telemetry reading IDs"
            )

        asset_ids = set(
            assets["asset_id"]
        )

        if not set(
            telemetry["asset_id"]
        ).issubset(asset_ids):
            raise ValueError(
                "Unknown telemetry asset reference"
            )

        reading_ids = set(
            telemetry["reading_id"]
        )

        if not anomalies.empty:
            if not set(
                anomalies["reading_id"]
            ).issubset(reading_ids):
                raise ValueError(
                    "Unknown anomaly reading reference"
                )

        if not alerts.empty:
            if not set(
                alerts["reading_id"]
            ).issubset(reading_ids):
                raise ValueError(
                    "Unknown alert reading reference"
                )

        if not telemetry[
            "health_score"
        ].between(0, 100).all():
            raise ValueError(
                "Invalid health scores"
            )

        if not telemetry[
            "degradation"
        ].between(0, 1).all():
            raise ValueError(
                "Invalid degradation values"
            )

        if not telemetry[
            "anomaly_label"
        ].isin([0, 1]).all():
            raise ValueError(
                "Invalid anomaly labels"
            )

        numeric_values = (
            telemetry.select_dtypes(
                include=[np.number]
            ).to_numpy()
        )

        if np.isinf(
            numeric_values
        ).any():
            raise ValueError(
                "Infinite telemetry values"
            )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def export(
        self,
        datasets: dict[str, pd.DataFrame],
    ) -> dict[str, str]:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths = {}

        for filename, dataframe in datasets.items():
            destination = (
                self.output_dir / filename
            )

            dataframe.to_csv(
                destination,
                index=False,
            )

            paths[filename] = str(destination)

        metadata = {
            "dataset_name": (
                "industrial_synthetic_telemetry"
            ),
            "synthetic": True,
            "generated_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "configuration": asdict(
                self.config
            ),
            "equipment_types": list(
                EQUIPMENT_PROFILES.keys()
            ),
            "sensor_definitions": (
                SENSOR_DEFINITIONS
            ),
            "files": {
                filename: len(dataframe)
                for filename, dataframe
                in datasets.items()
            },
            "limitations": [
                "All telemetry is synthetic",
                "Sensor correlations are illustrative",
                "Equipment health is simulator-derived",
                "Injected anomalies are artificial",
                "Sensor thresholds are not certified",
                "Power calculation assumes three-phase AC",
                "Energy integration is simplified",
                "Not calibrated for actual industrial equipment",
                "Use real operational data before deployment",
            ],
        }

        metadata_path = (
            self.output_dir
            / "dataset_metadata.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        paths[
            "dataset_metadata.json"
        ] = str(metadata_path)

        return paths

    # --------------------------------------------------------
    # GENERATE ALL
    # --------------------------------------------------------

    def generate_all(
        self,
    ) -> dict[str, str]:
        print(
            "[1/6] Generating industrial assets..."
        )

        assets = self.generate_assets()

        print(
            "[2/6] Generating sensor registry..."
        )

        sensors = self.generate_sensors(
            assets
        )

        print(
            "[3/6] Simulating IoT telemetry..."
        )

        (
            telemetry,
            anomalies,
            alerts,
            health,
        ) = self.generate_telemetry(
            assets
        )

        print(
            "[4/6] Generating telemetry summary..."
        )

        summary = self.generate_summary(
            telemetry
        )

        print(
            "[5/6] Validating datasets..."
        )

        self.validate_data(
            assets,
            sensors,
            telemetry,
            anomalies,
            alerts,
        )

        datasets = {
            "assets.csv": assets,
            "sensors.csv": sensors,
            "telemetry.csv": telemetry,
            "anomalies.csv": anomalies,
            "telemetry_alerts.csv": alerts,
            "asset_health.csv": health,
            "telemetry_summary.csv": summary,
        }

        print(
            "[6/6] Exporting datasets..."
        )

        paths = self.export(
            datasets
        )

        print("\nGeneration completed.")
        print("Assets:", len(assets))
        print("Sensors:", len(sensors))
        print("Telemetry readings:", len(telemetry))
        print("Injected anomalies:", len(anomalies))
        print("Generated alerts:", len(alerts))

        for filename, path in paths.items():
            print(f"{filename}: {path}")

        return paths


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic industrial "
            "IoT telemetry datasets"
        )
    )

    parser.add_argument(
        "--assets",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--readings-per-day",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--companies",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sites-per-company",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--anomaly-probability",
        type=float,
        default=0.025,
    )

    parser.add_argument(
        "--missing-probability",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic/telemetry",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TelemetryConfig(
        output_dir=args.output_dir,
        num_assets=args.assets,
        num_companies=args.companies,
        sites_per_company=args.sites_per_company,
        days=args.days,
        readings_per_day=args.readings_per_day,
        anomaly_probability=(
            args.anomaly_probability
        ),
        missing_probability=(
            args.missing_probability
        ),
        seed=args.seed,
    )

    generator = IndustrialTelemetryGenerator(
        config
    )

    generator.generate_all()


if __name__ == "__main__":
    main()