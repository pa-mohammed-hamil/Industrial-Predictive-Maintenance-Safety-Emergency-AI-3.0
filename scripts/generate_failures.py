"""
scripts/generate_failures.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Synthetic Equipment Failure Dataset Generator
==============================================

Python 3.11+

Dependencies:
    pip install numpy pandas

Generated files:
    data/synthetic/failures/
        assets.csv
        sensor_readings.csv
        failure_events.csv
        failure_predictions.csv
        rul_labels.csv
        maintenance_history.csv
        failure_summary.csv
        dataset_metadata.json

Features:
- Multi-company, multi-site equipment fleet
- Equipment-specific operating characteristics
- Time-series sensor telemetry
- Progressive equipment degradation
- Simulated equipment failure events
- Failure modes and severity
- Failure prediction labels
- Remaining Useful Life (RUL)
- Predictive maintenance events
- Downtime and repair estimates
- Reproducible random generation
- Dataset validation
- CSV and JSON exports

All data is synthetic.

Failure probabilities and degradation curves
are illustrative, not calibrated reliability models.
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
class FailureDataConfig:
    output_dir: str = "data/synthetic/failures"

    num_assets: int = 100
    num_companies: int = 5
    sites_per_company: int = 3

    days: int = 180
    readings_per_day: int = 4

    seed: int = 42
    start_date: str = "2026-01-01"

    prediction_horizon_hours: int = 72

    maintenance_probability: float = 0.03

    min_operating_hours_per_day: float = 8.0
    max_operating_hours_per_day: float = 24.0

    currency: str = "INR"


# ============================================================
# EQUIPMENT PROFILES
# ============================================================

EQUIPMENT_PROFILES: dict[str, dict[str, Any]] = {
    "pump": {
        "temperature": (45, 80),
        "vibration": (0.5, 4.0),
        "pressure": (2.0, 12.0),
        "current": (10, 80),
        "rpm": (900, 3600),
        "failure_modes": [
            "bearing_failure",
            "seal_leakage",
            "cavitation",
            "impeller_damage",
            "shaft_misalignment",
        ],
        "base_failure_rate": 0.00035,
        "replacement_cost": (80000, 700000),
    },
    "motor": {
        "temperature": (40, 90),
        "vibration": (0.4, 3.5),
        "pressure": (0, 0.1),
        "current": (8, 150),
        "rpm": (700, 3600),
        "failure_modes": [
            "bearing_failure",
            "winding_failure",
            "overheating",
            "insulation_failure",
            "rotor_imbalance",
        ],
        "base_failure_rate": 0.00030,
        "replacement_cost": (50000, 900000),
    },
    "compressor": {
        "temperature": (50, 105),
        "vibration": (0.8, 5.0),
        "pressure": (5, 25),
        "current": (20, 250),
        "rpm": (900, 3000),
        "failure_modes": [
            "bearing_failure",
            "valve_failure",
            "pressure_loss",
            "lubrication_failure",
            "seal_failure",
        ],
        "base_failure_rate": 0.00040,
        "replacement_cost": (250000, 3500000),
    },
    "conveyor": {
        "temperature": (30, 75),
        "vibration": (0.4, 4.0),
        "pressure": (0, 0.1),
        "current": (5, 100),
        "rpm": (50, 1200),
        "failure_modes": [
            "belt_failure",
            "bearing_failure",
            "motor_failure",
            "roller_failure",
            "misalignment",
        ],
        "base_failure_rate": 0.00032,
        "replacement_cost": (150000, 2000000),
    },
    "turbine": {
        "temperature": (70, 160),
        "vibration": (0.5, 4.5),
        "pressure": (5, 40),
        "current": (50, 500),
        "rpm": (1500, 6000),
        "failure_modes": [
            "blade_damage",
            "bearing_failure",
            "rotor_imbalance",
            "overheating",
            "lubrication_failure",
        ],
        "base_failure_rate": 0.00020,
        "replacement_cost": (2000000, 30000000),
    },
    "generator": {
        "temperature": (40, 100),
        "vibration": (0.4, 4.0),
        "pressure": (0, 0.1),
        "current": (50, 600),
        "rpm": (750, 3600),
        "failure_modes": [
            "bearing_failure",
            "insulation_failure",
            "overheating",
            "voltage_instability",
            "winding_failure",
        ],
        "base_failure_rate": 0.00025,
        "replacement_cost": (300000, 5000000),
    },
    "robotic_arm": {
        "temperature": (30, 85),
        "vibration": (0.2, 3.0),
        "pressure": (0, 8),
        "current": (5, 100),
        "rpm": (20, 600),
        "failure_modes": [
            "servo_failure",
            "joint_wear",
            "encoder_failure",
            "gearbox_failure",
            "controller_fault",
        ],
        "base_failure_rate": 0.00028,
        "replacement_cost": (700000, 12000000),
    },
    "cnc_machine": {
        "temperature": (30, 90),
        "vibration": (0.3, 4.5),
        "pressure": (0, 10),
        "current": (15, 200),
        "rpm": (100, 12000),
        "failure_modes": [
            "spindle_failure",
            "bearing_failure",
            "servo_failure",
            "cooling_failure",
            "tool_wear",
        ],
        "base_failure_rate": 0.00033,
        "replacement_cost": (1200000, 15000000),
    },
}


SEVERITY_LEVELS = [
    "low",
    "medium",
    "high",
    "critical",
]


# ============================================================
# GENERATOR
# ============================================================

class IndustrialFailureDataGenerator:
    def __init__(
        self,
        config: FailureDataConfig,
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
        if self.config.num_assets < 1:
            raise ValueError(
                "num_assets must be positive"
            )

        if self.config.num_companies < 1:
            raise ValueError(
                "num_companies must be positive"
            )

        if self.config.sites_per_company < 1:
            raise ValueError(
                "sites_per_company must be positive"
            )

        if self.config.days < 2:
            raise ValueError(
                "days must be at least 2"
            )

        if self.config.readings_per_day < 1:
            raise ValueError(
                "readings_per_day must be positive"
            )

        if not 0 <= self.config.maintenance_probability <= 1:
            raise ValueError(
                "maintenance_probability must be 0-1"
            )

        if self.config.prediction_horizon_hours <= 0:
            raise ValueError(
                "prediction_horizon_hours must be positive"
            )

        if not (
            0 < self.config.min_operating_hours_per_day
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
            self.rng.integers(low, high + 1)
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
            EQUIPMENT_PROFILES
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

            age_years = self.uniform(
                0.5,
                18,
            )

            initial_health = float(
                np.clip(
                    100 - age_years * 1.6
                    + self.rng.normal(0, 6),
                    45,
                    100,
                )
            )

            profile = EQUIPMENT_PROFILES[
                equipment_type
            ]

            replacement_cost = self.uniform(
                *profile["replacement_cost"]
            )

            operating_hours_per_day = self.uniform(
                self.config.min_operating_hours_per_day,
                self.config.max_operating_hours_per_day,
            )

            rows.append({
                "asset_id": f"ASSET-{index:05d}",
                "company_id": (
                    f"COMPANY-{company_number:03d}"
                ),
                "site_id": (
                    f"COMPANY-{company_number:03d}"
                    f"-SITE-{site_number:03d}"
                ),
                "equipment_type": equipment_type,
                "age_years": round(
                    age_years,
                    2,
                ),
                "initial_health_score": round(
                    initial_health,
                    2,
                ),
                "operating_hours_per_day": round(
                    operating_hours_per_day,
                    2,
                ),
                "replacement_cost": round(
                    replacement_cost,
                    2,
                ),
                "currency": self.config.currency,
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # FAILURE SEVERITY
    # --------------------------------------------------------

    def _failure_severity(
        self,
        degradation: float,
    ) -> str:
        if degradation >= 0.90:
            return "critical"

        if degradation >= 0.75:
            return "high"

        if degradation >= 0.55:
            return "medium"

        return "low"

    # --------------------------------------------------------
    # SENSOR VALUE GENERATION
    # --------------------------------------------------------

    def _sensor_values(
        self,
        equipment_type: str,
        degradation: float,
    ) -> dict[str, float]:
        profile = EQUIPMENT_PROFILES[
            equipment_type
        ]

        def baseline(
            key: str,
        ) -> float:
            low, high = profile[key]

            return self.uniform(
                low,
                high,
            )

        temperature = (
            baseline("temperature")
            + degradation * 28
            + self.rng.normal(0, 2)
        )

        vibration = (
            baseline("vibration")
            + degradation * 5
            + self.rng.normal(0, 0.25)
        )

        pressure = (
            baseline("pressure")
            * (1 - degradation * 0.12)
        )

        current = (
            baseline("current")
            * (1 + degradation * 0.22)
        )

        rpm = (
            baseline("rpm")
            * (1 - degradation * 0.08)
        )

        return {
            "temperature_c": round(
                max(0, temperature),
                3,
            ),
            "vibration_mm_s": round(
                max(0, vibration),
                3,
            ),
            "pressure_bar": round(
                max(0, pressure),
                3,
            ),
            "current_a": round(
                max(0, current),
                3,
            ),
            "rpm": round(
                max(0, rpm),
                3,
            ),
        }

    # --------------------------------------------------------
    # GENERATE TELEMETRY AND FAILURES
    # --------------------------------------------------------

    def generate_time_series(
        self,
        assets: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        sensor_rows = []
        failure_rows = []
        maintenance_rows = []

        failure_counter = 0
        maintenance_counter = 0

        for asset in assets.to_dict(
            orient="records"
        ):
            asset_id = asset["asset_id"]
            equipment_type = asset[
                "equipment_type"
            ]

            profile = EQUIPMENT_PROFILES[
                equipment_type
            ]

            health = float(
                asset["initial_health_score"]
            )

            degradation = (
                100 - health
            ) / 100

            cycle_count = 0

            for step in range(
                self.total_steps
            ):
                timestamp = (
                    self.start_date
                    + pd.Timedelta(
                        hours=step * self.step_hours
                    )
                )

                cycle_count += 1

                # Progressive wear model.
                degradation_increment = (
                    self.uniform(
                        0.0002,
                        0.0012,
                    )
                    * (
                        1
                        + asset["age_years"] / 20
                    )
                )

                degradation = float(
                    np.clip(
                        degradation
                        + degradation_increment,
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

                sensors = self._sensor_values(
                    equipment_type,
                    degradation,
                )

                # Hazard model is a per-reading
                # synthetic event probability.
                base_rate = profile[
                    "base_failure_rate"
                ]

                failure_probability = float(
                    np.clip(
                        base_rate
                        * (
                            1
                            + 12 * degradation ** 3
                        )
                        * (
                            self.step_hours / 6
                        ),
                        0,
                        0.95,
                    )
                )

                failed = bool(
                    self.rng.random()
                    < failure_probability
                )

                failure_event_id = None

                if failed:
                    failure_counter += 1

                    failure_event_id = (
                        f"FAIL-{failure_counter:06d}"
                    )

                    failure_mode = self.choice(
                        profile["failure_modes"]
                    )

                    severity = (
                        self._failure_severity(
                            degradation
                        )
                    )

                    downtime_hours = (
                        self.uniform(2, 48)
                        * (
                            1 + degradation
                        )
                    )

                    repair_cost = (
                        float(
                            asset[
                                "replacement_cost"
                            ]
                        )
                        * self.uniform(
                            0.02,
                            0.22,
                        )
                    )

                    failure_rows.append({
                        "failure_id": (
                            failure_event_id
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
                            equipment_type
                        ),
                        "failure_mode": (
                            failure_mode
                        ),
                        "severity": severity,
                        "health_score_before": (
                            round(health, 3)
                        ),
                        "degradation_before": (
                            round(degradation, 5)
                        ),
                        "downtime_hours": round(
                            downtime_hours,
                            3,
                        ),
                        "repair_cost": round(
                            repair_cost,
                            2,
                        ),
                        "currency": (
                            self.config.currency
                        ),
                    })

                    maintenance_counter += 1

                    maintenance_rows.append({
                        "maintenance_id": (
                            f"MAINT-"
                            f"{maintenance_counter:06d}"
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
                        "maintenance_type": (
                            "corrective"
                        ),
                        "failure_id": (
                            failure_event_id
                        ),
                        "health_before": round(
                            health,
                            3,
                        ),
                        "health_after": None,
                        "estimated_cost": round(
                            repair_cost,
                            2,
                        ),
                        "currency": (
                            self.config.currency
                        ),
                    })

                    # Simulated repair restores some
                    # health but not a new-equipment
                    # condition.
                    degradation *= self.uniform(
                        0.25,
                        0.60,
                    )

                    restored_health = (
                        100 * (1 - degradation)
                    )

                    maintenance_rows[-1][
                        "health_after"
                    ] = round(
                        restored_health,
                        3,
                    )

                else:
                    # Scheduled/condition-based
                    # maintenance opportunity.
                    maintenance_chance = (
                        self.config
                        .maintenance_probability
                        / self.config.readings_per_day
                    )

                    if (
                        degradation >= 0.25
                        and self.rng.random()
                        < maintenance_chance
                    ):
                        maintenance_counter += 1

                        maintenance_type = (
                            "predictive"
                            if degradation >= 0.50
                            else "preventive"
                        )

                        health_before = health

                        degradation *= self.uniform(
                            0.55,
                            0.90,
                        )

                        maintenance_rows.append({
                            "maintenance_id": (
                                f"MAINT-"
                                f"{maintenance_counter:06d}"
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
                            "maintenance_type": (
                                maintenance_type
                            ),
                            "failure_id": None,
                            "health_before": round(
                                health_before,
                                3,
                            ),
                            "health_after": round(
                                100 * (
                                    1 - degradation
                                ),
                                3,
                            ),
                            "estimated_cost": round(
                                float(
                                    asset[
                                        "replacement_cost"
                                    ]
                                )
                                * self.uniform(
                                    0.002,
                                    0.035,
                                ),
                                2,
                            ),
                            "currency": (
                                self.config.currency
                            ),
                        })

                sensor_rows.append({
                    "reading_id": (
                        f"{asset_id}-R-{step:06d}"
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
                        equipment_type
                    ),
                    "cycle_count": cycle_count,
                    "operating_hours": round(
                        cycle_count
                        * self.step_hours
                        * (
                            asset[
                                "operating_hours_per_day"
                            ] / 24
                        ),
                        3,
                    ),
                    **sensors,
                    "health_score": round(
                        health,
                        3,
                    ),
                    "degradation": round(
                        (
                            100 - health
                        ) / 100,
                        5,
                    ),
                    "simulated_failure_probability": (
                        round(
                            failure_probability,
                            8,
                        )
                    ),
                    "failure_event": int(
                        failed
                    ),
                    "failure_id": (
                        failure_event_id
                    ),
                })

        return (
            pd.DataFrame(sensor_rows),
            pd.DataFrame(failure_rows),
            pd.DataFrame(maintenance_rows),
        )

    # --------------------------------------------------------
    # GENERATE PREDICTION LABELS
    # --------------------------------------------------------

    def generate_prediction_labels(
        self,
        telemetry: pd.DataFrame,
        failures: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Produce labels based on the next observed
        failure for the same asset.

        Censored observations have no known next
        failure within the simulation window.

        The prediction horizon is inclusive.
        """

        data = telemetry.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            utc=True,
        )

        data = data.sort_values(
            ["asset_id", "timestamp"]
        ).reset_index(drop=True)

        data["next_failure_at"] = pd.NaT

        if not failures.empty:
            failure_data = failures.copy()

            failure_data["timestamp"] = (
                pd.to_datetime(
                    failure_data["timestamp"],
                    utc=True,
                )
            )

            for asset_id, group in data.groupby(
                "asset_id",
                sort=False,
            ):
                asset_failures = failure_data.loc[
                    failure_data["asset_id"]
                    == asset_id,
                    "timestamp",
                ].sort_values()

                if asset_failures.empty:
                    continue

                failure_times = (
                    asset_failures.to_numpy(
                        dtype="datetime64[ns]"
                    )
                )

                reading_times = (
                    group["timestamp"].to_numpy(
                        dtype="datetime64[ns]"
                    )
                )

                # Strictly future failures only.
                positions = np.searchsorted(
                    failure_times,
                    reading_times,
                    side="right",
                )

                valid = (
                    positions
                    < len(failure_times)
                )

                next_times = np.full(
                    len(group),
                    np.datetime64("NaT"),
                    dtype="datetime64[ns]",
                )

                next_times[valid] = (
                    failure_times[
                        positions[valid]
                    ]
                )

                data.loc[
                    group.index,
                    "next_failure_at",
                ] = pd.to_datetime(
                    next_times,
                    utc=True,
                ).to_numpy()

        data["rul_hours"] = (
            (
                data["next_failure_at"]
                - data["timestamp"]
            ).dt.total_seconds()
            / 3600
        )

        data["rul_observed"] = (
            data["rul_hours"].notna()
        )

        data["failure_within_horizon"] = (
            (
                data["rul_hours"]
                <= self.config.prediction_horizon_hours
            )
            & data["rul_observed"]
        ).astype(int)

        # Important:
        # Do not use degradation, health_score or
        # simulated_failure_probability as training
        # features unless the experiment explicitly
        # evaluates oracle/simulator-derived inputs.
        #
        # They are latent simulator variables and
        # can cause unrealistic label leakage.

        prediction_columns = [
            "reading_id",
            "timestamp",
            "asset_id",
            "company_id",
            "site_id",
            "equipment_type",
            "temperature_c",
            "vibration_mm_s",
            "pressure_bar",
            "current_a",
            "rpm",
            "operating_hours",
            "failure_within_horizon",
            "rul_observed",
        ]

        prediction_data = data[
            prediction_columns
        ].copy()

        rul_data = data[
            [
                "reading_id",
                "timestamp",
                "asset_id",
                "company_id",
                "site_id",
                "equipment_type",
                "rul_hours",
                "rul_observed",
            ]
        ].copy()

        return (
            prediction_data,
            rul_data,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def generate_summary(
        self,
        assets: pd.DataFrame,
        telemetry: pd.DataFrame,
        failures: pd.DataFrame,
        maintenance: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []

        for asset in assets.to_dict(
            orient="records"
        ):
            asset_id = asset["asset_id"]

            asset_telemetry = telemetry[
                telemetry["asset_id"]
                == asset_id
            ]

            asset_failures = (
                failures[
                    failures["asset_id"]
                    == asset_id
                ]
                if not failures.empty
                else failures
            )

            asset_maintenance = (
                maintenance[
                    maintenance["asset_id"]
                    == asset_id
                ]
                if not maintenance.empty
                else maintenance
            )

            rows.append({
                "asset_id": asset_id,
                "company_id": (
                    asset["company_id"]
                ),
                "site_id": asset["site_id"],
                "equipment_type": (
                    asset["equipment_type"]
                ),
                "reading_count": len(
                    asset_telemetry
                ),
                "failure_count": len(
                    asset_failures
                ),
                "maintenance_count": len(
                    asset_maintenance
                ),
                "average_temperature_c": round(
                    asset_telemetry[
                        "temperature_c"
                    ].mean(),
                    3,
                ),
                "average_vibration_mm_s": round(
                    asset_telemetry[
                        "vibration_mm_s"
                    ].mean(),
                    3,
                ),
                "final_health_score": float(
                    asset_telemetry.iloc[-1][
                        "health_score"
                    ]
                ),
                "total_downtime_hours": (
                    round(
                        asset_failures[
                            "downtime_hours"
                        ].sum(),
                        3,
                    )
                    if not asset_failures.empty
                    else 0.0
                ),
                "total_repair_cost": (
                    round(
                        asset_failures[
                            "repair_cost"
                        ].sum(),
                        2,
                    )
                    if not asset_failures.empty
                    else 0.0
                ),
                "currency": (
                    self.config.currency
                ),
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def validate_data(
        assets: pd.DataFrame,
        telemetry: pd.DataFrame,
        failures: pd.DataFrame,
        predictions: pd.DataFrame,
        rul: pd.DataFrame,
    ) -> None:
        if assets.empty:
            raise ValueError(
                "Asset dataset is empty"
            )

        if telemetry.empty:
            raise ValueError(
                "Telemetry dataset is empty"
            )

        if assets["asset_id"].duplicated().any():
            raise ValueError(
                "Duplicate asset IDs"
            )

        if telemetry["reading_id"].duplicated().any():
            raise ValueError(
                "Duplicate reading IDs"
            )

        if not telemetry["asset_id"].isin(
            assets["asset_id"]
        ).all():
            raise ValueError(
                "Telemetry references unknown assets"
            )

        if not failures.empty:
            if not failures["asset_id"].isin(
                assets["asset_id"]
            ).all():
                raise ValueError(
                    "Failure references unknown assets"
                )

        if not predictions[
            "failure_within_horizon"
        ].isin([0, 1]).all():
            raise ValueError(
                "Invalid prediction labels"
            )

        observed_rul = rul.loc[
            rul["rul_observed"],
            "rul_hours",
        ]

        if (observed_rul < 0).any():
            raise ValueError(
                "Negative observed RUL"
            )

        if not np.isfinite(
            telemetry.select_dtypes(
                include=[np.number]
            ).to_numpy()
        ).all():
            raise ValueError(
                "Invalid telemetry numeric values"
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
                "industrial_synthetic_failure_data"
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
            "files": {
                filename: len(dataframe)
                for filename, dataframe
                in datasets.items()
            },
            "limitations": [
                "Synthetic equipment behavior",
                "Illustrative degradation model",
                "Failure probabilities are not calibrated",
                "Sensor signals are simulator-generated",
                "RUL is censored when no future failure is observed",
                "Failure prediction labels use future outcomes",
                "Use time-based and asset-separated evaluation",
                "Do not use simulator latent state as ordinary ML features",
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
        print("[1/6] Generating assets...")

        assets = self.generate_assets()

        print(
            "[2/6] Simulating telemetry and failures..."
        )

        (
            telemetry,
            failures,
            maintenance,
        ) = self.generate_time_series(
            assets
        )

        print(
            "[3/6] Generating failure prediction labels..."
        )

        (
            predictions,
            rul,
        ) = self.generate_prediction_labels(
            telemetry,
            failures,
        )

        print(
            "[4/6] Calculating failure summary..."
        )

        summary = self.generate_summary(
            assets,
            telemetry,
            failures,
            maintenance,
        )

        print(
            "[5/6] Validating generated data..."
        )

        self.validate_data(
            assets,
            telemetry,
            failures,
            predictions,
            rul,
        )

        datasets = {
            "assets.csv": assets,
            "sensor_readings.csv": telemetry,
            "failure_events.csv": failures,
            "failure_predictions.csv": predictions,
            "rul_labels.csv": rul,
            "maintenance_history.csv": maintenance,
            "failure_summary.csv": summary,
        }

        print(
            "[6/6] Exporting datasets..."
        )

        paths = self.export(
            datasets
        )

        print("\nGeneration completed.")
        print("Assets:", len(assets))
        print("Telemetry readings:", len(telemetry))
        print("Failure events:", len(failures))
        print("Maintenance events:", len(maintenance))
        print(
            "Positive prediction labels:",
            int(
                predictions[
                    "failure_within_horizon"
                ].sum()
            ),
        )

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
            "equipment failure datasets"
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
        default=180,
    )

    parser.add_argument(
        "--readings-per-day",
        type=int,
        default=4,
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
        "--prediction-horizon-hours",
        type=int,
        default=72,
    )

    parser.add_argument(
        "--maintenance-probability",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic/failures",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = FailureDataConfig(
        output_dir=args.output_dir,
        num_assets=args.assets,
        num_companies=args.companies,
        sites_per_company=args.sites_per_company,
        days=args.days,
        readings_per_day=args.readings_per_day,
        prediction_horizon_hours=(
            args.prediction_horizon_hours
        ),
        maintenance_probability=(
            args.maintenance_probability
        ),
        seed=args.seed,
    )

    generator = IndustrialFailureDataGenerator(
        config
    )

    generator.generate_all()


if __name__ == "__main__":
    main()