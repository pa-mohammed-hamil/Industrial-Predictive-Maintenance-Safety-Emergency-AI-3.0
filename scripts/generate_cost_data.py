"""
scripts/generate_cost_data.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Synthetic Industrial Cost Dataset Generator
============================================

Python 3.11+

Dependencies:
    pip install numpy pandas

Outputs:
    data/synthetic/costs/
        maintenance_costs.csv
        repair_replacement_costs.csv
        downtime_costs.csv
        failure_costs.csv
        asset_financials.csv
        cost_summary.csv
        dataset_metadata.json

Features:
- Reproducible synthetic data
- Multi-company and multi-site datasets
- Industrial asset and equipment categories
- Repair and replacement economics
- Spare parts and labor costs
- Downtime and production loss
- Energy and emergency costs
- Failure probability and expected loss
- Predictive maintenance avoided cost
- Maintenance ROI
- Total cost of ownership
- CSV and JSON exports
- CLI configuration
- Data validation

All financial figures are synthetic and illustrative.
They are not actual supplier quotations or market prices.
"""

from __future__ import annotations

import argparse
import json

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CostDataConfig:
    output_dir: str = "data/synthetic/costs"

    num_assets: int = 200
    num_events: int = 2000

    num_companies: int = 5
    sites_per_company: int = 3

    seed: int = 42

    currency: str = "INR"

    start_date: str = "2025-01-01"
    end_date: str = "2026-12-31"

    min_hourly_labor_rate: float = 350.0
    max_hourly_labor_rate: float = 1500.0

    min_downtime_hourly_cost: float = 1500.0
    max_downtime_hourly_cost: float = 50000.0


# ============================================================
# EQUIPMENT PROFILES
# ============================================================

EQUIPMENT_PROFILES = {
    "pump": {
        "purchase_cost": (80000, 700000),
        "repair_ratio": (0.02, 0.18),
        "replacement_ratio": (0.85, 1.25),
        "failure_probability": (0.03, 0.20),
        "useful_life_years": (8, 15),
    },
    "motor": {
        "purchase_cost": (50000, 900000),
        "repair_ratio": (0.02, 0.20),
        "replacement_ratio": (0.85, 1.30),
        "failure_probability": (0.04, 0.22),
        "useful_life_years": (8, 18),
    },
    "compressor": {
        "purchase_cost": (250000, 3500000),
        "repair_ratio": (0.03, 0.25),
        "replacement_ratio": (0.90, 1.35),
        "failure_probability": (0.04, 0.25),
        "useful_life_years": (10, 20),
    },
    "conveyor": {
        "purchase_cost": (150000, 2000000),
        "repair_ratio": (0.02, 0.16),
        "replacement_ratio": (0.85, 1.30),
        "failure_probability": (0.03, 0.20),
        "useful_life_years": (10, 18),
    },
    "turbine": {
        "purchase_cost": (2000000, 30000000),
        "repair_ratio": (0.02, 0.22),
        "replacement_ratio": (0.90, 1.40),
        "failure_probability": (0.02, 0.15),
        "useful_life_years": (15, 30),
    },
    "generator": {
        "purchase_cost": (300000, 5000000),
        "repair_ratio": (0.02, 0.20),
        "replacement_ratio": (0.85, 1.35),
        "failure_probability": (0.03, 0.18),
        "useful_life_years": (10, 25),
    },
    "hvac": {
        "purchase_cost": (100000, 2500000),
        "repair_ratio": (0.02, 0.18),
        "replacement_ratio": (0.85, 1.25),
        "failure_probability": (0.04, 0.22),
        "useful_life_years": (8, 18),
    },
    "robotic_arm": {
        "purchase_cost": (700000, 12000000),
        "repair_ratio": (0.02, 0.20),
        "replacement_ratio": (0.90, 1.40),
        "failure_probability": (0.03, 0.18),
        "useful_life_years": (8, 15),
    },
    "cnc_machine": {
        "purchase_cost": (1200000, 15000000),
        "repair_ratio": (0.02, 0.22),
        "replacement_ratio": (0.90, 1.40),
        "failure_probability": (0.03, 0.20),
        "useful_life_years": (10, 20),
    },
    "boiler": {
        "purchase_cost": (500000, 10000000),
        "repair_ratio": (0.02, 0.22),
        "replacement_ratio": (0.85, 1.35),
        "failure_probability": (0.03, 0.20),
        "useful_life_years": (12, 25),
    },
}


FAILURE_MODES = [
    "bearing_failure",
    "overheating",
    "vibration_anomaly",
    "misalignment",
    "lubrication_failure",
    "electrical_fault",
    "mechanical_wear",
    "corrosion",
    "pressure_anomaly",
    "seal_leakage",
    "sensor_failure",
    "control_system_fault",
]


MAINTENANCE_TYPES = [
    "preventive",
    "predictive",
    "corrective",
    "emergency",
]


# ============================================================
# GENERATOR
# ============================================================

class IndustrialCostDataGenerator:
    def __init__(
        self,
        config: CostDataConfig,
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

        self.end_date = pd.Timestamp(
            config.end_date,
            tz="UTC",
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

        if self.config.num_events < 1:
            raise ValueError(
                "num_events must be positive"
            )

        if self.config.num_companies < 1:
            raise ValueError(
                "num_companies must be positive"
            )

        if self.config.sites_per_company < 1:
            raise ValueError(
                "sites_per_company must be positive"
            )

        if self.end_date < self.start_date:
            raise ValueError(
                "end_date must not precede start_date"
            )

        if (
            self.config.min_hourly_labor_rate < 0
            or self.config.max_hourly_labor_rate
            < self.config.min_hourly_labor_rate
        ):
            raise ValueError(
                "Invalid labor rate range"
            )

        if (
            self.config.min_downtime_hourly_cost < 0
            or self.config.max_downtime_hourly_cost
            < self.config.min_downtime_hourly_cost
        ):
            raise ValueError(
                "Invalid downtime cost range"
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
        probabilities: list[float] | None = None,
    ) -> str:
        return str(
            self.rng.choice(
                values,
                p=probabilities,
            )
        )

    def random_timestamp(self) -> str:
        duration_seconds = int(
            (
                self.end_date - self.start_date
            ).total_seconds()
        )

        offset = self.integer(
            0,
            duration_seconds,
        )

        timestamp = (
            self.start_date
            + pd.Timedelta(seconds=offset)
        )

        return timestamp.isoformat()

    @staticmethod
    def money(value: float) -> float:
        return round(max(0.0, value), 2)

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

            profile = EQUIPMENT_PROFILES[
                equipment_type
            ]

            company_number = self.integer(
                1,
                self.config.num_companies,
            )

            site_number = self.integer(
                1,
                self.config.sites_per_company,
            )

            purchase_cost = self.uniform(
                *profile["purchase_cost"]
            )

            useful_life = self.integer(
                *profile["useful_life_years"]
            )

            age_years = self.uniform(
                0,
                useful_life * 1.2,
            )

            age_ratio = min(
                age_years / useful_life,
                1.5,
            )

            condition_score = float(
                np.clip(
                    100
                    - age_ratio * 50
                    + self.rng.normal(0, 12),
                    5,
                    100,
                )
            )

            failure_probability = float(
                np.clip(
                    self.uniform(
                        *profile[
                            "failure_probability"
                        ]
                    )
                    + (100 - condition_score)
                    / 100
                    * 0.18,
                    0.001,
                    0.95,
                )
            )

            annual_energy_cost = (
                purchase_cost
                * self.uniform(0.03, 0.18)
            )

            annual_maintenance_cost = (
                purchase_cost
                * self.uniform(0.015, 0.08)
            )

            residual_value = (
                purchase_cost
                * self.uniform(0.05, 0.25)
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
                "purchase_cost": (
                    self.money(purchase_cost)
                ),
                "age_years": round(
                    age_years,
                    2,
                ),
                "useful_life_years": useful_life,
                "condition_score": round(
                    condition_score,
                    2,
                ),
                "failure_probability": round(
                    failure_probability,
                    6,
                ),
                "annual_energy_cost": (
                    self.money(annual_energy_cost)
                ),
                "annual_maintenance_cost": (
                    self.money(
                        annual_maintenance_cost
                    )
                ),
                "estimated_residual_value": (
                    self.money(residual_value)
                ),
                "currency": self.config.currency,
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # GENERATE MAINTENANCE EVENTS
    # --------------------------------------------------------

    def generate_maintenance_events(
        self,
        assets: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []

        asset_records = assets.to_dict(
            orient="records"
        )

        for index in range(
            1,
            self.config.num_events + 1,
        ):
            asset = asset_records[
                self.integer(
                    0,
                    len(asset_records) - 1,
                )
            ]

            profile = EQUIPMENT_PROFILES[
                asset["equipment_type"]
            ]

            maintenance_type = self.choice(
                MAINTENANCE_TYPES,
                probabilities=[
                    0.35,
                    0.30,
                    0.25,
                    0.10,
                ],
            )

            failure_mode = self.choice(
                FAILURE_MODES
            )

            severity = self.choice(
                ["low", "medium", "high", "critical"],
                probabilities=[
                    0.25,
                    0.40,
                    0.25,
                    0.10,
                ],
            )

            severity_factor = {
                "low": 0.5,
                "medium": 1.0,
                "high": 1.8,
                "critical": 3.0,
            }[severity]

            maintenance_factor = {
                "preventive": 0.45,
                "predictive": 0.65,
                "corrective": 1.0,
                "emergency": 1.8,
            }[maintenance_type]

            purchase_cost = float(
                asset["purchase_cost"]
            )

            base_repair_cost = (
                purchase_cost
                * self.uniform(
                    *profile["repair_ratio"]
                )
            )

            repair_cost = (
                base_repair_cost
                * severity_factor
                * maintenance_factor
            )

            labor_hours = self.uniform(
                1,
                16,
            ) * severity_factor

            hourly_labor_rate = self.uniform(
                self.config.min_hourly_labor_rate,
                self.config.max_hourly_labor_rate,
            )

            labor_cost = (
                labor_hours
                * hourly_labor_rate
            )

            spare_parts_cost = (
                repair_cost
                * self.uniform(0.20, 0.65)
            )

            service_cost = (
                repair_cost
                * self.uniform(0.05, 0.18)
            )

            inspection_cost = (
                purchase_cost
                * self.uniform(0.001, 0.006)
            )

            downtime_hours = (
                self.uniform(0.5, 12)
                * severity_factor
                * maintenance_factor
            )

            downtime_hourly_cost = self.uniform(
                self.config.min_downtime_hourly_cost,
                self.config.max_downtime_hourly_cost,
            )

            downtime_cost = (
                downtime_hours
                * downtime_hourly_cost
            )

            production_units_per_hour = (
                self.uniform(10, 500)
            )

            margin_per_unit = (
                self.uniform(20, 2000)
            )

            production_loss_units = (
                downtime_hours
                * production_units_per_hour
                * self.uniform(0.3, 1.0)
            )

            production_loss_cost = (
                production_loss_units
                * margin_per_unit
            )

            energy_cost = (
                downtime_hours
                * self.uniform(100, 1500)
            )

            emergency_cost = (
                self.uniform(10000, 200000)
                if maintenance_type == "emergency"
                else 0.0
            )

            scrap_cost = (
                purchase_cost
                * self.uniform(0.0, 0.015)
                * severity_factor
            )

            replacement_cost = (
                purchase_cost
                * self.uniform(
                    *profile["replacement_ratio"]
                )
            )

            direct_maintenance_cost = (
                labor_cost
                + spare_parts_cost
                + service_cost
                + inspection_cost
            )

            total_event_cost = (
                direct_maintenance_cost
                + downtime_cost
                + production_loss_cost
                + energy_cost
                + emergency_cost
                + scrap_cost
            )

            failure_probability = float(
                asset["failure_probability"]
            )

            expected_failure_cost = (
                failure_probability
                * (
                    repair_cost
                    + downtime_cost
                    + production_loss_cost
                    + emergency_cost
                )
            )

            avoided_cost = (
                expected_failure_cost
                * self.uniform(0.25, 0.80)
                if maintenance_type == "predictive"
                else 0.0
            )

            # Illustrative ROI using modeled avoided
            # loss and direct maintenance expenditure.
            # Not a causal estimate from field outcomes.
            roi_percent = (
                (
                    avoided_cost
                    - direct_maintenance_cost
                )
                / direct_maintenance_cost
                * 100
                if direct_maintenance_cost > 0
                else 0.0
            )

            repair_vs_replacement = (
                "repair"
                if (
                    repair_cost
                    + downtime_cost
                    < replacement_cost
                )
                else "replace"
            )

            rows.append({
                "event_id": f"EVENT-{index:06d}",
                "timestamp": self.random_timestamp(),
                "asset_id": asset["asset_id"],
                "company_id": asset["company_id"],
                "site_id": asset["site_id"],
                "equipment_type": (
                    asset["equipment_type"]
                ),
                "maintenance_type": maintenance_type,
                "failure_mode": failure_mode,
                "severity": severity,
                "failure_probability": (
                    failure_probability
                ),
                "labor_hours": round(
                    labor_hours,
                    3,
                ),
                "hourly_labor_rate": (
                    self.money(hourly_labor_rate)
                ),
                "labor_cost": self.money(
                    labor_cost
                ),
                "spare_parts_cost": self.money(
                    spare_parts_cost
                ),
                "service_cost": self.money(
                    service_cost
                ),
                "inspection_cost": self.money(
                    inspection_cost
                ),
                "repair_cost": self.money(
                    repair_cost
                ),
                "replacement_cost": self.money(
                    replacement_cost
                ),
                "downtime_hours": round(
                    downtime_hours,
                    3,
                ),
                "downtime_hourly_cost": (
                    self.money(
                        downtime_hourly_cost
                    )
                ),
                "downtime_cost": self.money(
                    downtime_cost
                ),
                "production_loss_units": round(
                    production_loss_units,
                    3,
                ),
                "margin_per_unit": self.money(
                    margin_per_unit
                ),
                "production_loss_cost": (
                    self.money(
                        production_loss_cost
                    )
                ),
                "energy_cost": self.money(
                    energy_cost
                ),
                "emergency_cost": self.money(
                    emergency_cost
                ),
                "scrap_cost": self.money(
                    scrap_cost
                ),
                "direct_maintenance_cost": (
                    self.money(
                        direct_maintenance_cost
                    )
                ),
                "total_event_cost": self.money(
                    total_event_cost
                ),
                "expected_failure_cost": (
                    self.money(
                        expected_failure_cost
                    )
                ),
                "avoided_cost": self.money(
                    avoided_cost
                ),
                "maintenance_roi_percent": round(
                    roi_percent,
                    2,
                ),
                "repair_vs_replacement": (
                    repair_vs_replacement
                ),
                "currency": self.config.currency,
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # ASSET FINANCIALS
    # --------------------------------------------------------

    def generate_asset_financials(
        self,
        assets: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        grouped = events.groupby(
            "asset_id",
            as_index=False,
        ).agg(
            event_count=("event_id", "count"),
            total_maintenance_cost=(
                "direct_maintenance_cost",
                "sum",
            ),
            total_downtime_cost=(
                "downtime_cost",
                "sum",
            ),
            total_production_loss=(
                "production_loss_cost",
                "sum",
            ),
            total_emergency_cost=(
                "emergency_cost",
                "sum",
            ),
            total_avoided_cost=(
                "avoided_cost",
                "sum",
            ),
        )

        financials = assets.merge(
            grouped,
            on="asset_id",
            how="left",
            validate="one_to_one",
        )

        numeric_columns = [
            "event_count",
            "total_maintenance_cost",
            "total_downtime_cost",
            "total_production_loss",
            "total_emergency_cost",
            "total_avoided_cost",
        ]

        financials[numeric_columns] = (
            financials[numeric_columns].fillna(0)
        )

        financials["event_count"] = (
            financials["event_count"].astype(int)
        )

        financials["remaining_life_years"] = (
            financials["useful_life_years"]
            - financials["age_years"]
        ).clip(lower=0)

        financials["future_maintenance_cost"] = (
            financials["annual_maintenance_cost"]
            * financials["remaining_life_years"]
        )

        financials["future_energy_cost"] = (
            financials["annual_energy_cost"]
            * financials["remaining_life_years"]
        )

        financials["historical_total_cost"] = (
            financials["purchase_cost"]
            + financials["total_maintenance_cost"]
            + financials["total_downtime_cost"]
            + financials["total_production_loss"]
            + financials["total_emergency_cost"]
        )

        financials["projected_tco"] = (
            financials["historical_total_cost"]
            + financials["future_maintenance_cost"]
            + financials["future_energy_cost"]
            - financials["estimated_residual_value"]
        )

        financials["projected_tco"] = (
            financials["projected_tco"].clip(
                lower=0
            )
        )

        return financials.round(2)

    # --------------------------------------------------------
    # EXPORT DATASETS
    # --------------------------------------------------------

    def export(
        self,
        assets: pd.DataFrame,
        events: pd.DataFrame,
        financials: pd.DataFrame,
    ) -> dict[str, str]:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        datasets = {
            "maintenance_costs.csv": events[
                [
                    "event_id",
                    "timestamp",
                    "asset_id",
                    "company_id",
                    "site_id",
                    "maintenance_type",
                    "labor_cost",
                    "spare_parts_cost",
                    "service_cost",
                    "inspection_cost",
                    "direct_maintenance_cost",
                    "total_event_cost",
                    "currency",
                ]
            ],
            "repair_replacement_costs.csv": events[
                [
                    "event_id",
                    "asset_id",
                    "equipment_type",
                    "failure_mode",
                    "severity",
                    "repair_cost",
                    "replacement_cost",
                    "repair_vs_replacement",
                    "currency",
                ]
            ],
            "downtime_costs.csv": events[
                [
                    "event_id",
                    "asset_id",
                    "downtime_hours",
                    "downtime_hourly_cost",
                    "downtime_cost",
                    "production_loss_units",
                    "margin_per_unit",
                    "production_loss_cost",
                    "currency",
                ]
            ],
            "failure_costs.csv": events[
                [
                    "event_id",
                    "asset_id",
                    "failure_mode",
                    "failure_probability",
                    "repair_cost",
                    "downtime_cost",
                    "production_loss_cost",
                    "emergency_cost",
                    "scrap_cost",
                    "expected_failure_cost",
                    "avoided_cost",
                    "maintenance_roi_percent",
                    "currency",
                ]
            ],
            "asset_financials.csv": financials,
        }

        summary = events.groupby(
            [
                "company_id",
                "site_id",
                "equipment_type",
                "maintenance_type",
            ],
            as_index=False,
        ).agg(
            event_count=("event_id", "count"),
            total_event_cost=(
                "total_event_cost",
                "sum",
            ),
            total_downtime_cost=(
                "downtime_cost",
                "sum",
            ),
            total_production_loss=(
                "production_loss_cost",
                "sum",
            ),
            total_avoided_cost=(
                "avoided_cost",
                "sum",
            ),
            average_roi_percent=(
                "maintenance_roi_percent",
                "mean",
            ),
        )

        summary["currency"] = (
            self.config.currency
        )

        datasets["cost_summary.csv"] = (
            summary.round(2)
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
                "industrial_synthetic_cost_data"
            ),
            "generated_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "synthetic": True,
            "currency": self.config.currency,
            "seed": self.config.seed,
            "num_assets": len(assets),
            "num_events": len(events),
            "companies": (
                self.config.num_companies
            ),
            "sites_per_company": (
                self.config.sites_per_company
            ),
            "equipment_types": list(
                EQUIPMENT_PROFILES.keys()
            ),
            "files": {
                filename: len(dataframe)
                for filename, dataframe
                in datasets.items()
            },
            "limitations": [
                "Synthetic values are illustrative",
                "Not validated against actual invoices",
                "Failure probabilities are simulated",
                "Avoided cost is a modeled estimate",
                "ROI is not a causal field estimate",
                "TCO uses simplified nominal costs",
                "Financial components should be checked "
                "for double counting before production use",
            ],
            "configuration": asdict(
                self.config
            ),
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

        paths["dataset_metadata.json"] = str(
            metadata_path
        )

        return paths

    # --------------------------------------------------------
    # VALIDATE GENERATED DATA
    # --------------------------------------------------------

    @staticmethod
    def validate_data(
        assets: pd.DataFrame,
        events: pd.DataFrame,
        financials: pd.DataFrame,
    ) -> None:
        if assets.empty or events.empty:
            raise ValueError(
                "Generated datasets cannot be empty"
            )

        if assets["asset_id"].duplicated().any():
            raise ValueError(
                "Duplicate asset IDs"
            )

        if events["event_id"].duplicated().any():
            raise ValueError(
                "Duplicate event IDs"
            )

        if not events["asset_id"].isin(
            assets["asset_id"]
        ).all():
            raise ValueError(
                "Unknown asset references in events"
            )

        if not events[
            "failure_probability"
        ].between(0, 1).all():
            raise ValueError(
                "Invalid failure probabilities"
            )

        nonnegative_columns = [
            "labor_cost",
            "spare_parts_cost",
            "service_cost",
            "inspection_cost",
            "repair_cost",
            "replacement_cost",
            "downtime_cost",
            "production_loss_cost",
            "energy_cost",
            "emergency_cost",
            "scrap_cost",
            "total_event_cost",
        ]

        for column in nonnegative_columns:
            if (events[column] < 0).any():
                raise ValueError(
                    f"Negative values in {column}"
                )

        if not np.isfinite(
            events.select_dtypes(
                include=[np.number]
            ).to_numpy()
        ).all():
            raise ValueError(
                "Non-finite numeric event values"
            )

        if (
            financials["projected_tco"] < 0
        ).any():
            raise ValueError(
                "Negative projected TCO"
            )

    # --------------------------------------------------------
    # GENERATE ALL
    # --------------------------------------------------------

    def generate_all(
        self,
    ) -> dict[str, str]:
        print(
            "[1/5] Generating industrial assets..."
        )

        assets = self.generate_assets()

        print(
            "[2/5] Generating maintenance costs..."
        )

        events = self.generate_maintenance_events(
            assets
        )

        print(
            "[3/5] Calculating asset financials..."
        )

        financials = (
            self.generate_asset_financials(
                assets,
                events,
            )
        )

        print(
            "[4/5] Validating datasets..."
        )

        self.validate_data(
            assets,
            events,
            financials,
        )

        print(
            "[5/5] Exporting CSV and JSON..."
        )

        paths = self.export(
            assets,
            events,
            financials,
        )

        print("\nGeneration completed.\n")

        for filename, path in paths.items():
            print(f"{filename}: {path}")

        print(
            "\nTotal simulated event cost:",
            f"{self.config.currency} "
            f"{events['total_event_cost'].sum():,.2f}",
        )

        return paths


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic industrial "
            "maintenance cost datasets"
        )
    )

    parser.add_argument(
        "--assets",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--events",
        type=int,
        default=2000,
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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--currency",
        type=str,
        default="INR",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic/costs",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = CostDataConfig(
        output_dir=args.output_dir,
        num_assets=args.assets,
        num_events=args.events,
        num_companies=args.companies,
        sites_per_company=args.sites_per_company,
        seed=args.seed,
        currency=args.currency,
    )

    generator = IndustrialCostDataGenerator(
        config
    )

    generator.generate_all()


if __name__ == "__main__":
    main()