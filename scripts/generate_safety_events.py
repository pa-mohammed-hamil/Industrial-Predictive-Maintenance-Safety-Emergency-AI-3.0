"""
scripts/generate_safety_events.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Synthetic Industrial Safety Event Dataset Generator
==================================================

Python 3.11+

Dependencies:
    pip install numpy pandas

Outputs:
    data/synthetic/safety/
        safety_events.csv
        safety_sensor_readings.csv
        cv_safety_detections.csv
        safety_incidents.csv
        emergency_responses.csv
        safety_summary.csv
        dataset_metadata.json

Features:
- Multi-company, multi-site, multi-zone data
- Industrial safety event simulation
- Fire and smoke events
- Gas leakage and overheating
- PPE violations
- Restricted-zone intrusion
- Worker fall and collision events
- Electrical and machinery hazards
- Simulated sensor measurements
- Computer-vision detection records
- Severity and risk scoring
- Emergency response records
- Incident outcomes
- Safety event classification labels
- Reproducible generation
- Data validation
- CSV and JSON exports

All data is synthetic and illustrative.

The generated events and risk scores must not be
used to trigger real industrial safety systems.
"""

from __future__ import annotations

import argparse
import json

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
class SafetyDataConfig:
    output_dir: str = "data/synthetic/safety"

    num_events: int = 5000

    num_companies: int = 5
    sites_per_company: int = 3
    zones_per_site: int = 10

    seed: int = 42

    start_date: str = "2025-01-01"
    end_date: str = "2026-12-31"

    currency: str = "INR"

    incident_probability: float = 0.12
    emergency_probability: float = 0.08


# ============================================================
# EVENT PROFILES
# ============================================================

SAFETY_EVENT_PROFILES: dict[
    str,
    dict[str, Any],
] = {
    "fire": {
        "base_risk": 0.95,
        "sensor": "temperature",
        "cv_class": "fire",
        "emergency_weight": 0.90,
    },
    "smoke": {
        "base_risk": 0.82,
        "sensor": "smoke",
        "cv_class": "smoke",
        "emergency_weight": 0.70,
    },
    "gas_leak": {
        "base_risk": 0.92,
        "sensor": "gas",
        "cv_class": "gas_leak_indicator",
        "emergency_weight": 0.90,
    },
    "overheating": {
        "base_risk": 0.75,
        "sensor": "temperature",
        "cv_class": "overheating",
        "emergency_weight": 0.45,
    },
    "ppe_violation": {
        "base_risk": 0.55,
        "sensor": "none",
        "cv_class": "missing_ppe",
        "emergency_weight": 0.05,
    },
    "restricted_zone_entry": {
        "base_risk": 0.70,
        "sensor": "motion",
        "cv_class": "restricted_person",
        "emergency_weight": 0.12,
    },
    "worker_fall": {
        "base_risk": 0.85,
        "sensor": "motion",
        "cv_class": "person_fall",
        "emergency_weight": 0.65,
    },
    "unsafe_motion": {
        "base_risk": 0.62,
        "sensor": "motion",
        "cv_class": "unsafe_motion",
        "emergency_weight": 0.10,
    },
    "collision": {
        "base_risk": 0.88,
        "sensor": "vibration",
        "cv_class": "collision",
        "emergency_weight": 0.70,
    },
    "electrical_hazard": {
        "base_risk": 0.90,
        "sensor": "current",
        "cv_class": "electrical_hazard",
        "emergency_weight": 0.75,
    },
    "chemical_leak": {
        "base_risk": 0.90,
        "sensor": "gas",
        "cv_class": "chemical_leak_indicator",
        "emergency_weight": 0.85,
    },
    "machine_guard_violation": {
        "base_risk": 0.78,
        "sensor": "motion",
        "cv_class": "machine_guard_violation",
        "emergency_weight": 0.20,
    },
}


ZONE_TYPES = [
    "production",
    "maintenance",
    "warehouse",
    "electrical",
    "chemical",
    "robotics",
    "restricted",
    "high_temperature",
    "loading",
    "general",
]


PPE_TYPES = [
    "helmet",
    "safety_glasses",
    "gloves",
    "safety_shoes",
    "high_visibility_vest",
    "respirator",
]


# ============================================================
# GENERATOR
# ============================================================

class IndustrialSafetyDataGenerator:
    def __init__(
        self,
        config: SafetyDataConfig,
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
        positive_fields = {
            "num_events": self.config.num_events,
            "num_companies": self.config.num_companies,
            "sites_per_company": (
                self.config.sites_per_company
            ),
            "zones_per_site": (
                self.config.zones_per_site
            ),
        }

        for name, value in positive_fields.items():
            if value < 1:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.end_date < self.start_date:
            raise ValueError(
                "end_date must not precede start_date"
            )

        for name, probability in {
            "incident_probability": (
                self.config.incident_probability
            ),
            "emergency_probability": (
                self.config.emergency_probability
            ),
        }.items():
            if not 0 <= probability <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1"
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

        return (
            self.start_date
            + pd.Timedelta(
                seconds=offset
            )
        ).isoformat()

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    @staticmethod
    def severity_from_score(
        score: float,
    ) -> str:
        if score >= 85:
            return "critical"

        if score >= 65:
            return "high"

        if score >= 40:
            return "medium"

        return "low"

    # --------------------------------------------------------
    # SENSOR MEASUREMENT
    # --------------------------------------------------------

    def generate_sensor_measurement(
        self,
        event_type: str,
        risk_score: float,
    ) -> dict[str, float]:
        severity_factor = risk_score / 100

        temperature = self.uniform(
            25,
            45,
        )

        smoke_ppm = self.uniform(
            0,
            8,
        )

        gas_ppm = self.uniform(
            0,
            15,
        )

        vibration = self.uniform(
            0.2,
            2.5,
        )

        current = self.uniform(
            5,
            80,
        )

        motion_score = self.uniform(
            0,
            0.3,
        )

        if event_type in {
            "fire",
            "overheating",
        }:
            temperature += (
                severity_factor
                * self.uniform(35, 130)
            )

        if event_type in {
            "fire",
            "smoke",
        }:
            smoke_ppm += (
                severity_factor
                * self.uniform(100, 800)
            )

        if event_type in {
            "gas_leak",
            "chemical_leak",
        }:
            gas_ppm += (
                severity_factor
                * self.uniform(150, 1200)
            )

        if event_type == "collision":
            vibration += (
                severity_factor
                * self.uniform(5, 20)
            )

        if event_type == "electrical_hazard":
            current += (
                severity_factor
                * self.uniform(80, 500)
            )

        if event_type in {
            "worker_fall",
            "unsafe_motion",
            "restricted_zone_entry",
            "machine_guard_violation",
        }:
            motion_score = self.uniform(
                0.65,
                1.0,
            )

        return {
            "temperature_c": round(
                temperature,
                3,
            ),
            "smoke_ppm": round(
                smoke_ppm,
                3,
            ),
            "gas_ppm": round(
                gas_ppm,
                3,
            ),
            "vibration_mm_s": round(
                vibration,
                3,
            ),
            "current_a": round(
                current,
                3,
            ),
            "motion_score": round(
                motion_score,
                5,
            ),
        }

    # --------------------------------------------------------
    # GENERATE EVENTS
    # --------------------------------------------------------

    def generate_events(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        event_rows = []
        sensor_rows = []
        cv_rows = []
        incident_rows = []
        emergency_rows = []

        event_types = list(
            SAFETY_EVENT_PROFILES.keys()
        )

        for index in range(
            1,
            self.config.num_events + 1,
        ):
            event_id = (
                f"SAFE-{index:07d}"
            )

            event_type = self.choice(
                event_types
            )

            profile = SAFETY_EVENT_PROFILES[
                event_type
            ]

            company_number = self.integer(
                1,
                self.config.num_companies,
            )

            site_number = self.integer(
                1,
                self.config.sites_per_company,
            )

            zone_number = self.integer(
                1,
                self.config.zones_per_site,
            )

            company_id = (
                f"COMPANY-{company_number:03d}"
            )

            site_id = (
                f"{company_id}"
                f"-SITE-{site_number:03d}"
            )

            zone_id = (
                f"{site_id}"
                f"-ZONE-{zone_number:03d}"
            )

            zone_type = self.choice(
                ZONE_TYPES
            )

            timestamp = (
                self.random_timestamp()
            )

            risk_score = float(
                np.clip(
                    profile["base_risk"] * 100
                    + self.rng.normal(0, 15),
                    5,
                    100,
                )
            )

            severity = (
                self.severity_from_score(
                    risk_score
                )
            )

            sensor_values = (
                self.generate_sensor_measurement(
                    event_type,
                    risk_score,
                )
            )

            cv_confidence = float(
                np.clip(
                    self.uniform(0.55, 0.99),
                    0,
                    1,
                )
            )

            incident_probability = float(
                np.clip(
                    self.config.incident_probability
                    * (0.5 + risk_score / 100),
                    0,
                    1,
                )
            )

            is_incident = bool(
                self.rng.random()
                < incident_probability
            )

            emergency_probability = float(
                np.clip(
                    self.config.emergency_probability
                    * profile["emergency_weight"]
                    * (0.5 + risk_score / 100),
                    0,
                    1,
                )
            )

            is_emergency = bool(
                self.rng.random()
                < emergency_probability
            )

            # An emergency event is always also
            # represented as an incident.
            if is_emergency:
                is_incident = True

            event_rows.append({
                "event_id": event_id,
                "timestamp": timestamp,
                "company_id": company_id,
                "site_id": site_id,
                "zone_id": zone_id,
                "zone_type": zone_type,
                "event_type": event_type,
                "severity": severity,
                "risk_score": round(
                    risk_score,
                    3,
                ),
                "detection_source": (
                    self.choice([
                        "sensor",
                        "computer_vision",
                        "sensor_fusion",
                        "manual_report",
                    ])
                ),
                "cv_confidence": round(
                    cv_confidence,
                    5,
                ),
                "incident_label": int(
                    is_incident
                ),
                "emergency_label": int(
                    is_emergency
                ),
                "status": "detected",
                "synthetic": True,
            })

            sensor_rows.append({
                "reading_id": (
                    f"SENSOR-{index:07d}"
                ),
                "event_id": event_id,
                "timestamp": timestamp,
                "company_id": company_id,
                "site_id": site_id,
                "zone_id": zone_id,
                "event_type": event_type,
                **sensor_values,
            })

            # Illustrative CV metadata only.
            # This does not create image files.
            x1 = self.uniform(0.02, 0.55)
            y1 = self.uniform(0.02, 0.55)

            width = self.uniform(0.10, 0.35)
            height = self.uniform(0.10, 0.35)

            x2 = min(
                x1 + width,
                1.0,
            )

            y2 = min(
                y1 + height,
                1.0,
            )

            missing_ppe = (
                self.choice(PPE_TYPES)
                if event_type == "ppe_violation"
                else None
            )

            cv_rows.append({
                "detection_id": (
                    f"CV-{index:07d}"
                ),
                "event_id": event_id,
                "timestamp": timestamp,
                "company_id": company_id,
                "site_id": site_id,
                "zone_id": zone_id,
                "camera_id": (
                    f"CAM-{zone_id}"
                ),
                "detection_class": (
                    profile["cv_class"]
                ),
                "confidence": round(
                    cv_confidence,
                    5,
                ),
                "bbox_x1": round(x1, 5),
                "bbox_y1": round(y1, 5),
                "bbox_x2": round(x2, 5),
                "bbox_y2": round(y2, 5),
                "missing_ppe": missing_ppe,
                "image_path": None,
                "synthetic_metadata_only": True,
            })

            if is_incident:
                incident_id = (
                    f"INC-{index:07d}"
                )

                injury_count = (
                    self.integer(0, 2)
                    if event_type in {
                        "worker_fall",
                        "collision",
                        "fire",
                        "electrical_hazard",
                    }
                    else 0
                )

                downtime_hours = (
                    self.uniform(0.5, 24)
                    * risk_score / 100
                )

                estimated_cost = (
                    downtime_hours
                    * self.uniform(
                        1000,
                        25000,
                    )
                )

                incident_rows.append({
                    "incident_id": incident_id,
                    "event_id": event_id,
                    "timestamp": timestamp,
                    "company_id": company_id,
                    "site_id": site_id,
                    "zone_id": zone_id,
                    "incident_type": event_type,
                    "severity": severity,
                    "injury_count": injury_count,
                    "downtime_hours": round(
                        downtime_hours,
                        3,
                    ),
                    "estimated_cost": round(
                        estimated_cost,
                        2,
                    ),
                    "currency": (
                        self.config.currency
                    ),
                    "investigation_status": (
                        "pending"
                    ),
                })

            if is_emergency:
                response_minutes = (
                    self.uniform(1, 20)
                    * (
                        1.3 - risk_score / 200
                    )
                )

                emergency_rows.append({
                    "emergency_id": (
                        f"EMERG-{index:07d}"
                    ),
                    "event_id": event_id,
                    "timestamp": timestamp,
                    "company_id": company_id,
                    "site_id": site_id,
                    "zone_id": zone_id,
                    "emergency_type": event_type,
                    "severity": severity,
                    "response_time_minutes": round(
                        response_minutes,
                        3,
                    ),
                    "evacuation_required": (
                        event_type in {
                            "fire",
                            "smoke",
                            "gas_leak",
                            "chemical_leak",
                        }
                    ),
                    "response_status": (
                        "simulated"
                    ),
                    "human_review_required": True,
                })

        return (
            pd.DataFrame(event_rows),
            pd.DataFrame(sensor_rows),
            pd.DataFrame(cv_rows),
            pd.DataFrame(incident_rows),
            pd.DataFrame(emergency_rows),
        )

    # --------------------------------------------------------
    # GENERATE SUMMARY
    # --------------------------------------------------------

    def generate_summary(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        summary = events.groupby(
            [
                "company_id",
                "site_id",
                "zone_id",
                "event_type",
                "severity",
            ],
            as_index=False,
        ).agg(
            event_count=(
                "event_id",
                "count",
            ),
            average_risk_score=(
                "risk_score",
                "mean",
            ),
            average_cv_confidence=(
                "cv_confidence",
                "mean",
            ),
            incident_count=(
                "incident_label",
                "sum",
            ),
            emergency_count=(
                "emergency_label",
                "sum",
            ),
        )

        return summary.round(3)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def validate_data(
        events: pd.DataFrame,
        sensors: pd.DataFrame,
        detections: pd.DataFrame,
        incidents: pd.DataFrame,
        emergencies: pd.DataFrame,
    ) -> None:
        if events.empty:
            raise ValueError(
                "Safety events dataset is empty"
            )

        if events["event_id"].duplicated().any():
            raise ValueError(
                "Duplicate safety event IDs"
            )

        event_ids = set(
            events["event_id"]
        )

        for name, dataframe in {
            "sensors": sensors,
            "detections": detections,
            "incidents": incidents,
            "emergencies": emergencies,
        }.items():
            if not set(
                dataframe["event_id"]
            ).issubset(event_ids):
                raise ValueError(
                    f"{name} references unknown events"
                )

        if not events[
            "risk_score"
        ].between(0, 100).all():
            raise ValueError(
                "Invalid risk scores"
            )

        if not events[
            "cv_confidence"
        ].between(0, 1).all():
            raise ValueError(
                "Invalid CV confidence"
            )

        if not events[
            "incident_label"
        ].isin([0, 1]).all():
            raise ValueError(
                "Invalid incident labels"
            )

        if not events[
            "emergency_label"
        ].isin([0, 1]).all():
            raise ValueError(
                "Invalid emergency labels"
            )

        if (
            events["emergency_label"]
            > events["incident_label"]
        ).any():
            raise ValueError(
                "Emergency without incident label"
            )

        for coordinate in [
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ]:
            if not detections[
                coordinate
            ].between(0, 1).all():
                raise ValueError(
                    f"Invalid bounding box: {coordinate}"
                )

        if (
            detections["bbox_x1"]
            >= detections["bbox_x2"]
        ).any():
            raise ValueError(
                "Invalid bounding box width"
            )

        if (
            detections["bbox_y1"]
            >= detections["bbox_y2"]
        ).any():
            raise ValueError(
                "Invalid bounding box height"
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
                "industrial_synthetic_safety_events"
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
            "event_types": list(
                SAFETY_EVENT_PROFILES.keys()
            ),
            "files": {
                filename: len(dataframe)
                for filename, dataframe
                in datasets.items()
            },
            "limitations": [
                "All events are synthetic",
                "Risk scores are illustrative",
                "Sensor units are simulated",
                "CV bounding boxes are synthetic metadata",
                "No real images are generated",
                "No real emergency responses occur",
                "Not suitable for safety certification",
                "Not calibrated to actual incident frequencies",
                "Use real labeled data for production evaluation",
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
            "[1/5] Generating industrial safety events..."
        )

        (
            events,
            sensors,
            detections,
            incidents,
            emergencies,
        ) = self.generate_events()

        print(
            "[2/5] Generating safety analytics..."
        )

        summary = self.generate_summary(
            events
        )

        print(
            "[3/5] Validating datasets..."
        )

        self.validate_data(
            events,
            sensors,
            detections,
            incidents,
            emergencies,
        )

        print(
            "[4/5] Preparing datasets..."
        )

        datasets = {
            "safety_events.csv": events,
            "safety_sensor_readings.csv": sensors,
            "cv_safety_detections.csv": detections,
            "safety_incidents.csv": incidents,
            "emergency_responses.csv": emergencies,
            "safety_summary.csv": summary,
        }

        print(
            "[5/5] Exporting CSV and JSON..."
        )

        paths = self.export(
            datasets
        )

        print("\nGeneration completed.")
        print("Safety events:", len(events))
        print("Sensor readings:", len(sensors))
        print("CV detections:", len(detections))
        print("Incidents:", len(incidents))
        print("Emergencies:", len(emergencies))

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
            "safety event datasets"
        )
    )

    parser.add_argument(
        "--events",
        type=int,
        default=5000,
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
        "--zones-per-site",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--incident-probability",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--emergency-probability",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic/safety",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SafetyDataConfig(
        output_dir=args.output_dir,
        num_events=args.events,
        num_companies=args.companies,
        sites_per_company=args.sites_per_company,
        zones_per_site=args.zones_per_site,
        incident_probability=(
            args.incident_probability
        ),
        emergency_probability=(
            args.emergency_probability
        ),
        seed=args.seed,
    )

    generator = IndustrialSafetyDataGenerator(
        config
    )

    generator.generate_all()


if __name__ == "__main__":
    main()