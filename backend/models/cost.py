"""
app/ai/ml/models/cost.py

Industrial Asset Cost Intelligence Models
=========================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant cost records
- Repair and replacement cost estimates
- Labor, spare parts and material costs
- Downtime and production loss
- Energy, scrap and emergency costs
- Expected failure cost calculation
- Repair-versus-replacement comparison
- Avoided cost and ROI analysis
- Cost breakdown and financial reporting
- Currency-safe Decimal arithmetic
- JSON-compatible request/response schemas

IMPORTANT:
    These are deterministic financial calculation
    and validation models, not trained ML models.

    Estimated inputs must come from actual asset
    records, supplier quotes, historical costs,
    engineering assumptions or validated models.

    ROI and cost calculations are estimates and
    should be reviewed before financial decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# CONSTANTS
# ============================================================

MONEY_PRECISION = Decimal("0.01")
ZERO = Decimal("0.00")
ONE_HUNDRED = Decimal("100")


def money(value: Decimal | int | str) -> Decimal:
    """Round monetary values to two decimal places."""
    return Decimal(str(value)).quantize(
        MONEY_PRECISION,
        rounding=ROUND_HALF_UP,
    )


# ============================================================
# ENUMS
# ============================================================

class CostType(str, Enum):
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    SPARE_PARTS = "spare_parts"
    LABOR = "labor"
    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"
    EMERGENCY = "emergency"
    ENERGY = "energy"
    SCRAP = "scrap"
    MAINTENANCE = "maintenance"
    INSPECTION = "inspection"
    TRANSPORTATION = "transportation"
    INSTALLATION = "installation"
    EXPECTED_FAILURE = "expected_failure"
    FUTURE_COST = "future_cost"
    OTHER = "other"


class CostStatus(str, Enum):
    DRAFT = "draft"
    ESTIMATED = "estimated"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    ACTUAL = "actual"
    CANCELLED = "cancelled"


class CostSource(str, Enum):
    MANUAL = "manual"
    HISTORICAL = "historical"
    SUPPLIER_QUOTE = "supplier_quote"
    ML_PREDICTION = "ml_prediction"
    RULE_ENGINE = "rule_engine"
    ERP = "erp"
    CMMS = "cmms"


class MaintenanceDecision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    INSPECT = "inspect"
    MONITOR = "monitor"
    ENGINEERING_REVIEW = "engineering_review"


class CostConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


# ============================================================
# BASE MODEL
# ============================================================

class CostBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# COST CONTEXT
# ============================================================

class CostContext(CostBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_id: str = Field(
        min_length=1,
        max_length=100,
    )

    equipment_id: str | None = None
    component_id: str | None = None

    site_id: str | None = None
    work_order_id: str | None = None

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )


# ============================================================
# COST LINE ITEM
# ============================================================

class CostLineItem(CostBaseModel):
    item_id: UUID = Field(
        default_factory=uuid4
    )

    cost_type: CostType

    description: str = Field(
        min_length=1,
        max_length=500,
    )

    quantity: Decimal = Field(
        default=Decimal("1"),
        gt=0,
    )

    unit: str = Field(
        default="unit",
        min_length=1,
        max_length=50,
    )

    unit_cost: Decimal = Field(
        ge=0,
    )

    tax_rate_percent: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        le=100,
    )

    discount_percent: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        le=100,
    )

    source: CostSource = CostSource.MANUAL

    supplier_id: str | None = None
    reference_id: str | None = None

    @property
    def subtotal(self) -> Decimal:
        return money(
            self.quantity * self.unit_cost
        )

    @property
    def discount_amount(self) -> Decimal:
        return money(
            self.subtotal
            * self.discount_percent
            / ONE_HUNDRED
        )

    @property
    def taxable_amount(self) -> Decimal:
        return money(
            self.subtotal
            - self.discount_amount
        )

    @property
    def tax_amount(self) -> Decimal:
        return money(
            self.taxable_amount
            * self.tax_rate_percent
            / ONE_HUNDRED
        )

    @property
    def total(self) -> Decimal:
        return money(
            self.taxable_amount
            + self.tax_amount
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")

        data.update({
            "subtotal": str(self.subtotal),
            "discount_amount": str(
                self.discount_amount
            ),
            "tax_amount": str(self.tax_amount),
            "total": str(self.total),
        })

        return data


# ============================================================
# LABOR COST
# ============================================================

class LaborCost(CostBaseModel):
    technician_count: int = Field(
        default=1,
        ge=1,
    )

    hours_per_technician: Decimal = Field(
        ge=0,
    )

    hourly_rate: Decimal = Field(
        ge=0,
    )

    overtime_hours_per_technician: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    overtime_multiplier: Decimal = Field(
        default=Decimal("1.5"),
        ge=1,
    )

    @property
    def total(self) -> Decimal:
        regular = (
            Decimal(self.technician_count)
            * self.hours_per_technician
            * self.hourly_rate
        )

        overtime = (
            Decimal(self.technician_count)
            * self.overtime_hours_per_technician
            * self.hourly_rate
            * self.overtime_multiplier
        )

        return money(regular + overtime)


# ============================================================
# DOWNTIME COST
# ============================================================

class DowntimeCost(CostBaseModel):
    downtime_hours: Decimal = Field(
        ge=0,
    )

    cost_per_hour: Decimal = Field(
        ge=0,
    )

    additional_restart_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    @property
    def total(self) -> Decimal:
        return money(
            self.downtime_hours
            * self.cost_per_hour
            + self.additional_restart_cost
        )


# ============================================================
# PRODUCTION LOSS
# ============================================================

class ProductionLossCost(CostBaseModel):
    lost_units: Decimal = Field(
        ge=0,
    )

    contribution_margin_per_unit: Decimal = Field(
        ge=0,
    )

    additional_waste_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    @property
    def total(self) -> Decimal:
        return money(
            self.lost_units
            * self.contribution_margin_per_unit
            + self.additional_waste_cost
        )


# ============================================================
# ENERGY COST
# ============================================================

class EnergyCost(CostBaseModel):
    additional_energy_kwh: Decimal = Field(
        ge=0,
    )

    tariff_per_kwh: Decimal = Field(
        ge=0,
    )

    @property
    def total(self) -> Decimal:
        return money(
            self.additional_energy_kwh
            * self.tariff_per_kwh
        )


# ============================================================
# REPAIR ESTIMATE
# ============================================================

class RepairCostEstimate(CostBaseModel):
    estimate_id: UUID = Field(
        default_factory=uuid4
    )

    context: CostContext

    parts: list[CostLineItem] = Field(
        default_factory=list
    )

    labor: LaborCost | None = None

    downtime: DowntimeCost | None = None

    production_loss: ProductionLossCost | None = None

    inspection_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    transportation_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    other_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    status: CostStatus = CostStatus.ESTIMATED

    confidence: CostConfidence = (
        CostConfidence.UNKNOWN
    )

    estimated_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    assumptions: list[str] = Field(
        default_factory=list
    )

    @property
    def parts_cost(self) -> Decimal:
        return money(
            sum(
                (item.total for item in self.parts),
                ZERO,
            )
        )

    @property
    def total(self) -> Decimal:
        return money(
            self.parts_cost
            + (
                self.labor.total
                if self.labor
                else ZERO
            )
            + (
                self.downtime.total
                if self.downtime
                else ZERO
            )
            + (
                self.production_loss.total
                if self.production_loss
                else ZERO
            )
            + self.inspection_cost
            + self.transportation_cost
            + self.other_cost
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")

        data["parts_cost"] = str(
            self.parts_cost
        )

        data["total"] = str(
            self.total
        )

        return data


# ============================================================
# REPLACEMENT ESTIMATE
# ============================================================

class ReplacementCostEstimate(CostBaseModel):
    estimate_id: UUID = Field(
        default_factory=uuid4
    )

    context: CostContext

    purchase_cost: Decimal = Field(
        ge=0,
    )

    installation_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    transportation_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    commissioning_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    disposal_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    downtime: DowntimeCost | None = None

    production_loss: ProductionLossCost | None = None

    salvage_value: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    status: CostStatus = CostStatus.ESTIMATED

    confidence: CostConfidence = (
        CostConfidence.UNKNOWN
    )

    estimated_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    assumptions: list[str] = Field(
        default_factory=list
    )

    @property
    def total(self) -> Decimal:
        gross = (
            self.purchase_cost
            + self.installation_cost
            + self.transportation_cost
            + self.commissioning_cost
            + self.disposal_cost
            + (
                self.downtime.total
                if self.downtime
                else ZERO
            )
            + (
                self.production_loss.total
                if self.production_loss
                else ZERO
            )
        )

        return money(
            gross - self.salvage_value
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["total"] = str(self.total)
        return data


# ============================================================
# EXPECTED FAILURE COST
# ============================================================

class ExpectedFailureCost(CostBaseModel):
    failure_probability: Decimal = Field(
        ge=0,
        le=1,
    )

    repair_cost_if_failure: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    downtime_cost_if_failure: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    production_loss_if_failure: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    emergency_cost_if_failure: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    other_cost_if_failure: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    probability_horizon_hours: float = Field(
        gt=0,
    )

    @property
    def failure_consequence_cost(self) -> Decimal:
        return money(
            self.repair_cost_if_failure
            + self.downtime_cost_if_failure
            + self.production_loss_if_failure
            + self.emergency_cost_if_failure
            + self.other_cost_if_failure
        )

    @property
    def expected_cost(self) -> Decimal:
        return money(
            self.failure_probability
            * self.failure_consequence_cost
        )


# ============================================================
# REPAIR VS REPLACEMENT
# ============================================================

class RepairReplacementComparison(CostBaseModel):
    comparison_id: UUID = Field(
        default_factory=uuid4
    )

    repair: RepairCostEstimate
    replacement: ReplacementCostEstimate

    decision: MaintenanceDecision = (
        MaintenanceDecision.ENGINEERING_REVIEW
    )

    engineering_review_required: bool = True

    compared_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    @model_validator(mode="after")
    def validate_context(
        self,
    ) -> "RepairReplacementComparison":
        a = self.repair.context
        b = self.replacement.context

        if (
            a.company_id != b.company_id
            or a.asset_id != b.asset_id
            or a.equipment_id != b.equipment_id
            or a.component_id != b.component_id
            or a.currency != b.currency
        ):
            raise ValueError(
                "Repair and replacement estimates "
                "must refer to the same asset, "
                "component and currency"
            )

        return self

    @property
    def cost_difference(self) -> Decimal:
        return money(
            self.replacement.total
            - self.repair.total
        )

    @property
    def lower_initial_cost_option(self) -> str:
        if self.repair.total < self.replacement.total:
            return "repair"

        if self.replacement.total < self.repair.total:
            return "replace"

        return "equal"

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")

        data.update({
            "repair_total": str(
                self.repair.total
            ),
            "replacement_total": str(
                self.replacement.total
            ),
            "cost_difference": str(
                self.cost_difference
            ),
            "lower_initial_cost_option": (
                self.lower_initial_cost_option
            ),
        })

        return data


# ============================================================
# ROI ANALYSIS
# ============================================================

class MaintenanceROI(CostBaseModel):
    analysis_id: UUID = Field(
        default_factory=uuid4
    )

    context: CostContext

    investment_cost: Decimal = Field(
        gt=0,
    )

    avoided_failure_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    downtime_savings: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    energy_savings: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    maintenance_savings: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    other_savings: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    analysis_horizon_months: int = Field(
        default=12,
        ge=1,
    )

    @property
    def total_benefits(self) -> Decimal:
        return money(
            self.avoided_failure_cost
            + self.downtime_savings
            + self.energy_savings
            + self.maintenance_savings
            + self.other_savings
        )

    @property
    def net_benefit(self) -> Decimal:
        return money(
            self.total_benefits
            - self.investment_cost
        )

    @property
    def roi_percent(self) -> Decimal:
        return money(
            self.net_benefit
            / self.investment_cost
            * ONE_HUNDRED
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")

        data.update({
            "total_benefits": str(
                self.total_benefits
            ),
            "net_benefit": str(
                self.net_benefit
            ),
            "roi_percent": str(
                self.roi_percent
            ),
        })

        return data


# ============================================================
# GENERAL COST RECORD
# ============================================================

class IndustrialCostRecord(CostBaseModel):
    cost_id: UUID = Field(
        default_factory=uuid4
    )

    context: CostContext

    cost_type: CostType

    description: str = Field(
        min_length=1,
        max_length=1000,
    )

    line_items: list[CostLineItem] = Field(
        default_factory=list
    )

    additional_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    status: CostStatus = CostStatus.DRAFT

    source: CostSource = CostSource.MANUAL

    confidence: CostConfidence = (
        CostConfidence.UNKNOWN
    )

    approved_by: str | None = None
    approved_at: datetime | None = None

    reference_id: str | None = None

    assumptions: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    @property
    def total(self) -> Decimal:
        return money(
            sum(
                (
                    item.total
                    for item in self.line_items
                ),
                ZERO,
            )
            + self.additional_cost
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["total"] = str(self.total)
        return data


# ============================================================
# CREATE / UPDATE / RESPONSE
# ============================================================

class CostCreate(CostBaseModel):
    context: CostContext

    cost_type: CostType

    description: str = Field(
        min_length=1,
        max_length=1000,
    )

    line_items: list[CostLineItem] = Field(
        default_factory=list
    )

    additional_cost: Decimal = Field(
        default=ZERO,
        ge=0,
    )

    source: CostSource = CostSource.MANUAL

    assumptions: list[str] = Field(
        default_factory=list
    )


class CostUpdate(CostBaseModel):
    description: str | None = None

    status: CostStatus | None = None

    line_items: list[CostLineItem] | None = None

    additional_cost: Decimal | None = Field(
        default=None,
        ge=0,
    )

    confidence: CostConfidence | None = None


class CostResponse(IndustrialCostRecord):
    pass


# ============================================================
# COST SUMMARY
# ============================================================

class CostSummary(CostBaseModel):
    company_id: str
    asset_id: str

    currency: str

    total_repair_cost: Decimal = ZERO
    total_replacement_cost: Decimal = ZERO

    total_maintenance_cost: Decimal = ZERO
    total_downtime_cost: Decimal = ZERO

    total_production_loss: Decimal = ZERO
    total_emergency_cost: Decimal = ZERO

    total_energy_cost: Decimal = ZERO
    total_scrap_cost: Decimal = ZERO

    @property
    def grand_total(self) -> Decimal:
        return money(
            self.total_repair_cost
            + self.total_replacement_cost
            + self.total_maintenance_cost
            + self.total_downtime_cost
            + self.total_production_loss
            + self.total_emergency_cost
            + self.total_energy_cost
            + self.total_scrap_cost
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")

        data["grand_total"] = str(
            self.grand_total
        )

        return data


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    context = CostContext(
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="PUMP-101",
        component_id="BEARING-101",
        currency="INR",
    )

    repair = RepairCostEstimate(
        context=context,
        parts=[
            CostLineItem(
                cost_type=CostType.SPARE_PARTS,
                description="Replacement bearing",
                quantity=Decimal("2"),
                unit_cost=Decimal("4500"),
            )
        ],
        labor=LaborCost(
            technician_count=2,
            hours_per_technician=Decimal("4"),
            hourly_rate=Decimal("800"),
        ),
        downtime=DowntimeCost(
            downtime_hours=Decimal("6"),
            cost_per_hour=Decimal("15000"),
        ),
        inspection_cost=Decimal("2000"),
    )

    replacement = ReplacementCostEstimate(
        context=context,
        purchase_cost=Decimal("120000"),
        installation_cost=Decimal("15000"),
        transportation_cost=Decimal("5000"),
        downtime=DowntimeCost(
            downtime_hours=Decimal("10"),
            cost_per_hour=Decimal("15000"),
        ),
        salvage_value=Decimal("10000"),
    )

    comparison = RepairReplacementComparison(
        repair=repair,
        replacement=replacement,
    )

    roi = MaintenanceROI(
        context=context,
        investment_cost=Decimal("25000"),
        avoided_failure_cost=Decimal("90000"),
        downtime_savings=Decimal("45000"),
        maintenance_savings=Decimal("12000"),
    )

    print(
        json.dumps(
            comparison.to_dict(),
            indent=2,
        )
    )

    print(
        json.dumps(
            roi.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()