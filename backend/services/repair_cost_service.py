
"""
app/services/repair_cost_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
========================================================

Repair Cost Estimation and Analysis Service

Features:
- Repair cost estimation
- Spare parts and material costing
- Technician labor costing
- Diagnostic and inspection costs
- Equipment and external service costs
- Downtime and production-loss costing
- Emergency repair surcharges
- Taxes, discounts, and contingency reserves
- Repair estimate scenarios
- Actual vs estimated repair costs
- Repair vs replacement comparison
- Batch repair cost analysis
- Optional async repository persistence
- Optional audit integration

Financial rules:
- Uses Decimal for monetary calculations.
- Currency is explicit.
- Direct repair cost is separated from economic impact.
- Downtime and production losses must not overlap.
- Replacement comparisons must use consistent cost scope.
- Estimates are advisory, not supplier quotations.

Expected repository:
    async create_repair_cost(data)
    async get_repair_cost(repair_cost_id)
    async list_repair_costs(filters)

Optional audit service:
    async record_event(data)
"""

from __future__ import annotations

import inspect
import logging

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)

MONEY_QUANT = Decimal("0.01")
ZERO = Decimal("0")
ONE_HUNDRED = Decimal("100")


# ============================================================
# EXCEPTIONS
# ============================================================

class RepairCostServiceError(Exception):
    """Base repair cost service exception."""


class RepairCostValidationError(RepairCostServiceError):
    """Invalid repair cost input."""


class RepairCostNotFoundError(RepairCostServiceError):
    """Repair cost record not found."""


class RepairCostRepositoryError(RepairCostServiceError):
    """Repair cost repository operation failed."""


# ============================================================
# ENUMS
# ============================================================

class RepairType(str, Enum):
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"
    OVERHAUL = "overhaul"
    COMPONENT_REPLACEMENT = "component_replacement"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    SOFTWARE = "software"
    OTHER = "other"


class RepairPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RepairStatus(str, Enum):
    ESTIMATED = "estimated"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RepairScenario(str, Enum):
    OPTIMISTIC = "optimistic"
    BASE = "base"
    PESSIMISTIC = "pessimistic"


class RepairDecision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    REVIEW_REQUIRED = "review_required"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def decimal_value(
    value: Any,
    name: str,
    minimum: Decimal | None = ZERO,
) -> Decimal:
    if isinstance(value, bool):
        raise RepairCostValidationError(
            f"{name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise RepairCostValidationError(
            f"Invalid numeric value for {name}"
        ) from exc

    if not result.is_finite():
        raise RepairCostValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise RepairCostValidationError(
            f"{name} must be >= {minimum}"
        )

    return result


def money(value: Any) -> Decimal:
    return decimal_value(
        value,
        "money",
    ).quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def required_text(
    value: Any,
    name: str,
    max_length: int = 255,
) -> str:
    if not isinstance(value, str):
        raise RepairCostValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not result:
        raise RepairCostValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > max_length:
        raise RepairCostValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def optional_text(
    value: str | None,
    name: str,
    max_length: int = 255,
) -> str | None:
    if value is None:
        return None

    return required_text(value, name, max_length)


def currency_code(value: str) -> str:
    code = required_text(
        value,
        "currency",
        3,
    ).upper()

    if len(code) != 3 or not code.isalpha():
        raise RepairCostValidationError(
            "Currency must be a three-letter code"
        )

    return code


def percent_value(
    value: Any,
    name: str,
) -> Decimal:
    result = decimal_value(value, name)

    if result > ONE_HUNDRED:
        raise RepairCostValidationError(
            f"{name} cannot exceed 100"
        )

    return result


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:
    try:
        return enum_class(value)
    except (ValueError, TypeError) as exc:
        raise RepairCostValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def serialize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)

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

    if isinstance(value, (list, tuple)):
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
class SparePartCost:
    part_id: str
    name: str
    quantity: Decimal
    unit_cost: Decimal
    total_cost: Decimal

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class LaborCost:
    technician_role: str
    hours: Decimal
    hourly_rate: Decimal
    total_cost: Decimal

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class RepairCostBreakdown:
    spare_parts_cost: Decimal
    labor_cost: Decimal
    diagnostic_cost: Decimal
    inspection_cost: Decimal
    equipment_cost: Decimal
    external_service_cost: Decimal
    transport_cost: Decimal
    miscellaneous_cost: Decimal

    direct_cost_subtotal: Decimal
    emergency_surcharge: Decimal
    discount: Decimal
    contingency_reserve: Decimal
    tax: Decimal

    total_direct_repair_cost: Decimal

    downtime_cost: Decimal
    production_loss_cost: Decimal

    total_economic_impact: Decimal

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class RepairCostEstimate:
    repair_cost_id: str
    company_id: str
    site_id: str
    asset_id: str

    repair_type: str
    priority: str
    status: str
    scenario: str
    currency: str

    description: str

    spare_parts: list[SparePartCost]
    labor_items: list[LaborCost]

    breakdown: RepairCostBreakdown

    estimated_duration_hours: Decimal

    machine_id: str | None = None
    component_id: str | None = None
    maintenance_id: str | None = None
    incident_id: str | None = None

    created_by: str = "system"
    created_at: datetime = field(default_factory=utc_now)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class RepairCostVariance:
    repair_cost_id: str
    currency: str
    estimated_cost: Decimal
    actual_cost: Decimal
    variance: Decimal
    variance_percent: Decimal
    over_budget: bool

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class RepairReplacementComparison:
    currency: str
    repair_cost: Decimal
    replacement_cost: Decimal
    repair_to_replacement_ratio_percent: Decimal
    estimated_savings: Decimal
    recommendation: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# REPAIR COST SERVICE
# ============================================================

class RepairCostService:
    """
    Industrial repair cost estimation and analysis.

    All calculations can run without a database.

    Optional repository integration supports persistence.

    Caller responsibilities:
    - Enforce RBAC and tenant isolation.
    - Validate supplier quotations.
    - Prevent duplicate cost entries.
    - Verify actual repair completion.
    - Apply jurisdiction-specific tax rules.
    """

    SCENARIO_MULTIPLIERS = {
        RepairScenario.OPTIMISTIC: Decimal("0.85"),
        RepairScenario.BASE: Decimal("1.00"),
        RepairScenario.PESSIMISTIC: Decimal("1.25"),
    }

    def __init__(
        self,
        repair_cost_repository: Any = None,
        audit_service: Any = None,
        default_currency: str = "INR",
    ):
        self.repository = repair_cost_repository
        self.audit = audit_service
        self.default_currency = currency_code(
            default_currency
        )

    # ========================================================
    # SPARE PART COST
    # ========================================================

    @staticmethod
    def calculate_spare_part_cost(
        *,
        part_id: str,
        name: str,
        quantity: Any,
        unit_cost: Any,
    ) -> SparePartCost:
        qty = decimal_value(
            quantity,
            "quantity",
        )

        if qty <= ZERO:
            raise RepairCostValidationError(
                "Spare part quantity must be positive"
            )

        price = decimal_value(
            unit_cost,
            "unit_cost",
        )

        return SparePartCost(
            part_id=required_text(
                part_id,
                "part_id",
            ),
            name=required_text(
                name,
                "name",
            ),
            quantity=qty,
            unit_cost=money(price),
            total_cost=money(qty * price),
        )

    # ========================================================
    # LABOR COST
    # ========================================================

    @staticmethod
    def calculate_labor_cost(
        *,
        technician_role: str,
        hours: Any,
        hourly_rate: Any,
    ) -> LaborCost:
        duration = decimal_value(
            hours,
            "hours",
        )

        rate = decimal_value(
            hourly_rate,
            "hourly_rate",
        )

        return LaborCost(
            technician_role=required_text(
                technician_role,
                "technician_role",
            ),
            hours=duration,
            hourly_rate=money(rate),
            total_cost=money(duration * rate),
        )

    # ========================================================
    # COMPLETE REPAIR ESTIMATE
    # ========================================================

    def estimate_repair_cost(
        self,
        *,
        company_id: str,
        site_id: str,
        asset_id: str,
        description: str,

        spare_parts: Sequence[Mapping[str, Any]] | None = None,
        labor_items: Sequence[Mapping[str, Any]] | None = None,

        diagnostic_cost: Any = 0,
        inspection_cost: Any = 0,
        equipment_cost: Any = 0,
        external_service_cost: Any = 0,
        transport_cost: Any = 0,
        miscellaneous_cost: Any = 0,

        emergency_surcharge_percent: Any = 0,
        discount_percent: Any = 0,
        contingency_percent: Any = 10,
        tax_percent: Any = 0,

        downtime_cost: Any = 0,
        production_loss_cost: Any = 0,

        estimated_duration_hours: Any = 0,

        repair_type: str | RepairType = RepairType.CORRECTIVE,
        priority: str | RepairPriority = RepairPriority.MEDIUM,
        scenario: str | RepairScenario = RepairScenario.BASE,

        currency: str | None = None,

        machine_id: str | None = None,
        component_id: str | None = None,
        maintenance_id: str | None = None,
        incident_id: str | None = None,

        created_by: str = "system",
        metadata: Mapping[str, Any] | None = None,
    ) -> RepairCostEstimate:

        selected_type = parse_enum(
            RepairType,
            repair_type,
            "repair_type",
        )

        selected_priority = parse_enum(
            RepairPriority,
            priority,
            "priority",
        )

        selected_scenario = parse_enum(
            RepairScenario,
            scenario,
            "scenario",
        )

        selected_currency = currency_code(
            currency or self.default_currency
        )

        # ----------------------------------------------------
        # SPARE PARTS
        # ----------------------------------------------------

        parts: list[SparePartCost] = []

        for item in spare_parts or []:
            if not isinstance(item, Mapping):
                raise RepairCostValidationError(
                    "Each spare part must be a mapping"
                )

            parts.append(
                self.calculate_spare_part_cost(
                    part_id=item.get("part_id"),
                    name=item.get("name"),
                    quantity=item.get("quantity"),
                    unit_cost=item.get("unit_cost"),
                )
            )

        parts_total = sum(
            (item.total_cost for item in parts),
            ZERO,
        )

        # ----------------------------------------------------
        # LABOR
        # ----------------------------------------------------

        labor: list[LaborCost] = []

        for item in labor_items or []:
            if not isinstance(item, Mapping):
                raise RepairCostValidationError(
                    "Each labor item must be a mapping"
                )

            labor.append(
                self.calculate_labor_cost(
                    technician_role=item.get(
                        "technician_role"
                    ),
                    hours=item.get("hours"),
                    hourly_rate=item.get("hourly_rate"),
                )
            )

        labor_total = sum(
            (item.total_cost for item in labor),
            ZERO,
        )

        # ----------------------------------------------------
        # ADDITIONAL DIRECT COSTS
        # ----------------------------------------------------

        diagnostic = money(diagnostic_cost)
        inspection = money(inspection_cost)
        equipment = money(equipment_cost)
        external = money(external_service_cost)
        transport = money(transport_cost)
        miscellaneous = money(miscellaneous_cost)

        downtime = money(downtime_cost)
        production_loss = money(production_loss_cost)

        # ----------------------------------------------------
        # PERCENTAGES
        # ----------------------------------------------------

        emergency_rate = percent_value(
            emergency_surcharge_percent,
            "emergency_surcharge_percent",
        )

        discount_rate = percent_value(
            discount_percent,
            "discount_percent",
        )

        contingency_rate = percent_value(
            contingency_percent,
            "contingency_percent",
        )

        tax_rate = percent_value(
            tax_percent,
            "tax_percent",
        )

        # ----------------------------------------------------
        # DIRECT COST SUBTOTAL
        # ----------------------------------------------------

        subtotal = money(
            parts_total
            + labor_total
            + diagnostic
            + inspection
            + equipment
            + external
            + transport
            + miscellaneous
        )

        # Scenario adjustments are applied to direct
        # cost estimates only, not downtime or production
        # loss, which may come from separate models.

        multiplier = self.SCENARIO_MULTIPLIERS[
            selected_scenario
        ]

        adjusted_subtotal = money(
            subtotal * multiplier
        )

        emergency_surcharge = money(
            adjusted_subtotal
            * emergency_rate
            / ONE_HUNDRED
        )

        gross_direct_cost = money(
            adjusted_subtotal
            + emergency_surcharge
        )

        discount = money(
            gross_direct_cost
            * discount_rate
            / ONE_HUNDRED
        )

        net_direct_cost = money(
            gross_direct_cost - discount
        )

        contingency = money(
            net_direct_cost
            * contingency_rate
            / ONE_HUNDRED
        )

        taxable_amount = money(
            net_direct_cost + contingency
        )

        tax = money(
            taxable_amount
            * tax_rate
            / ONE_HUNDRED
        )

        total_direct_cost = money(
            taxable_amount + tax
        )

        total_economic_impact = money(
            total_direct_cost
            + downtime
            + production_loss
        )

        # ----------------------------------------------------
        # BREAKDOWN
        # ----------------------------------------------------

        breakdown = RepairCostBreakdown(
            spare_parts_cost=money(parts_total),
            labor_cost=money(labor_total),
            diagnostic_cost=diagnostic,
            inspection_cost=inspection,
            equipment_cost=equipment,
            external_service_cost=external,
            transport_cost=transport,
            miscellaneous_cost=miscellaneous,
            direct_cost_subtotal=adjusted_subtotal,
            emergency_surcharge=emergency_surcharge,
            discount=discount,
            contingency_reserve=contingency,
            tax=tax,
            total_direct_repair_cost=total_direct_cost,
            downtime_cost=downtime,
            production_loss_cost=production_loss,
            total_economic_impact=total_economic_impact,
        )

        duration = decimal_value(
            estimated_duration_hours,
            "estimated_duration_hours",
        )

        return RepairCostEstimate(
            repair_cost_id=str(uuid4()),
            company_id=required_text(
                company_id,
                "company_id",
            ),
            site_id=required_text(
                site_id,
                "site_id",
            ),
            asset_id=required_text(
                asset_id,
                "asset_id",
            ),
            repair_type=selected_type.value,
            priority=selected_priority.value,
            status=RepairStatus.ESTIMATED.value,
            scenario=selected_scenario.value,
            currency=selected_currency,
            description=required_text(
                description,
                "description",
                5000,
            ),
            spare_parts=parts,
            labor_items=labor,
            breakdown=breakdown,
            estimated_duration_hours=duration,
            machine_id=optional_text(
                machine_id,
                "machine_id",
            ),
            component_id=optional_text(
                component_id,
                "component_id",
            ),
            maintenance_id=optional_text(
                maintenance_id,
                "maintenance_id",
            ),
            incident_id=optional_text(
                incident_id,
                "incident_id",
            ),
            created_by=required_text(
                created_by,
                "created_by",
            ),
            metadata={
                **dict(metadata or {}),
                "scenario_multiplier": str(multiplier),
                "emergency_surcharge_percent": str(
                    emergency_rate
                ),
                "discount_percent": str(
                    discount_rate
                ),
                "contingency_percent": str(
                    contingency_rate
                ),
                "tax_percent": str(tax_rate),
                "cost_scope": (
                    "direct_repair_plus_separately_"
                    "provided_economic_losses"
                ),
            },
        )

    # ========================================================
    # ESTIMATED VS ACTUAL COST
    # ========================================================

    @staticmethod
    def calculate_cost_variance(
        *,
        repair_cost_id: str,
        estimated_cost: Any,
        actual_cost: Any,
        currency: str = "INR",
    ) -> RepairCostVariance:

        estimated = money(estimated_cost)
        actual = money(actual_cost)

        variance = actual - estimated

        if estimated > ZERO:
            variance_percent = (
                variance
                / estimated
                * ONE_HUNDRED
            ).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )
        else:
            variance_percent = ZERO

        return RepairCostVariance(
            repair_cost_id=required_text(
                repair_cost_id,
                "repair_cost_id",
            ),
            currency=currency_code(currency),
            estimated_cost=estimated,
            actual_cost=actual,
            variance=money(variance),
            variance_percent=variance_percent,
            over_budget=actual > estimated,
        )

    # ========================================================
    # REPAIR VS REPLACEMENT
    # ========================================================

    @staticmethod
    def compare_repair_vs_replacement(
        *,
        repair_cost: Any,
        replacement_cost: Any,
        currency: str = "INR",
        repair_threshold_percent: Any = 60,
    ) -> RepairReplacementComparison:

        repair = money(repair_cost)
        replacement = money(replacement_cost)

        threshold = percent_value(
            repair_threshold_percent,
            "repair_threshold_percent",
        )

        if replacement <= ZERO:
            raise RepairCostValidationError(
                "Replacement cost must be positive"
            )

        ratio = (
            repair
            / replacement
            * ONE_HUNDRED
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        savings = money(
            replacement - repair
        )

        if ratio < threshold:
            decision = RepairDecision.REPAIR
            explanation = (
                "Repair cost is below the configured "
                "replacement-cost threshold. Review "
                "reliability and lifecycle factors "
                "before approval."
            )

        elif ratio >= ONE_HUNDRED:
            decision = RepairDecision.REPLACE
            explanation = (
                "Repair cost equals or exceeds "
                "replacement cost. Replacement may "
                "be more economical, subject to "
                "asset condition and lifecycle review."
            )

        else:
            decision = RepairDecision.REVIEW_REQUIRED
            explanation = (
                "Repair cost is near the configured "
                "replacement threshold. Evaluate "
                "remaining useful life, warranty, "
                "downtime and future failure risk."
            )

        return RepairReplacementComparison(
            currency=currency_code(currency),
            repair_cost=repair,
            replacement_cost=replacement,
            repair_to_replacement_ratio_percent=ratio,
            estimated_savings=savings,
            recommendation=decision.value,
            explanation=explanation,
        )

    # ========================================================
    # MULTI-SCENARIO ESTIMATION
    # ========================================================

    def compare_repair_scenarios(
        self,
        **estimate_kwargs: Any,
    ) -> dict[str, Any]:

        results = {}

        for scenario in RepairScenario:
            estimate = self.estimate_repair_cost(
                **{
                    **estimate_kwargs,
                    "scenario": scenario,
                }
            )

            results[scenario.value] = {
                "repair_cost_id": estimate.repair_cost_id,
                "direct_repair_cost": str(
                    estimate.breakdown.total_direct_repair_cost
                ),
                "total_economic_impact": str(
                    estimate.breakdown.total_economic_impact
                ),
                "currency": estimate.currency,
            }

        return results

    # ========================================================
    # AGGREGATE REPAIR COSTS
    # ========================================================

    @staticmethod
    def aggregate_repair_costs(
        estimates: Sequence[RepairCostEstimate],
    ) -> dict[str, Any]:

        if not estimates:
            return {
                "total_repairs": 0,
                "currency": None,
                "total_direct_repair_cost": "0.00",
                "total_economic_impact": "0.00",
                "average_direct_repair_cost": "0.00",
            }

        currencies = {
            item.currency
            for item in estimates
        }

        if len(currencies) != 1:
            raise RepairCostValidationError(
                "Cannot aggregate different currencies"
            )

        ids = [
            item.repair_cost_id
            for item in estimates
        ]

        if len(ids) != len(set(ids)):
            raise RepairCostValidationError(
                "Duplicate repair estimates detected"
            )

        direct_total = sum(
            (
                item.breakdown.total_direct_repair_cost
                for item in estimates
            ),
            ZERO,
        )

        economic_total = sum(
            (
                item.breakdown.total_economic_impact
                for item in estimates
            ),
            ZERO,
        )

        average = direct_total / Decimal(
            len(estimates)
        )

        return {
            "total_repairs": len(estimates),
            "currency": next(iter(currencies)),
            "total_direct_repair_cost": str(
                money(direct_total)
            ),
            "total_economic_impact": str(
                money(economic_total)
            ),
            "average_direct_repair_cost": str(
                money(average)
            ),
        }

    # ========================================================
    # AUDIT
    # ========================================================

    async def _audit(
        self,
        *,
        repair_cost_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:

        if self.audit is None:
            return

        method = getattr(
            self.audit,
            "record_event",
            None,
        )

        if method is None:
            return

        try:
            await invoke(
                method,
                {
                    "event_id": str(uuid4()),
                    "entity_type": "repair_cost",
                    "entity_id": repair_cost_id,
                    "actor_id": actor_id,
                    "action": action,
                    "details": dict(details or {}),
                    "timestamp": utc_now().isoformat(),
                },
            )
        except Exception:
            logger.exception(
                "Repair cost audit failed: %s",
                repair_cost_id,
            )

    # ========================================================
    # SAVE ESTIMATE
    # ========================================================

    async def save_repair_estimate(
        self,
        estimate: RepairCostEstimate,
    ) -> Any:

        if self.repository is None:
            raise RepairCostRepositoryError(
                "Repair cost repository is not configured"
            )

        if not isinstance(
            estimate,
            RepairCostEstimate,
        ):
            raise RepairCostValidationError(
                "Expected RepairCostEstimate"
            )

        saved = await invoke(
            self.repository.create_repair_cost,
            estimate.to_dict(),
        )

        await self._audit(
            repair_cost_id=estimate.repair_cost_id,
            actor_id=estimate.created_by,
            action="repair_cost_estimated",
            details={
                "asset_id": estimate.asset_id,
                "currency": estimate.currency,
                "direct_repair_cost": str(
                    estimate.breakdown.total_direct_repair_cost
                ),
            },
        )

        return saved

    # ========================================================
    # GET ESTIMATE
    # ========================================================

    async def get_repair_estimate(
        self,
        repair_cost_id: str,
    ) -> Any:

        if self.repository is None:
            raise RepairCostRepositoryError(
                "Repair cost repository is not configured"
            )

        record = await invoke(
            self.repository.get_repair_cost,
            required_text(
                repair_cost_id,
                "repair_cost_id",
            ),
        )

        if record is None:
            raise RepairCostNotFoundError(
                f"Repair estimate not found: {repair_cost_id}"
            )

        return record

    # ========================================================
    # LIST ESTIMATES
    # ========================================================

    async def list_repair_estimates(
        self,
        *,
        company_id: str | None = None,
        site_id: str | None = None,
        asset_id: str | None = None,
        repair_type: str | None = None,
    ) -> Any:

        if self.repository is None:
            raise RepairCostRepositoryError(
                "Repair cost repository is not configured"
            )

        filters: dict[str, Any] = {}

        for name, value in {
            "company_id": company_id,
            "site_id": site_id,
            "asset_id": asset_id,
        }.items():
            if value is not None:
                filters[name] = required_text(
                    value,
                    name,
                )

        if repair_type is not None:
            filters["repair_type"] = parse_enum(
                RepairType,
                repair_type,
                "repair_type",
            ).value

        return await invoke(
            self.repository.list_repair_costs,
            filters,
        )


# ============================================================
# IN-MEMORY REPOSITORY
# ============================================================

class InMemoryRepairCostRepository:
    """
    Development-only repository.

    Not durable or concurrency-safe.
    """

    def __init__(self):
        self.records: dict[str, dict[str, Any]] = {}

    async def create_repair_cost(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        import copy

        record = copy.deepcopy(data)

        repair_cost_id = record["repair_cost_id"]

        if repair_cost_id in self.records:
            raise RepairCostRepositoryError(
                "Duplicate repair cost ID"
            )

        self.records[repair_cost_id] = record

        return copy.deepcopy(record)

    async def get_repair_cost(
        self,
        repair_cost_id: str,
    ) -> dict[str, Any] | None:

        import copy

        record = self.records.get(repair_cost_id)

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def list_repair_costs(
        self,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:

        import copy

        return [
            copy.deepcopy(record)
            for record in self.records.values()
            if all(
                record.get(key) == value
                for key, value in filters.items()
            )
        ]


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    import asyncio
    import json

    async def main() -> None:

        logging.basicConfig(level=logging.INFO)

        repository = InMemoryRepairCostRepository()

        service = RepairCostService(
            repair_cost_repository=repository,
            default_currency="INR",
        )

        estimate = service.estimate_repair_cost(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_id="MACHINE-001",
            machine_id="MOTOR-001",
            maintenance_id="MAINTENANCE-001",
            description=(
                "Industrial motor bearing replacement "
                "and cooling system inspection"
            ),
            repair_type="corrective",
            priority="high",
            scenario="base",
            spare_parts=[
                {
                    "part_id": "BEARING-001",
                    "name": "Industrial motor bearing",
                    "quantity": 2,
                    "unit_cost": 2500,
                },
                {
                    "part_id": "SEAL-001",
                    "name": "Mechanical seal",
                    "quantity": 1,
                    "unit_cost": 1200,
                },
            ],
            labor_items=[
                {
                    "technician_role": "Mechanical Technician",
                    "hours": 4,
                    "hourly_rate": 650,
                },
                {
                    "technician_role": "Electrical Technician",
                    "hours": 2,
                    "hourly_rate": 800,
                },
            ],
            diagnostic_cost=1500,
            inspection_cost=750,
            equipment_cost=1000,
            external_service_cost=0,
            transport_cost=500,
            miscellaneous_cost=300,
            emergency_surcharge_percent=0,
            discount_percent=5,
            contingency_percent=10,
            tax_percent=18,
            downtime_cost=5000,
            production_loss_cost=8000,
            estimated_duration_hours=6,
            created_by="ENGINEER-001",
        )

        print("\nREPAIR COST ESTIMATE")

        print(
            json.dumps(
                estimate.to_dict(),
                indent=2,
            )
        )

        await service.save_repair_estimate(
            estimate
        )

        comparison = (
            service.compare_repair_vs_replacement(
                repair_cost=(
                    estimate.breakdown.total_direct_repair_cost
                ),
                replacement_cost=65000,
                currency="INR",
                repair_threshold_percent=60,
            )
        )

        print("\nREPAIR VS REPLACEMENT")

        print(
            json.dumps(
                comparison.to_dict(),
                indent=2,
            )
        )

        variance = service.calculate_cost_variance(
            repair_cost_id=estimate.repair_cost_id,
            estimated_cost=(
                estimate.breakdown.total_direct_repair_cost
            ),
            actual_cost=18500,
            currency="INR",
        )

        print("\nCOST VARIANCE")

        print(
            json.dumps(
                variance.to_dict(),
                indent=2,
            )
        )

        scenarios = service.compare_repair_scenarios(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_id="MACHINE-001",
            description="Motor bearing replacement",
            spare_parts=[
                {
                    "part_id": "BEARING-001",
                    "name": "Motor bearing",
                    "quantity": 2,
                    "unit_cost": 2500,
                }
            ],
            labor_items=[
                {
                    "technician_role": "Technician",
                    "hours": 4,
                    "hourly_rate": 650,
                }
            ],
            contingency_percent=10,
            tax_percent=18,
        )

        print("\nSCENARIO COMPARISON")

        print(
            json.dumps(
                scenarios,
                indent=2,
            )
        )

        records = await service.list_repair_estimates(
            asset_id="MACHINE-001"
        )

        print("\nSAVED REPAIR ESTIMATES")
        print(f"Total records: {len(records)}")

    asyncio.run(main())
