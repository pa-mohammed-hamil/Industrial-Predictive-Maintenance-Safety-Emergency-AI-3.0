
"""
app/services/replacement_cost_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
========================================================

Replacement Cost Analysis Service

Features:
- Industrial asset and component replacement estimation
- Equipment purchase and procurement costs
- Shipping, installation, commissioning and training
- Decommissioning, disposal and recycling
- Salvage and trade-in credits
- Taxes, discounts and contingency reserves
- Replacement-related downtime and production losses
- Emergency replacement surcharges
- Optimistic, base and pessimistic scenarios
- Replacement cost forecasting
- Repair vs replacement financial comparison
- Lifecycle cost and discounted cash-flow analysis
- Actual vs estimated replacement cost variance
- Batch replacement cost aggregation
- Optional asynchronous repository persistence
- Optional audit logging

Financial assumptions:
- Decimal is used for monetary calculations.
- All monetary inputs must use the same currency.
- Tax rules are simplified and must be reviewed locally.
- Salvage credits are applied separately from direct costs.
- Downtime and production loss must not overlap.
- Lifecycle costs are scenario estimates, not guarantees.
- Financial recommendations are advisory only.

Expected repository methods:
    async create_replacement_cost(data)
    async get_replacement_cost(replacement_cost_id)
    async list_replacement_costs(filters)

Optional audit method:
    async record_event(data)
"""

from __future__ import annotations

import copy
import inspect
import logging

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)

ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")
MONEY_QUANT = Decimal("0.01")
PERCENT_QUANT = Decimal("0.01")


# ============================================================
# EXCEPTIONS
# ============================================================

class ReplacementCostServiceError(Exception):
    """Base replacement cost service exception."""


class ReplacementCostValidationError(ReplacementCostServiceError):
    """Invalid replacement cost input."""


class ReplacementCostNotFoundError(ReplacementCostServiceError):
    """Replacement cost record not found."""


class ReplacementCostRepositoryError(ReplacementCostServiceError):
    """Repository operation failed."""


# ============================================================
# ENUMS
# ============================================================

class ReplacementType(str, Enum):
    FULL_ASSET = "full_asset"
    MACHINE = "machine"
    COMPONENT = "component"
    SUBSYSTEM = "subsystem"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    ELECTRONIC = "electronic"
    SAFETY_SYSTEM = "safety_system"
    UPGRADE = "upgrade"
    OTHER = "other"


class ReplacementPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReplacementStatus(str, Enum):
    ESTIMATED = "estimated"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    ORDERED = "ordered"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReplacementScenario(str, Enum):
    OPTIMISTIC = "optimistic"
    BASE = "base"
    PESSIMISTIC = "pessimistic"


class ReplacementDecision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    REVIEW_REQUIRED = "review_required"


# ============================================================
# VALIDATION AND SERIALIZATION
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_decimal(
    value: Any,
    name: str,
    minimum: Optional[Decimal] = ZERO,
) -> Decimal:
    if isinstance(value, bool):
        raise ReplacementCostValidationError(
            f"{name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ReplacementCostValidationError(
            f"Invalid numeric value for {name}"
        ) from exc

    if not result.is_finite():
        raise ReplacementCostValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise ReplacementCostValidationError(
            f"{name} must be >= {minimum}"
        )

    return result


def money(value: Any) -> Decimal:
    return to_decimal(
        value,
        "money",
        minimum=None,
    ).quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def percentage(
    value: Any,
    name: str,
) -> Decimal:
    result = to_decimal(value, name)

    if result > HUNDRED:
        raise ReplacementCostValidationError(
            f"{name} cannot exceed 100"
        )

    return result


def required_text(
    value: Any,
    name: str,
    max_length: int = 255,
) -> str:
    if not isinstance(value, str):
        raise ReplacementCostValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not result:
        raise ReplacementCostValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > max_length:
        raise ReplacementCostValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def optional_text(
    value: Optional[str],
    name: str,
    max_length: int = 255,
) -> Optional[str]:
    if value is None:
        return None

    return required_text(
        value,
        name,
        max_length,
    )


def currency_code(value: str) -> str:
    code = required_text(
        value,
        "currency",
        3,
    ).upper()

    if len(code) != 3 or not code.isascii() or not code.isalpha():
        raise ReplacementCostValidationError(
            "Currency must be a three-letter ASCII code"
        )

    return code


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:
    try:
        return enum_class(value)
    except (ValueError, TypeError) as exc:
        raise ReplacementCostValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def validate_positive_int(
    value: Any,
    name: str,
    maximum: int = 100,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise ReplacementCostValidationError(
            f"{name} must be an integer from 1 to {maximum}"
        )

    return value


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
class ReplacementItem:
    item_id: str
    name: str
    quantity: Decimal
    unit_cost: Decimal
    total_cost: Decimal
    supplier_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class ReplacementCostBreakdown:
    equipment_cost: Decimal
    shipping_cost: Decimal
    installation_cost: Decimal
    commissioning_cost: Decimal
    training_cost: Decimal
    decommissioning_cost: Decimal
    disposal_cost: Decimal
    integration_cost: Decimal
    permits_cost: Decimal
    other_cost: Decimal

    base_cost_subtotal: Decimal
    emergency_surcharge: Decimal
    discount: Decimal
    contingency_reserve: Decimal
    tax: Decimal

    gross_replacement_cost: Decimal
    salvage_credit: Decimal
    trade_in_credit: Decimal
    net_replacement_cost: Decimal

    downtime_cost: Decimal
    production_loss_cost: Decimal
    total_economic_impact: Decimal

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class ReplacementCostEstimate:
    replacement_cost_id: str
    company_id: str
    site_id: str
    asset_id: str

    replacement_type: str
    priority: str
    status: str
    scenario: str
    currency: str

    description: str
    replacement_items: list[ReplacementItem]
    breakdown: ReplacementCostBreakdown

    estimated_duration_hours: Decimal
    expected_useful_life_years: int

    machine_id: Optional[str] = None
    component_id: Optional[str] = None
    maintenance_id: Optional[str] = None
    incident_id: Optional[str] = None

    created_by: str = "system"
    created_at: datetime = field(default_factory=utc_now)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class ReplacementCostVariance:
    replacement_cost_id: str
    currency: str
    estimated_cost: Decimal
    actual_cost: Decimal
    variance: Decimal
    variance_percent: Optional[Decimal]
    over_budget: bool

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class ReplacementLifecycleResult:
    currency: str
    analysis_years: int
    discount_rate_percent: Decimal
    initial_replacement_cost: Decimal
    annual_operating_cost: Decimal
    annual_maintenance_cost: Decimal
    annual_energy_cost: Decimal
    annual_cost_growth_percent: Decimal
    end_of_life_disposal_cost: Decimal
    residual_value: Decimal
    operating_cost_npv: Decimal
    terminal_cost_npv: Decimal
    total_lifecycle_cost_npv: Decimal
    annual_cash_flows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class RepairReplacementDecision:
    currency: str
    repair_cost: Decimal
    replacement_cost: Decimal
    repair_to_replacement_ratio_percent: Decimal
    initial_cost_difference: Decimal
    recommendation: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# REPLACEMENT COST SERVICE
# ============================================================

class ReplacementCostService:
    """
    Industrial equipment replacement financial service.

    Supports standalone calculations and optional
    repository persistence.

    Production integrations must enforce:
        - Tenant isolation
        - Authorization
        - Supplier quotation verification
        - Transactional persistence
        - Audit reliability
        - Cost accounting consistency
    """

    SCENARIO_MULTIPLIERS = {
        ReplacementScenario.OPTIMISTIC: Decimal("0.90"),
        ReplacementScenario.BASE: Decimal("1.00"),
        ReplacementScenario.PESSIMISTIC: Decimal("1.20"),
    }

    def __init__(
        self,
        replacement_cost_repository: Any = None,
        audit_service: Any = None,
        default_currency: str = "INR",
    ):
        self.repository = replacement_cost_repository
        self.audit = audit_service
        self.default_currency = currency_code(
            default_currency
        )

    # ========================================================
    # REPLACEMENT ITEM COST
    # ========================================================

    @staticmethod
    def calculate_replacement_item_cost(
        *,
        item_id: str,
        name: str,
        quantity: Any,
        unit_cost: Any,
        supplier_id: Optional[str] = None,
    ) -> ReplacementItem:

        qty = to_decimal(
            quantity,
            "quantity",
        )

        if qty <= ZERO:
            raise ReplacementCostValidationError(
                "Replacement quantity must be positive"
            )

        price = to_decimal(
            unit_cost,
            "unit_cost",
        )

        return ReplacementItem(
            item_id=required_text(
                item_id,
                "item_id",
            ),
            name=required_text(
                name,
                "name",
            ),
            quantity=qty,
            unit_cost=money(price),
            total_cost=money(qty * price),
            supplier_id=optional_text(
                supplier_id,
                "supplier_id",
            ),
        )

    # ========================================================
    # REPLACEMENT COST ESTIMATION
    # ========================================================

    def estimate_replacement_cost(
        self,
        *,
        company_id: str,
        site_id: str,
        asset_id: str,
        description: str,

        replacement_items: Sequence[
            Mapping[str, Any]
        ] | None = None,

        shipping_cost: Any = 0,
        installation_cost: Any = 0,
        commissioning_cost: Any = 0,
        training_cost: Any = 0,
        decommissioning_cost: Any = 0,
        disposal_cost: Any = 0,
        integration_cost: Any = 0,
        permits_cost: Any = 0,
        other_cost: Any = 0,

        emergency_surcharge_percent: Any = 0,
        discount_percent: Any = 0,
        contingency_percent: Any = 10,
        tax_percent: Any = 0,

        salvage_credit: Any = 0,
        trade_in_credit: Any = 0,

        downtime_cost: Any = 0,
        production_loss_cost: Any = 0,

        estimated_duration_hours: Any = 0,
        expected_useful_life_years: int = 10,

        replacement_type: str | ReplacementType = (
            ReplacementType.FULL_ASSET
        ),
        priority: str | ReplacementPriority = (
            ReplacementPriority.MEDIUM
        ),
        scenario: str | ReplacementScenario = (
            ReplacementScenario.BASE
        ),

        currency: Optional[str] = None,

        machine_id: Optional[str] = None,
        component_id: Optional[str] = None,
        maintenance_id: Optional[str] = None,
        incident_id: Optional[str] = None,

        created_by: str = "system",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ReplacementCostEstimate:

        selected_type = parse_enum(
            ReplacementType,
            replacement_type,
            "replacement_type",
        )

        selected_priority = parse_enum(
            ReplacementPriority,
            priority,
            "priority",
        )

        selected_scenario = parse_enum(
            ReplacementScenario,
            scenario,
            "scenario",
        )

        selected_currency = currency_code(
            currency or self.default_currency
        )

        useful_life = validate_positive_int(
            expected_useful_life_years,
            "expected_useful_life_years",
            maximum=100,
        )

        duration = to_decimal(
            estimated_duration_hours,
            "estimated_duration_hours",
        )

        # ----------------------------------------------------
        # EQUIPMENT / COMPONENT COSTS
        # ----------------------------------------------------

        items: list[ReplacementItem] = []

        for item in replacement_items or []:
            if not isinstance(item, Mapping):
                raise ReplacementCostValidationError(
                    "Replacement items must be mappings"
                )

            items.append(
                self.calculate_replacement_item_cost(
                    item_id=item.get("item_id"),
                    name=item.get("name"),
                    quantity=item.get("quantity"),
                    unit_cost=item.get("unit_cost"),
                    supplier_id=item.get("supplier_id"),
                )
            )

        if not items:
            raise ReplacementCostValidationError(
                "At least one replacement item is required"
            )

        equipment_total = money(
            sum(
                (item.total_cost for item in items),
                ZERO,
            )
        )

        # ----------------------------------------------------
        # ADDITIONAL COSTS
        # ----------------------------------------------------

        shipping = money(
            to_decimal(shipping_cost, "shipping_cost")
        )

        installation = money(
            to_decimal(
                installation_cost,
                "installation_cost",
            )
        )

        commissioning = money(
            to_decimal(
                commissioning_cost,
                "commissioning_cost",
            )
        )

        training = money(
            to_decimal(training_cost, "training_cost")
        )

        decommissioning = money(
            to_decimal(
                decommissioning_cost,
                "decommissioning_cost",
            )
        )

        disposal = money(
            to_decimal(disposal_cost, "disposal_cost")
        )

        integration = money(
            to_decimal(
                integration_cost,
                "integration_cost",
            )
        )

        permits = money(
            to_decimal(permits_cost, "permits_cost")
        )

        other = money(
            to_decimal(other_cost, "other_cost")
        )

        salvage = money(
            to_decimal(
                salvage_credit,
                "salvage_credit",
            )
        )

        trade_in = money(
            to_decimal(
                trade_in_credit,
                "trade_in_credit",
            )
        )

        downtime = money(
            to_decimal(
                downtime_cost,
                "downtime_cost",
            )
        )

        production_loss = money(
            to_decimal(
                production_loss_cost,
                "production_loss_cost",
            )
        )

        # ----------------------------------------------------
        # PERCENTAGES
        # ----------------------------------------------------

        emergency_rate = percentage(
            emergency_surcharge_percent,
            "emergency_surcharge_percent",
        )

        discount_rate = percentage(
            discount_percent,
            "discount_percent",
        )

        contingency_rate = percentage(
            contingency_percent,
            "contingency_percent",
        )

        tax_rate = percentage(
            tax_percent,
            "tax_percent",
        )

        # ----------------------------------------------------
        # BASE COST
        # ----------------------------------------------------

        base_subtotal = money(
            equipment_total
            + shipping
            + installation
            + commissioning
            + training
            + decommissioning
            + disposal
            + integration
            + permits
            + other
        )

        multiplier = self.SCENARIO_MULTIPLIERS[
            selected_scenario
        ]

        adjusted_subtotal = money(
            base_subtotal * multiplier
        )

        emergency_surcharge = money(
            adjusted_subtotal
            * emergency_rate
            / HUNDRED
        )

        gross_before_discount = money(
            adjusted_subtotal
            + emergency_surcharge
        )

        discount = money(
            gross_before_discount
            * discount_rate
            / HUNDRED
        )

        after_discount = money(
            gross_before_discount - discount
        )

        contingency = money(
            after_discount
            * contingency_rate
            / HUNDRED
        )

        taxable_amount = money(
            after_discount + contingency
        )

        tax = money(
            taxable_amount
            * tax_rate
            / HUNDRED
        )

        gross_replacement_cost = money(
            taxable_amount + tax
        )

        total_credits = money(
            salvage + trade_in
        )

        if total_credits > gross_replacement_cost:
            raise ReplacementCostValidationError(
                "Salvage and trade-in credits cannot "
                "exceed gross replacement cost"
            )

        net_replacement_cost = money(
            gross_replacement_cost
            - total_credits
        )

        total_economic_impact = money(
            net_replacement_cost
            + downtime
            + production_loss
        )

        # ----------------------------------------------------
        # BREAKDOWN
        # ----------------------------------------------------

        breakdown = ReplacementCostBreakdown(
            equipment_cost=equipment_total,
            shipping_cost=shipping,
            installation_cost=installation,
            commissioning_cost=commissioning,
            training_cost=training,
            decommissioning_cost=decommissioning,
            disposal_cost=disposal,
            integration_cost=integration,
            permits_cost=permits,
            other_cost=other,
            base_cost_subtotal=adjusted_subtotal,
            emergency_surcharge=emergency_surcharge,
            discount=discount,
            contingency_reserve=contingency,
            tax=tax,
            gross_replacement_cost=gross_replacement_cost,
            salvage_credit=salvage,
            trade_in_credit=trade_in,
            net_replacement_cost=net_replacement_cost,
            downtime_cost=downtime,
            production_loss_cost=production_loss,
            total_economic_impact=total_economic_impact,
        )

        return ReplacementCostEstimate(
            replacement_cost_id=str(uuid4()),
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
            replacement_type=selected_type.value,
            priority=selected_priority.value,
            status=ReplacementStatus.ESTIMATED.value,
            scenario=selected_scenario.value,
            currency=selected_currency,
            description=required_text(
                description,
                "description",
                5000,
            ),
            replacement_items=items,
            breakdown=breakdown,
            estimated_duration_hours=duration,
            expected_useful_life_years=useful_life,
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
                "discount_percent": str(discount_rate),
                "contingency_percent": str(
                    contingency_rate
                ),
                "tax_percent": str(tax_rate),
                "tax_model": (
                    "simplified_uniform_tax_on_"
                    "net_cost_plus_contingency"
                ),
                "cost_scope": (
                    "net_replacement_plus_separately_"
                    "provided_economic_losses"
                ),
            },
        )

    # ========================================================
    # SCENARIO COMPARISON
    # ========================================================

    def compare_replacement_scenarios(
        self,
        **estimate_kwargs: Any,
    ) -> dict[str, Any]:

        results: dict[str, Any] = {}

        for scenario in ReplacementScenario:
            estimate = self.estimate_replacement_cost(
                **{
                    **estimate_kwargs,
                    "scenario": scenario,
                }
            )

            results[scenario.value] = {
                "replacement_cost_id": (
                    estimate.replacement_cost_id
                ),
                "net_replacement_cost": str(
                    estimate.breakdown.net_replacement_cost
                ),
                "total_economic_impact": str(
                    estimate.breakdown.total_economic_impact
                ),
                "currency": estimate.currency,
            }

        return results

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
    ) -> RepairReplacementDecision:

        repair = money(
            to_decimal(repair_cost, "repair_cost")
        )

        replacement = money(
            to_decimal(
                replacement_cost,
                "replacement_cost",
            )
        )

        threshold = percentage(
            repair_threshold_percent,
            "repair_threshold_percent",
        )

        if replacement <= ZERO:
            raise ReplacementCostValidationError(
                "Replacement cost must be positive"
            )

        ratio = (
            repair
            / replacement
            * HUNDRED
        ).quantize(
            PERCENT_QUANT,
            rounding=ROUND_HALF_UP,
        )

        difference = money(
            replacement - repair
        )

        if ratio < threshold:
            recommendation = ReplacementDecision.REPAIR
            explanation = (
                "Repair cost is below the configured "
                "replacement threshold. Verify remaining "
                "useful life and future reliability."
            )

        elif ratio >= HUNDRED:
            recommendation = ReplacementDecision.REPLACE
            explanation = (
                "Repair cost equals or exceeds replacement "
                "cost. Replacement is financially attractive "
                "on initial cost, subject to lifecycle and "
                "operational review."
            )

        else:
            recommendation = (
                ReplacementDecision.REVIEW_REQUIRED
            )
            explanation = (
                "Repair cost exceeds the preferred repair "
                "threshold but remains below replacement "
                "cost. Compare lifecycle costs, downtime, "
                "warranty and failure risk."
            )

        return RepairReplacementDecision(
            currency=currency_code(currency),
            repair_cost=repair,
            replacement_cost=replacement,
            repair_to_replacement_ratio_percent=ratio,
            initial_cost_difference=difference,
            recommendation=recommendation.value,
            explanation=explanation,
        )

    # ========================================================
    # LIFECYCLE COST ANALYSIS
    # ========================================================

    @staticmethod
    def calculate_lifecycle_cost(
        *,
        initial_replacement_cost: Any,
        annual_operating_cost: Any,
        annual_maintenance_cost: Any,
        annual_energy_cost: Any,
        analysis_years: int,
        discount_rate_percent: Any = 8,
        annual_cost_growth_percent: Any = 3,
        end_of_life_disposal_cost: Any = 0,
        residual_value: Any = 0,
        currency: str = "INR",
    ) -> ReplacementLifecycleResult:

        years = validate_positive_int(
            analysis_years,
            "analysis_years",
            maximum=50,
        )

        initial = to_decimal(
            initial_replacement_cost,
            "initial_replacement_cost",
        )

        operating = to_decimal(
            annual_operating_cost,
            "annual_operating_cost",
        )

        maintenance = to_decimal(
            annual_maintenance_cost,
            "annual_maintenance_cost",
        )

        energy = to_decimal(
            annual_energy_cost,
            "annual_energy_cost",
        )

        discount_rate = percentage(
            discount_rate_percent,
            "discount_rate_percent",
        )

        growth_rate = percentage(
            annual_cost_growth_percent,
            "annual_cost_growth_percent",
        )

        disposal = to_decimal(
            end_of_life_disposal_cost,
            "end_of_life_disposal_cost",
        )

        residual = to_decimal(
            residual_value,
            "residual_value",
        )

        discount_factor_base = (
            ONE + discount_rate / HUNDRED
        )

        growth_factor_base = (
            ONE + growth_rate / HUNDRED
        )

        annual_base_cost = (
            operating + maintenance + energy
        )

        operating_npv = ZERO
        annual_cash_flows: list[dict[str, Any]] = []

        for year in range(1, years + 1):
            annual_cost = (
                annual_base_cost
                * growth_factor_base ** (year - 1)
            )

            present_value = (
                annual_cost
                / discount_factor_base ** year
            )

            operating_npv += present_value

            annual_cash_flows.append({
                "year": year,
                "nominal_operating_cost": str(
                    money(annual_cost)
                ),
                "discounted_operating_cost": str(
                    money(present_value)
                ),
            })

        terminal_net_cost = disposal - residual

        terminal_npv = (
            terminal_net_cost
            / discount_factor_base ** years
        )

        lifecycle_npv = (
            initial
            + operating_npv
            + terminal_npv
        )

        return ReplacementLifecycleResult(
            currency=currency_code(currency),
            analysis_years=years,
            discount_rate_percent=discount_rate,
            initial_replacement_cost=money(initial),
            annual_operating_cost=money(operating),
            annual_maintenance_cost=money(maintenance),
            annual_energy_cost=money(energy),
            annual_cost_growth_percent=growth_rate,
            end_of_life_disposal_cost=money(disposal),
            residual_value=money(residual),
            operating_cost_npv=money(operating_npv),
            terminal_cost_npv=money(terminal_npv),
            total_lifecycle_cost_npv=money(
                lifecycle_npv
            ),
            annual_cash_flows=annual_cash_flows,
        )

    # ========================================================
    # COST VARIANCE
    # ========================================================

    @staticmethod
    def calculate_cost_variance(
        *,
        replacement_cost_id: str,
        estimated_cost: Any,
        actual_cost: Any,
        currency: str = "INR",
    ) -> ReplacementCostVariance:

        estimated = to_decimal(
            estimated_cost,
            "estimated_cost",
        )

        actual = to_decimal(
            actual_cost,
            "actual_cost",
        )

        variance = actual - estimated

        variance_percent: Optional[Decimal]

        if estimated > ZERO:
            variance_percent = (
                variance
                / estimated
                * HUNDRED
            ).quantize(
                PERCENT_QUANT,
                rounding=ROUND_HALF_UP,
            )
        else:
            variance_percent = None

        return ReplacementCostVariance(
            replacement_cost_id=required_text(
                replacement_cost_id,
                "replacement_cost_id",
            ),
            currency=currency_code(currency),
            estimated_cost=money(estimated),
            actual_cost=money(actual),
            variance=money(variance),
            variance_percent=variance_percent,
            over_budget=actual > estimated,
        )

    # ========================================================
    # AGGREGATE REPLACEMENT COSTS
    # ========================================================

    @staticmethod
    def aggregate_replacement_costs(
        estimates: Sequence[ReplacementCostEstimate],
    ) -> dict[str, Any]:

        if not estimates:
            return {
                "total_replacements": 0,
                "currency": None,
                "total_net_replacement_cost": "0.00",
                "total_economic_impact": "0.00",
                "average_replacement_cost": "0.00",
            }

        currencies = {
            estimate.currency
            for estimate in estimates
        }

        if len(currencies) != 1:
            raise ReplacementCostValidationError(
                "Cannot aggregate different currencies"
            )

        identifiers = [
            estimate.replacement_cost_id
            for estimate in estimates
        ]

        if len(identifiers) != len(set(identifiers)):
            raise ReplacementCostValidationError(
                "Duplicate replacement estimates detected"
            )

        net_total = sum(
            (
                estimate.breakdown.net_replacement_cost
                for estimate in estimates
            ),
            ZERO,
        )

        economic_total = sum(
            (
                estimate.breakdown.total_economic_impact
                for estimate in estimates
            ),
            ZERO,
        )

        average = net_total / Decimal(
            len(estimates)
        )

        return {
            "total_replacements": len(estimates),
            "currency": next(iter(currencies)),
            "total_net_replacement_cost": str(
                money(net_total)
            ),
            "total_economic_impact": str(
                money(economic_total)
            ),
            "average_replacement_cost": str(
                money(average)
            ),
        }

    # ========================================================
    # AUDIT
    # ========================================================

    async def _audit(
        self,
        *,
        replacement_cost_id: str,
        actor_id: str,
        action: str,
        details: Optional[Mapping[str, Any]] = None,
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
                    "entity_type": "replacement_cost",
                    "entity_id": replacement_cost_id,
                    "actor_id": actor_id,
                    "action": action,
                    "details": dict(details or {}),
                    "timestamp": utc_now().isoformat(),
                },
            )

        except Exception:
            logger.exception(
                "Replacement cost audit failed: %s",
                replacement_cost_id,
            )

    # ========================================================
    # SAVE ESTIMATE
    # ========================================================

    async def save_replacement_estimate(
        self,
        estimate: ReplacementCostEstimate,
    ) -> Any:

        if self.repository is None:
            raise ReplacementCostRepositoryError(
                "Replacement cost repository not configured"
            )

        if not isinstance(
            estimate,
            ReplacementCostEstimate,
        ):
            raise ReplacementCostValidationError(
                "Expected ReplacementCostEstimate"
            )

        saved = await invoke(
            self.repository.create_replacement_cost,
            estimate.to_dict(),
        )

        await self._audit(
            replacement_cost_id=(
                estimate.replacement_cost_id
            ),
            actor_id=estimate.created_by,
            action="replacement_cost_estimated",
            details={
                "asset_id": estimate.asset_id,
                "currency": estimate.currency,
                "net_replacement_cost": str(
                    estimate.breakdown.net_replacement_cost
                ),
            },
        )

        return saved

    # ========================================================
    # GET ESTIMATE
    # ========================================================

    async def get_replacement_estimate(
        self,
        replacement_cost_id: str,
    ) -> Any:

        if self.repository is None:
            raise ReplacementCostRepositoryError(
                "Replacement cost repository not configured"
            )

        identifier = required_text(
            replacement_cost_id,
            "replacement_cost_id",
        )

        record = await invoke(
            self.repository.get_replacement_cost,
            identifier,
        )

        if record is None:
            raise ReplacementCostNotFoundError(
                f"Replacement estimate not found: {identifier}"
            )

        return record

    # ========================================================
    # LIST ESTIMATES
    # ========================================================

    async def list_replacement_estimates(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        replacement_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Any:

        if self.repository is None:
            raise ReplacementCostRepositoryError(
                "Replacement cost repository not configured"
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

        if replacement_type is not None:
            filters["replacement_type"] = parse_enum(
                ReplacementType,
                replacement_type,
                "replacement_type",
            ).value

        if status is not None:
            filters["status"] = parse_enum(
                ReplacementStatus,
                status,
                "status",
            ).value

        return await invoke(
            self.repository.list_replacement_costs,
            filters,
        )


# ============================================================
# IN-MEMORY REPOSITORY
# ============================================================

class InMemoryReplacementCostRepository:
    """
    Development-only repository.

    Not durable or concurrency-safe.
    """

    def __init__(self):
        self.records: dict[str, dict[str, Any]] = {}

    async def create_replacement_cost(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        record = copy.deepcopy(data)

        identifier = record["replacement_cost_id"]

        if identifier in self.records:
            raise ReplacementCostRepositoryError(
                "Duplicate replacement cost ID"
            )

        self.records[identifier] = record

        return copy.deepcopy(record)

    async def get_replacement_cost(
        self,
        replacement_cost_id: str,
    ) -> Optional[dict[str, Any]]:

        record = self.records.get(
            replacement_cost_id
        )

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def list_replacement_costs(
        self,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:

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

        repository = (
            InMemoryReplacementCostRepository()
        )

        service = ReplacementCostService(
            replacement_cost_repository=repository,
            default_currency="INR",
        )

        estimate = service.estimate_replacement_cost(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_id="MACHINE-001",
            machine_id="MOTOR-001",
            description=(
                "Replace aging industrial motor "
                "with a high-efficiency motor"
            ),
            replacement_type="machine",
            priority="high",
            scenario="base",
            replacement_items=[
                {
                    "item_id": "MOTOR-NEW-001",
                    "name": "High-efficiency industrial motor",
                    "quantity": 1,
                    "unit_cost": 120000,
                    "supplier_id": "SUPPLIER-001",
                },
                {
                    "item_id": "CONTROL-001",
                    "name": "Motor control module",
                    "quantity": 1,
                    "unit_cost": 18000,
                },
            ],
            shipping_cost=5000,
            installation_cost=10000,
            commissioning_cost=6000,
            training_cost=2000,
            decommissioning_cost=3000,
            disposal_cost=1500,
            integration_cost=4000,
            permits_cost=1000,
            other_cost=1000,
            contingency_percent=10,
            discount_percent=5,
            tax_percent=18,
            salvage_credit=8000,
            trade_in_credit=0,
            downtime_cost=10000,
            production_loss_cost=15000,
            estimated_duration_hours=12,
            expected_useful_life_years=12,
            created_by="ENGINEER-001",
        )

        print("\nREPLACEMENT COST ESTIMATE")

        print(
            json.dumps(
                estimate.to_dict(),
                indent=2,
            )
        )

        await service.save_replacement_estimate(
            estimate
        )

        comparison = (
            service.compare_repair_vs_replacement(
                repair_cost=95000,
                replacement_cost=(
                    estimate.breakdown.net_replacement_cost
                ),
                currency="INR",
            )
        )

        print("\nREPAIR VS REPLACEMENT")

        print(
            json.dumps(
                comparison.to_dict(),
                indent=2,
            )
        )

        lifecycle = service.calculate_lifecycle_cost(
            initial_replacement_cost=(
                estimate.breakdown.net_replacement_cost
            ),
            annual_operating_cost=20000,
            annual_maintenance_cost=12000,
            annual_energy_cost=35000,
            analysis_years=10,
            discount_rate_percent=8,
            annual_cost_growth_percent=3,
            end_of_life_disposal_cost=5000,
            residual_value=15000,
            currency="INR",
        )

        print("\nLIFECYCLE COST ANALYSIS")

        print(
            json.dumps(
                lifecycle.to_dict(),
                indent=2,
            )
        )

        variance = service.calculate_cost_variance(
            replacement_cost_id=(
                estimate.replacement_cost_id
            ),
            estimated_cost=(
                estimate.breakdown.net_replacement_cost
            ),
            actual_cost=185000,
            currency="INR",
        )

        print("\nCOST VARIANCE")

        print(
            json.dumps(
                variance.to_dict(),
                indent=2,
            )
        )

        scenarios = (
            service.compare_replacement_scenarios(
                company_id="COMPANY-001",
                site_id="SITE-001",
                asset_id="MACHINE-001",
                description="Industrial motor replacement",
                replacement_items=[
                    {
                        "item_id": "MOTOR-NEW-001",
                        "name": "Industrial motor",
                        "quantity": 1,
                        "unit_cost": 120000,
                    }
                ],
                installation_cost=10000,
                contingency_percent=10,
                tax_percent=18,
            )
        )

        print("\nSCENARIO COMPARISON")

        print(
            json.dumps(
                scenarios,
                indent=2,
            )
        )

        records = await service.list_replacement_estimates(
            asset_id="MACHINE-001"
        )

        print("\nSAVED REPLACEMENT ESTIMATES")
        print(f"Total records: {len(records)}")

    asyncio.run(main())
