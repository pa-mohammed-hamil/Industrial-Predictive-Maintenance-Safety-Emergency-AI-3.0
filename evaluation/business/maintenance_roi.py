from app.cost_engine.downtime_savings import (
    DowntimeSavingsInput,
    DowntimeType,
    downtime_savings_engine,
)


result = downtime_savings_engine.calculate(
    DowntimeSavingsInput(
        asset_id="MOTOR-001",
        site_id="FACTORY-01",

        downtime_type=(
            DowntimeType.FAILURE
        ),

        # Without predictive maintenance:
        downtime_before_hours=18,

        # After early detection + repair:
        downtime_after_hours=4,

        downtime_cost_per_hour=5_000,

        production_units_per_hour=120,
        contribution_margin_per_unit=75,

        idle_workers=8,
        labor_cost_per_worker_hour=350,

        idle_energy_kw=25,
        electricity_cost_per_kwh=8,

        scrap_cost_per_downtime_hour=2_500,

        penalty_cost_per_hour=1_500,

        overtime_hours_before=10,
        overtime_hours_after=2,
        overtime_cost_per_hour=1_200,

        rental_cost_per_hour=500,
        logistics_cost_per_hour=300,

        intervention_cost=40_000,

        currency="INR",

        prediction_id="PRED-001",
        work_order_id="WO-001",
    )
)

print(result.to_dict())