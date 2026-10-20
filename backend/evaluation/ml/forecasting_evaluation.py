from app.evaluation.forecasting_evaluation import (
    ForecastEvaluationSample,
    ForecastTarget,
    forecasting_evaluator,
)


samples = [
    ForecastEvaluationSample(
        sample_id="FC-001",
        asset_id="MOTOR-001",
        sensor_id="TEMP-001",
        target=ForecastTarget.TEMPERATURE,

        actual_value=82.5,
        predicted_value=80.8,

        lower_bound=77.0,
        upper_bound=85.0,

        forecast_horizon=1,
        timestamp_ms=1000,

        model_name="time-series-transformer",
        model_version="2.0",
    ),

    ForecastEvaluationSample(
        sample_id="FC-002",
        asset_id="MOTOR-001",
        sensor_id="TEMP-001",
        target=ForecastTarget.TEMPERATURE,

        actual_value=86.2,
        predicted_value=84.9,

        lower_bound=81.0,
        upper_bound=89.0,

        forecast_horizon=2,
        timestamp_ms=2000,

        model_name="time-series-transformer",
        model_version="2.0",
    ),

    ForecastEvaluationSample(
        sample_id="FC-003",
        asset_id="MOTOR-001",
        sensor_id="TEMP-001",
        target=ForecastTarget.TEMPERATURE,

        actual_value=91.0,
        predicted_value=88.6,

        lower_bound=84.0,
        upper_bound=93.0,

        forecast_horizon=3,
        timestamp_ms=3000,

        model_name="time-series-transformer",
        model_version="2.0",
    ),
]


result = forecasting_evaluator.evaluate(
    samples
)

print(result.to_dict())