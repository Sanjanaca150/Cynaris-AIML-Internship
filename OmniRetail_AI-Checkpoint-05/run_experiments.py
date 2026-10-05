import mlflow
import pandas as pd
from prophet import Prophet


mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment_name = "OmniRetail_Checkpoint_5_Forecasting"

mlflow.set_experiment(experiment_name)

df = pd.read_csv("data/demand_data.csv")
df["date"] = pd.to_datetime(df["date"])


configs = [
    {
        "name": "Forecast_SKU_001",
        "sku": "SKU-001",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "Forecast_SKU_002",
        "sku": "SKU-002",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "Forecast_SKU_003",
        "sku": "SKU-003",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "Forecast_SKU_004",
        "sku": "SKU-004",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "Forecast_SKU_005",
        "sku": "SKU-005",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "Forecast_SKU_006",
        "sku": "SKU-006",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "Forecast_SKU_007",
        "sku": "SKU-007",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "Forecast_SKU_008",
        "sku": "SKU-008",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "Forecast_SKU_009",
        "sku": "SKU-009",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "Forecast_SKU_010",
        "sku": "SKU-010",
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
    },
]


def calculate_mape(actual, predicted):
    actual = pd.Series(actual)
    predicted = pd.Series(predicted)

    mask = actual != 0

    return (
        abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        ).mean()
        * 100
    )


for config in configs:

    sku_df = df[
        df["sku_id"] == config["sku"]
    ].sort_values("date").copy()

    sku_df = sku_df[
        ["date", "demand"]
    ].rename(
        columns={
            "date": "ds",
            "demand": "y",
        }
    )

    train_df = sku_df.iloc[:-60]
    test_df = sku_df.iloc[-60:]

    model = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=(
            config["changepoint_prior_scale"]
        ),
        seasonality_prior_scale=(
            config["seasonality_prior_scale"]
        ),
        interval_width=0.95,
    )

    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=3,
    )

    model.fit(train_df)

    future = model.make_future_dataframe(
        periods=60,
        freq="D",
    )

    forecast = model.predict(future)

    predictions = forecast.tail(60)["yhat"].clip(
        lower=0
    )

    mape = calculate_mape(
        test_df["y"].values,
        predictions.values,
    )

    with mlflow.start_run(
        run_name=config["name"]
    ):

        mlflow.log_param(
            "model",
            "Prophet",
        )

        mlflow.log_param(
            "sku_id",
            config["sku"],
        )

        mlflow.log_param(
            "changepoint_prior_scale",
            config["changepoint_prior_scale"],
        )

        mlflow.log_param(
            "seasonality_prior_scale",
            config["seasonality_prior_scale"],
        )

        mlflow.log_param(
            "test_days",
            60,
        )

        mlflow.log_metric(
            "MAPE",
            round(float(mape), 4),
        )

        mlflow.log_metric(
            "train_rows",
            len(train_df),
        )

        mlflow.log_metric(
            "test_rows",
            len(test_df),
        )

    print(
        f'{config["name"]}: '
        f'MAPE = {mape:.2f}%'
    )


print()
print("MLflow experiment tracking complete.")
print("Total experiments logged: 10")