import os

import pandas as pd
from prophet import Prophet


DATA_FILE = "data/demand_data.csv"
MODEL_DIR = "models"


def load_demand_data():
    """Load the retail demand dataset."""
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])

    return df


def calculate_mape(actual, predicted):
    """Calculate MAPE while ignoring zero-demand days."""
    actual = pd.Series(actual).astype(float)
    predicted = pd.Series(predicted).astype(float)

    non_zero = actual != 0

    if non_zero.sum() == 0:
        return 0.0

    percentage_errors = (
        abs(
            (actual[non_zero] - predicted[non_zero])
            / actual[non_zero]
        )
        * 100
    )

    return float(percentage_errors.mean())


def train_forecast(sku_id, test_days=60):
    """Train and evaluate a Prophet demand forecasting model."""

    df = load_demand_data()

    sku_df = df[df["sku_id"] == sku_id].copy()
    sku_df = sku_df.sort_values("date")

    if sku_df.empty:
        raise ValueError(f"SKU not found: {sku_id}")

    if len(sku_df) <= test_days:
        raise ValueError(
            "Not enough historical data for the requested test period."
        )

    # Prophet requires columns named ds and y.
    prophet_df = sku_df[
        ["date", "demand"]
    ].rename(
        columns={
            "date": "ds",
            "demand": "y",
        }
    )

    # Chronological train/test split.
    train_df = prophet_df.iloc[:-test_days].copy()
    test_df = prophet_df.iloc[-test_days:].copy()

    # Tuned Prophet configuration.
    model = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=5.0,
        interval_width=0.95,
    )

    # Add a monthly pattern because retail demand can have
    # short-term monthly variation.
    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=3,
    )

    model.fit(train_df)

    # Generate the required 60-day forecast.
    future = model.make_future_dataframe(
        periods=test_days,
        freq="D",
    )

    forecast = model.predict(future)

    predictions = forecast[
        ["ds", "yhat"]
    ].tail(test_days).copy()

    predictions["y"] = test_df["y"].values

    # Demand cannot be negative.
    predictions["yhat"] = predictions["yhat"].clip(
        lower=0
    )

    # Calculate actual MAPE.
    mape = calculate_mape(
        predictions["y"],
        predictions["yhat"],
    )

    # Save the trained Prophet model.
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        f"prophet_{sku_id}.json",
    )

    from prophet.serialize import model_to_json

    with open(
        model_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(model_to_json(model))

    return {
        "sku_id": sku_id,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "mape": round(mape, 2),
        "predictions": predictions,
        "model_path": model_path,
    }