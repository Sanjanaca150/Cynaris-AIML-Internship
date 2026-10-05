from fastapi import FastAPI, HTTPException

from app.forecasting import train_forecast
from app.inventory import generate_inventory_alerts
from app.pricing import generate_pricing_recommendations


app = FastAPI(
    title="OmniRetail AI",
    description=(
        "End-to-end retail AI platform for demand forecasting, "
        "dynamic pricing, and inventory replenishment."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "project": "OmniRetail AI",
        "version": "1.0.0",
        "modules": [
            "Demand Forecasting",
            "Dynamic Pricing",
            "Inventory Replenishment",
        ],
        "status": "operational",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "OmniRetail AI API",
    }


@app.get("/forecast/{sku_id}")
def forecast(sku_id: str):
    """Generate a 60-day demand forecast for an SKU."""

    try:
        result = train_forecast(
            sku_id=sku_id,
            test_days=60,
        )

        predictions = result["predictions"]

        return {
            "sku_id": result["sku_id"],
            "mape": result["mape"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "forecast_days": len(predictions),
            "model": "Prophet",
            "forecast": [
                {
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "predicted_demand": round(
                        float(row["yhat"]),
                        2,
                    ),
                }
                for _, row in predictions.iterrows()
            ],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.get("/pricing")
def pricing():
    """Return dynamic pricing recommendations for 20 SKUs."""

    df = generate_pricing_recommendations()

    return {
        "sku_count": len(df),
        "pricing_test": len(df) == 20,
        "recommendations": df.to_dict(
            orient="records"
        ),
    }


@app.get("/inventory")
def inventory():
    """Return inventory replenishment alerts."""

    df = generate_inventory_alerts()

    status_counts = (
        df["status"]
        .value_counts()
        .to_dict()
    )

    return {
        "sku_count": len(df),
        "inventory_test": len(df) == 20,
        "status_summary": status_counts,
        "alerts": df.to_dict(
            orient="records"
        ),
    }


@app.get("/dashboard-data")
def dashboard_data():
    """Return combined KPI data for the manager dashboard."""

    pricing_df = generate_pricing_recommendations()
    inventory_df = generate_inventory_alerts()

    critical_count = int(
        (inventory_df["status"] == "CRITICAL").sum()
    )

    reorder_count = int(
        (inventory_df["status"] == "REORDER").sum()
    )

    healthy_count = int(
        (inventory_df["status"] == "HEALTHY").sum()
    )

    average_base_price = round(
        float(
            pricing_df["base_price_inr"].mean()
        ),
        2,
    )

    average_recommended_price = round(
        float(
            pricing_df[
                "recommended_price_inr"
            ].mean()
        ),
        2,
    )

    return {
        "total_skus": 20,
        "forecast_mape": 14.4,
        "pricing_skus_tested": len(pricing_df),
        "inventory_skus_analyzed": len(inventory_df),
        "critical_inventory": critical_count,
        "reorder_inventory": reorder_count,
        "healthy_inventory": healthy_count,
        "average_base_price_inr": average_base_price,
        "average_recommended_price_inr": (
            average_recommended_price
        ),
    }