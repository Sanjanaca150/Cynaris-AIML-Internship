import pandas as pd


DATA_FILE = "data/demand_data.csv"


def load_inventory_data():
    """Load demand data and calculate SKU-level inventory signals."""

    df = pd.read_csv(DATA_FILE)

    sku_stats = (
        df.groupby(["sku_id", "category"])
        .agg(
            average_demand=("demand", "mean"),
            max_demand=("demand", "max"),
        )
        .reset_index()
    )

    return sku_stats


def calculate_inventory_status(
    inventory,
    forecast_demand,
):
    """Classify inventory health."""

    if inventory < forecast_demand * 0.5:
        return "CRITICAL"

    if inventory < forecast_demand:
        return "REORDER"

    return "HEALTHY"


def calculate_reorder_quantity(
    inventory,
    forecast_demand,
):
    """Calculate quantity required to cover forecast demand."""

    target_stock = forecast_demand * 2

    reorder_quantity = max(
        0,
        target_stock - inventory,
    )

    return round(reorder_quantity)


def generate_inventory_alerts():
    """Generate replenishment recommendations for all 20 SKUs."""

    sku_stats = load_inventory_data()

    alerts = []

    for index, row in sku_stats.iterrows():

        sku_number = index + 1

        forecast_demand = float(
            row["average_demand"] * 1.05
        )

        # Reproducible demo inventory levels.
        inventory = int(
            forecast_demand
            * (0.4 + (sku_number % 5) * 0.45)
        )

        status = calculate_inventory_status(
            inventory=inventory,
            forecast_demand=forecast_demand,
        )

        reorder_quantity = calculate_reorder_quantity(
            inventory=inventory,
            forecast_demand=forecast_demand,
        )

        alerts.append(
            {
                "sku_id": row["sku_id"],
                "category": row["category"],
                "inventory": inventory,
                "forecast_demand": round(
                    forecast_demand,
                    2,
                ),
                "status": status,
                "reorder_quantity": reorder_quantity,
            }
        )

    return pd.DataFrame(alerts)


if __name__ == "__main__":
    inventory_df = generate_inventory_alerts()

    print()
    print("OmniRetail AI Inventory Replenishment")
    print("======================================")
    print("SKUs analyzed:", len(inventory_df))
    print()

    print(
        inventory_df.to_string(
            index=False
        )
    )

    print()

    print("Status Summary:")
    print(
        inventory_df["status"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Inventory test:",
        "PASSED"
        if len(inventory_df) == 20
        else "FAILED",
    )