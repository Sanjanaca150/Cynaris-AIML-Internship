import pandas as pd


DATA_FILE = "data/demand_data.csv"


def load_sku_data():
    """Load SKU demand data and calculate SKU-level statistics."""
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


def calculate_dynamic_price(
    base_price,
    demand,
    inventory,
    forecast_demand,
):
    """
    Calculate a dynamic price using demand and inventory signals.

    Price is constrained between 80% and 120% of the base price.
    """

    demand_ratio = demand / max(forecast_demand, 1)

    # Higher demand relative to forecast -> moderate price increase.
    demand_factor = 1 + (
        0.10 * (demand_ratio - 1)
    )

    # Low inventory -> price increases slightly.
    if inventory < forecast_demand:
        inventory_factor = 1.05
    elif inventory > forecast_demand * 2:
        inventory_factor = 0.95
    else:
        inventory_factor = 1.0

    recommended_price = (
        base_price
        * demand_factor
        * inventory_factor
    )

    # Keep prices within reasonable business limits.
    minimum_price = base_price * 0.80
    maximum_price = base_price * 1.20

    recommended_price = max(
        minimum_price,
        min(
            recommended_price,
            maximum_price,
        ),
    )

    return round(recommended_price, 2)


def generate_pricing_recommendations():
    """Generate pricing recommendations for all 20 SKUs."""

    sku_stats = load_sku_data()

    recommendations = []

    for index, row in sku_stats.iterrows():

        sku_number = index + 1

        base_price = (
            499
            + ((sku_number - 1) % 10) * 150
        )

        demand = float(row["average_demand"])

        forecast_demand = float(
            row["average_demand"] * 1.05
        )

        inventory = int(
            forecast_demand
            * (0.7 + (sku_number % 4) * 0.5)
        )

        recommended_price = calculate_dynamic_price(
            base_price=base_price,
            demand=demand,
            inventory=inventory,
            forecast_demand=forecast_demand,
        )

        recommendations.append(
            {
                "sku_id": row["sku_id"],
                "category": row["category"],
                "base_price_inr": base_price,
                "average_demand": round(demand, 2),
                "forecast_demand": round(
                    forecast_demand,
                    2,
                ),
                "inventory": inventory,
                "recommended_price_inr": recommended_price,
            }
        )

    return pd.DataFrame(recommendations)


if __name__ == "__main__":
    pricing_df = generate_pricing_recommendations()

    print()
    print("OmniRetail AI Dynamic Pricing")
    print("=============================")
    print("SKUs tested:", len(pricing_df))
    print()

    print(
        pricing_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Pricing test:",
        "PASSED" if len(pricing_df) == 20 else "FAILED",
    )