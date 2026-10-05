import os

import numpy as np
import pandas as pd


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

INPUT_FILE = "data/aiml_ecommerce_clickstream.csv"
OUTPUT_FILE = "data/demand_data.csv"


def build_demand_dataset():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    # Aggregate the supplied clickstream into daily retail activity.
    daily = (
        df.groupby("date")
        .agg(
            purchases=("purchase_made", "sum"),
            sessions=("session_id", "count"),
            revenue=("order_value_inr", "sum"),
        )
        .reset_index()
    )

    # Create a continuous daily timeline.
    date_range = pd.date_range(
        daily["date"].min(),
        daily["date"].max(),
        freq="D",
    )

    daily = (
        daily.set_index("date")
        .reindex(date_range)
        .fillna(0)
        .rename_axis("date")
        .reset_index()
    )

    sku_categories = [
        "Electronics",
        "Grocery",
        "Beauty",
        "Fashion",
        "Home",
    ]

    records = []

    for sku_num in range(1, 21):
        sku_id = f"SKU-{sku_num:03d}"

        category = sku_categories[
            (sku_num - 1) % len(sku_categories)
        ]

        # Each SKU has a different demand scale.
        sku_scale = 1.0 + ((sku_num - 1) % 5) * 0.18

        # Small SKU-specific trend.
        sku_trend = 0.0005 * sku_num

        for day_index, row in daily.iterrows():

            # Strong weekly retail seasonality.
            weekly = 1 + 0.12 * np.sin(
                2 * np.pi * day_index / 7
            )

            # Smooth yearly/seasonal pattern.
            yearly = 1 + 0.06 * np.sin(
                2 * np.pi * day_index / 365.25
            )

            # Long-term trend.
            trend = 1 + sku_trend * day_index

            # Clickstream-derived demand signal.
            activity = (
                2.0
                + row["purchases"] * 0.45
                + row["sessions"] * 0.015
            )

            demand = (
                activity
                * sku_scale
                * weekly
                * yearly
                * trend
            )

            # Very small noise keeps the data realistic while
            # preserving a strong predictable signal.
            noise = np.random.normal(
                loc=0,
                scale=max(0.15, demand * 0.025),
            )

            final_demand = max(
                1,
                round(demand + noise),
            )

            records.append(
                {
                    "date": row["date"],
                    "sku_id": sku_id,
                    "category": category,
                    "demand": int(final_demand),
                }
            )

    demand_df = pd.DataFrame(records)

    os.makedirs("data", exist_ok=True)

    demand_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("Demand dataset created successfully.")
    print("Rows:", len(demand_df))
    print("SKUs:", demand_df["sku_id"].nunique())
    print(
        "Date range:",
        demand_df["date"].min(),
        "to",
        demand_df["date"].max(),
    )
    print("Saved to:", OUTPUT_FILE)
    print()
    print(demand_df.head(10))


if __name__ == "__main__":
    build_demand_dataset()